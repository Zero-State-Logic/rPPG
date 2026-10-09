"""ProPOS: Projection of Rotated Orthogonal bases in POS.

Faithful reproduction of:
    Rao, Fang, Zhao, Bai, "Measurement of heart rate from long-distance videos
    via projection of rotated orthogonal bases in POS (ProPOS)",
    Medical Engineering & Physics 138 (2025) 104326, Algorithm 1.

Pipeline (per segment):
    1. raw RGB trace -> global per-channel z-normalization                (Eq. 2)
    2. rotate the POS projection plane into a bank of candidate bases      (Eqs. 3-9)
    3. project the trace onto every basis -> candidate pulses             (Eqs. 10-11)
    4. select the candidate with the highest SNR, using the heart rate from a
       canonical POS pass as the reference frequency                      (Eqs. 12-14)
    5. refine the chosen pulse with a two-harmonic sinusoidal fit          (Eqs. 15-16)

Documented deviations (for reproducibility):
    * The paper equates the central candidate p~(0,0) with "the signal obtained
      by POS" and uses its HR as the SNR reference. We compute the reference HR
      from canonical overlap-add POS (pos.pos_pulse) instead -- the robust
      reading of that statement, decoupled from the exact R1/R2 constants.
    * Sampling follows the Section 4.1 implementation details
      (gamma in {5,10,15,20,25} deg, theta in {36,...,360} deg -> 50 bases),
      not the wider ranges in the Step 3/4 prose.

Pure functions over NumPy arrays. No I/O.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ...hr.metrics import estimate_hr_fft, snr_db
from .pos import _POS_PROJECTION, _as_rgb_trace, pos_pulse

_EPS = 1e-9
# Fixed alignment rotations (Eqs. 3-4): bring a projection axis onto skin tone [1,1,1].
_ALPHA = np.pi / 4.0
_BETA = float(np.arctan(np.sqrt(2.0)))
_DEFAULT_THETAS_DEG: tuple[float, ...] = tuple(float(d) for d in range(36, 361, 36))
_DEFAULT_GAMMAS_DEG: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_projection_bank(
    thetas_deg: tuple[float, ...] = _DEFAULT_THETAS_DEG,
    gammas_deg: tuple[float, ...] = _DEFAULT_GAMMAS_DEG,
) -> np.ndarray:
    """Return stacked projection matrices P~(theta, gamma), shape (K, 2, 3)."""
    base = _rx(_ALPHA) @ _ry(_BETA)                       # R1 R2 (Eqs. 3-4)
    banks = []
    for theta in thetas_deg:
        rz = _rz(np.deg2rad(theta))                       # R3 (Eq. 5)
        for gamma in gammas_deg:
            r = base @ rz @ _rx(np.deg2rad(gamma))        # R = R1 R2 R3 R4 (Eq. 7)
            banks.append(_POS_PROJECTION @ r)             # P~ = P R (Eq. 8)
    return np.asarray(banks, dtype=np.float64)


def _normalize_trace(rgb: np.ndarray) -> np.ndarray:
    """Global per-channel z-normalization (Eq. 2)."""
    mu = rgb.mean(axis=1, keepdims=True)
    sd = rgb.std(axis=1, keepdims=True)
    sd[sd < _EPS] = _EPS
    return (rgb - mu) / sd


def _two_harmonic_fit(pulse: np.ndarray, fps: float, hr_hz: float) -> np.ndarray:
    """Refine with a1 sin(wt+b1) + a2 sin(2wt+b2) (Eqs. 15-16)."""
    n = pulse.size
    t = np.arange(n, dtype=np.float64) / fps
    amp = float(pulse.std() * np.sqrt(2.0))
    w0 = 2.0 * np.pi * (hr_hz if np.isfinite(hr_hz) and hr_hz > 0 else 1.2)

    def model(tt: np.ndarray, w: float, a1: float, a2: float,
              b1: float, b2: float) -> np.ndarray:
        return a1 * np.sin(w * tt + b1) + a2 * np.sin(2.0 * w * tt + b2)

    try:
        popt, _ = curve_fit(model, t, pulse,
                            p0=[w0, amp, 0.3 * amp, 0.0, 0.0], maxfev=600)
        return model(t, *popt)
    except (RuntimeError, ValueError):
        return pulse


def propos_pulse(
    rgb: np.ndarray,
    fps: float,
    thetas_deg: tuple[float, ...] = _DEFAULT_THETAS_DEG,
    gammas_deg: tuple[float, ...] = _DEFAULT_GAMMAS_DEG,
    do_fit: bool = True,
) -> np.ndarray:
    """Extract the ProPOS pulse from an RGB trace (shape (3, T)); returns (T,)."""
    trace = _as_rgb_trace(rgb)
    sn = _normalize_trace(trace)                          # (3, T)

    bank = build_projection_bank(thetas_deg, gammas_deg)  # (K, 2, 3)
    projected = np.einsum("kij,jt->kit", bank, sn)        # (K, 2, T)
    s1 = projected[:, 0, :]
    s2 = projected[:, 1, :]
    std2 = s2.std(axis=1)
    std2[std2 < _EPS] = _EPS
    alpha = s1.std(axis=1) / std2                         # (K,)
    candidates = s1 + alpha[:, None] * s2                 # (K, T)  (Eq. 11)

    # Reference HR from canonical POS (Eq. 12).
    ref_hr_bpm = estimate_hr_fft(pos_pulse(trace, fps), fps)
    ref_hr_hz = ref_hr_bpm / 60.0 if np.isfinite(ref_hr_bpm) else float("nan")

    # SNR-based candidate selection (Eqs. 13-14).
    snrs = np.array([snr_db(candidates[k], fps, ref_hr_hz)
                     for k in range(candidates.shape[0])])
    best = int(np.nanargmax(snrs)) if np.any(np.isfinite(snrs)) else 0
    optimal = candidates[best]

    return _two_harmonic_fit(optimal, fps, ref_hr_hz) if do_fit else optimal
