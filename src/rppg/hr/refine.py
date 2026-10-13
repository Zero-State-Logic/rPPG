"""High-resolution heart-rate estimation: band-pass + sub-bin spectral peak + fusion.

The plain periodogram used in :mod:`rppg.hr.metrics` lands the HR peak on a coarse
frequency grid, so it cannot resolve a few-bpm difference on a short window. This
module closes that gap with three proven steps:

* **Band-pass** the pulse to the cardiac band (Butterworth, zero-phase) so
  out-of-band noise never competes with the real peak.
* **Sub-bin peak** location: a Hann window + heavy zero-padding + parabolic
  interpolation around the spectral maximum give ~0.3 bpm resolution.
* **SNR fusion**: each rPPG method (POS / CHROM / ProPOS) yields an ``(hr, snr)``;
  :func:`fuse_estimates` trusts the strongest-pulse estimate (SNR-weighted,
  outlier-guarded) so no single method's bad moment dominates.

Pure functions over 1-D NumPy arrays. No I/O.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

_EPS = 1e-9
_LO_HZ, _HI_HZ = 0.7, 4.0          # 42-240 bpm cardiac band
_PEAK_HALF_HZ = 0.15               # +/- band counted as "signal" for SNR


def bandpass(sig: np.ndarray, fps: float, lo: float = _LO_HZ,
             hi: float = _HI_HZ, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass; falls back to mean-removal if too short."""
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    nyq = 0.5 * fps
    lo_n, hi_n = max(1e-3, lo / nyq), min(0.99, hi / nyq)
    # filtfilt needs len(x) > its padlen (= 3 * filter length = 3*(2*order+1));
    # too-short windows (early provisional reading / low FPS) skip filtering.
    if hi_n <= lo_n or x.size <= 3 * (2 * order + 1):
        return x
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x)


def hr_peak_interp(sig: np.ndarray, fps: float, lo: float = _LO_HZ,
                   hi: float = _HI_HZ) -> tuple[float, float]:
    """Return (hr_bpm, snr_db) from a Hann-windowed, zero-padded, interpolated peak."""
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 8:
        return float("nan"), float("nan")
    nfft = int(2 ** np.ceil(np.log2(n * 8)))          # heavy zero-pad -> fine grid
    spec = np.abs(np.fft.rfft(x * np.hanning(n), n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band):
        return float("nan"), float("nan")
    band_idx = np.where(band)[0]
    k = int(band_idx[np.argmax(spec[band_idx])])

    delta = 0.0
    if 0 < k < spec.size - 1:                          # parabolic sub-bin refinement
        a0, b0, c0 = spec[k - 1], spec[k], spec[k + 1]
        denom = a0 - 2.0 * b0 + c0
        if abs(denom) > _EPS:
            delta = 0.5 * (a0 - c0) / denom
    f_peak = freqs[k] + delta * (freqs[1] - freqs[0])

    gate = (freqs >= f_peak - _PEAK_HALF_HZ) & (freqs <= f_peak + _PEAK_HALF_HZ)
    sig_power = float(spec[gate & band].sum())
    noise_power = float(spec[band].sum()) - sig_power
    snr = 10.0 * np.log10((sig_power + _EPS) / (noise_power + _EPS))
    return float(f_peak * 60.0), float(snr)


def fuse_estimates(estimates: list[tuple[float, float]]) -> tuple[float, float]:
    """Fuse ``(hr, snr)`` pairs: SNR-weighted mean of estimates near the best one.

    The highest-SNR estimate anchors the result; others within 10 bpm of it are
    merged with softmax-over-SNR weights, so agreeing methods reinforce each other
    while a lone outlier (e.g. a method that locked onto a harmonic) is dropped.
    """
    valid = [(hr, snr) for hr, snr in estimates
             if np.isfinite(hr) and np.isfinite(snr)]
    if not valid:
        return float("nan"), float("nan")
    anchor_hr, anchor_snr = max(valid, key=lambda e: e[1])
    near = [(hr, snr) for hr, snr in valid if abs(hr - anchor_hr) <= 10.0]
    snrs = np.array([s for _, s in near])
    weights = np.exp(snrs - snrs.max())
    weights /= weights.sum()
    hr = float(np.sum(weights * np.array([h for h, _ in near])))
    return hr, float(anchor_snr)
