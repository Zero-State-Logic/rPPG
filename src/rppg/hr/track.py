"""Causal spectral heart-rate tracking (online Viterbi over a 1-bpm grid).

Taking the tallest spectral peak of every window independently lets one bad window
(room-light flicker, a head turn, a harmonic) throw the reading anywhere in the band.
Heart rate, however, changes smoothly. :class:`SpectralTracker` keeps a running
max-product (Viterbi) score over a bpm grid: each new window's normalised spectrum is
the evidence, and a Gaussian random-walk prior limits how far the rate may move per
unit time. A brief outlier therefore cannot win, while a real, sustained change is
followed within a few seconds.

Live use differs from offline windows: the demo re-estimates every ~0.4 s over a 10 s
buffer, so consecutive spectra share most of their samples. To avoid counting the same
evidence many times, each update's log-evidence is **tempered** by ``dt / ref_dt`` and
the random-walk variance grows with ``dt``; a stream of 0.4 s updates then behaves like
the 2 s hop the parameters were tuned on (MCD-rPPG bench, D:\\rppg_eval\\code\\bench).

Pure NumPy, no I/O.
"""
from __future__ import annotations

import numpy as np

from .refine import bandpass

_LO_BPM, _HI_BPM, _STEP_BPM = 42.0, 180.0, 1.0   # 0.7-3.0 Hz: tracked band
_SIGMA_BPM = 4.0       # random-walk std of the HR per reference step (bpm); MCD dev CV
_REF_DT = 2.0          # s: reference step the sigma / evidence weight refer to
_FLOOR = 1e-2          # evidence floor relative to the window's peak (log stability)
_REFINE_BPM = 3.0      # sub-bin refinement searches the spectrum this close to the track
_NFFT = 8192
_EPS = 1e-12


def pulse_spectrum(pulse: np.ndarray, fps: float, grid: np.ndarray) -> np.ndarray:
    """Band-passed, Hann-windowed power of ``pulse`` sampled on ``grid`` (bpm).

    The band is the grid's own range; the result is non-negative and sums to 1. When
    the pulse carries no in-band power (a flat / constant input) it is all-NaN, which
    :meth:`SpectralTracker.update` treats as "no evidence" -- never as a rate.
    """
    grid = np.asarray(grid, dtype=np.float64)
    lo, hi = grid[0] / 60.0, grid[-1] / 60.0
    x = bandpass(np.asarray(pulse, dtype=np.float64), fps, lo, hi)
    x = (x - x.mean()) * np.hanning(x.size)
    power = np.abs(np.fft.rfft(x, n=max(_NFFT, x.size))) ** 2
    freqs = np.fft.rfftfreq(max(_NFFT, x.size), d=1.0 / fps)
    spec = np.interp(grid / 60.0, freqs, power)
    total = float(spec.sum())
    if not np.isfinite(total) or total <= _EPS or spec.max() - spec.min() <= _EPS * total:
        return np.full(grid.size, np.nan)
    return spec / total


class SpectralTracker:
    """Online Viterbi heart-rate tracker over normalised window spectra."""

    def __init__(self, lo_bpm: float = _LO_BPM, hi_bpm: float = _HI_BPM,
                 step_bpm: float = _STEP_BPM, sigma_bpm: float = _SIGMA_BPM,
                 ref_dt: float = _REF_DT, floor: float = _FLOOR) -> None:
        if not (0 < lo_bpm < hi_bpm) or step_bpm <= 0 or sigma_bpm <= 0 or ref_dt <= 0:
            raise ValueError("invalid tracker parameters")
        self.grid = np.arange(lo_bpm, hi_bpm + step_bpm / 2.0, step_bpm)
        self._d2 = (self.grid[:, None] - self.grid[None, :]) ** 2
        self.sigma_bpm = float(sigma_bpm)
        self._sigma2 = self.sigma_bpm ** 2
        self._ref_dt = float(ref_dt)
        self._floor = float(floor)
        self._alpha: np.ndarray | None = None
        self.hr: float = float("nan")

    def reset(self) -> None:
        """Forget the history (e.g. when the tracked face is lost)."""
        self._alpha = None
        self.hr = float("nan")

    def update(self, spec: np.ndarray, dt: float) -> float:
        """Fold one window spectrum (on :attr:`grid`) observed ``dt`` s after the last.

        Returns the tracked heart rate in bpm, refined to sub-bin precision on the
        current spectrum near the tracked bin.
        """
        spec = np.asarray(spec, dtype=np.float64)
        if spec.shape != self.grid.shape:
            raise ValueError(f"spectrum shape {spec.shape} != grid shape {self.grid.shape}")
        if not np.all(np.isfinite(spec)) or spec.max() - spec.min() <= _EPS:
            return self.hr                                   # no evidence: hold
        if self._alpha is not None and np.isfinite(dt) and dt <= 0:
            return self.hr               # same / out-of-order instant: never re-count it
        if not np.isfinite(dt):
            dt = self._ref_dt
        ratio = dt / self._ref_dt
        logp = min(1.0, ratio) * np.log(spec / spec.max() + self._floor)
        if self._alpha is None:
            alpha = logp
        else:
            trans = -self._d2 / (2.0 * self._sigma2 * max(ratio, 1e-3))
            alpha = logp + np.max(self._alpha[None, :] + trans, axis=1)
        self._alpha = alpha - alpha.max()
        self.hr = self._refine(spec, int(np.argmax(self._alpha)))
        return self.hr

    def _refine(self, spec: np.ndarray, k: int) -> float:
        """Parabolic peak of ``spec`` within ``_REFINE_BPM`` of grid bin ``k``."""
        step = float(self.grid[1] - self.grid[0]) if self.grid.size > 1 else 1.0
        reach = max(1, int(round(_REFINE_BPM / step)))
        lo, hi = max(0, k - reach), min(spec.size, k + reach + 1)
        j = lo + int(np.argmax(spec[lo:hi]))
        if j in (lo, hi - 1) or not (0 < j < spec.size - 1):
            return float(self.grid[k])                       # no local peak near track
        a, b, c = spec[j - 1], spec[j], spec[j + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > _EPS else 0.0
        return float(self.grid[j] + delta * step)          # |delta| <= 0.5 at a local max
