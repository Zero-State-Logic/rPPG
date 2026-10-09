"""Heart-rate estimation from a pulse signal, plus HR evaluation metrics.

Pure functions over 1-D NumPy arrays. No I/O.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

_EPS = 1e-9


def estimate_hr_fft(pulse: np.ndarray, fps: float,
                    lo_hz: float = 0.7, hi_hz: float = 4.0,
                    nfft: int | None = None) -> float:
    """Return heart rate in bpm as 60 * dominant frequency in [lo_hz, hi_hz]."""
    sig = np.asarray(pulse, dtype=np.float64)
    sig = sig - sig.mean()
    n = sig.size
    if n < 4:
        return float("nan")
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(max(256, n * 2))))
    freqs, psd = periodogram(sig, fs=fps, nfft=nfft, detrend=False)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(band):
        return float("nan")
    peak = freqs[band][int(np.argmax(psd[band]))]
    return float(peak * 60.0)


def snr_db(pulse: np.ndarray, fps: float, hr_hz: float,
           f_window: float = 0.4, lo_hz: float = 0.7, hi_hz: float = 5.0,
           nfft: int | None = None) -> float:
    """Signal-to-noise ratio (dB) around the HR fundamental + first harmonic.

    Matches the SNR used by CHROM/POS/ProPOS (de Haan & Jeanne 2013): power
    inside +/- f_window/2 of the fundamental and its first harmonic, over the
    remaining power in [lo_hz, hi_hz].
    """
    sig = np.asarray(pulse, dtype=np.float64)
    sig = sig - sig.mean()
    if sig.size < 4 or not np.isfinite(hr_hz):
        return float("nan")
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(max(512, sig.size * 2))))
    freqs, psd = periodogram(sig, fs=fps, nfft=nfft, detrend=False)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    f = freqs[band]
    p = psd[band]
    half = f_window / 2.0
    gate = (((f >= hr_hz - half) & (f <= hr_hz + half)) |
            ((f >= 2.0 * hr_hz - half) & (f <= 2.0 * hr_hz + half)))
    signal_power = float(np.sum(p[gate]))
    noise_power = float(np.sum(p[~gate]))
    return float(10.0 * np.log10((signal_power + _EPS) / (noise_power + _EPS)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation; NaN when a series is constant or too short."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.size < 2 or pred.std() < _EPS or true.std() < _EPS:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def accuracy_within(pred: np.ndarray, true: np.ndarray, tol_bpm: float = 5.0) -> float:
    """Percentage of estimates within +/- tol_bpm of ground truth."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - true) <= tol_bpm) * 100.0)
