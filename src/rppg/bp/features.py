"""Blood-pressure-relevant morphology features from a BVP/PPG pulse.

Features that the BP literature links to blood pressure: heart rate, pulse
amplitude/area, 1st & 2nd derivative (VPG / APG) markers, and beat-interval
statistics (an HRV proxy). Computed on a filtered, averaged pulse so they survive
rPPG noise. These feed the baseline BP regressor; richer APG a-e indices and
calibration come later. Pure functions over 1-D arrays, no I/O.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from ..hr.metrics import estimate_hr_fft

FEATURE_NAMES: tuple[str, ...] = (
    "hr_bpm", "amp_std", "skewness", "kurtosis",
    "vpg_max", "vpg_min_abs", "apg_max", "apg_min_abs", "apg_vpg_ratio",
    "pulse_area", "n_beats", "ibi_mean_s", "ibi_std_s", "sys_amp_mean",
)

_EPS = 1e-9


def pulse_features(pulse: np.ndarray, fps: float) -> dict[str, float]:
    """Return a dict of BP-relevant features for one pulse segment."""
    x = np.asarray(pulse, dtype=np.float64)
    x = x - x.mean()
    sd = float(x.std())
    xn = x / sd if sd > _EPS else x

    vpg = np.gradient(xn) * fps
    apg = np.gradient(vpg) * fps
    vpg_max = float(vpg.max())
    apg_max = float(apg.max())

    min_dist = max(1, int(round(0.33 * fps)))              # <=180 bpm
    peaks, props = find_peaks(xn, distance=min_dist, prominence=0.1)
    if peaks.size >= 2:
        ibi = np.diff(peaks) / fps
        ibi_mean, ibi_std = float(ibi.mean()), float(ibi.std())
        sys_amp = float(np.mean(props["prominences"]))
    else:
        ibi_mean = ibi_std = sys_amp = float("nan")

    return {
        "hr_bpm": float(estimate_hr_fft(x, fps)),
        "amp_std": sd,
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),
        "vpg_max": vpg_max,
        "vpg_min_abs": float(abs(vpg.min())),
        "apg_max": apg_max,
        "apg_min_abs": float(abs(apg.min())),
        "apg_vpg_ratio": float(apg_max / vpg_max) if vpg_max > _EPS else float("nan"),
        "pulse_area": float(np.mean(np.abs(xn))),
        "n_beats": float(peaks.size),
        "ibi_mean_s": ibi_mean,
        "ibi_std_s": ibi_std,
        "sys_amp_mean": sys_amp,
    }


def features_vector(pulse: np.ndarray, fps: float) -> np.ndarray:
    """Return the features as a fixed-order 1-D vector (for ML models)."""
    feats = pulse_features(pulse, fps)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
