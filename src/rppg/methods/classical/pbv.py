"""PBV (blood-volume-pulse signature) rPPG pulse extraction.

Reference:
    de Haan & van Leest, "Improved motion robustness of remote-PPG by using the
    blood volume pulse signature", Physiol. Meas. 35(9):1913-1926, 2014.

Data-driven PBV signature (std of the temporally-normalized channels), as used
in common rPPG toolkits. One of the baselines the ProPOS paper compares against.
"""
from __future__ import annotations

import numpy as np

from .pos import _EPS, _as_rgb_trace


def pbv_pulse(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Extract the PBV pulse from an RGB trace (shape (3, T)); returns (T,)."""
    c = _as_rgb_trace(rgb)
    mean = c.mean(axis=1, keepdims=True)
    mean[np.abs(mean) < _EPS] = _EPS
    cn = c / mean                                        # temporal normalization

    pbv_n = np.array([cn[0].std(), cn[1].std(), cn[2].std()], dtype=np.float64)
    pbv_d = float(np.sqrt(np.sum(pbv_n ** 2)))
    pbv = pbv_n / (pbv_d if pbv_d > _EPS else _EPS)      # blood-volume signature

    q = cn @ cn.T                                        # (3, 3)
    weights = np.linalg.solve(q + _EPS * np.eye(3), pbv)
    numerator = weights @ cn                             # (T,)
    denominator = float(weights @ pbv)
    bvp = numerator / (denominator if abs(denominator) > _EPS else _EPS)
    return bvp - bvp.mean()
