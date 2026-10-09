"""CHROM (chrominance-based) rPPG pulse extraction.

Reference:
    de Haan & Jeanne, "Robust pulse rate from chrominance-based rPPG",
    IEEE TBME 60(10):2878-2886, 2013.

A motion-robust baseline that the ProPOS paper compares against. Overlap-add
windowed formulation, consistent with our POS implementation.
"""
from __future__ import annotations

import numpy as np

from .pos import _EPS, _as_rgb_trace


def chrom_pulse(rgb: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """Extract the CHROM pulse from an RGB trace (shape (3, T)); returns (T,)."""
    c = _as_rgb_trace(rgb)
    n_frames = c.shape[1]
    win = max(2, min(int(round(window_sec * fps)), n_frames))

    pulse = np.zeros(n_frames, dtype=np.float64)
    for end in range(win - 1, n_frames):
        start = end - win + 1
        block = c[:, start:end + 1]
        mean = block.mean(axis=1, keepdims=True)
        mean[np.abs(mean) < _EPS] = _EPS
        cn = block / mean                                # temporal normalization
        r, g, b = cn[0], cn[1], cn[2]
        xs = 3.0 * r - 2.0 * g
        ys = 1.5 * r + g - 1.5 * b
        std_y = ys.std()
        alpha = xs.std() / (std_y if std_y > _EPS else _EPS)
        s = xs - alpha * ys
        s = s - s.mean()
        pulse[start:end + 1] += s                        # overlap-add
    return pulse
