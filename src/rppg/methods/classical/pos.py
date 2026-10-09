"""POS (Plane-Orthogonal-to-Skin) rPPG pulse extraction.

Reference:
    Wang, den Brinker, Stuijk, de Haan, "Algorithmic Principles of Remote PPG",
    IEEE TBME 64(7):1479-1491, 2017.

POS is the baseline the ProPOS paper (Rao et al., Med. Eng. & Phys. 138, 2025)
extends. We implement the canonical overlap-add formulation so that:
  * it is a standalone classical baseline, and
  * ProPOS can reuse the POS pulse to derive the reference heart rate used in its
    SNR-based candidate selection.

All functions are pure: they take an in-memory RGB trace (shape (3, T)) and return
NumPy arrays. No file or camera I/O lives here.
"""
from __future__ import annotations

import numpy as np

# Canonical POS projection matrix (Wang et al. 2017, S = P @ Cn).
_POS_PROJECTION = np.array([[0.0, 1.0, -1.0],
                            [-2.0, 1.0, 1.0]], dtype=np.float64)

_EPS = 1e-9


def _as_rgb_trace(rgb: np.ndarray) -> np.ndarray:
    """Validate and coerce input to a float64 (3, T) RGB trace."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"rgb trace must have shape (3, T); got {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError("rgb trace must contain at least 2 frames")
    return arr


def pos_pulse(rgb: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """Extract the POS pulse signal from an RGB trace via overlap-add.

    Args:
        rgb: Raw spatially-averaged channel trace, shape (3, T), order R, G, B.
        fps: Frames per second of the trace.
        window_sec: Sliding window length in seconds (Wang et al. use 1.6 s).

    Returns:
        Pulse signal of shape (T,), zero-mean.
    """
    c = _as_rgb_trace(rgb)
    n_frames = c.shape[1]
    win = int(round(window_sec * fps))
    win = max(2, min(win, n_frames))

    pulse = np.zeros(n_frames, dtype=np.float64)
    for end in range(win - 1, n_frames):
        start = end - win + 1
        block = c[:, start:end + 1]                      # (3, win)
        mean = block.mean(axis=1, keepdims=True)
        mean[np.abs(mean) < _EPS] = _EPS
        c_norm = block / mean                            # temporal normalization
        s = _POS_PROJECTION @ c_norm                     # (2, win)
        s1, s2 = s[0], s[1]
        std2 = s2.std()
        alpha = s1.std() / (std2 if std2 > _EPS else _EPS)
        h = s1 + alpha * s2
        h = h - h.mean()
        pulse[start:end + 1] += h                        # overlap-add
    return pulse
