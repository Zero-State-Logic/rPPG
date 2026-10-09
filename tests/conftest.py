"""Shared pytest fixtures and the src/ import shim for Phase-1 classical tests."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

# Make the src/ layout importable without installation (Phase-1 convenience;
# replaced by `pip install -e .` once pyproject.toml lands).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def synthetic_rgb_trace(hr_bpm: float, fps: float = 30.0, seconds: float = 12.0,
                        noise: float = 0.002, seed: int = 0) -> np.ndarray:
    """Generate a (3, T) RGB trace carrying a pulse at hr_bpm with a 2nd harmonic.

    Models skin-tone DC per channel + small per-channel pulsatile gain + noise,
    which is what POS/CHROM/ProPOS consume.
    """
    rng = np.random.default_rng(seed)
    n = int(round(fps * seconds))
    t = np.arange(n) / fps
    f = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.0 * f * t)
    dc = np.array([0.6, 0.5, 0.4])
    gain = np.array([0.012, 0.030, 0.018])
    trace = dc[:, None] + gain[:, None] * pulse[None, :]
    trace = trace + noise * rng.standard_normal((3, n))
    return trace


@pytest.fixture
def make_rgb():
    """Return the synthetic RGB-trace generator for parametrized tests."""
    return synthetic_rgb_trace
