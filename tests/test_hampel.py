"""Unit tests for the Hampel spike filter used against sudden optical noise.

The filter must knock a single large transient (a lighting flicker / specular
glint) back toward the underlying pulse while leaving clean, in-band samples
essentially untouched.
"""
from __future__ import annotations

import numpy as np

from rppg.live.estimate import _hampel


def _clean_sinusoid(n: int = 300) -> np.ndarray:
    """A smooth length-``n`` sinusoid standing in for a clean pulse trace."""
    t = np.linspace(0.0, 6.0 * np.pi, n)
    return np.sin(t).astype(np.float64)


def test_hampel_pulls_injected_spike_toward_clean_value():
    # Arrange: clean trace, plus a copy with one large mid-array spike.
    clean = _clean_sinusoid(300)
    spiked = clean.copy()
    idx = 150
    spiked[idx] = clean[idx] + 10.0  # far outside the +/-1 signal range

    # Act
    filtered = _hampel(spiked, 5)

    # Assert: the spike is removed, landing close to the clean sample.
    spike_residual = abs(filtered[idx] - clean[idx])
    assert spike_residual < 0.5
    # And it is a huge reduction from the injected deviation.
    assert spike_residual < abs(spiked[idx] - clean[idx]) / 10.0


def test_hampel_leaves_clean_samples_essentially_unchanged():
    # Arrange
    clean = _clean_sinusoid(300)
    spiked = clean.copy()
    idx = 150
    spiked[idx] = clean[idx] + 10.0

    # Act
    filtered = _hampel(spiked, 5)

    # Assert: every non-spike index stays very close to the original clean value.
    mask = np.ones(clean.size, dtype=bool)
    mask[idx] = False
    max_clean_drift = float(np.max(np.abs(filtered[mask] - clean[mask])))
    assert max_clean_drift < 1e-6


def test_hampel_returns_input_when_buffer_too_short():
    # Arrange: fewer than 2*half+1 samples -> guard returns the array unchanged.
    half = 5
    short = np.arange(2 * half, dtype=np.float64)  # length 10 < 11

    # Act
    out = _hampel(short, half)

    # Assert
    np.testing.assert_array_equal(out, short)
