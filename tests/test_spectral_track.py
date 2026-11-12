"""Tests for the causal spectral heart-rate tracker (rppg.hr.track).

The tracker replaces per-window "take the tallest peak" with an online Viterbi over a
1-bpm grid: a single bad window (flicker / motion peak elsewhere in the band) must not
pull the reading away, a real, sustained heart-rate change must still be followed, and
the evidence of overlapping live windows (one estimate every 0.4 s over a 10 s buffer)
must not count as many independent observations.
"""
from __future__ import annotations

import numpy as np
import pytest

from rppg.hr.track import SpectralTracker, pulse_spectrum

FPS = 30.0


def _sine(bpm: float, seconds: float = 10.0, fps: float = FPS, noise: float = 0.0,
          seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * fps)) / fps
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * bpm / 60.0 * t) + noise * rng.standard_normal(t.size)


def test_pulse_spectrum_is_normalised_and_peaks_at_the_pulse_rate():
    tracker = SpectralTracker()
    spec = pulse_spectrum(_sine(72.0), FPS, tracker.grid)
    assert spec.shape == tracker.grid.shape
    assert spec.sum() == pytest.approx(1.0)
    assert abs(tracker.grid[int(np.argmax(spec))] - 72.0) <= 1.0


def test_pulse_spectrum_is_zero_outside_the_band():
    tracker = SpectralTracker(lo_bpm=42.0, hi_bpm=180.0)
    spec = pulse_spectrum(_sine(200.0), FPS, tracker.grid)
    # all power of a 200 bpm tone falls outside 42-180 bpm: nothing may peak at 200
    assert tracker.grid.max() <= 180.0
    assert np.all(np.isfinite(spec))


def test_first_update_returns_the_spectral_peak_with_sub_bin_precision():
    tracker = SpectralTracker()
    hr = tracker.update(pulse_spectrum(_sine(72.4), FPS, tracker.grid), dt=2.0)
    assert hr == pytest.approx(72.4, abs=0.6)


def test_single_outlier_window_does_not_move_the_tracked_rate():
    tracker = SpectralTracker()
    good = pulse_spectrum(_sine(75.0, noise=0.3), FPS, tracker.grid)
    bad = pulse_spectrum(_sine(140.0), FPS, tracker.grid)
    for _ in range(5):
        tracker.update(good, dt=2.0)
    hr = tracker.update(bad, dt=2.0)
    assert abs(hr - 75.0) <= 2.0


def test_sustained_change_is_followed():
    tracker = SpectralTracker()
    for _ in range(5):
        tracker.update(pulse_spectrum(_sine(70.0), FPS, tracker.grid), dt=2.0)
    hr = float("nan")
    for _ in range(8):                               # 16 s at the new rate
        hr = tracker.update(pulse_spectrum(_sine(95.0), FPS, tracker.grid), dt=2.0)
    assert abs(hr - 95.0) <= 2.0


def test_live_cadence_needs_a_similar_wall_clock_time_to_switch():
    """0.4 s updates must not make an outlier 5x more convincing than 2 s updates."""
    def switch_time(dt: float) -> float:
        tracker = SpectralTracker()
        for _ in range(int(10 / dt)):
            tracker.update(pulse_spectrum(_sine(70.0), FPS, tracker.grid), dt=dt)
        t = 0.0
        while t < 60.0:
            t += dt
            hr = tracker.update(pulse_spectrum(_sine(110.0), FPS, tracker.grid), dt=dt)
            if abs(hr - 110.0) <= 2.0:
                return t
        return float("inf")

    slow, fast = switch_time(2.0), switch_time(0.4)
    assert np.isfinite(slow) and np.isfinite(fast)
    assert fast >= 0.5 * slow                    # outlier evidence is not over-counted


def test_reset_forgets_the_history():
    tracker = SpectralTracker()
    for _ in range(5):
        tracker.update(pulse_spectrum(_sine(70.0), FPS, tracker.grid), dt=2.0)
    tracker.reset()
    hr = tracker.update(pulse_spectrum(_sine(120.0), FPS, tracker.grid), dt=2.0)
    assert hr == pytest.approx(120.0, abs=1.0)


def test_flat_input_gives_no_rate_instead_of_the_band_edge():
    tracker = SpectralTracker()
    spec = pulse_spectrum(np.zeros(300), FPS, tracker.grid)
    assert not np.any(np.isfinite(spec))                 # "no in-band power"
    assert np.isnan(tracker.update(spec, dt=2.0))        # never a fake 42 bpm


def test_zero_or_negative_dt_does_not_double_count_a_window():
    tracker = SpectralTracker()
    good = pulse_spectrum(_sine(75.0), FPS, tracker.grid)
    bad = pulse_spectrum(_sine(140.0), FPS, tracker.grid)
    tracker.update(good, dt=2.0)
    before = tracker.update(good, dt=2.0)
    for _ in range(20):                                  # the same instant, repeated
        assert tracker.update(bad, dt=0.0) == before
        assert tracker.update(bad, dt=-1.0) == before


def test_default_parameters_match_the_mcd_bench_choice():
    tracker = SpectralTracker()
    assert tracker.grid[0] == 42.0 and tracker.grid[-1] == 180.0
    assert tracker.sigma_bpm == 4.0        # both CV folds picked sigma 4 (face skin, POS)


def test_rejects_a_spectrum_of_the_wrong_shape():
    tracker = SpectralTracker()
    with pytest.raises(ValueError):
        tracker.update(np.ones(5), dt=2.0)
