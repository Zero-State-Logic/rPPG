"""Behaviour tests for the live per-person estimator (rppg.live.estimate.PersonSignal).

These pin the 2026-11-12 accuracy fixes measured on MCD-rPPG:
  * no Hampel spike filter by default (it turned 10 Hz room-light flicker into fake
    in-band waves); still available on request for legacy comparisons;
  * heart rate is tracked across estimates (a short corrupted stretch must not throw
    the reading away from the true rate);
  * the reported rate stays inside the 42-180 bpm live band.
"""
from __future__ import annotations

import numpy as np

from rppg.live.estimate import Calibration, PersonSignal, _hampel

FPS = 30.0
_BASE = np.array([170.0, 125.0, 100.0])          # skin RGB
_PBV = np.array([0.33, 0.77, 0.53])              # blood-volume colour signature


def _face(bpm: float, seconds: float, fps: float = FPS, amp: float = 0.004,
          noise: float = 0.15, seed: int = 0, t0: float = 0.0) -> np.ndarray:
    """``(T, 3)`` skin RGB with a pulse of ``bpm`` plus sensor noise."""
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(int(round(seconds * fps))) / fps
    pulse = np.sin(2 * np.pi * bpm / 60.0 * t)
    rgb = _BASE * (1.0 + amp * pulse[:, None] * _PBV)
    return rgb + noise * rng.standard_normal(rgb.shape)


def _flicker(rgb: np.ndarray, depth: float = 0.01) -> np.ndarray:
    """Every third frame brighter on all channels (10 Hz flicker at 30 fps)."""
    out = rgb.copy()
    out[::3] *= 1.0 + depth
    return out


def _run(signal: PersonSignal, rgb: np.ndarray, fps: float = FPS, t0: float = 0.0,
         every: float = 0.4) -> list:
    """Feed samples like the demo and estimate every ``every`` seconds."""
    readings, last = [], -1e9
    for i, sample in enumerate(rgb):
        t = t0 + i / fps
        signal.add(t, sample)
        if t - last >= every:
            readings.append(signal.estimate(Calibration()))
            last = t
    return readings


def test_spike_filter_is_off_by_default_and_available_on_request():
    rgb = _face(72.0, 10.0)
    rgb[150] += 25.0                                       # one large glint
    plain, legacy = PersonSignal(), PersonSignal(spike_filter=True)
    for i, s in enumerate(rgb):
        plain.add(i / FPS, s)
        legacy.add(i / FPS, s)
    raw, _ = plain._uniform_trace()
    filt, _ = legacy._uniform_trace()
    assert plain.spike_filter is False
    assert abs(raw[0, 150] - rgb[150, 0]) < 1e-9           # untouched without the filter
    half = max(2, int(round(0.2 * FPS)))
    np.testing.assert_allclose(filt[0], _hampel(raw[0], half))


def test_flicker_does_not_corrupt_the_heart_rate():
    readings = _run(PersonSignal(), _flicker(_face(72.0, 20.0)))
    final = readings[-1]
    assert final.ready
    assert abs(final.hr - 72.0) <= 3.0


def test_brief_corruption_does_not_move_the_tracked_rate():
    clean = _face(72.0, 24.0, seed=1)
    burst = np.arange(int(14 * FPS), int(16.5 * FPS))       # 2.5 s of strong junk
    t = burst / FPS
    clean[burst] += (3.0 * np.sin(2 * np.pi * 2.2 * t))[:, None] * np.array([1.0, 0.6, 0.9])
    readings = _run(PersonSignal(), clean)
    hrs = [r.hr for r in readings if r.ready and r.seconds >= 10.0]
    assert hrs, "estimator never became ready"
    assert max(abs(h - 72.0) for h in hrs) <= 5.0


def test_sustained_change_is_followed():
    first = _face(70.0, 20.0, seed=2)
    second = _face(100.0, 25.0, seed=3, t0=20.0)
    signal = PersonSignal()
    _run(signal, first)
    readings = _run(signal, second, t0=20.0)
    assert abs(readings[-1].hr - 100.0) <= 3.0


def test_rate_stays_inside_the_live_band():
    readings = _run(PersonSignal(), _face(200.0, 15.0, amp=0.01))
    for r in readings:
        if r.ready:
            assert 42.0 <= r.hr <= 180.0


def test_constant_colour_is_not_reported_as_a_heart_rate():
    flat = np.tile(_BASE, (int(12 * FPS), 1))
    readings = _run(PersonSignal(), flat)
    assert all(not r.ready for r in readings)


def test_long_gap_restarts_the_buffer_and_the_tracker():
    signal = PersonSignal()
    _run(signal, _face(70.0, 20.0, seed=4))
    readings = _run(signal, _face(120.0, 12.0, seed=5, t0=60.0), t0=60.0)  # 40 s later
    assert readings[0].ready is False                    # old samples were dropped
    assert abs(readings[-1].hr - 120.0) <= 3.0           # no pull from the old track


def test_not_ready_before_three_seconds():
    readings = _run(PersonSignal(), _face(72.0, 2.5))
    assert all(not r.ready for r in readings)
