"""Tests for the window-level reliability feature table."""
from __future__ import annotations

import numpy as np

from rppg.eval.frame_cache import FrameCache
from rppg.eval.window_table import (
    COLUMNS,
    _kinematics,
    _spread,
    build_table,
    load_table,
    save_table,
    window_records,
)

_FPS = 30.0
_HR_HZ = 1.5                                   # 90 bpm
_BASE = np.array([200.0, 150.0, 120.0])        # skin RGB
_PBV = np.array([0.33, 0.77, 0.53])            # blood-volume colour signature


def _pulse_cache(seconds: float = 20.0, box=None) -> FrameCache:
    """A clean synthetic subject: the same chromatic 90 bpm pulse in all 3 ROIs."""
    t = np.arange(int(seconds * _FPS)) / _FPS
    pulse = np.sin(2 * np.pi * _HR_HZ * t)
    rgb = _BASE[None, :] * (1.0 + 0.01 * pulse[:, None] * _PBV[None, :])   # (T, 3)
    roi = np.repeat(rgb[:, None, :], 3, axis=1)                            # (T, 3, 3)
    n = t.size
    if box is None:
        box = np.tile([100.0, 100.0, 80.0, 80.0], (n, 1))
    return FrameCache(
        fps=_FPS, roi_rgb=roi, roi_raw=roi.copy(), roi_npx=np.full((n, 3), 500),
        roi_area=np.full((n, 3), 500), bg_rgb=np.full((n, 3), 50.0), box=box,
        landmarks=np.zeros((n, 5, 2)), detected=np.ones(n, dtype=bool),
        ppg=pulse.copy(), gt_hr=np.full(n, 90.0))


def test_spread_is_max_minus_min_of_finite_values():
    assert _spread([70.0, 75.0, 72.0]) == 5.0
    assert _spread([70.0, float("nan"), 74.0]) == 4.0
    assert np.isnan(_spread([70.0, float("nan")]))


def test_kinematics_zero_for_a_still_face_and_positive_for_motion():
    still = np.tile([100.0, 100.0, 80.0, 80.0], (10, 1))
    moving = still.copy()
    moving[:, 0] += np.arange(10) * 4.0          # drifts 4 px/frame = 5% of width

    speed0, scale0 = _kinematics(still)
    speed1, _ = _kinematics(moving)

    assert speed0 == 0.0 and scale0 == 0.0
    np.testing.assert_allclose(speed1, 5.0)


def test_window_records_recover_a_known_90_bpm_pulse():
    # Arrange: 20 s of clean pulse -> windows of 10 s every 5 s -> starts 0, 150, 300
    cache = _pulse_cache()

    # Act
    recs = window_records(cache, "synthetic", win_sec=10.0, step_sec=5.0)

    # Assert
    assert [r["start"] for r in recs] == [0, 150, 300]
    for r in recs:
        assert abs(r["pred_hr"] - 90.0) < 2.0
        assert abs(r["ref_hr"] - 90.0) < 1.0
        assert r["abs_err"] < 3.0
        assert r["roi_spread"] < 2.0 and r["method_spread"] < 2.0
        assert r["det_rate"] == 1.0
        np.testing.assert_allclose(r["ox_hr"], 90.0)


def test_build_table_and_round_trip(tmp_path):
    recs = window_records(_pulse_cache(), "synthetic", win_sec=10.0, step_sec=5.0)

    table = build_table(recs)
    path = tmp_path / "w.npz"
    save_table(path, table)
    back = load_table(path)

    assert set(COLUMNS) <= set(back)
    assert all(back[c].shape == (3,) for c in COLUMNS)
    assert list(back["subject"]) == ["synthetic"] * 3
    np.testing.assert_allclose(back["pred_hr"], table["pred_hr"])
