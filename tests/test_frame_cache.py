"""Tests for the offline frame-feature cache and windowed-evaluation helpers."""
from __future__ import annotations

import numpy as np
import pytest

from rppg.eval.frame_cache import (
    FrameCache,
    background_mean,
    load_cache,
    read_ground_truth,
    roi_skin_mean,
    save_cache,
)
from rppg.eval.windows import window_starts

_SKIN_BGR = (120, 150, 200)    # RGB (200, 150, 120): Cr~155, Cb~105 -> passes the gate
_BLUE_BGR = (255, 0, 0)        # Cr~107 -> fails the skin gate


def _frame(h: int = 40, w: int = 40, bgr=(0, 0, 0)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_roi_skin_mean_returns_skin_colour_and_counts_pixels():
    # Arrange
    frame = _frame(bgr=_SKIN_BGR)

    # Act
    skin, raw, n_skin, area = roi_skin_mean(frame, (5, 5, 10, 8))

    # Assert
    assert n_skin == area == 80
    np.testing.assert_allclose(skin, [200, 150, 120])
    np.testing.assert_allclose(raw, [200, 150, 120])


def test_roi_skin_mean_non_skin_patch_has_nan_skin_but_raw_mean():
    frame = _frame(bgr=_BLUE_BGR)

    skin, raw, n_skin, area = roi_skin_mean(frame, (0, 0, 10, 10))

    assert n_skin == 0 and area == 100
    assert np.all(np.isnan(skin))
    np.testing.assert_allclose(raw, [0, 0, 255])


def test_roi_skin_mean_empty_patch_is_all_nan():
    frame = _frame()

    skin, raw, n_skin, area = roi_skin_mean(frame, (10, 10, 0, 0))

    assert n_skin == 0 and area == 0
    assert np.all(np.isnan(skin)) and np.all(np.isnan(raw))


def test_background_mean_averages_the_four_corners():
    frame = _frame(h=100, w=100, bgr=(10, 20, 30))   # uniform frame

    bg = background_mean(frame)

    np.testing.assert_allclose(bg, [30, 20, 10])     # BGR -> RGB


def _cache(roi_rgb, roi_raw, roi_npx, roi_area) -> FrameCache:
    t = len(roi_npx)
    return FrameCache(
        fps=30.0, roi_rgb=np.asarray(roi_rgb, float), roi_raw=np.asarray(roi_raw, float),
        roi_npx=np.asarray(roi_npx), roi_area=np.asarray(roi_area),
        bg_rgb=np.zeros((t, 3)), box=np.zeros((t, 4)), landmarks=np.zeros((t, 5, 2)),
        detected=np.ones(t, dtype=bool), ppg=np.zeros(t), gt_hr=np.empty(0))


def test_pooled_rgb_weights_skin_means_by_pixel_count():
    nan3 = [np.nan] * 3
    cache = _cache(roi_rgb=[[[10, 20, 30], [30, 40, 50], nan3]],
                   roi_raw=[[[0, 0, 0]] * 3], roi_npx=[[100, 100, 0]],
                   roi_area=[[100, 100, 100]])

    pooled = cache.pooled_rgb()

    np.testing.assert_allclose(pooled[0], [20, 30, 40])


def test_pooled_rgb_falls_back_to_area_weighted_raw_when_skin_is_scarce():
    nan3 = [np.nan] * 3
    cache = _cache(roi_rgb=[[[9, 9, 9], [9, 9, 9], nan3]],
                   roi_raw=[[[1, 1, 1], [3, 3, 3], [5, 5, 5]]],
                   roi_npx=[[10, 10, 0]], roi_area=[[100, 100, 200]])

    pooled = cache.pooled_rgb()

    np.testing.assert_allclose(pooled[0], [3.5, 3.5, 3.5])   # (100+300+1000)/400


def test_read_ground_truth_parses_ppg_and_hr_rows(tmp_path):
    gt = tmp_path / "ground_truth.txt"
    gt.write_text("0.1 0.2 0.3\n70 71 72\n0.0 0.033 0.066\n", encoding="utf-8")

    ppg, hr = read_ground_truth(gt)

    np.testing.assert_allclose(ppg, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(hr, [70, 71, 72])


def test_read_ground_truth_without_hr_row_returns_empty_hr(tmp_path):
    gt = tmp_path / "ground_truth.txt"
    gt.write_text("0.1 0.2\n", encoding="utf-8")

    _, hr = read_ground_truth(gt)

    assert hr.size == 0


def test_read_ground_truth_rejects_empty_file(tmp_path):
    gt = tmp_path / "ground_truth.txt"
    gt.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_ground_truth(gt)


def test_save_and_load_cache_round_trip(tmp_path):
    nan3 = [np.nan] * 3
    cache = _cache(roi_rgb=[[[1, 2, 3], [4, 5, 6], nan3]] * 4,
                   roi_raw=[[[7, 8, 9]] * 3] * 4, roi_npx=[[5, 6, 0]] * 4,
                   roi_area=[[9, 9, 9]] * 4)
    path = tmp_path / "s.npz"

    save_cache(path, cache)
    back = load_cache(path)

    assert back.fps == cache.fps and back.n_frames == 4
    np.testing.assert_array_equal(back.roi_npx, cache.roi_npx)
    np.testing.assert_allclose(back.roi_rgb, cache.roi_rgb)   # NaNs compare equal


def test_window_starts_covers_all_full_windows():
    assert window_starts(600, 30.0, 10.0, 2.0) == [0, 60, 120, 180, 240, 300]
    assert window_starts(300, 30.0, 10.0, 2.0) == [0]
    assert window_starts(100, 30.0, 10.0, 2.0) == []
