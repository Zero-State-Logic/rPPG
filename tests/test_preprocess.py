"""Unit tests for the trace reducer and video I/O error handling.

The MediaPipe ROI extractor and the full video pipeline are validated against a
real face video (UBFC) in a separate integration step; here we use a fake
extractor so these stay fast and dependency-light.
"""
from __future__ import annotations

import numpy as np
import pytest

from rppg.io.video import read_meta
from rppg.preprocess.rgb_trace import frames_to_rgb_trace, masked_channel_mean


class _FullFrameExtractor:
    """ROI = the whole frame; lets us test the reducer without MediaPipe."""

    def roi_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        return np.ones(frame_rgb.shape[:2], dtype=bool)


def test_masked_channel_mean_basic():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 100, 50, 25
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    value = masked_channel_mean(frame, mask, reject_specular=False)
    assert np.allclose(value, [100.0, 50.0, 25.0])


def test_masked_channel_mean_empty():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert masked_channel_mean(frame, None) is None
    assert masked_channel_mean(frame, np.zeros((4, 4), dtype=bool)) is None


def test_frames_to_rgb_trace_shape_and_values():
    n = 20
    frames = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    for t in range(n):
        frames[t, :, :, 1] = 40 + t                 # green channel ramps
    trace = frames_to_rgb_trace(frames, _FullFrameExtractor())
    assert trace.shape == (3, n)
    assert np.allclose(trace[1], 40 + np.arange(n))


def test_read_meta_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_meta(tmp_path / "does_not_exist.avi")
