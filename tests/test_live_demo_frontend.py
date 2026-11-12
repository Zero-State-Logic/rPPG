"""Tests for the live demo's colour front end (scripts/live_demo.py).

Since 2026-11-12 the demo samples the whole face box (optionally its skin pixels)
instead of the small forehead/cheek patches, and background illumination division is
opt-in. These pin the helpers and the CLI defaults.
"""
from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "live_demo.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("live_demo_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame() -> np.ndarray:
    """100x100 BGR frame: grey background, a skin-coloured square at (20..80)."""
    f = np.full((100, 100, 3), 40, np.uint8)
    f[20:80, 20:80] = (120, 150, 200)          # BGR skin tone (passes the YCrCb gate)
    f[45:55, 45:55] = (255, 0, 0)              # a non-skin blue blob inside the box
    return f


def test_face_rgb_is_the_box_mean_in_rgb_order(demo):
    f = _frame()
    out = demo._face_rgb(f, (20, 20, 60, 60))
    expect = f[20:80, 20:80].reshape(-1, 3).mean(axis=0)[::-1]
    np.testing.assert_allclose(out, expect)


def test_face_rgb_skin_mode_ignores_non_skin_pixels(demo):
    out = demo._face_rgb(_frame(), (20, 20, 60, 60), skin=True)
    np.testing.assert_allclose(out, [200.0, 150.0, 120.0])


def test_face_rgb_clips_to_the_frame_and_handles_empty_boxes(demo):
    f = _frame()
    assert demo._face_rgb(f, (200, 200, 10, 10)) is None
    clipped = demo._face_rgb(f, (-10, -10, 40, 40))
    np.testing.assert_allclose(clipped, f[0:30, 0:30].reshape(-1, 3).mean(axis=0)[::-1])


def test_roi_rgb_dispatches_on_mode(demo):
    f = _frame()
    det = SimpleNamespace(box=(20, 20, 60, 60), rois=((25, 25, 10, 10),))
    np.testing.assert_allclose(demo._roi_rgb(f, det, "face"), demo._face_rgb(f, det.box))
    np.testing.assert_allclose(demo._roi_rgb(f, det, "patches"),
                               demo._pooled_skin_rgb(f, det.rois))


def test_cli_defaults_face_skin_without_illumination_division(demo):
    args = demo.build_parser().parse_args([])
    assert args.roi == "face-skin"     # MCD-rPPG dev: 3.64 vs 4.48 bpm for the raw box
    assert args.illum_norm is False
    legacy = demo.build_parser().parse_args(["--no-illum-norm"])   # still accepted
    assert legacy.no_illum_norm is True
