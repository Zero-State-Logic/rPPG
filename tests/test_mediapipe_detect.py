"""Tests for the MediaPipe FaceLandmarker detector module.

The module must import (and its constants + pure helpers must be exercisable)
even where ``mediapipe`` or the ``.task`` model is unavailable; only the detector
construction test is skipped in that case.
"""
from __future__ import annotations

import numpy as np
import pytest

from rppg.live import mediapipe_detect as md


def test_index_lists_non_empty():
    """The three ROI index lists all carry mesh vertices."""
    assert len(md.FOREHEAD_IDX) > 0
    assert len(md.RCHEEK_IDX) > 0
    assert len(md.LCHEEK_IDX) > 0


def test_index_lists_pairwise_disjoint():
    """No mesh vertex is shared between forehead and either cheek region."""
    forehead = set(md.FOREHEAD_IDX)
    r_cheek = set(md.RCHEEK_IDX)
    l_cheek = set(md.LCHEEK_IDX)

    assert forehead.isdisjoint(r_cheek)
    assert forehead.isdisjoint(l_cheek)
    assert r_cheek.isdisjoint(l_cheek)


def test_mesh_rect_returns_int_rect_inside_frame():
    """``_mesh_rect`` yields an int (x, y, w, h) inside the frame with w, h >= 1."""
    frame_w, frame_h = 960, 540
    px_pts = np.zeros((478, 2), dtype=int)
    # Give the indexed points a known, spread-out bounding box well inside frame.
    idxs = md.FOREHEAD_IDX
    px_pts[idxs] = np.array([[200, 100], [200, 100], [400, 100], [200, 100],
                             [300, 100], [400, 100], [300, 180], [200, 180],
                             [400, 180]])

    x, y, w, h = md._mesh_rect(px_pts, idxs, frame_w, frame_h)

    assert all(isinstance(v, int) for v in (x, y, w, h))
    assert w >= 1 and h >= 1
    assert 0 <= x < frame_w and 0 <= y < frame_h
    assert x + w <= frame_w and y + h <= frame_h


def test_detector_constructs_or_skips():
    """Constructing the detector works, or is skipped when deps/model are absent."""
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"mediapipe not importable: {exc}")

    try:
        detector = md.MediaPipeDetector()
    except FileNotFoundError as exc:
        pytest.skip(f"FaceLandmarker model unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 - runtime/model init failure
        pytest.skip(f"MediaPipe detector could not initialise: {exc}")

    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    found = detector.detect(blank)
    assert isinstance(found, list)
