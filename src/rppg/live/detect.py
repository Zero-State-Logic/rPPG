"""Multi-face detection + forehead/cheek ROI + a stable per-person tracker.

Detection uses OpenCV's YuNet (``cv2.FaceDetectorYN``), which is fast, handles
several faces at once, and stays reliable across a range of distances -- exactly
what the "moving / multiple people / far-and-near" demo scenarios need. From each
face box a mid-face ROI (cheeks + upper lip, avoiding eyes, mouth and hairline) is
derived for the rPPG colour trace.

A greedy centroid tracker assigns each face a persistent integer id so the signal
estimator can accumulate one RGB buffer per person while everyone moves around.

The YuNet ONNX model lives in ``models/`` and is downloaded on first use.
"""
from __future__ import annotations

import pathlib
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

_MODEL_DIR = pathlib.Path(__file__).resolve().parents[3] / "models"
_YUNET_NAME = "face_detection_yunet_2023mar.onnx"
_YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet/face_detection_yunet_2023mar.onnx")

# Mid-face ROI as fractions of the face box: central cheeks + upper lip. This band
# sits below the eyes and above the mouth, dodging blinks, teeth and the hairline.
_ROI_X, _ROI_W = 0.25, 0.50
_ROI_Y, _ROI_H = 0.55, 0.25

_MAX_MISSED = 15      # drop a track after this many consecutive unseen frames
_MATCH_FACTOR = 0.6   # a detection matches a track within this * face-width pixels


def _ensure_model() -> str:
    """Return the YuNet model path, downloading it to ``models/`` if absent."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / _YUNET_NAME
    if not path.exists():
        urllib.request.urlretrieve(_YUNET_URL, str(path))
    return str(path)


def _clamp_roi(x: int, y: int, w: int, h: int,
               frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Clamp an (x, y, w, h) rect so it stays inside the frame with w, h >= 1."""
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def _roi_from_box(x: int, y: int, w: int, h: int,
                  frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Derive the mid-face ROI rect from a face box, clamped to the frame."""
    return _clamp_roi(int(x + _ROI_X * w), int(y + _ROI_Y * h),
                      int(_ROI_W * w), int(_ROI_H * h), frame_w, frame_h)


@dataclass
class Detection:
    """One detected face: boxes are (x, y, w, h); landmarks are (5, 2) pixels."""

    box: tuple[int, int, int, int]
    roi: tuple[int, int, int, int]
    landmarks: np.ndarray
    score: float


@dataclass
class Track:
    """A tracked face with a persistent id and its latest detection."""

    id: int
    det: Detection
    centroid: tuple[float, float]
    missed: int = 0


class YuNetDetector:
    """Thin wrapper over ``cv2.FaceDetectorYN`` returning :class:`Detection` list."""

    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3,
                 top_k: int = 50) -> None:
        self._det = cv2.FaceDetectorYN.create(
            _ensure_model(), "", (320, 320), score_threshold, nms_threshold, top_k)
        self._size = (320, 320)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect all faces in a BGR frame; returns an empty list when none."""
        frame_h, frame_w = frame_bgr.shape[:2]
        if (frame_w, frame_h) != self._size:
            self._det.setInputSize((frame_w, frame_h))
            self._size = (frame_w, frame_h)
        _, faces = self._det.detect(frame_bgr)
        detections: list[Detection] = []
        if faces is None:
            return detections
        for f in faces:
            x, y, w, h = (max(0, int(v)) for v in f[:4])
            roi = _roi_from_box(x, y, w, h, frame_w, frame_h)
            detections.append(Detection((x, y, w, h), roi,
                                        f[4:14].reshape(5, 2), float(f[14])))
        return detections


class CentroidTracker:
    """Greedy nearest-centroid tracker giving each face a stable id over time."""

    def __init__(self, max_missed: int = _MAX_MISSED,
                 match_factor: float = _MATCH_FACTOR) -> None:
        self._next_id = 0
        self._tracks: dict[int, Track] = {}
        self._max_missed = max_missed
        self._match_factor = match_factor

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the tracker one frame; returns the currently visible tracks."""
        centroids = [(d.box[0] + d.box[2] / 2.0, d.box[1] + d.box[3] / 2.0)
                     for d in detections]
        unmatched = set(range(len(detections)))

        for track in self._tracks.values():
            best, best_dist = None, float("inf")
            for i in unmatched:
                dist = ((centroids[i][0] - track.centroid[0]) ** 2
                        + (centroids[i][1] - track.centroid[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best, best_dist = i, dist
            radius = self._match_factor * detections[best].box[2] if best is not None else 0.0
            if best is not None and best_dist <= max(radius, 20.0):
                track.det, track.centroid, track.missed = (
                    detections[best], centroids[best], 0)
                unmatched.discard(best)
            else:
                track.missed += 1

        for i in unmatched:
            self._tracks[self._next_id] = Track(self._next_id, detections[i],
                                                centroids[i], 0)
            self._next_id += 1

        for stale in [tid for tid, t in self._tracks.items()
                      if t.missed > self._max_missed]:
            del self._tracks[stale]
        return [t for t in self._tracks.values() if t.missed == 0]


if __name__ == "__main__":  # smoke test without a camera
    detector = YuNetDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    found = detector.detect(blank)
    print(f"YuNet loaded OK; faces on a blank frame: {len(found)}")
