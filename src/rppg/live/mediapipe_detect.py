"""MediaPipe FaceLandmarker detector, emitting the same ``Detection`` interface.

A selectable, higher-precision alternative to the default YuNet detector. It runs
Google's MediaPipe **FaceLandmarker** (the dense 478-point face mesh with iris
refinement) and distils each mesh into the exact forehead + right-cheek +
left-cheek ROI triplet the rest of the pipeline consumes, so nothing downstream
changes: ``detect()`` returns ``list[Detection]`` just like ``YuNetDetector``.

The forehead/cheek rectangles are taken straight from dense mesh vertices (not a
coarse 5-point estimate), inset slightly to stay on skin, and the cheeks' medial
edges are pushed clear of the nose. Landmarks reported on each ``Detection`` are a
YuNet-compatible 5-point set (both iris centres, nose tip, both mouth corners).

MediaPipe is imported lazily inside the detector so this module (and its constants
and pure helpers) import cleanly even where ``mediapipe`` is not installed. The
``models/face_landmarker.task`` asset must already be present.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np

from .detect import Detection, _clamp_roi

_MODEL_DIR = pathlib.Path(__file__).resolve().parents[3] / "models"
_LANDMARKER_NAME = "face_landmarker.task"

# Dense-mesh vertex indices (canonical 478-point FaceLandmarker topology).
FOREHEAD_IDX = [10, 109, 338, 108, 151, 337, 9, 107, 336]
RCHEEK_IDX = [50, 101, 118, 117, 123, 147, 187, 205, 36, 142]
LCHEEK_IDX = [280, 330, 347, 346, 352, 376, 411, 425, 266, 371]
# YuNet-compatible 5-point set: right iris, left iris, nose tip, R/L mouth corners.
FIVE_PT_IDX = [468, 473, 4, 61, 291]

_NOSE_MARGIN = 0.22   # cheek medial edge kept this * inter-ocular distance off the nose
_INSET = 0.12         # shrink each mesh rect this fraction per side to stay on skin
_INFER_LONG_SIDE = 512  # run inference on a copy downscaled to this long side (speed)

# Detector precision/robustness (highest sensible precision for the 478 mesh).
_NUM_FACES = 3
_MIN_DETECT_CONF = 0.6
_MIN_PRESENCE_CONF = 0.6
_MIN_TRACKING_CONF = 0.7


def _ensure_landmarker_model() -> str:
    """Return the FaceLandmarker ``.task`` path; raise if the asset is missing."""
    path = _MODEL_DIR / _LANDMARKER_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"MediaPipe FaceLandmarker model not found at {path}. Place "
            f"'{_LANDMARKER_NAME}' in the models/ directory before using "
            "--detector mediapipe."
        )
    return str(path)


def _mesh_rect(px_pts: np.ndarray, idxs: list[int], frame_w: int,
               frame_h: int) -> tuple[int, int, int, int]:
    """Clamped (x, y, w, h) bbox of ``px_pts[idxs]``, inset ``_INSET`` toward centre."""
    pts = px_pts[idxs]
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    w, h = x1 - x0, y1 - y0
    ix, iy = _INSET * w, _INSET * h
    return _clamp_roi(int(round(x0 + ix)), int(round(y0 + iy)),
                      int(round(w - 2 * ix)), int(round(h - 2 * iy)),
                      frame_w, frame_h)


def _clamp_off_nose(rect: tuple[int, int, int, int], nose_x: float, gap: float,
                    frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Push a cheek rect's medial (nose-side) edge ``gap`` px clear of ``nose_x``."""
    x, y, w, h = rect
    if x + w / 2.0 <= nose_x:                       # cheek on the smaller-x side of nose
        new_right = min(float(x + w), nose_x - gap)
        return _clamp_roi(int(x), int(y), int(round(new_right - x)), int(h),
                          frame_w, frame_h)
    new_left = max(float(x), nose_x + gap)          # cheek on the larger-x side of nose
    return _clamp_roi(int(round(new_left)), int(y),
                      int(round((x + w) - new_left)), int(h), frame_w, frame_h)


def _iod(px_pts: np.ndarray) -> float:
    """Inter-ocular distance from the two iris centres, with a five-point fallback."""
    d = float(np.hypot(*(px_pts[468] - px_pts[473]).astype(float)))
    if not np.isfinite(d) or d < 1.0:
        eye0, eye1 = px_pts[FIVE_PT_IDX[0]], px_pts[FIVE_PT_IDX[1]]
        d = float(np.hypot(*(eye0 - eye1).astype(float)))
    return d


class MediaPipeDetector:
    """FaceLandmarker (478 mesh) wrapper emitting ``list[Detection]`` like YuNet."""

    def __init__(self, num_faces: int = _NUM_FACES,
                 long_side: int = _INFER_LONG_SIDE) -> None:
        # Lazy imports: keep this module importable without mediapipe installed.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._long = long_side
        self._ts = 0
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=_ensure_landmarker_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=_MIN_DETECT_CONF,
            min_face_presence_confidence=_MIN_PRESENCE_CONF,
            min_tracking_confidence=_MIN_TRACKING_CONF,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _next_ts(self) -> int:
        """A strictly monotonic millisecond timestamp (VIDEO mode requires it)."""
        self._ts = max(self._ts + 1, int(round(time.time() * 1000)))
        return self._ts

    def _infer(self, frame_bgr: np.ndarray):
        """Run FaceLandmarker on a downscaled RGB copy; returns the raw result."""
        import cv2

        frame_h, frame_w = frame_bgr.shape[:2]
        scale = self._long / max(frame_w, frame_h) if self._long else 1.0
        small = (cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else frame_bgr)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(image, self._next_ts())

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect all faces in a BGR frame; returns an empty list when none found."""
        frame_h, frame_w = frame_bgr.shape[:2]
        result = self._infer(frame_bgr)
        faces = getattr(result, "face_landmarks", None)
        if not faces:
            return []

        detections: list[Detection] = []
        for face in faces:
            # Normalised mesh -> FULL-frame pixel coords (independent of downscale).
            px_pts = np.rint(np.array(
                [[lm.x * frame_w, lm.y * frame_h] for lm in face])).astype(int)
            box = _clamp_roi(int(px_pts[:, 0].min()), int(px_pts[:, 1].min()),
                             int(px_pts[:, 0].max() - px_pts[:, 0].min()),
                             int(px_pts[:, 1].max() - px_pts[:, 1].min()),
                             frame_w, frame_h)

            iod = _iod(px_pts)
            gap = _NOSE_MARGIN * iod
            nose_x = float(px_pts[4][0])
            forehead = _mesh_rect(px_pts, FOREHEAD_IDX, frame_w, frame_h)
            r_cheek = _clamp_off_nose(
                _mesh_rect(px_pts, RCHEEK_IDX, frame_w, frame_h),
                nose_x, gap, frame_w, frame_h)
            l_cheek = _clamp_off_nose(
                _mesh_rect(px_pts, LCHEEK_IDX, frame_w, frame_h),
                nose_x, gap, frame_w, frame_h)

            landmarks = px_pts[FIVE_PT_IDX].astype(float).reshape(5, 2)
            score = float(min(1.0, iod / 80.0))
            detections.append(
                Detection(box, (forehead, r_cheek, l_cheek), landmarks, score))
        return detections


if __name__ == "__main__":  # smoke test without a camera
    detector = MediaPipeDetector()
    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    found = detector.detect(blank)
    print(f"MediaPipe FaceLandmarker loaded OK; faces on a blank frame: {len(found)}")
