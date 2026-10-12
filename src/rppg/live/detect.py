"""Multi-face detection + forehead/cheek ROIs + a stable per-person tracker.

Detection uses OpenCV's YuNet (``cv2.FaceDetectorYN``) on a **downscaled** copy of
the frame (boxes + landmarks are scaled back up), so detection cost stays low at
any capture resolution -- measured ~13 ms at a 480-wide downscale vs ~53 ms at
720p. From YuNet's 5 landmarks a **forehead + left-cheek + right-cheek** ROI
triplet is built: the regions the rPPG literature (Kumar 2015; Kossack ICCVW 2021;
R2I-rPPG 2024) finds carry the strongest, most motion-robust pulse. Mouth/chin are
deliberately excluded (speech/expression motion). ROIs are anchored to
inter-ocular distance so they stay correct under head roll and distance, with a
face-box-fraction fallback when landmarks are unreliable (tiny/far faces).

A greedy centroid tracker assigns each face a persistent integer id. The YuNet
ONNX model lives in ``models/`` and is downloaded on first use.
"""
from __future__ import annotations

import math
import pathlib
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

_MODEL_DIR = pathlib.Path(__file__).resolve().parents[3] / "models"
_YUNET_NAME = "face_detection_yunet_2023mar.onnx"
_YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet/face_detection_yunet_2023mar.onnx")

_DETECT_LONG_SIDE = 480   # detect on a frame downscaled to this long side (speed)
_MIN_IOD = 8.0            # below this inter-ocular distance, use box-fraction ROIs

# Forehead + cheek ROI geometry, in multiples of inter-ocular distance (landmarks).
_FOREHEAD_UP = 0.60       # forehead centre sits this * iod above the eye line
_FOREHEAD_W, _FOREHEAD_H = 1.00, 0.45
_CHEEK_W, _CHEEK_H = 0.45, 0.60

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


def _rect_center(cx: float, cy: float, w: float, h: float,
                 frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Clamped (x, y, w, h) rect centred on (cx, cy)."""
    return _clamp_roi(int(cx - w / 2), int(cy - h / 2), int(w), int(h),
                      frame_w, frame_h)


def _rois_from_box(box: tuple[int, int, int, int], frame_w: int,
                   frame_h: int) -> tuple[tuple[int, int, int, int], ...]:
    """Fallback forehead + cheek rects from face-box fractions (far/tiny faces)."""
    x, y, w, h = box
    forehead = _clamp_roi(int(x + 0.30 * w), int(y + 0.07 * h),
                          int(0.40 * w), int(0.13 * h), frame_w, frame_h)
    r_cheek = _clamp_roi(int(x + 0.12 * w), int(y + 0.52 * h),
                         int(0.24 * w), int(0.20 * h), frame_w, frame_h)
    l_cheek = _clamp_roi(int(x + 0.64 * w), int(y + 0.52 * h),
                         int(0.24 * w), int(0.20 * h), frame_w, frame_h)
    return (forehead, r_cheek, l_cheek)


def _rois_from_landmarks(lm: np.ndarray, box: tuple[int, int, int, int],
                         frame_w: int, frame_h: int
                         ) -> tuple[tuple[int, int, int, int], ...]:
    """Forehead + both cheeks from YuNet's 5 landmarks, scaled by inter-ocular dist.

    Landmark order: [0]=right eye, [1]=left eye, [2]=nose, [3]=right mouth corner,
    [4]=left mouth corner ("right"/"left" are the subject's own sides).
    """
    re, le, _, rm, lm_ = lm[0], lm[1], lm[2], lm[3], lm[4]
    iod = math.hypot(le[0] - re[0], le[1] - re[1])
    if not math.isfinite(iod) or iod < _MIN_IOD:
        return _rois_from_box(box, frame_w, frame_h)

    eye_cx, eye_cy = (re[0] + le[0]) / 2.0, (re[1] + le[1]) / 2.0
    mouth_cx, mouth_cy = (rm[0] + lm_[0]) / 2.0, (rm[1] + lm_[1]) / 2.0
    ex, ey = (le[0] - re[0]) / iod, (le[1] - re[1]) / iod   # eye-line unit vector
    ux, uy = ey, -ex                                        # perpendicular ("up")
    if ux * (mouth_cx - eye_cx) + uy * (mouth_cy - eye_cy) > 0:  # point away from mouth
        ux, uy = -ux, -uy

    forehead = _rect_center(eye_cx + ux * _FOREHEAD_UP * iod,
                            eye_cy + uy * _FOREHEAD_UP * iod,
                            _FOREHEAD_W * iod, _FOREHEAD_H * iod, frame_w, frame_h)
    r_cheek = _rect_center((re[0] + rm[0]) / 2.0, (re[1] + rm[1]) / 2.0,
                           _CHEEK_W * iod, _CHEEK_H * iod, frame_w, frame_h)
    l_cheek = _rect_center((le[0] + lm_[0]) / 2.0, (le[1] + lm_[1]) / 2.0,
                           _CHEEK_W * iod, _CHEEK_H * iod, frame_w, frame_h)
    return (forehead, r_cheek, l_cheek)


@dataclass
class Detection:
    """One detected face: ``box`` and each ``rois`` rect are (x, y, w, h) pixels.

    ``rois`` is (forehead, right cheek, left cheek); ``.roi`` returns the forehead
    for callers that still expect a single rect.
    """

    box: tuple[int, int, int, int]
    rois: tuple[tuple[int, int, int, int], ...]
    landmarks: np.ndarray
    score: float

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return self.rois[0]


@dataclass
class Track:
    """A tracked face with a persistent id and its latest detection."""

    id: int
    det: Detection
    centroid: tuple[float, float]
    missed: int = 0


class YuNetDetector:
    """YuNet wrapper that detects on a downscaled frame and rescales the results."""

    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3,
                 top_k: int = 20, detect_long_side: int = _DETECT_LONG_SIDE) -> None:
        self._det = cv2.FaceDetectorYN.create(
            _ensure_model(), "", (320, 320), score_threshold, nms_threshold, top_k)
        self._size = (320, 320)
        self._long = detect_long_side

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect all faces in a BGR frame; returns an empty list when none."""
        frame_h, frame_w = frame_bgr.shape[:2]
        scale = self._long / max(frame_w, frame_h) if self._long else 1.0
        if scale < 1.0:
            small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small, scale = frame_bgr, 1.0
        small_h, small_w = small.shape[:2]
        if (small_w, small_h) != self._size:
            self._det.setInputSize((small_w, small_h))
            self._size = (small_w, small_h)
        _, faces = self._det.detect(small)
        detections: list[Detection] = []
        if faces is None:
            return detections
        inv = 1.0 / scale
        for f in faces:
            x, y, w, h = (max(0, int(v * inv)) for v in f[:4])
            lm = f[4:14].reshape(5, 2) * inv
            rois = _rois_from_landmarks(lm, (x, y, w, h), frame_w, frame_h)
            detections.append(Detection((x, y, w, h), rois, lm, float(f[14])))
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
    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    found = detector.detect(blank)
    print(f"YuNet loaded OK; faces on a blank frame: {len(found)}")
