"""Facial skin-ROI extraction via MediaPipe FaceMesh.

Produces a per-frame boolean skin mask over the face (convex hull minus eyes and
lips, intersected with a YCrCb skin-colour gate). Averaging only these pixels is
what raises rPPG SNR; excluding eyes/mouth/specular pixels removes non-pulsatile
contamination.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - mediapipe is a hard dep for this module
    mp = None

_SKIN_LO = (0, 133, 77)
_SKIN_HI = (255, 173, 127)


def _indices(connections) -> list[int]:
    idx: set[int] = set()
    for a, b in connections:
        idx.add(a)
        idx.add(b)
    return sorted(idx)


class FaceRoiExtractor:
    """Extracts a skin-ROI boolean mask from an RGB frame via FaceMesh."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        if mp is None:
            raise ImportError("mediapipe is required for FaceRoiExtractor")
        self._fm = mp.solutions.face_mesh
        self._mesh = self._fm.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )
        self._exclude = (self._fm.FACEMESH_LEFT_EYE, self._fm.FACEMESH_RIGHT_EYE,
                         self._fm.FACEMESH_LIPS)

    def close(self) -> None:
        self._mesh.close()

    def roi_mask(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Return a boolean (H, W) skin mask, or None if no face is found."""
        h, w = frame_rgb.shape[:2]
        res = self._mesh.process(np.ascontiguousarray(frame_rgb))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts.reshape(-1, 1, 2)), 255)
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)

        for region in self._exclude:
            region_pts = pts[_indices(region)].reshape(-1, 1, 2)
            cv2.fillConvexPoly(mask, cv2.convexHull(region_pts), 0)

        ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
        skin = cv2.inRange(ycrcb, _SKIN_LO, _SKIN_HI)
        mask = cv2.bitwise_and(mask, skin)
        return mask > 0
