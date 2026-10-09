"""Video decoding to RGB frames (thin OpenCV wrapper). No model logic.

OpenCV decodes BGR; we always hand back RGB uint8 so downstream code never has
to remember the channel order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    n_frames: int
    width: int
    height: int


def read_meta(path: str | Path) -> VideoMeta:
    """Return container metadata without decoding frames."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video: {p}")
        return VideoMeta(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def load_frames_rgb(path: str | Path, max_frames: int | None = None,
                    stride: int = 1) -> tuple[np.ndarray, float]:
    """Decode a video to an (T, H, W, 3) uint8 RGB array and its fps."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    cap = cv2.VideoCapture(str(p))
    frames: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video: {p}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise OSError(f"no frames decoded from {p}")
    return np.stack(frames, axis=0), fps
