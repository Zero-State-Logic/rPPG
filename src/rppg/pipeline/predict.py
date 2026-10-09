"""End-to-end convenience: a video file -> heart rate via a classical method."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..face.roi import FaceRoiExtractor
from ..hr.metrics import estimate_hr_fft
from ..io.video import load_frames_rgb
from ..methods.classical.pos import pos_pulse
from ..methods.classical.propos import propos_pulse
from ..preprocess.rgb_trace import frames_to_rgb_trace

_METHODS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "pos": pos_pulse,
    "propos": propos_pulse,
}


def video_to_trace(path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Decode a video and reduce it to a (3, T) skin RGB trace + fps."""
    frames, fps = load_frames_rgb(path, max_frames=max_frames)
    extractor = FaceRoiExtractor()
    try:
        trace = frames_to_rgb_trace(frames, extractor)
    finally:
        extractor.close()
    return trace, fps


def estimate_hr_from_video(path: str | Path, method: str = "propos",
                           max_frames: int | None = None) -> float:
    """Estimate heart rate (bpm) from a face video using the named method."""
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    trace, fps = video_to_trace(path, max_frames=max_frames)
    pulse = _METHODS[method](trace, fps)
    return estimate_hr_fft(pulse, fps)
