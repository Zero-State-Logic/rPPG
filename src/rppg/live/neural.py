"""Optional trained-ProPOSNet HR readout for the live demo (CPU).

Buffers the resized ROI crops per face, builds a **Standardized** clip (matching
the training preprocessing), runs ProPOSNet to a pulse, and reads HR with the same
sub-bin peak as the classical path. Loaded only when ``--neural`` is passed -- the
demo works fully without it. This is how the *trained* model is shown running live
alongside the classical ProPOS estimate.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import torch

from ..hr.refine import bandpass, hr_peak_interp
from ..models.proposnet import ProPOSNet

_T = 160          # clip length used in training (train_deep --chunk-len default)
_RESIZE = 72
_EPS = 1e-7


class NeuralHR:
    """Per-face ProPOSNet heart-rate estimator from buffered ROI crops."""

    def __init__(self, ckpt_path: str, resize: int = _RESIZE,
                 clip_len: int = _T) -> None:
        self.model = ProPOSNet()
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.model.eval()
        self.resize = resize
        self.clip_len = clip_len
        self._buf: dict[int, deque] = {}

    def add(self, face_id: int, timestamp: float, roi_bgr: np.ndarray) -> None:
        """Buffer one resized RGB ROI crop for a face."""
        if roi_bgr.size == 0:
            return
        crop = cv2.resize(roi_bgr, (self.resize, self.resize))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        self._buf.setdefault(face_id, deque(maxlen=self.clip_len * 2)).append(
            (timestamp, rgb))

    def estimate(self, face_id: int) -> float | None:
        """Neural HR (bpm) for a face, or None until enough frames are buffered."""
        buf = self._buf.get(face_id)
        if not buf or len(buf) < self.clip_len:
            return None
        items = list(buf)[-self.clip_len:]
        ts = np.array([t for t, _ in items])
        span = float(ts[-1] - ts[0])
        if span <= 0.5:
            return None
        fps = (len(ts) - 1) / span
        frames = np.stack([f for _, f in items], axis=0)         # (T, H, W, 3)
        std = (frames - frames.mean()) / (frames.std() + _EPS)   # Standardized
        x = torch.from_numpy(std.transpose(3, 0, 1, 2)[None]).float()  # (1,3,T,H,W)
        with torch.no_grad():
            pulse = self.model(x)[0].cpu().numpy()
        hr, _ = hr_peak_interp(bandpass(pulse, fps), fps)
        return hr if np.isfinite(hr) else None

    def prune(self, live_ids: set[int]) -> None:
        """Forget buffers whose faces are no longer tracked."""
        for gone in [i for i in self._buf if i not in live_ids]:
            del self._buf[gone]
