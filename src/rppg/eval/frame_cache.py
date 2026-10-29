"""Frame-level feature cache for offline rPPG research on UBFC.

Decoding video and running face detection dominate experiment time, so this module
does it ONCE per subject and stores compact per-frame features:

* per-ROI skin-mean RGB for the forehead / right-cheek / left-cheek patches
  (YCrCb-gated exactly like the live demo, with the raw patch mean as a fallback),
* the 4-corner background RGB (an ambient-light reference),
* face kinematics -- the box ``(x, y, w, h)`` and the 5 detector landmarks -- which
  give a camera-side motion reference,
* a per-frame detection flag, and
* the contact-PPG ground truth (and oximeter HR when present) aligned to frames.

Every downstream experiment (fused HR, inter-ROI phase, calibration, motion-reference
cancellation) then reads these arrays in milliseconds instead of re-decoding video.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import cv2
import numpy as np

_SKIN_LO = (0, 133, 77)        # YCrCb skin gate, identical to the live demo
_SKIN_HI = (255, 173, 127)
_BG_FRAC = 0.14                # corner patch size as a fraction of the frame side
_N_ROIS = 3                    # forehead, right cheek, left cheek
_N_LANDMARKS = 5
_MIN_POOLED_SKIN = 50          # below this many skin pixels the pool falls back to raw


@dataclass(frozen=True)
class FrameCache:
    """Per-frame features of one subject video (``T`` frames)."""

    fps: float
    roi_rgb: np.ndarray      # (T, 3, 3) skin-mean RGB per ROI (NaN when no skin px)
    roi_raw: np.ndarray      # (T, 3, 3) raw patch-mean RGB per ROI
    roi_npx: np.ndarray      # (T, 3) skin-pixel count per ROI
    roi_area: np.ndarray     # (T, 3) patch area (pixels) per ROI
    bg_rgb: np.ndarray       # (T, 3) 4-corner background RGB
    box: np.ndarray          # (T, 4) face box x, y, w, h (held through misses)
    landmarks: np.ndarray    # (T, 5, 2) detector landmarks (held through misses)
    detected: np.ndarray     # (T,) True where the detector found a face this frame
    ppg: np.ndarray          # (T,) contact PPG aligned to the frames
    gt_hr: np.ndarray        # (T,) oximeter HR in bpm, or empty when absent

    @property
    def n_frames(self) -> int:
        """Number of cached frames."""
        return int(self.roi_rgb.shape[0])

    def pooled_rgb(self) -> np.ndarray:
        """(T, 3) RGB pooled over all ROIs' skin pixels, as the live demo computes it.

        Skin means are weighted by their pixel counts; frames with fewer than
        ``_MIN_POOLED_SKIN`` skin pixels fall back to the area-weighted raw patch mean,
        so a reading is never lost under hard light or darker skin tones.
        """
        w_skin = self.roi_npx.astype(np.float64)
        skin = np.nan_to_num(self.roi_rgb)
        tot_skin = w_skin.sum(axis=1)
        pooled = (skin * w_skin[:, :, None]).sum(axis=1) / np.maximum(tot_skin, 1.0)[:, None]
        w_raw = self.roi_area.astype(np.float64)
        raw = (np.nan_to_num(self.roi_raw) * w_raw[:, :, None]).sum(axis=1)
        raw = raw / np.maximum(w_raw.sum(axis=1), 1.0)[:, None]
        use_raw = tot_skin < _MIN_POOLED_SKIN
        return np.where(use_raw[:, None], raw, pooled)


def roi_skin_mean(frame_bgr: np.ndarray, roi: tuple[int, int, int, int]
                  ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Skin-gated and raw mean RGB of one rectangular ROI.

    Returns ``(skin_rgb, raw_rgb, n_skin, area)``; ``skin_rgb`` is NaN when no pixel
    passes the YCrCb gate and both are NaN for an empty patch.
    """
    nan3 = np.full(3, np.nan)
    x, y, w, h = (int(v) for v in roi)
    patch = frame_bgr[max(y, 0):max(y, 0) + max(h, 0), max(x, 0):max(x, 0) + max(w, 0)]
    if patch.size == 0:
        return nan3, nan3.copy(), 0, 0
    flat = patch.reshape(-1, 3).astype(np.float64)
    raw = flat.mean(axis=0)[::-1]                                   # BGR -> RGB
    ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    mask = np.all((ycc >= _SKIN_LO) & (ycc <= _SKIN_HI), axis=1)
    n_skin = int(mask.sum())
    skin = flat[mask].mean(axis=0)[::-1] if n_skin else nan3
    return skin, raw, n_skin, int(flat.shape[0])


def background_mean(frame_bgr: np.ndarray, frac: float = _BG_FRAC) -> np.ndarray:
    """Mean RGB of the four frame corners (an ambient-light reference)."""
    h, w = frame_bgr.shape[:2]
    bw, bh = max(1, int(w * frac)), max(1, int(h * frac))
    corners = (frame_bgr[:bh, :bw], frame_bgr[:bh, w - bw:],
               frame_bgr[h - bh:, :bw], frame_bgr[h - bh:, w - bw:])
    px = np.concatenate([c.reshape(-1, 3) for c in corners]).astype(np.float64)
    return px.mean(axis=0)[::-1]


def read_ground_truth(gt_path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ppg, oximeter_hr)`` from a UBFC ``ground_truth.txt``.

    Row 0 is the contact PPG waveform, row 1 (when present) the oximeter heart rate
    in bpm. A missing HR row yields an empty array.
    """
    with open(gt_path, encoding="utf-8") as fh:
        rows = [line.split() for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"empty ground truth file: {gt_path}")
    ppg = np.asarray(rows[0], dtype=np.float64)
    hr = np.asarray(rows[1], dtype=np.float64) if len(rows) > 1 else np.empty(0)
    return ppg, hr


def extract_subject(video_path: str | pathlib.Path, gt_path: str | pathlib.Path,
                    detector, max_frames: int | None = None) -> FrameCache:
    """Decode one subject video and build its :class:`FrameCache`.

    ``detector`` is any object with ``detect(frame) -> list[Detection]`` (YuNet or
    MediaPipe). The largest face is used; on a missed frame the last face geometry
    is held (and ``detected`` is False) so the colour trace stays continuous.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    rows: dict[str, list] = {k: [] for k in
                             ("rgb", "raw", "npx", "area", "bg", "box", "lm", "det")}
    last = None
    try:
        while max_frames is None or len(rows["det"]) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detector.detect(frame)
            found = bool(dets)
            if found:
                last = max(dets, key=lambda d: d.box[2] * d.box[3])
            rows["det"].append(found)
            rows["bg"].append(background_mean(frame))
            if last is None:
                _append_missing(rows)
                continue
            per_roi = [roi_skin_mean(frame, r) for r in last.rois[:_N_ROIS]]
            rows["rgb"].append([p[0] for p in per_roi])
            rows["raw"].append([p[1] for p in per_roi])
            rows["npx"].append([p[2] for p in per_roi])
            rows["area"].append([p[3] for p in per_roi])
            rows["box"].append(np.asarray(last.box, dtype=np.float64))
            rows["lm"].append(np.asarray(last.landmarks, dtype=np.float64).reshape(_N_LANDMARKS, 2))
    finally:
        cap.release()

    ppg, gt_hr = read_ground_truth(gt_path)
    n = min(len(rows["det"]), ppg.size)
    if n == 0:
        raise ValueError(f"no aligned frames for {video_path}")
    return FrameCache(
        fps=fps,
        roi_rgb=np.asarray(rows["rgb"][:n], dtype=np.float64),
        roi_raw=np.asarray(rows["raw"][:n], dtype=np.float64),
        roi_npx=np.asarray(rows["npx"][:n], dtype=np.int64),
        roi_area=np.asarray(rows["area"][:n], dtype=np.int64),
        bg_rgb=np.asarray(rows["bg"][:n], dtype=np.float64),
        box=np.asarray(rows["box"][:n], dtype=np.float64),
        landmarks=np.asarray(rows["lm"][:n], dtype=np.float64),
        detected=np.asarray(rows["det"][:n], dtype=bool),
        ppg=ppg[:n],
        gt_hr=gt_hr[:n] if gt_hr.size else gt_hr,
    )


def _append_missing(rows: dict[str, list]) -> None:
    """Placeholder row before the first face is ever found (all NaN / zero)."""
    nan33 = np.full((_N_ROIS, 3), np.nan)
    rows["rgb"].append(nan33)
    rows["raw"].append(nan33.copy())
    rows["npx"].append([0] * _N_ROIS)
    rows["area"].append([0] * _N_ROIS)
    rows["box"].append(np.full(4, np.nan))
    rows["lm"].append(np.full((_N_LANDMARKS, 2), np.nan))


def save_cache(path: str | pathlib.Path, cache: FrameCache) -> None:
    """Write a :class:`FrameCache` to a compressed ``.npz``."""
    np.savez_compressed(
        path, fps=np.float64(cache.fps), roi_rgb=cache.roi_rgb, roi_raw=cache.roi_raw,
        roi_npx=cache.roi_npx, roi_area=cache.roi_area, bg_rgb=cache.bg_rgb,
        box=cache.box, landmarks=cache.landmarks, detected=cache.detected,
        ppg=cache.ppg, gt_hr=cache.gt_hr)


def load_cache(path: str | pathlib.Path) -> FrameCache:
    """Read a :class:`FrameCache` written by :func:`save_cache`."""
    with np.load(path) as z:
        return FrameCache(
            fps=float(z["fps"]), roi_rgb=z["roi_rgb"], roi_raw=z["roi_raw"],
            roi_npx=z["roi_npx"], roi_area=z["roi_area"], bg_rgb=z["bg_rgb"],
            box=z["box"], landmarks=z["landmarks"], detected=z["detected"],
            ppg=z["ppg"], gt_hr=z["gt_hr"])
