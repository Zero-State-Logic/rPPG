"""PyTorch datasets turning UBFC-rPPG and PURE recordings into model-ready clips.

Each recording is decoded and resized to 72x72 **once** at construction time and
cached in memory as uint8 frames (~0.5 GB for 16 UBFC subjects), so every epoch
slices clips from RAM instead of re-opening/seeking the video file. Clips are
``chunk_len`` frames, encoded as a ``DiffNormalized`` and/or ``Standardized``
stream per the shared preprocessing contract; the per-frame BVP label is sliced
and z-scored.

Layouts -- UBFC: ``<root>/<subject>/vid.avi`` + ``ground_truth.txt`` (three
whitespace rows: PPG waveform, HR bpm, timestamps; only the waveform is used).
PURE: ``<root>/<subject>/`` with ``*.png`` frames plus a ``*.json`` holding a
``"/FullPackage"`` list of ``{"Timestamp", "Value": {"waveform"}}`` samples and an
``"/Image"`` list of frame timestamps. No writes or network at import time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

_EPS = 1e-7
_RESIZE = 72
_CHUNK = 160
_DATA_TYPES = ("DiffNormalized", "Standardized", "both")


@dataclass(frozen=True)
class SampleSource:
    """One full recording; ``kind`` selects the loader, paths point at the media."""

    kind: str          # "ubfc" or "pure"
    subject_id: str
    media_path: str    # vid.avi (ubfc) or frames directory (pure)
    label_path: str    # ground_truth.txt (ubfc) or metadata .json (pure)


def list_ubfc_subjects(root: str | Path) -> list[tuple[str, str, str]]:
    """Return ``(subject_id, video_path, gt_path)`` for every UBFC subject dir."""
    base = Path(root)
    out: list[tuple[str, str, str]] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        vid, gt = sub / "vid.avi", sub / "ground_truth.txt"
        if vid.exists() and gt.exists():
            out.append((sub.name, str(vid), str(gt)))
    return out


def list_pure_subjects(root: str | Path) -> list[tuple[str, str, str]]:
    """Return ``(subject_id, frames_dir, json_path)`` for every PURE subject dir."""
    base = Path(root)
    out: list[tuple[str, str, str]] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        if not any(sub.glob("*.png")):
            continue
        cand = sub / f"{sub.name}.json"
        if not cand.exists():
            js = sorted(sub.glob("*.json"))
            if not js:
                continue
            cand = js[0]
        out.append((sub.name, str(sub), str(cand)))
    return out


def ubfc_sources(root: str | Path) -> list[SampleSource]:
    """Wrap :func:`list_ubfc_subjects` results as :class:`SampleSource` objects."""
    return [SampleSource("ubfc", s, v, g) for s, v, g in list_ubfc_subjects(root)]


def pure_sources(root: str | Path) -> list[SampleSource]:
    """Wrap :func:`list_pure_subjects` results as :class:`SampleSource` objects."""
    return [SampleSource("pure", s, d, j) for s, d, j in list_pure_subjects(root)]


def _ubfc_label(gt_path: str, n_frames: int) -> np.ndarray:
    """Per-frame PPG label (length ``n_frames``) from the first row of a UBFC gt."""
    with open(gt_path, "r", encoding="utf-8") as fh:
        wave = np.asarray(fh.readline().split(), dtype=np.float64)
    if wave.size == 0 or n_frames <= 0:
        return np.zeros(max(n_frames, 0), dtype=np.float64)
    src = np.linspace(0.0, 1.0, wave.size)
    return np.interp(np.linspace(0.0, 1.0, n_frames), src, wave)


def _pure_label(json_path: str, n_frames: int) -> tuple[np.ndarray, float]:
    """Per-frame BVP label (length ``n_frames``) and fps from a PURE json."""
    with open(json_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    full = meta.get("/FullPackage") or meta.get("FullPackage") or []
    imgs = meta.get("/Image") or meta.get("Image") or []

    ts_full: list[float] = []
    wave: list[float] = []
    for entry in full:
        val = entry.get("Value", {}) if isinstance(entry, dict) else {}
        w = val.get("waveform") if isinstance(val, dict) else None
        if w is None:
            continue
        ts_full.append(float(entry.get("Timestamp", len(ts_full))))
        wave.append(float(w))
    ts_img = [float(e.get("Timestamp", i)) for i, e in enumerate(imgs)
              if isinstance(e, dict)]

    fps = 30.0
    if len(ts_img) >= 2 and ts_img[-1] > ts_img[0]:
        fps = (len(ts_img) - 1) * 1e9 / (ts_img[-1] - ts_img[0])  # PURE ts are ns
    if not wave:
        return np.zeros(max(n_frames, 0), dtype=np.float64), fps

    ts_a = np.asarray(ts_full)
    wave_a = np.asarray(wave, dtype=np.float64)
    order = np.argsort(ts_a)
    ts_a, wave_a = ts_a[order], wave_a[order]
    if ts_img:
        idx = np.clip(np.searchsorted(ts_a, np.asarray(ts_img)), 0, wave_a.size - 1)
        per_img = wave_a[idx]
    else:
        per_img = wave_a
    if per_img.size and n_frames > 0 and per_img.size != n_frames:
        src = np.linspace(0.0, 1.0, per_img.size)
        per_img = np.interp(np.linspace(0.0, 1.0, n_frames), src, per_img)
    return per_img.astype(np.float64), fps


def _read_all_video_frames(path: str, size: int) -> tuple[np.ndarray, float]:
    """Decode a whole video to ``(T, size, size, 3)`` uint8 RGB + fps."""
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if not frames:
        raise OSError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0).astype(np.uint8), fps


def _read_all_png_frames(files: Sequence[str], size: int) -> np.ndarray:
    """Read all PURE png frames to ``(T, size, size, 3)`` uint8 RGB."""
    frames: list[np.ndarray] = []
    for p in files:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
    if not frames:
        raise OSError("no png frames read")
    return np.stack(frames, axis=0).astype(np.uint8)


def _diff_normalized(frames: np.ndarray) -> np.ndarray:
    """``(f[t+1]-f[t])/(f[t+1]+f[t]+eps)`` z-scored, zero-padded back to length T."""
    num = frames[1:] - frames[:-1]
    den = frames[1:] + frames[:-1] + _EPS
    diff = num / den
    diff = (diff - float(diff.mean())) / (float(diff.std()) + _EPS)
    pad = np.zeros((1,) + diff.shape[1:], dtype=np.float32)
    return np.concatenate([diff.astype(np.float32), pad], axis=0)


def _standardized(frames: np.ndarray) -> np.ndarray:
    """Z-score the raw frames across the whole clip."""
    return ((frames - float(frames.mean())) / (float(frames.std()) + _EPS)
            ).astype(np.float32)


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """Clip or edge-repeat ``arr`` along axis 0 so it has exactly ``length`` rows."""
    if arr.shape[0] >= length:
        return arr[:length]
    tail = np.repeat(arr[-1:], length - arr.shape[0], axis=0)
    return np.concatenate([arr, tail], axis=0)


@dataclass
class _Record:
    """Resolved per-recording state: cached resized frames + per-frame label."""

    source: SampleSource
    frames: np.ndarray    # (T, size, size, 3) uint8 resized RGB, held in RAM
    label: np.ndarray     # per-frame, length == frames.shape[0]
    fps: float
    length: int


class ClipDataset(torch.utils.data.Dataset):
    """Model-ready clips drawn from a list of :class:`SampleSource`.

    Each item is ``(input, bvp_label)`` with ``input`` shaped ``(C, T, 72, 72)``
    float32 (``C`` = 3, or 6 for ``data_type="both"``) and ``bvp_label`` a z-scored
    ``(T,)`` float32 tensor. Frames are decoded once at construction and cached.
    """

    def __init__(self, sources: Sequence[SampleSource], chunk_len: int = _CHUNK,
                 resize: int = _RESIZE, data_type: str = "DiffNormalized") -> None:
        if data_type not in _DATA_TYPES:
            raise ValueError(f"unknown data_type: {data_type!r}")
        if chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")
        self.chunk_len = int(chunk_len)
        self.resize = int(resize)
        self.data_type = data_type
        self._records = [self._build_record(s) for s in sources]
        self._index: list[tuple[int, int]] = []
        for r_idx, rec in enumerate(self._records):
            if rec.length <= 0:
                continue
            n_chunks = max(1, rec.length // self.chunk_len)   # robust to short clips
            for c in range(n_chunks):
                self._index.append((r_idx, c * self.chunk_len))

    def _build_record(self, source: SampleSource) -> _Record:
        if source.kind == "ubfc":
            frames, fps = _read_all_video_frames(source.media_path, self.resize)
            n = int(frames.shape[0])
            return _Record(source, frames, _ubfc_label(source.label_path, n), fps, n)
        if source.kind == "pure":
            files = tuple(str(p) for p in sorted(Path(source.media_path).glob("*.png")))
            frames = _read_all_png_frames(files, self.resize)
            n = int(frames.shape[0])
            label, fps = _pure_label(source.label_path, n)
            return _Record(source, frames, label, fps, n)
        raise ValueError(f"unknown source kind: {source.kind!r}")

    def __len__(self) -> int:
        return len(self._index)

    def subject_ids(self) -> list[str]:
        """Subject id behind every clip, in index order (for subject-wise splits)."""
        return [self._records[r].source.subject_id for r, _ in self._index]

    def _encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode float32 ``(T, H, W, 3)`` frames to a ``(C, T, H, W)`` model input."""
        if self.data_type == "DiffNormalized":
            clip = _diff_normalized(frames)
        elif self.data_type == "Standardized":
            clip = _standardized(frames)
        else:
            clip = np.concatenate([_diff_normalized(frames),
                                   _standardized(frames)], axis=-1)
        return np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r_idx, start = self._index[idx]
        rec = self._records[r_idx]
        clip = _pad_to(rec.frames[start:start + self.chunk_len].astype(np.float32),
                       self.chunk_len)
        label = _pad_to(np.asarray(rec.label[start:start + self.chunk_len],
                                   dtype=np.float64), self.chunk_len)
        label = (label - float(label.mean())) / (float(label.std()) + _EPS)
        return (torch.from_numpy(self._encode(clip)),
                torch.from_numpy(label.astype(np.float32)))
