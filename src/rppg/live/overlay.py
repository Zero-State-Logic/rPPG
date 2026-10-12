"""OpenCV overlay: per-person box, ROI, HR, experimental BP, quality and pulse.

Pure drawing on an in-memory BGR frame. Colour encodes signal quality (green =
strong pulse, amber = usable, red/grey = too weak or still measuring) so that
during the moving / far-away scenarios the viewer can see confidence drop and
recover honestly. Blood pressure is always tagged ``exp`` to keep it clearly
experimental.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detect import Track
from .estimate import Reading

_GOOD = (90, 220, 120)
_FAIR = (60, 200, 240)
_POOR = (120, 120, 235)
_IDLE = (170, 170, 170)
_PANEL = (28, 28, 30)
_TEXT = (240, 240, 240)
_ACCENT = (230, 180, 80)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_SNR_GOOD = 3.0
_SNR_FAIR = 0.0


def _quality(reading: Reading) -> tuple[tuple[int, int, int], str]:
    """Map a reading to a (colour, label) pair for the overlay."""
    if not reading.ready:
        return _IDLE, f"measuring {reading.seconds:0.0f}s"
    if reading.snr >= _SNR_GOOD:
        return _GOOD, "strong"
    if reading.snr >= _SNR_FAIR:
        return _FAIR, "fair"
    return _POOR, "weak"


def _sparkline(frame: np.ndarray, pulse: np.ndarray,
               x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    """Draw a normalized pulse waveform inside the (x, y, w, h) box."""
    if pulse.size < 4:
        return
    sig = pulse - pulse.min()
    span = sig.max()
    if span < 1e-9:
        return
    sig = sig / span
    xs = np.linspace(x, x + w, sig.size).astype(np.int32)
    ys = (y + h - sig * h).astype(np.int32)
    cv2.polylines(frame, [np.stack([xs, ys], axis=1)], False, color, 1, cv2.LINE_AA)


def _text(frame: np.ndarray, s: str, org: tuple[int, int], scale: float,
          color: tuple[int, int, int], thick: int = 1) -> None:
    """putText with a dark stroke underneath for legibility on any background."""
    cv2.putText(frame, s, org, _FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, _FONT, scale, color, thick, cv2.LINE_AA)


def draw_face(frame: np.ndarray, track: Track, reading: Reading) -> None:
    """Draw one person's box, ROI, readings and pulse sparkline."""
    x, y, w, h = track.det.box
    rx, ry, rw, rh = track.det.roi
    color, qlabel = _quality(reading)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), _ACCENT, 1)

    panel_w = max(w, 168)
    panel_h = 70
    py = y - panel_h - 6
    if py < 2:
        py = y + h + 6
    px = max(2, min(x, frame.shape[1] - panel_w - 2))
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), _PANEL, -1)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), color, 1)

    if reading.ready:
        _text(frame, f"P{track.id}  {reading.hr:0.0f} bpm", (px + 8, py + 22),
              0.6, _TEXT, 1)
        _text(frame, f"BP {reading.bp_sys:0.0f}/{reading.bp_dia:0.0f} exp",
              (px + 8, py + 42), 0.48, (180, 210, 255), 1)
        _text(frame, f"{qlabel}  SNR {reading.snr:0.1f}dB", (px + 8, py + 60),
              0.42, color, 1)
        _sparkline(frame, reading.pulse, px + panel_w - 60, py + 8, 54, 20, color)
    else:
        _text(frame, f"P{track.id}", (px + 8, py + 22), 0.6, _TEXT, 1)
        _text(frame, qlabel, (px + 8, py + 46), 0.5, _IDLE, 1)


def draw_header(frame: np.ndarray, fps: float, n_people: int) -> None:
    """Top banner: title, live FPS, people count and the BP disclaimer."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 34), _PANEL, -1)
    _text(frame, "rPPG live  -  HR from ProPOS", (10, 23), 0.6, _ACCENT, 1)
    _text(frame, f"{fps:0.0f} fps   people: {n_people}", (w - 230, 23), 0.55,
          _TEXT, 1)
    _text(frame, "BP = experimental, not a medical measurement",
          (10, frame.shape[0] - 12), 0.46, (170, 190, 255), 1)
