"""Reduce ROI-masked frames to a (3, T) RGB trace via spatial averaging.

Specular-highlight / deep-shadow pixels are dropped (they carry no pulse and bias
the mean); frames with no detected face are interpolated over so downstream
filters see a uniform time series.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class RoiExtractor(Protocol):
    def roi_mask(self, frame_rgb: np.ndarray) -> np.ndarray | None: ...


def masked_channel_mean(frame_rgb: np.ndarray, mask: np.ndarray | None,
                        reject_specular: bool = True) -> np.ndarray | None:
    """Spatial mean of skin pixels per channel -> (3,), or None if mask empty."""
    if mask is None or not np.any(mask):
        return None
    pix = frame_rgb[mask].astype(np.float64)        # (N, 3)
    if reject_specular and pix.shape[0] > 20:
        lum = pix.mean(axis=1)
        lo, hi = np.percentile(lum, [5, 95])
        keep = (lum >= lo) & (lum <= hi)
        if int(keep.sum()) > 10:
            pix = pix[keep]
    return pix.mean(axis=0)


def frames_to_rgb_trace(frames_rgb: np.ndarray, extractor: RoiExtractor,
                        reject_specular: bool = True) -> np.ndarray:
    """Map (T, H, W, 3) frames to a (3, T) RGB trace; NaN gaps are interpolated."""
    n = frames_rgb.shape[0]
    trace = np.full((3, n), np.nan, dtype=np.float64)
    for t in range(n):
        mask = extractor.roi_mask(frames_rgb[t])
        value = masked_channel_mean(frames_rgb[t], mask, reject_specular)
        if value is not None:
            trace[:, t] = value

    index = np.arange(n)
    for c in range(3):
        row = trace[c]
        missing = np.isnan(row)
        if missing.all():
            row[:] = 0.0
        elif missing.any():
            row[missing] = np.interp(index[missing], index[~missing], row[~missing])
    return trace
