"""Window-level feature table for rPPG reliability and abstention research.

For every sliding window of every cached subject this records the live estimate,
the ground truth and a set of reference-free reliability features, so calibration
and abstention experiments (leave-subject-out, risk-coverage) run in milliseconds:

* the fused classical HR + SNR (exactly what the live demo shows) and each method's
  (POS, CHROM, ProPOS) HR + SNR on the pooled skin trace -- method agreement;
* each ROI's (forehead, right cheek, left cheek) fused HR + SNR on its own skin
  trace -- spatial agreement between independent skin regions;
* method spread and ROI spread (max - min HR) and the median ROI HR;
* face-motion speed and scale change (camera-side kinematics);
* the face detection rate inside the window; and
* the contact-PPG reference HR and the oximeter HR (ground truth) plus the error.
"""
from __future__ import annotations

import pathlib

import numpy as np

from ..hr.refine import fuse_estimates
from .frame_cache import FrameCache
from .windows import method_estimates, reference_hr, window_starts

METHODS = ("pos", "chrom", "propos")
ROIS = ("fh", "rc", "lc")          # forehead, right cheek, left cheek

COLUMNS = (
    "start", "t_sec", "ref_hr", "ox_hr", "pred_hr", "pred_snr",
    *(f"{m}_{k}" for m in METHODS for k in ("hr", "snr")),
    *(f"{r}_{k}" for r in ROIS for k in ("hr", "snr")),
    "method_spread", "roi_spread", "roi_median_hr",
    "motion", "scale_change", "det_rate", "abs_err",
)


def roi_trace(cache: FrameCache, roi: int) -> np.ndarray:
    """``(3, T)`` RGB trace of one ROI: skin mean, raw patch mean where no skin."""
    rgb = cache.roi_rgb[:, roi, :].copy()
    missing = np.isnan(rgb)
    rgb[missing] = cache.roi_raw[:, roi, :][missing]
    return rgb.T


def _spread(values: list[float]) -> float:
    """``max - min`` of the finite values (NaN when fewer than two are finite)."""
    finite = [v for v in values if np.isfinite(v)]
    return float(max(finite) - min(finite)) if len(finite) >= 2 else float("nan")


def _kinematics(box: np.ndarray) -> tuple[float, float]:
    """Mean face-centre speed and box-width std, both in % of the mean face width."""
    width = float(np.nanmean(box[:, 2])) if box.size else float("nan")
    if not np.isfinite(width) or width <= 0 or box.shape[0] < 2:
        return float("nan"), float("nan")
    cx = box[:, 0] + box[:, 2] / 2.0
    cy = box[:, 1] + box[:, 3] / 2.0
    speed = float(np.nanmean(np.hypot(np.diff(cx), np.diff(cy)))) / width * 100.0
    scale = float(np.nanstd(box[:, 2])) / width * 100.0
    return speed, scale


def window_records(cache: FrameCache, subject: str, win_sec: float = 10.0,
                   step_sec: float = 2.0) -> list[dict]:
    """One feature record per full sliding window of ``cache`` (see module doc)."""
    fps = cache.fps
    win = int(round(win_sec * fps))
    pooled = cache.pooled_rgb().T
    rois = [roi_trace(cache, i) for i in range(len(ROIS))]
    records = []
    for st in window_starts(cache.n_frames, fps, win_sec, step_sec):
        sl = slice(st, st + win)
        ests = method_estimates(pooled[:, sl], fps)
        pred_hr, pred_snr = fuse_estimates(ests)
        roi_fused = [fuse_estimates(method_estimates(t[:, sl], fps)) for t in rois]
        ref = reference_hr(cache.ppg[sl], fps)
        ox = float(np.mean(cache.gt_hr[sl])) if cache.gt_hr.size else float("nan")
        motion, scale = _kinematics(cache.box[sl])
        rec = {"subject": subject, "start": st, "t_sec": st / fps, "ref_hr": ref,
               "ox_hr": ox, "pred_hr": pred_hr, "pred_snr": pred_snr}
        for name, (hr, snr) in zip(METHODS, ests):
            rec[f"{name}_hr"], rec[f"{name}_snr"] = hr, snr
        for name, (hr, snr) in zip(ROIS, roi_fused):
            rec[f"{name}_hr"], rec[f"{name}_snr"] = hr, snr
        roi_hrs = [hr for hr, _ in roi_fused]
        rec["method_spread"] = _spread([hr for hr, _ in ests])
        rec["roi_spread"] = _spread(roi_hrs)
        finite_roi = [h for h in roi_hrs if np.isfinite(h)]
        rec["roi_median_hr"] = float(np.median(finite_roi)) if finite_roi else float("nan")
        rec["motion"], rec["scale_change"] = motion, scale
        rec["det_rate"] = float(np.mean(cache.detected[sl]))
        rec["abs_err"] = (abs(pred_hr - ref)
                          if np.isfinite(pred_hr) and np.isfinite(ref) else float("nan"))
        records.append(rec)
    return records


def build_table(records: list[dict]) -> dict[str, np.ndarray]:
    """Column-oriented table: ``subject`` (str) plus every name in ``COLUMNS``."""
    table = {"subject": np.asarray([r["subject"] for r in records], dtype=str)}
    for col in COLUMNS:
        table[col] = np.asarray([r[col] for r in records], dtype=np.float64)
    return table


def save_table(path: str | pathlib.Path, table: dict[str, np.ndarray]) -> None:
    """Write a window table to a compressed ``.npz``."""
    np.savez_compressed(path, **table)


def load_table(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Read a window table written by :func:`save_table`."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
