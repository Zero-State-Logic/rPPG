"""Measure real HR accuracy on UBFC subjects with the live fused estimator.

For every ``<data-root>/<subject>/`` it decodes the video, finds the face each
frame with YuNet, builds the mid-face ROI colour trace, and over sliding windows
computes HR with the exact live method (POS+CHROM+ProPOS band-passed, sub-bin
peak, SNR fusion). Ground-truth HR per window is derived from the contact-PPG
waveform (row 0 of ``ground_truth.txt``) with the same sub-bin peak, so predicted
and reference HR are measured identically. Reports per-subject and overall MAE,
RMSE, within ±5 bpm and within ±3 bpm.

Usage:
    python scripts/eval_ubfc_fused.py --data-root C:\\rppg_data\\UBFC --max-frames 900
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.hr.metrics import accuracy_within, mae, rmse  # noqa: E402
from rppg.hr.refine import bandpass, fuse_estimates, hr_peak_interp  # noqa: E402
from rppg.live.detect import YuNetDetector  # noqa: E402
from rppg.live.estimate import HRTracker  # noqa: E402
from rppg.methods.classical.chrom import chrom_pulse  # noqa: E402
from rppg.methods.classical.pos import pos_pulse  # noqa: E402
from rppg.methods.classical.propos import propos_pulse  # noqa: E402


def roi_rgb_trace(video_path: str, detector: YuNetDetector,
                  max_frames: int) -> tuple[np.ndarray, float]:
    """Decode a video, return (3, T) mid-face ROI RGB trace + fps via YuNet."""
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    rgb: list[np.ndarray] = []
    last_roi: tuple[int, int, int, int] | None = None
    try:
        while len(rgb) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detector.detect(frame)
            if dets:
                biggest = max(dets, key=lambda d: d.box[2] * d.box[3])
                last_roi = biggest.roi
            if last_roi is None:
                rgb.append(rgb[-1] if rgb else np.zeros(3))
                continue
            x, y, w, h = last_roi
            patch = frame[y:y + h, x:x + w]
            mean = patch.reshape(-1, 3).mean(axis=0)[::-1] if patch.size else None
            rgb.append(mean if mean is not None else (rgb[-1] if rgb else np.zeros(3)))
    finally:
        cap.release()
    return np.asarray(rgb, dtype=np.float64).T, fps


def fused_hr(trace: np.ndarray, fps: float) -> tuple[float, float]:
    """(HR bpm, SNR dB) from the (3, T) trace via POS/CHROM/ProPOS SNR fusion."""
    methods = (pos_pulse, chrom_pulse, lambda r, f: propos_pulse(r, f, do_fit=False))
    ests = [hr_peak_interp(bandpass(fn(trace, fps), fps), fps) for fn in methods]
    return fuse_estimates(ests)


def ppg_hr(ppg_window: np.ndarray, fps: float) -> float:
    """Reference HR (bpm) from a contact-PPG window, same sub-bin peak method."""
    return hr_peak_interp(bandpass(ppg_window, fps), fps)[0]


def eval_subject(subject_dir: pathlib.Path, detector: YuNetDetector,
                 max_frames: int, win_sec: float, step_sec: float,
                 track: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted, ground-truth) HR arrays over all sliding windows.

    With ``track`` the live :class:`HRTracker` is applied across windows, matching
    exactly what the webcam demo does (harmonic + spike rejection, motion hold).
    """
    trace, fps = roi_rgb_trace(str(subject_dir / "vid.avi"), detector, max_frames)
    with open(subject_dir / "ground_truth.txt", encoding="utf-8") as fh:
        ppg = np.array(fh.readline().split(), dtype=np.float64)
    n = min(trace.shape[1], ppg.size)
    trace, ppg = trace[:, :n], ppg[:n]
    w, s = int(round(win_sec * fps)), max(1, int(round(step_sec * fps)))
    tracker = HRTracker() if track else None
    preds, gts = [], []
    for st in range(0, n - w + 1, s):
        hr, snr = fused_hr(trace[:, st:st + w], fps)
        if tracker is not None:
            hr = tracker.update(hr, snr)
            hr = float("nan") if hr is None else hr
        preds.append(hr)
        gts.append(ppg_hr(ppg[st:st + w], fps))
    return np.array(preds), np.array(gts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--step-sec", type=float, default=2.0)
    ap.add_argument("--track", action="store_true",
                    help="apply the live HR tracker across windows (matches the demo)")
    args = ap.parse_args()

    detector = YuNetDetector()
    root = pathlib.Path(args.data_root)
    subjects = sorted(d for d in root.iterdir()
                      if (d / "vid.avi").exists() and (d / "ground_truth.txt").exists())
    if not subjects:
        sys.exit(f"No UBFC subjects with vid.avi + ground_truth.txt under {root}")

    all_pred, all_gt = [], []
    for sub in subjects:
        pred, gt = eval_subject(sub, detector, args.max_frames,
                                args.win_sec, args.step_sec, track=args.track)
        ok = np.isfinite(pred) & np.isfinite(gt)
        all_pred.append(pred[ok])
        all_gt.append(gt[ok])
        if np.any(ok):
            print(f"{sub.name:>10}: n={int(ok.sum()):2d}  "
                  f"MAE={mae(pred[ok], gt[ok]):5.2f}  RMSE={rmse(pred[ok], gt[ok]):5.2f}  "
                  f"within5={accuracy_within(pred[ok], gt[ok], 5.0):3.0f}%  "
                  f"within3={accuracy_within(pred[ok], gt[ok], 3.0):3.0f}%")
        else:
            print(f"{sub.name:>10}: no valid windows (no face / bad signal)")

    pred = np.concatenate(all_pred) if all_pred else np.array([])
    gt = np.concatenate(all_gt) if all_gt else np.array([])
    if pred.size:
        print(f"\nOVERALL ({len(subjects)} subjects, {pred.size} windows): "
              f"MAE={mae(pred, gt):.2f} bpm  RMSE={rmse(pred, gt):.2f}  "
              f"within±5={accuracy_within(pred, gt, 5.0):.0f}%  "
              f"within±3={accuracy_within(pred, gt, 3.0):.0f}%")


if __name__ == "__main__":
    main()
