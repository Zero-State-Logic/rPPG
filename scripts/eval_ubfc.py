"""Real-video sanity check: evaluate POS and ProPOS on a UBFC-rPPG subject.

UBFC DATASET_2 ground_truth.txt layout (3 whitespace-separated rows):
    row 0: PPG waveform samples
    row 1: instantaneous heart rate (bpm), one value per sample
    row 2: timestamps in seconds, one per sample
Samples are synchronized to the ~30 fps video.

Usage:
    python scripts/eval_ubfc.py --video D:/rppg_data/ubfc/subject1/vid.avi \
        --gt D:/rppg_data/ubfc/subject1/ground_truth.txt --max-frames 900
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.face.roi import FaceRoiExtractor  # noqa: E402
from rppg.hr.metrics import accuracy_within, estimate_hr_fft, mae, rmse  # noqa: E402
from rppg.io.video import load_frames_rgb  # noqa: E402
from rppg.methods.classical.chrom import chrom_pulse  # noqa: E402
from rppg.methods.classical.pbv import pbv_pulse  # noqa: E402
from rppg.methods.classical.pos import pos_pulse  # noqa: E402
from rppg.methods.classical.propos import propos_pulse  # noqa: E402
from rppg.preprocess.rgb_trace import frames_to_rgb_trace  # noqa: E402

_METHODS = {"pos": pos_pulse, "chrom": chrom_pulse, "pbv": pbv_pulse,
            "propos": propos_pulse}


def load_ground_truth(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(np.array([float(x) for x in line.split()]))
    return rows[0], rows[1], rows[2]


def windowed_hr(trace: np.ndarray, fps: float, method_fn, win_sec: float,
                step_sec: float) -> tuple[np.ndarray, np.ndarray]:
    n = trace.shape[1]
    w = int(round(win_sec * fps))
    s = max(1, int(round(step_sec * fps)))
    starts, hrs = [], []
    for st in range(0, n - w + 1, s):
        hrs.append(estimate_hr_fft(method_fn(trace[:, st:st + w], fps), fps))
        starts.append(st)
    return np.array(starts), np.array(hrs)


def gt_hr_for_windows(starts: np.ndarray, fps: float, win_sec: float,
                      gt_hr: np.ndarray, gt_times: np.ndarray) -> np.ndarray:
    out = []
    for st in starts:
        t0, t1 = st / fps, st / fps + win_sec
        mask = (gt_times >= t0) & (gt_times < t1)
        out.append(float(np.mean(gt_hr[mask])) if np.any(mask) else np.nan)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--step-sec", type=float, default=5.0)
    args = ap.parse_args()

    print(f"[1/3] decoding up to {args.max_frames} frames ...")
    frames, fps = load_frames_rgb(args.video, max_frames=args.max_frames)
    print(f"      {frames.shape[0]} frames @ {fps:.1f} fps, "
          f"size {frames.shape[2]}x{frames.shape[1]}")

    print("[2/3] extracting skin-ROI trace (MediaPipe) ...")
    extractor = FaceRoiExtractor()
    try:
        trace = frames_to_rgb_trace(frames, extractor)
    finally:
        extractor.close()

    _, gt_hr, gt_times = load_ground_truth(args.gt)
    print(f"      ground-truth HR mean = {np.mean(gt_hr):.1f} bpm (full clip)")

    print(f"[3/3] per-window HR (win={args.win_sec}s, step={args.step_sec}s):")
    for name, fn in _METHODS.items():
        starts, hrs = windowed_hr(trace, fps, fn, args.win_sec, args.step_sec)
        gts = gt_hr_for_windows(starts, fps, args.win_sec, gt_hr, gt_times)
        ok = np.isfinite(hrs) & np.isfinite(gts)
        if not np.any(ok):
            print(f"  {name:>7}: no valid windows")
            continue
        print(f"  {name:>7}: n={int(ok.sum())}  MAE={mae(hrs[ok], gts[ok]):.2f}  "
              f"RMSE={rmse(hrs[ok], gts[ok]):.2f}  "
              f"within±5bpm={accuracy_within(hrs[ok], gts[ok], 5.0):.0f}%")
        print("           est:", np.round(hrs[ok], 1).tolist())
        print("           gt :", np.round(gts[ok], 1).tolist())


if __name__ == "__main__":
    main()
