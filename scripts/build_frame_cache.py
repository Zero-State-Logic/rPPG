"""Build the per-frame feature cache for every UBFC subject (run once).

Usage:
    .venv\\Scripts\\python.exe scripts\\build_frame_cache.py --data-root C:\\rppg_data\\UBFC --out C:\\rppg_data\\cache

For each ``<data-root>/<subject>/`` with ``vid.avi`` + ``ground_truth.txt`` it decodes
the video once (YuNet face + forehead/cheek ROIs) and writes ``<out>/<subject>.npz``
(see :mod:`rppg.eval.frame_cache`). It then sanity-checks the cache: the live fused HR
(POS + CHROM + ProPOS) on the cached pooled trace vs the contact-PPG reference HR over
sliding windows, which should reproduce the known ~1.4 bpm MAE. Use ``--force`` to
rebuild existing caches.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.eval.frame_cache import extract_subject, load_cache, save_cache  # noqa: E402
from rppg.eval.windows import fused_hr, reference_hr, window_starts  # noqa: E402
from rppg.live.detect import YuNetDetector  # noqa: E402


def sanity_mae(cache, win_sec: float, step_sec: float) -> tuple[float, int]:
    """MAE (bpm) of the fused HR on the cached pooled trace vs the PPG reference."""
    trace = cache.pooled_rgb().T                      # (3, T)
    win = int(round(win_sec * cache.fps))
    errs = []
    for st in window_starts(cache.n_frames, cache.fps, win_sec, step_sec):
        pred, _ = fused_hr(trace[:, st:st + win], cache.fps)
        ref = reference_hr(cache.ppg[st:st + win], cache.fps)
        if np.isfinite(pred) and np.isfinite(ref):
            errs.append(abs(pred - ref))
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--step-sec", type=float, default=2.0)
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = ap.parse_args()

    root, out = pathlib.Path(args.data_root), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subjects = sorted(d for d in root.iterdir()
                      if (d / "vid.avi").exists() and (d / "ground_truth.txt").exists())
    if not subjects:
        sys.exit(f"No UBFC subjects with vid.avi + ground_truth.txt under {root}")

    detector = YuNetDetector()
    all_err: list[float] = []
    for sub in subjects:
        path = out / f"{sub.name}.npz"
        t0 = time.time()
        if args.force or not path.exists():
            cache = extract_subject(sub / "vid.avi", sub / "ground_truth.txt", detector)
            save_cache(path, cache)
            built = f"built in {time.time() - t0:5.1f}s"
        else:
            cache = load_cache(path)
            built = "cached"
        err, n = sanity_mae(cache, args.win_sec, args.step_sec)
        det_rate = 100.0 * float(np.mean(cache.detected))
        print(f"{sub.name:>10}: frames={cache.n_frames:5d} fps={cache.fps:4.1f} "
              f"face={det_rate:5.1f}%  windows={n:3d}  sanity MAE={err:5.2f} bpm  ({built})")
        if np.isfinite(err):
            all_err.append(err)
    if all_err:
        print(f"\nmean per-subject sanity MAE over {len(all_err)} subjects: "
              f"{np.mean(all_err):.2f} bpm  -> cache at {out}")


if __name__ == "__main__":
    main()
