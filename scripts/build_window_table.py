"""Build the window-level reliability feature table from the frame cache.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_window_table.py --cache C:\\rppg_data\\cache

Reads every ``<cache>/<subject>.npz`` frame cache (built by build_frame_cache.py),
computes one feature record per sliding window (see :mod:`rppg.eval.window_table`)
and writes ``<cache>/windows_<win>s_<step>s.npz``. Prints a per-subject summary of
the estimate's error so the table can be sanity-checked against the cache build.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.eval.frame_cache import load_cache  # noqa: E402
from rppg.eval.window_table import build_table, save_table, window_records  # noqa: E402

_FAIL_BPM = 10.0   # a window is "failing" when its absolute HR error exceeds this


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="directory of <subject>.npz caches")
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--step-sec", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="output .npz (default inside --cache)")
    args = ap.parse_args()

    cache_dir = pathlib.Path(args.cache)
    paths = sorted(p for p in cache_dir.glob("*.npz") if not p.name.startswith("windows_"))
    if not paths:
        sys.exit(f"No frame caches (*.npz) under {cache_dir}")

    records: list[dict] = []
    for path in paths:
        t0 = time.time()
        recs = window_records(load_cache(path), path.stem, args.win_sec, args.step_sec)
        err = np.array([r["abs_err"] for r in recs], dtype=float)
        ok = np.isfinite(err)
        print(f"{path.stem:>10}: windows={len(recs):3d}  MAE={np.mean(err[ok]):6.2f} bpm  "
              f"failing(>{_FAIL_BPM:.0f})={int(np.sum(err[ok] > _FAIL_BPM)):2d}  "
              f"({time.time() - t0:4.1f}s)")
        records.extend(recs)

    table = build_table(records)
    out = (pathlib.Path(args.out) if args.out else
           cache_dir / f"windows_{args.win_sec:g}s_{args.step_sec:g}s.npz")
    save_table(out, table)
    err = table["abs_err"][np.isfinite(table["abs_err"])]
    print(f"\n{len(records)} windows from {len(paths)} subjects -> {out}")
    print(f"all-window MAE {np.mean(err):.2f} bpm, failing {int(np.sum(err > _FAIL_BPM))}")


if __name__ == "__main__":
    main()
