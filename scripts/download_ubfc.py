"""Robust UBFC-rPPG subject downloader for the cloud (Colab / Kaggle).

Google Drive rate-limits popular files ("Too many users have downloaded this
file recently"), so ``gdown.download_folder`` over the whole dataset crashes
part-way. This script downloads subjects one at a time, skips any that hit the
quota, resumes already-present ones, and keeps going -- so training gets as many
subjects as are currently available.

Usage:
    python scripts/download_ubfc.py --out data/UBFC --limit 25
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Public Google Drive folder ids for UBFC-rPPG DATASET_2 (each holds vid.avi +
# ground_truth.txt). Source folder: the official UBFC_DATASET share.
SUBJECT_FOLDER_IDS: dict[str, str] = {
    "subject1": "1q6U7jKRBUtr6N2rVs1vwZ2wuUT2VDoRy",
    "subject3": "1tJdI-E143hW0q0NHxmPPkLcr0UvZgeNz",
    "subject4": "1vjXMUmi9CDWvAnrLNoteFNMoxU48Dhzt",
    "subject5": "1y9wb_8Vn1bt34gSMBuc7ux987AzpNEeg",
    "subject8": "1yK1wZdt33J6OxBSyv8kRtNieaH0SgSpi",
    "subject9": "1yR3hc7nfVqJt8VsXz9AIM1r2R5H-pdGC",
    "subject10": "1qCvEJjNRfMVuz4VoQgbDS-RMvgh5Hn9a",
    "subject11": "1qRx5WQvLOwiHM_4w09aKqSLeWVbG4qHE",
    "subject12": "1qavAF-3T7IQdnytHSxqCesdfGqzgavlJ",
    "subject13": "1ql-e1MDCCr4VFLEPPbMpAOXPbYCIX95f",
    "subject14": "1rCP6v2FotK0-SVJGSNgjeA6ppbBb3z0L",
    "subject15": "1rGs5PpY62WdHet_PH6fBGqTMVJvxGAzt",
    "subject16": "1rgia82iJ_6igB0U3tj522jxDa0pdqZTz",
    "subject17": "1rjLl2Nm5ZsDLAsSCCYPQCf08HV5JJwYX",
    "subject18": "1rrd41TEOi9H1_k0cV8pQFZURUE5yS5pb",
    "subject20": "1s46xn-OMg1pEVXnMvl3yB1bJfwBmMX-g",
    "subject22": "1sJXdmzk9mTG_kpCjo00lPkcDKyo5Aqhs",
    "subject23": "1sWjC0FyshavhK8ztwg4I95bF4QU6nybu",
    "subject24": "1soR7P6uArXdJEeaxER7bBKTPuzlhsq1b",
    "subject25": "1szbMEUYHPF_VJjGtdtgUjo4NL7OPKSZ7",
    "subject26": "1t2usXJ0VjLnM3wV7raDUnVGc4Z_eCz3V",
    "subject27": "1tAyikar9Zeu6jjJZmx5m7LcyxIeq7XKX",
    "subject30": "1tWXN4qYLmzHa5_hx_elqU_sgl1aYy0vO",
    "subject31": "1twV7BMwxwF14XGZfxzkS-9P4ezfZty1P",
    "subject32": "1u3oijGKzB0G_5o1Ju2knvn2KGUtEwOEk",
    "subject33": "1uMT6Cv-JnlXhOVoPZ8nxPtqPQYy6Gdva",
    "subject34": "1uOfj5OY22f1vMo_Gc5no9H7Nbb6U3dXK",
    "subject35": "1uggXwmu9pOHOS1Ds2CO4pEgfOZoVT57j",
    "subject36": "1v5hfhYpjBsUDmrQVpaDEUZmtWZw06etA",
    "subject37": "1vEDM3WM0vbaiWdiZcIRbtE9a0plR2Tsc",
    "subject38": "1vLl3Sd3Vfrml6BwXuMAqzYDB82v27pBv",
    "subject39": "1vb1dYR72dGH0pcr-LVhtJSKIQJb7GOep",
    "subject40": "1w2In1mSSLtHqOMPwnsepOQ9n6gzYJKPd",
    "subject41": "1wHgO1SZw9dI4MOlslapzSXr4FO6xcowj",
    "subject42": "1wSYMwaW7l4wUe5OCF37CgsC4_T4YM6qE",
    "subject43": "1wf6sjVYvYoJaUmK0PJa11y7TKQEd2wRd",
    "subject44": "1x3OJT8ArvFpso54_EK0DNPpy2aSJ_UTL",
    "subject45": "1x8szB8zXsqAQgHnXKQWOUdhtkYiGmN_J",
    "subject46": "1xH-vlaB5aggjVVlm-rvydVOfoJWn7YwO",
    "subject47": "1xSXY4if2UbdoBw47FuUM8LOdKzavwIkY",
    "subject48": "1xxOONETbO_zyFq7TMcOdLlvc8vDuu0Aa",
    "subject49": "1y3az0As2ePyGGwT4OzGwmafAVw3EdJqk",
}

_MIN_VIDEO_BYTES = 1_000_000  # a real vid.avi is >1 GB; anything tiny = failed


def _has_video(subject_dir: pathlib.Path) -> bool:
    video = subject_dir / "vid.avi"
    return video.exists() and video.stat().st_size > _MIN_VIDEO_BYTES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/UBFC")
    ap.add_argument("--limit", type=int, default=0, help="max subjects (0 = all)")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    import gdown

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    items = list(SUBJECT_FOLDER_IDS.items())
    if args.subjects:
        wanted = set(args.subjects)
        items = [(sid, fid) for sid, fid in items if sid in wanted]
    if args.limit:
        items = items[:args.limit]

    ready, skipped = [], []
    for sid, fid in items:
        dest = out_root / sid
        if _has_video(dest):
            print(f"[skip] {sid} already present")
            ready.append(sid)
            continue
        try:
            gdown.download_folder(id=fid, output=str(dest), quiet=True,
                                  remaining_ok=True)
        except Exception as exc:  # noqa: BLE001 - keep going past quota errors
            print(f"[fail] {sid}: {str(exc)[:90]}")
            skipped.append(sid)
            continue
        if _has_video(dest):
            print(f"[ok]   {sid}")
            ready.append(sid)
        else:
            print(f"[miss] {sid} (video not retrieved - likely Drive quota)")
            skipped.append(sid)

    print(f"\nDONE: {len(ready)} subjects ready, {len(skipped)} skipped")
    print("ready:", ", ".join(ready) if ready else "(none)")
    if not ready:
        sys.exit("No subjects downloaded - Drive quota may be fully tripped; "
                 "retry later or use a different data source.")


if __name__ == "__main__":
    main()
