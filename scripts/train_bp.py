"""Train the blood-pressure regressor on the noninvasivebp contact-PPG dataset.

Each record (``<data-root>/N.json``) holds a long fingertip-PPG waveform
(``data_PPG``, ~500 Hz) and a few cuff readings (``data_BP`` = ``[{SBP,DBP},...]``).
We resample the PPG down to the **camera-rPPG rate (30 Hz)** so its morphology
features match exactly what the live demo extracts, split each waveform into
chronological segments aligned to the cuff readings, window each segment, extract
features (:mod:`rppg.bp.features`), and fit :class:`rppg.bp.regressor.BPRegressor`
with **subject-independent** validation and BHS / AAMI clinical reporting.

Honest scope: trained on CONTACT fingertip PPG; applied to camera rPPG it is a
rough estimate (a real domain gap). The validation numbers it prints are on
held-out contact subjects -- the fair bar for the method, not for live rPPG.

Usage:
    python scripts/train_bp.py --data-root C:\\rppg_data\\bp --out artifacts/models/bp_model.pkl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.bp.features import features_vector  # noqa: E402
from rppg.bp.regressor import (  # noqa: E402
    BPRegressor, bhs_aami_report, mean_predictor_mae)
from rppg.data.splits import subject_independent_split  # noqa: E402

_SRC_FPS = 500      # inferred PPG sampling rate of the dataset
_TGT_FPS = 30       # resample to the camera-rPPG rate so features transfer
_WIN_SEC = 8.0
_STEP_SEC = 4.0


def _load_records(root: str):
    """Yield (uid, raw PPG array, bp_list) for every usable JSON record."""
    for path in sorted(pathlib.Path(root).glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip unreadable/partial files
            continue
        ppg, bp = rec.get("data_PPG"), rec.get("data_BP")
        if ppg and bp:
            yield rec.get("UID", path.stem), np.asarray(ppg, dtype=np.float64), bp


def _windows(ppg30: np.ndarray, bp_list: list):
    """Yield (features, sbp, dbp) over windows, BP aligned to time segments."""
    n = ppg30.size
    win, step = int(_WIN_SEC * _TGT_FPS), int(_STEP_SEC * _TGT_FPS)
    n_bp = len(bp_list)
    seg_len = n // max(1, n_bp)
    for si in range(n_bp):
        sbp, dbp = float(bp_list[si]["SBP"]), float(bp_list[si]["DBP"])
        if not (60 <= sbp <= 220 and 30 <= dbp <= 140):
            continue
        seg = ppg30[si * seg_len:(si + 1) * seg_len]
        for st in range(0, seg.size - win + 1, step):
            feats = features_vector(seg[st:st + win], float(_TGT_FPS))
            if np.isfinite(feats[0]):               # HR feature present
                yield feats, sbp, dbp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=r"C:\rppg_data\bp")
    ap.add_argument("--out", default="artifacts/models/bp_model.pkl")
    args = ap.parse_args()

    feats, sbps, dbps, subs = [], [], [], []
    for uid, ppg, bp in _load_records(args.data_root):
        ppg30 = resample_poly(ppg - ppg.mean(), _TGT_FPS, _SRC_FPS)
        for vec, sbp, dbp in _windows(ppg30, bp):
            feats.append(vec)
            sbps.append(sbp)
            dbps.append(dbp)
            subs.append(uid)
    if not feats:
        sys.exit(f"No usable windows found under {args.data_root}")

    X = np.asarray(feats)
    y_sbp, y_dbp = np.asarray(sbps), np.asarray(dbps)
    subs = np.asarray(subs)
    hr_col = X[:, 0]
    print(f"[data] {X.shape[0]} windows / {len(set(subs))} subjects; "
          f"HR feature median={np.nanmedian(hr_col):.0f} bpm "
          f"(sanity-check the fps); SBP {y_sbp.mean():.0f}+-{y_sbp.std():.0f}, "
          f"DBP {y_dbp.mean():.0f}+-{y_dbp.std():.0f}")

    uids = sorted(set(subs.tolist()))
    split = subject_independent_split(uids, val_frac=0.0, test_frac=0.25)
    test_ids = set(split["test"])
    is_test = np.array([s in test_ids for s in subs])
    is_train = ~is_test
    print(f"[split] subject-independent: {is_train.sum()} train / "
          f"{is_test.sum()} test windows ({len(test_ids)} test subjects)")

    reg = BPRegressor().fit(X[is_train], y_sbp[is_train], y_dbp[is_train])
    sbp_pred, dbp_pred = reg.predict(X[is_test])
    for name, pred, true, ytr, yte in (
            ("SBP", sbp_pred, y_sbp[is_test], y_sbp[is_train], y_sbp[is_test]),
            ("DBP", dbp_pred, y_dbp[is_test], y_dbp[is_train], y_dbp[is_test])):
        rep = bhs_aami_report(pred, true)
        base = mean_predictor_mae(ytr, yte)
        print(f"[{name}] MAE={rep['mae']:.1f}  err={rep['mean_err']:+.1f}+-"
              f"{rep['std_err']:.1f}  within5/10/15="
              f"{rep['within_5']:.0f}/{rep['within_10']:.0f}/{rep['within_15']:.0f}%"
              f"  BHS={rep['bhs_grade']}  AAMI={'pass' if rep['aami_pass'] else 'fail'}"
              f"  (mean-predictor MAE={base:.1f})")

    reg_all = BPRegressor().fit(X, y_sbp, y_dbp)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"model": reg_all, "src_fps": _SRC_FPS, "tgt_fps": _TGT_FPS}, fh)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
