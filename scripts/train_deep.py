"""Train a deep rPPG model on UBFC / PURE with subject-independent or
cross-dataset ("uneven data") evaluation.

Examples
--------
Subject-independent (train/val/test are disjoint subjects of one dataset):
    python scripts/train_deep.py --dataset ubfc --data-root data/UBFC \
        --model proposnet --eval-mode subject --epochs 30

Cross-dataset (train on UBFC, test on the *different* PURE distribution):
    python scripts/train_deep.py --dataset ubfc --data-root data/UBFC \
        --model proposnet --eval-mode cross --cross-dataset pure --cross-root data/PURE

Picks CUDA automatically when available (cloud GPU); runs on CPU otherwise.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.data.datasets import ClipDataset, pure_sources, ubfc_sources  # noqa: E402
from rppg.data.splits import subject_independent_split  # noqa: E402
from rppg.eval.deep_eval import evaluate_hr  # noqa: E402
from rppg.models.physnet import PhysNet  # noqa: E402
from rppg.models.proposnet import ProPOSNet  # noqa: E402
from rppg.models.tscan import TSCAN  # noqa: E402
from rppg.train.trainer import train_model  # noqa: E402

MODELS = {"proposnet": ProPOSNet, "physnet": PhysNet, "tscan": TSCAN}
SOURCES = {"ubfc": ubfc_sources, "pure": pure_sources}


def _subset(sources, ids):
    keep = set(ids)
    return [s for s in sources if s.subject_id in keep]


def _loader(sources, args, shuffle):
    dataset = ClipDataset(sources, chunk_len=args.chunk_len, resize=args.resize,
                          data_type=args.data_type)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, drop_last=shuffle)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(SOURCES), default="ubfc")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", choices=list(MODELS), default="proposnet")
    ap.add_argument("--eval-mode", choices=["subject", "cross"], default="subject")
    ap.add_argument("--cross-dataset", choices=list(SOURCES), default="pure")
    ap.add_argument("--cross-root", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-len", type=int, default=160)
    ap.add_argument("--resize", type=int, default=72)
    ap.add_argument("--data-type", default="Standardized",
                    choices=["Standardized", "DiffNormalized"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; 0 avoids OpenCV fork crashes on Colab")
    ap.add_argument("--out-dir", default="artifacts/models")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    import cv2  # OpenCV is not fork-safe; disable its threads to avoid heap
    cv2.setNumThreads(0)  # corruption ("free(): invalid next size") in loaders
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} model={args.model} "
          f"dataset={args.dataset} eval={args.eval_mode}")

    sources = SOURCES[args.dataset](args.data_root)
    subject_ids = sorted({s.subject_id for s in sources})
    print(f"[data] {len(sources)} sources / {len(subject_ids)} subjects")

    split = subject_independent_split(subject_ids, seed=args.seed)
    train_src = _subset(sources, split["train"])
    val_src = _subset(sources, split["val"])

    if args.eval_mode == "subject":
        test_src = _subset(sources, split["test"])
    else:
        cross_root = args.cross_root or args.data_root
        test_src = SOURCES[args.cross_dataset](cross_root)
        print(f"[data] cross-dataset test = {args.cross_dataset} "
              f"({len(test_src)} sources)")

    train_loader = _loader(train_src, args, shuffle=True)
    val_loader = _loader(val_src, args, shuffle=False) if val_src else None
    test_loader = _loader(test_src, args, shuffle=False)

    model = MODELS[args.model]()
    print(f"[model] {args.model} params={sum(p.numel() for p in model.parameters())}")
    train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                device=device, out_dir=args.out_dir)

    metrics = evaluate_hr(model, test_loader, fps=args.fps, device=device)
    print("=== TEST METRICS (held-out) ===")
    for key in ("mae", "rmse", "within5", "pearson", "n"):
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
