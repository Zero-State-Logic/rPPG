"""Generate a synthetic UBFC-format dataset for end-to-end pipeline validation.

Each 'subject' is a short video whose frame brightness oscillates with a known
pulse at a random heart rate (plus noise), paired with a ``ground_truth.txt`` in
UBFC layout. This lets the full training + evaluation pipeline run and produce
real metrics when the actual UBFC Drive files are quota-blocked. It is a pipeline
sanity check, NOT a real-data result -- swap in real UBFC once Drive access
returns. Output matches ``list_ubfc_subjects`` (``<out>/<subject>/vid.avi`` +
``ground_truth.txt``).
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np

# UBFC skin-ish base colour in BGR (OpenCV writes BGR frames).
_BASE_BGR = np.array([120.0, 150.0, 190.0], dtype=np.float32)
_MODULATION = 0.05   # 5% brightness swing -> survives MJPG and is learnable
_PIXEL_NOISE = 2.0


def make_subject(path: pathlib.Path, hr_bpm: float, n_frames: int,
                 fps: int, size: int, seed: int) -> None:
    """Write one synthetic subject (``vid.avi`` + ``ground_truth.txt``)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    f = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    pulse_n = (pulse - pulse.mean()) / (pulse.std() + 1e-8)

    path.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path / "vid.avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    if not writer.isOpened():
        raise OSError(f"cannot open VideoWriter for {path}")
    for i in range(n_frames):
        colour = _BASE_BGR * (1.0 + _MODULATION * pulse_n[i])
        frame = np.ones((size, size, 3), dtype=np.float32) * colour
        frame += rng.normal(0.0, _PIXEL_NOISE, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    with open(path / "ground_truth.txt", "w", encoding="utf-8") as fh:
        fh.write(" ".join(f"{x:.6f}" for x in pulse) + "\n")
        fh.write(" ".join(f"{hr_bpm:.2f}" for _ in range(n_frames)) + "\n")
        fh.write(" ".join(f"{x:.5f}" for x in t) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/UBFC")
    ap.add_argument("--subjects", type=int, default=16)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=72)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    rng = np.random.default_rng(42)
    for s in range(args.subjects):
        hr = float(rng.uniform(55.0, 105.0))
        make_subject(out / f"subject{s + 1}", hr, args.frames, args.fps,
                     args.size, seed=s)
        print(f"[ok] subject{s + 1}  hr={hr:.1f} bpm")
    print(f"DONE: {args.subjects} synthetic subjects written to {out}")


if __name__ == "__main__":
    main()
