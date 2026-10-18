"""Live webcam rPPG demo: per-person heart rate (+ experimental BP) on your cam.

Run (from the repo root, using the project venv so mediapipe/opencv/torch resolve):

    .venv\\Scripts\\python.exe scripts\\live_demo.py                 # default camera
    .venv\\Scripts\\python.exe scripts\\live_demo.py --camera 1       # pick a camera
    .venv\\Scripts\\python.exe scripts\\live_demo.py --lock-exposure  # cleanest signal

Keys while running:  q / Esc = quit,  s = save a screenshot to artifacts/.

Demo scenarios this supports: a moving person (the ROI tracks the face and HR is
smoothed), several people at once (one readout per tracked face) and near/far
subjects (YuNet detects across distances; quality drops honestly when a face gets
too small). Blood pressure is an experimental proxy and is labelled as such.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.live.detect import CentroidTracker, YuNetDetector  # noqa: E402
from rppg.live.estimate import Calibration, Reading, SignalBank  # noqa: E402
from rppg.live.overlay import draw_face, draw_header  # noqa: E402
from rppg.live.select import SourceArbiter  # noqa: E402

_WINDOW = "rPPG live"
_EST_EVERY = 0.4   # seconds between classical re-estimates per person
_NN_EVERY = 1.0    # seconds between (heavier) neural ProPOSNet re-estimates
_NN_MAX_FACES = 2  # run the neural net only on the N nearest faces (keeps it smooth)

# YCrCb skin gate (reused from src/rppg/face/roi.py) for the multi-ROI colour read.
_SKIN_LO = (0, 133, 77)
_SKIN_HI = (255, 173, 127)
_MIN_SKIN_PX = 50


def _pooled_skin_rgb(frame: np.ndarray, rois) -> np.ndarray | None:
    """Mean RGB over skin pixels pooled across the ROI patches (forehead + cheeks).

    Returns an (R, G, B) float vector, or None when the patches are empty. Falls
    back to the raw patch mean when too few skin pixels pass the gate, so a reading
    is never lost under hard lighting or darker skin tones.
    """
    skin_sum = np.zeros(3, dtype=np.float64)
    skin_n = 0
    raw_sum = np.zeros(3, dtype=np.float64)
    raw_n = 0
    for rx, ry, rw, rh in rois:
        patch = frame[ry:ry + rh, rx:rx + rw]
        if patch.size == 0:
            continue
        flat = patch.reshape(-1, 3).astype(np.float64)
        raw_sum += flat.sum(axis=0)
        raw_n += flat.shape[0]
        ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        mask = ((ycc[:, 0] >= _SKIN_LO[0]) & (ycc[:, 0] <= _SKIN_HI[0]) &
                (ycc[:, 1] >= _SKIN_LO[1]) & (ycc[:, 1] <= _SKIN_HI[1]) &
                (ycc[:, 2] >= _SKIN_LO[2]) & (ycc[:, 2] <= _SKIN_HI[2]))
        if mask.any():
            skin_sum += flat[mask].sum(axis=0)
            skin_n += int(mask.sum())
    if skin_n >= _MIN_SKIN_PX:
        return (skin_sum / skin_n)[::-1]   # BGR -> RGB
    if raw_n > 0:
        return (raw_sum / raw_n)[::-1]
    return None


def _background_rgb(frame, boxes, frac=0.14):
    h, w = frame.shape[:2]
    bw, bh = int(w * frac), int(h * frac)
    patches = [frame[0:bh, 0:bw], frame[0:bh, w - bw:w], frame[h - bh:h, 0:bw], frame[h - bh:h, w - bw:w]]
    acc = np.zeros(3, dtype=np.float64)
    n = 0
    for p in patches:
        if p.size:
            acc += p.reshape(-1, 3).sum(axis=0)
            n += p.shape[0] * p.shape[1]
    if n == 0:
        return None
    return (acc / n)[::-1]


def _open_camera(index: int, width: int, height: int,
                 lock_exposure: bool) -> cv2.VideoCapture:
    """Open a webcam, prefer DirectShow on Windows, disable auto white balance."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # avoid YUY2 FPS cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # drop stale frames -> low latency
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)            # steady colour -> cleaner rPPG
    if lock_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DSHOW: 0.25 = manual exposure
    return cap


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--no-flip", action="store_true",
                    help="do not mirror the frame (mirror is on by default)")
    ap.add_argument("--lock-exposure", action="store_true",
                    help="lock camera exposure for the cleanest signal (needs light)")
    ap.add_argument("--no-illum-norm", action="store_true",
                    help="disable background illumination normalization")
    ap.add_argument("--ref-sbp", type=float, default=120.0,
                    help="reference systolic BP for single-point calibration (key 'c')")
    ap.add_argument("--ref-dbp", type=float, default=80.0,
                    help="reference diastolic BP for single-point calibration (key 'c')")
    ap.add_argument("--neural", nargs="?", const="artifacts/models/best.pt",
                    default=None,
                    help="also run the trained ProPOSNet (optional checkpoint path)")
    ap.add_argument("--bp-model", default="artifacts/models/bp_model.pkl",
                    help="trained BP model to use instead of the heuristic (if present)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cap = _open_camera(args.camera, args.width, args.height, args.lock_exposure)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1.")

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"[camera] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"fourcc={''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))} "
          f"reported_fps={cap.get(cv2.CAP_PROP_FPS):.0f}  "
          f"(low light forces long exposure -> low FPS; use bright, white light)")

    detector = YuNetDetector()
    tracker = CentroidTracker()
    bank = SignalBank()
    arbiter = SourceArbiter()   # picks the main HR source (neural vs classical)
    calib = Calibration()
    calibrated = False
    neural = None
    if args.neural:
        from rppg.live.neural import NeuralHR  # lazy: only import torch if needed
        try:
            neural = NeuralHR(args.neural)
            print(f"[neural] loaded ProPOSNet from {args.neural}")
        except Exception as exc:  # noqa: BLE001 - demo should run without the net
            print(f"[neural] disabled (could not load {args.neural}): {exc}")
    bp_model = None
    if args.bp_model and pathlib.Path(args.bp_model).exists():
        from rppg.live.bp_model import BPModel
        try:
            bp_model = BPModel(args.bp_model)
            print(f"[bp] using trained model {args.bp_model} (experimental on rPPG)")
        except Exception as exc:  # noqa: BLE001 - fall back to the heuristic
            print(f"[bp] heuristic BP (could not load {args.bp_model}): {exc}")
    last_est: dict[int, float] = {}
    last_nn: dict[int, float] = {}
    readings: dict[int, Reading] = {}
    nn_hr: dict = {}   # track id -> NeuralReading | None (trained ProPOSNet reading)
    subj_nums: dict[int, int] = {}  # track id -> stable 1-based "Subj N" label
    fps = 0.0
    prev = time.time()
    shots = pathlib.Path("artifacts")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            now = time.time()

            tracks = tracker.update(detector.detect(frame))
            bg = None if args.no_illum_norm else _background_rgb(frame, [t.det.box for t in tracks])
            live_ids = set()
            neural_ids: set[int] = set()
            if neural is not None and tracks:
                nearest = sorted(tracks, key=lambda t: t.det.box[2] * t.det.box[3],
                                 reverse=True)[:_NN_MAX_FACES]
                neural_ids = {t.id for t in nearest}
            for tr in tracks:
                live_ids.add(tr.id)
                if tr.id not in subj_nums:          # assign the lowest free number
                    used = set(subj_nums.values())
                    n = 1
                    while n in used:
                        n += 1
                    subj_nums[tr.id] = n
                rgb = _pooled_skin_rgb(frame, tr.det.rois)
                if rgb is not None:
                    if bg is not None:
                        rgb = rgb / np.maximum(bg, 1.0)   # normalize out ambient-light swings
                    bank.add(tr.id, now, rgb)   # forehead + cheeks, skin-pooled
                if tr.id in neural_ids:
                    bx, by, bw, bh = tr.det.box
                    face_crop = frame[by:by + bh, bx:bx + bw]
                    if face_crop.size:
                        neural.add(tr.id, now, face_crop)  # net trained on face frames
                fresh_c = False
                fresh_n = False
                if now - last_est.get(tr.id, 0.0) >= _EST_EVERY:
                    readings[tr.id] = bank.estimate(tr.id, calib, bp_model)
                    last_est[tr.id] = now
                    fresh_c = True
                if tr.id in neural_ids and now - last_nn.get(tr.id, 0.0) >= _NN_EVERY:
                    nn_hr[tr.id] = neural.estimate(tr.id)
                    last_nn[tr.id] = now
                    fresh_n = True
                decision = arbiter.update(
                    tr.id, readings.get(tr.id, Reading(False, 0.0)),
                    nn_hr.get(tr.id) if tr.id in neural_ids else None,
                    fresh_c, fresh_n)
                bp_tag = "cal" if calibrated else ("mdl" if bp_model else "exp")
                draw_face(frame, tr, readings.get(tr.id, Reading(False, 0.0)),
                          bp_tag=bp_tag,
                          decision=decision,
                          subj_num=subj_nums[tr.id])
            bank.prune(live_ids)
            arbiter.prune(live_ids)
            if neural is not None:
                neural.prune(live_ids)
            for gone in [tid for tid in subj_nums if tid not in live_ids]:
                del subj_nums[gone]

            dt = now - prev
            prev = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            draw_header(frame, fps, len(tracks))

            cv2.imshow(_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                shots.mkdir(exist_ok=True)
                name = shots / f"rppg-{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(name), frame)
                print(f"[saved] {name}")
            if key == ord("c") and tracks:
                primary = max(tracks, key=lambda t: t.det.box[2] * t.det.box[3])
                raw = bank.estimate(primary.id, Calibration(), bp_model)
                if raw.ready:
                    calib = Calibration(args.ref_sbp - raw.bp_sys,
                                        args.ref_dbp - raw.bp_dia)
                    calibrated = True
                    print(f"[calibrated] P{primary.id} baseline set to "
                          f"{args.ref_sbp:.0f}/{args.ref_dbp:.0f} mmHg")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
