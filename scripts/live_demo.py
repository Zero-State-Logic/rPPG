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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rppg.live.detect import CentroidTracker, YuNetDetector  # noqa: E402
from rppg.live.estimate import Calibration, Reading, SignalBank  # noqa: E402
from rppg.live.overlay import draw_face, draw_header  # noqa: E402

_WINDOW = "rPPG live"
_EST_EVERY = 0.4  # seconds between classical re-estimates per person
_NN_EVERY = 1.0   # seconds between (heavier) neural ProPOSNet re-estimates


def _open_camera(index: int, width: int, height: int,
                 lock_exposure: bool) -> cv2.VideoCapture:
    """Open a webcam, prefer DirectShow on Windows, disable auto white balance."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
    ap.add_argument("--ref-sbp", type=float, default=120.0,
                    help="reference systolic BP for single-point calibration (key 'c')")
    ap.add_argument("--ref-dbp", type=float, default=80.0,
                    help="reference diastolic BP for single-point calibration (key 'c')")
    ap.add_argument("--neural", nargs="?", const="artifacts/models/best.pt",
                    default=None,
                    help="also run the trained ProPOSNet (optional checkpoint path)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cap = _open_camera(args.camera, args.width, args.height, args.lock_exposure)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1.")

    detector = YuNetDetector()
    tracker = CentroidTracker()
    bank = SignalBank()
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
    last_est: dict[int, float] = {}
    last_nn: dict[int, float] = {}
    readings: dict[int, Reading] = {}
    nn_hr: dict[int, float | None] = {}
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
            live_ids = set()
            for tr in tracks:
                live_ids.add(tr.id)
                rx, ry, rw, rh = tr.det.roi
                roi = frame[ry:ry + rh, rx:rx + rw]
                if roi.size:
                    bgr = roi.reshape(-1, 3).mean(axis=0)
                    bank.add(tr.id, now, bgr[::-1])  # BGR -> RGB for the methods
                    if neural is not None:
                        neural.add(tr.id, now, roi)
                if now - last_est.get(tr.id, 0.0) >= _EST_EVERY:
                    readings[tr.id] = bank.estimate(tr.id, calib)
                    last_est[tr.id] = now
                if neural is not None and now - last_nn.get(tr.id, 0.0) >= _NN_EVERY:
                    nn_hr[tr.id] = neural.estimate(tr.id)
                    last_nn[tr.id] = now
                draw_face(frame, tr, readings.get(tr.id, Reading(False, 0.0)),
                          bp_tag="cal" if calibrated else "exp",
                          nn_hr=nn_hr.get(tr.id) if neural is not None else None)
            bank.prune(live_ids)
            if neural is not None:
                neural.prune(live_ids)

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
                raw = bank.estimate(primary.id, Calibration())
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
