# Live webcam demo — open in VS Code & run

This runs the **rPPG live demo** on your laptop camera: it finds every face, and
for each person shows a **heart rate** (from our ProPOS method) plus an
**experimental** blood-pressure estimate.

---

## 1. Open the project in VS Code

- **File → Open Folder…** → select `E:\RPPG`
- (First time) install the Python extension if VS Code prompts you.

## 2. Point VS Code at the project's Python (`.venv`)

The demo needs the project venv (it has mediapipe, opencv, torch, scipy).

- Press **Ctrl+Shift+P** → type **"Python: Select Interpreter"** → choose
  **`.\.venv\Scripts\python.exe`** (Python 3.11.0).

## 3. Run it

Open a terminal in VS Code (**Ctrl+`**) and run, from the `E:\RPPG` folder:

```bash
.venv\Scripts\python.exe scripts\live_demo.py
```

A window titled **"rPPG live"** opens showing your camera. Sit still facing the
camera in decent light; after ~5 seconds a heart rate appears above your face.

**Useful options:**

```bash
.venv\Scripts\python.exe scripts\live_demo.py --camera 1       # if the wrong camera opens
.venv\Scripts\python.exe scripts\live_demo.py --lock-exposure  # cleanest signal (needs good light)
```

**Keys while it runs:** `q` / `Esc` quit · `s` screenshot to `artifacts\` · `c` calibrate BP.

---

## Showing the three demo scenarios

1. **Moving person** — move your head/body slowly; the colored box and ROI follow
   you and the HR stays locked. Fast motion briefly drops the quality label to
   "fair/weak" (this is honest — the signal really is noisier in motion).
2. **Multiple people** — have 2–3 people in frame. Each face gets its own
   labelled box (`P0`, `P1`, …) with an independent HR.
3. **Far & near** — start close, then step back. YuNet keeps detecting as you move
   away; the ROI shrinks with your face and quality drops gracefully when you are
   too far / too small for a clean pulse.

## About the blood-pressure number

It is labelled **`exp`** and the banner says *"BP = experimental, not a medical
measurement."* Present it that way. Cuff-grade BP needs the blood-pressure video
datasets on our roadmap (academic access, in progress); this value is a
heart-rate + waveform heuristic so the demo can show the intended output.

**Single-point calibration (for a stronger BP demo):** take one real reading with
a cuff, pass it in, and press **`c`** while the subject is on camera — the display
re-anchors to your reference and then tracks changes around it (the tag switches
from `exp` to `cal`):

```bash
.venv\Scripts\python.exe scripts\live_demo.py --ref-sbp 118 --ref-dbp 78
```

This is an honest "calibrate once, track changes" demo, not an absolute
measurement — say so.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Wrong camera / black window | Try `--camera 1` (then `2`). |
| "Could not open camera" | Close Zoom/Teams/other apps using the webcam, then retry. |
| HR slow to appear | Needs ~5 s of a steady, well-lit face to fill its buffer. |
| HR jumpy | Improve lighting; avoid backlight (bright window behind you). |
| Window doesn't close | Click the video window, press `q`. |
| Low FPS with many faces | Lower resolution: add `--width 640 --height 360`. |

> First run downloads a 227 KB face-detection model into `models\` automatically —
> needs internet that one time.
