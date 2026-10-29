# rPPG system architecture — code walkthrough

A presentation-ready explanation of **what each part of the code does and how it works**.
The system estimates **heart rate** (and an experimental **blood-pressure** proxy) from a
plain face camera — no contact sensor — using remote photoplethysmography (rPPG): tiny,
pulse-synchronous colour changes in facial skin as blood volume rises and falls.

---

## 1. The one-line data flow (put this on a slide)

```
Camera  →  face box  →  forehead/cheek skin RGB  →  10 s pulse buffer  →
POS / CHROM / ProPOS  →  spectral peak + SNR  →  fuse  →
HR tracker + display smoother  →  confidence gate  →  number on screen
```

Every arrow is one stage of the block diagram; every stage maps to one file below.

---

## 2. Where the code lives (thin demo vs. reusable library)

| Path | Purpose |
|------|---------|
| **`scripts/live_demo.py`** | **The main runnable program.** Entry point `main()`. Opens the camera, runs the per-frame loop, draws the overlay. Thin — it only *orchestrates*. |
| `src/rppg/live/` | The live engine: detection, per-person estimation, source arbitration, overlay. |
| `src/rppg/hr/` | Heart-rate maths: band-pass, spectral peak, SNR, fusion. |
| `src/rppg/methods/classical/` | The rPPG algorithms (POS, CHROM, ProPOS, PBV). |
| `src/rppg/models/`, `src/rppg/live/neural.py` | The trained neural net (ProPOSNet) and its wrapper. |
| `src/rppg/bp/`, `src/rppg/live/bp_model.py` | The (experimental) blood-pressure pipeline. |
| `src/rppg/data/`, `src/rppg/eval/`, `scripts/` | Offline dataset loading, evaluation and training (not used by the live demo). |

**Design point for the talk:** the demo is a thin shell; the science is a reusable library
under `src/rppg/`. You can import and test the algorithms with no camera.

---

## 3. The main file — `scripts/live_demo.py`

Run it with the project virtual environment:

```
.venv\Scripts\python.exe scripts\live_demo.py --neural --detector mediapipe
```

### 3.1 Helper functions (top of the file)

| Function | Line | What it does |
|----------|------|--------------|
| `_pooled_skin_rgb(frame, rois)` | 44 | **The colour front-end.** For each ROI patch (forehead + 2 cheeks) it keeps only *skin* pixels (a YCrCb colour gate), and returns their **mean (R, G, B)**. This mean colour, sampled every frame, *is* the raw rPPG signal. Falls back to the raw patch mean if too few skin pixels pass, so a reading is never lost under hard light or darker skin tones. |
| `_background_rgb(frame, boxes)` | 76 | Samples the **four frame corners** as an ambient-light reference, used to divide out room-light flicker. |
| `_open_camera(...)` | 91 | Opens the webcam; forces **MJPG** (avoids a low-FPS codec), 30 fps, a 1-frame buffer (low latency) and fixed white balance for a steadier colour signal. |
| `build_parser()` | 108 | All command-line flags (`--camera`, `--detector`, `--neural`, `--bp-model`, `--lock-exposure`, …). |
| `_build_detector(name)` | 135 | Builds the chosen face detector; falls back to YuNet if MediaPipe can't load. |

### 3.2 `main()` — the orchestration loop (line 148)

| Lines | Section | What happens |
|-------|---------|--------------|
| 149–159 | **Start-up** | Parse args, open the camera, print a diagnostic. |
| 161–182 | **Build components** | Create the detector, `CentroidTracker`, `SignalBank`, `SourceArbiter`; optionally load the neural net and the trained BP model (both wrapped in try/except so the demo runs without them). |
| 193–199 | **Grab a frame** | Read → mirror → timestamp. |
| 201–208 | **Detect + track** | `detector.detect()` finds faces; `tracker.update()` gives each a stable ID; the nearest N faces are marked for the heavy neural net. |
| 210–222 | **Per-face colour read** | Build the skin-pooled RGB (`_pooled_skin_rgb`), divide by the background reference to cancel light swings, push the sample into that person's rolling buffer (`bank.add`). |
| 228–241 | **Estimate + decide** | Every 0.4 s recompute the classical reading (`bank.estimate`), every 1 s the neural reading; `arbiter.update()` picks the headline source and runs the **confidence gate** → a `Decision`. |
| 242–246 | **Draw** | `draw_face()` renders the box, ROIs, HR, BP, quality for that person. |
| 247–258 | **Housekeeping** | Prune dead tracks, compute smoothed FPS, draw the header. |
| 260–277 | **Display + keys** | Show the window; `q`/`Esc` quit, `s` screenshot, `c` calibrate BP to a reference. |

The loop is **per-person**: each tracked face has its own buffer, estimator state and gate,
so several people can be measured at once.

---

## 4. File-by-file reference (the "sections" to explain)

### 4.1 `src/rppg/live/detect.py` — detection, tracking, ROI geometry
- **`YuNetDetector`** — runs OpenCV's YuNet face model (ONNX, 5 landmarks). Fast, works
  across distances; brightens very dark frames before detecting.
- **`Detection`** — one face: bounding `box`, three `rois` (forehead + 2 cheeks), 5
  `landmarks`, a `score`. The cheek ROIs are placed on the malar bone and their inner edge
  is clamped off the nose so they stay on pulsatile skin.
- **`CentroidTracker`** — matches detections across frames by centre distance, giving each
  face a **stable integer ID** so its signal buffer persists while the person moves.

### 4.2 `src/rppg/live/mediapipe_detect.py` — higher-precision detector
- **`MediaPipeDetector`** — Google's 478-point **face mesh** (with iris). It distils the
  dense mesh into the same forehead/cheek ROI triplet, so the rest of the pipeline is
  unchanged. Selected with `--detector mediapipe`; more precise ROIs, heavier on CPU.

### 4.3 `src/rppg/methods/classical/` — the rPPG algorithms
These turn the per-frame RGB trace into a 1-D **blood-volume-pulse waveform**:
- **`pos.py` — POS** (Plane-Orthogonal-to-Skin, Wang 2017): projects normalized RGB onto a
  plane orthogonal to the skin-tone direction, isolating the pulse from intensity changes.
- **`chrom.py` — CHROM** (de Haan 2013): a chrominance combination of RGB that cancels
  specular/motion components.
- **`propos.py` — ProPOS**: rotates the POS projection plane into several candidate bases and
  keeps the one with the strongest pulse (best SNR).
- **`pbv.py` — PBV**: uses the known blood-volume colour signature as a third, independent
  method. (Available in the library; wired into the live fuse as the project adds it.)

### 4.4 `src/rppg/hr/refine.py` — pulse → heart rate
- **`bandpass(sig, fps)`** — zero-phase Butterworth band-pass to the cardiac band
  (0.7–4 Hz ≈ 42–240 bpm), so out-of-band noise can't win the peak.
- **`hr_peak_interp(sig, fps)`** — the core HR+quality step: Hann-window → heavy zero-pad →
  FFT → find the tallest in-band spectral peak → **parabolic sub-bin interpolation** for
  ~0.3 bpm resolution. Returns `(hr_bpm, snr_db)`, where **SNR** = power in a narrow band
  around the peak over the rest of the band (a self-referential spectral-purity measure — it
  is reused for signal quality, not a ground-truth comparison).
- **`fuse_estimates(list of (hr, snr))`** — SNR-weighted blend of the methods near the
  strongest one, so agreeing methods reinforce and a lone harmonic-lock outlier is dropped.

### 4.5 `src/rppg/hr/metrics.py` — offline SNR
The de Haan SNR definition (anchored to a *known* reference HR). Used for **offline
evaluation** on labelled datasets, not for the live gate.

### 4.6 `src/rppg/live/estimate.py` — the per-person engine (the heart of the system)
- **`SignalBank`** — one `PersonSignal` per tracked face id; prunes faces that leave.
- **`PersonSignal`** — a rolling **10 s** buffer of `(timestamp, RGB)`. On `estimate()` it:
  1. **Resamples** the irregular samples to a uniform time grid and applies a **Hampel
     filter** to remove optical spikes (`_uniform_trace`).
  2. Runs **POS, CHROM, ProPOS**, band-passes each, scores `(hr, snr)` and **fuses** them.
  3. Feeds the raw HR to the **`HRTracker`**, then the **`DisplayStabilizer`**.
  4. Packs everything into a **`Reading`**.
- **`HRTracker`** — the *lock* logic: folds 2× / ½× harmonic locks back toward the current
  rate, rate-limits jumps, and only follows a large jump once several confident readings
  agree. Low-confidence readings just hold the last value instead of showing noise.
- **`DisplayStabilizer`** — the *presentation* smoother: a rolling **median (5)** →
  **EMA (α 0.2)** → **slew clamp (2.5 bpm/update)** on the committed HR, so the on-screen
  number is steady. SOTA readouts smooth the bpm time-series rather than showing the raw
  per-window FFT peak; a genuine HR change still fully appears within ~7–10 s.
- **`Reading`** — the output struct: `ready`, `hr`, `bp_sys/bp_dia`, `snr`, `provisional`,
  `pulse` (for the on-screen waveform).
- **`_experimental_bp(...)`** — the fallback BP heuristic from HR + pulse morphology
  (clearly labelled experimental).

### 4.7 `src/rppg/live/neural.py` + `src/rppg/models/proposnet.py` — trained neural HR
- **`NeuralHR`** — buffers face crops and runs the learned **ProPOSNet** to produce an HR +
  SNR. Enabled with `--neural`. `models/` also holds `physnet.py`, `tscan.py`, `losses.py`
  for training/experiments.

### 4.8 `src/rppg/live/select.py` — source choice + confidence gate
- **`SourceArbiter`** — per face, scores the neural vs. classical source on recent
  **stability** (low jitter), **SNR** and cross-source **agreement**, and shows the neural
  HR as the headline unless classical is clearly and persistently steadier.
- **The confidence gate** — a **Schmitt-trigger / leaky-dwell** state machine. Entering
  "trusted" needs sustained good signal; leaving it needs sustained bad signal, with an
  asymmetric floor, so the headline cannot flicker between a number and a warning. When not
  trusted it returns a reason (`acquiring`, `low signal`, `sources disagree`, …) so the demo
  shows an honest state instead of a misleading rate.
- **`Decision`** — what the overlay consumes: `main` source, `hr`, the other source's HR,
  `trusted`, `gate_reason`.

### 4.9 `src/rppg/live/bp_model.py` + `src/rppg/bp/` — trained blood-pressure model
- **`BPModel`** wraps a trained regressor (`bp/regressor.py`) over pulse-morphology features
  (`bp/features.py`). **Experimental on camera rPPG** — there is a real domain gap, and the
  overlay always tags BP as experimental.

### 4.10 `src/rppg/live/overlay.py` — drawing
- **`draw_face`** — per person: the box, the three ROI rectangles, the HR, the experimental
  BP, a quality label + colour (green/amber/red = strong/usable/weak), and a small pulse
  sparkline. Colour is the honest confidence cue.
- **`draw_header`** — the top banner (title, FPS, people count, BP disclaimer).

---

## 5. The algorithms, in one paragraph each (for Q&A)

- **rPPG principle:** each heartbeat pushes blood into facial capillaries, so skin reflects
  slightly more/less light in sync with the pulse. Averaging many skin pixels (spatial
  averaging) lifts this sub-percent signal out of camera noise.
- **POS / CHROM / ProPOS:** different linear projections of the RGB trace that cancel
  illumination and motion while keeping the pulse. Running three and fusing by SNR is a
  cheap robustness win — a method having a bad moment is outvoted.
- **Spectral HR + sub-bin peak:** the pulse is quasi-periodic, so its FFT has a peak at the
  heart rate; zero-padding + parabolic interpolation locate it to a fraction of a bpm on a
  short window.
- **HR tracker vs. display stabilizer:** two different jobs. The tracker decides *what the
  rate is* (reject harmonics, confirm big changes). The stabilizer decides *what to show*
  (median + EMA + slew) so the number doesn't twitch.
- **Confidence gate:** separates *truth* from *presentation* — only commit a number when the
  signal is sustainedly good, otherwise show an honest "acquiring / too weak" state.

---

## 6. Honest limitations (say these — they strengthen the talk)

- **Blood pressure is an experimental, unvalidated proxy**, not a medical measurement.
- **Dim + heavy motion is a physical floor:** low light forces a long camera exposure → low,
  jittery frame rate → little pulse captured. No algorithm manufactures a pulse the sensor
  did not record; the honest behaviour there is to say "too weak — add light."
- **Best accuracy (≤ ~3 bpm) is for cooperative conditions** (still, lit, near the camera);
  motion and distance degrade it, and the overlay shows that honestly.

---

## 7. Suggested slide order

1. Problem + rPPG principle (what skin colour tells us).
2. The block diagram (§1 / the 8-stage figure).
3. Thin demo vs. library (§2).
4. The colour front-end and ROIs (§3.1, §4.1–4.2).
5. The rPPG methods + spectral HR (§4.3–4.4).
6. Tracker + display stabilizer + confidence gate (§4.6, §4.8).
7. Honest limitations (§6).
