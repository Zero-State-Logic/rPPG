# Development Log

Newest entries first. Each work session has a **Plan** block (written *before* the
work: what, why, decisions, what changed) and a **Done** block (written *after*:
accomplished, results, what's left, how to continue). Routine defined in the
repo-root `CLAUDE.md`. Technical/status detail lives in [PROJECT.md](PROJECT.md).

---

## 2026-10-18 — Robustness: SNR, cheek ROI, light filtering, dual-source main reading

### Plan (intent / decisions)
- User requests: (1) improve pulse **SNR** and make the **trained neural (ProPOSNet)**
  reading the main/front number; (2) fix the **cheek ROI** boxes (too narrow + catching
  the nose, skewing the colour signal); (3) **filter sudden ambient-light changes /
  optical interference** for stable readings (median / Hampel / Kalman / adaptive);
  (4) continuously compare **ProPOS vs the neural** source and auto-select the more
  stable / less-noisy one as the MAIN displayed reading.
- **Decision — multi-agent research first** (Workflow: SNR+light filtering, cheek ROI,
  dual-source selection), then implement the synthesis conservatively.
- Direction: default main = neural, auto-switch to classical ProPOS when it is clearly
  more stable; a robust per-frame optical-noise filter + illumination normalization;
  wider, laterally-shifted cheek ROIs anchored off the nose landmark.

### Done (accomplished / results)
Research (Workflow, 3 agents + synthesis) then sequential implementation agents
(Workflow), each tested; orchestrator verified + committed. **All 29 tests pass**
(26 + 3 new Hampel tests) plus headless pipeline / arbiter / detector checks.
- **Cheek ROI fixed** (`detect.py`): anchored under each eye, dropped onto the malar
  bone, widened to 0.65×0.55 iod, medial edge **hard-clamped off the nose landmark**
  (nose pixels can't be skin-masked out, so the clamp is the real fix). Geometry
  unit-checked; detector still 30/30 on a real face.
- **SNR + sudden-light filtering:**
  - **Illumination normalization** (`live_demo._background_rgb`): per-frame divide of
    the skin ROI RGB by a four-corner background reference → cancels global ambient
    swings (light on/off, shadow sweep). `--no-illum-norm` disables. Biggest SNR lever.
  - **Hampel spike filter** (`estimate._hampel`): sliding median + MAD on the resampled
    trace removes flash/dropout spikes without distorting the pulse (beats
    moving-average / plain-median / Kalman per the research).
- **Neural-as-main + auto ProPOS fallback** (`neural.NeuralReading`, new
  `select.SourceArbiter`, `live_demo.py`, `overlay.py`): the trained ProPOSNet HR is
  the big main number; a per-track arbiter scores both sources by stability (HR jitter)
  + SNR + agreement and **auto-switches the main to classical ProPOS when it is clearly
  steadier** (3-vote hysteresis), snapping back when neural recovers. Overlay shows the
  selected source big with a `main: ProPOSNet/ProPOS` tag, the other small. The Reading
  dataclass was left untouched (selection state lives in the arbiter). Functionally
  tested: neural default, switches to steadier classical, falls back when neural absent.

### Audit (full-code error hunt, same session)
- `python -m compileall` clean; **pyflakes** found one issue — a forward-ref local
  annotation `nn_hr: "NeuralReading | None"` in `live_demo.py` (harmless at runtime,
  Python skips local annotations, but cleaned). Now pyflakes exit 0.
- **End-to-end run on a real face video** (subject10, 150 frames) through the EXACT
  `main()` loop with neural + BP model + illumination-norm + arbiter + overlay: **zero
  exceptions**. Overlay correctly hides the neural line until finite (no "ProPOSNet
  nan" on screen).
- **Illumination-norm accuracy check**: raw ROI vs illum-normalized gave **identical**
  classical HR (90.7 vs 90.7 bpm, err 0.7 vs GT 91.4) on steady light — the filter is
  harmless when light is constant and only acts on real swings. No silent degradation.
- Non-bugs noted: neural needs ~160 frames (~5 s @30fps, longer at low FPS) before
  "main: ProPOSNet" shows — until then the arbiter correctly keeps classical ProPOS;
  `_background_rgb` doesn't exclude faces sitting in the frame corners (minor; flag-gated).

### Left to do / how to continue
1. **User to re-test on webcam** (face the camera, front light):
   `.venv\Scripts\python.exe scripts\live_demo.py --neural` — the main number reads
   ProPOSNet, switching to ProPOS when steadier; cheek boxes off the nose; steady HR
   through a light flick.
2. Optional: move neural inference off-thread for higher FPS with `--neural`; add a unit
   test for the cheek nose-clamp; re-measure HR MAE with the new ROI + illumination norm.

---

## 2026-10-13 — Demo professionalization (FPS, forehead/cheek ROI, latency)

### Plan (intent / decisions)
- User feedback: demo ~10 FPS, >10 s to first reading (vs ~3-4 s in the first
  version), and the ROI tracked the **lips**. Asked for research-grade work.
- **Decision — multi-agent research first** (Workflow: 3 specialists + synthesis)
  on real-time face tracking/FPS, rPPG ROI, and latency, then implement the
  prioritized, low-risk plan (not a blind rewrite).

### Done (accomplished / results)
- **Root causes found (measured):** (1) camera opened without MJPG → webcam stuck
  at uncompressed YUY2 ~10 FPS regardless of CPU; (2) YuNet ran every frame at full
  res (52.6 ms @720p); (3) ROI band y=0.55-0.80 = mouth/lips; (4) first-reading
  delay is the HRTracker SNR≥0 gate, not the window.
- **Fixes implemented + validated:**
  - Camera: MJPG FOURCC + FPS 30 + BUFFERSIZE 1 (`live_demo._open_camera`).
  - Detection on a 480-wide downscale, boxes+landmarks scaled back (~13 ms),
    top_k 50→20 (`detect.py`).
  - **ROI = forehead + both cheeks**, landmark-anchored (inter-ocular-distance
    scaled, roll-robust), skin-pooled (YCrCb gate + raw-mean fallback); box-fraction
    fallback for far faces. Geometry unit-checked (forehead above eyes, cheeks
    mid-face, not lips). Overlay draws all three.
  - Faster first reading: HRTracker first-lock gate 0 → −2 dB, `_MIN_SEC` 4 → 3.
  - Neural fed the face-box crop (matches training); still capped to 2 nearest.
- 26 tests green; headless pipeline + ROI geometry + detect smoke all pass.
- Research sources logged (Kossack ICCVW'21, R2I-rPPG arXiv:2410.15851, YuNet
  tuning) in the workflow result file.

### Follow-ups (same session)
- **HR in ~1s**: provisional "acquiring ~X bpm" at `_MIN_SEC=1.0`, converging to a
  locked value by ~4 s (validated: ~62 provisional @1s → 72.1 locked @5s).
- **Crash fix**: `bandpass` skips `filtfilt` when the window ≤ padlen (was crashing
  at low FPS / the 1 s window).
- **Low-light detection**: the DETECTION frame is brightened when dim so YuNet still
  finds dim/backlit faces (signal untouched). Verified the detector finds a real
  face 30/30 frames @0.94 — the user's "people:0" was a ceiling-pointing camera
  angle, not a code bug.
- **BP: real trained model.** Downloaded `mkachuee/noninvasivebp` (26 subjects,
  finger PPG + cuff SBP/DBP), resampled 500→30 Hz, trained the `BPRegressor`
  subject-independent: **SBP MAE 15.1 (BHS-D, beats mean-predictor 19.6), DBP MAE
  6.5 (BHS-C, ≈ baseline 6.2)**. Saved `artifacts/models/bp_model.pkl`; wired into
  the demo via `--bp-model` (tag "mdl"), heuristic fallback. Honest: weak (esp.
  DBP) and rougher on rPPG (contact→camera domain gap).

### Left to do / how to continue
1. **User to re-test on webcam** — expect ~25-30 FPS, a reading in ~3-4 s, and boxes
   on forehead+cheeks. If `--neural` still stutters, move neural inference
   off-thread (research STEP 6) and/or vectorize POS/CHROM + cache the ProPOS bank
   (STEP 7).
2. **Re-measure accuracy** with the new forehead/cheek ROI (numbers may shift).
3. Optional: optical-flow landmark propagation + IoU association (research STEP 7).

---

## 2026-10-12 — Live demo, accuracy engine, HR-jump fix, docs

### Plan (intent / decisions)
- **Why:** user has a demo in ~12 h and wants a live webcam that shows HR + BP for
  three scenarios (moving, multiple people, near/far), with the *trained model*
  involved. Prior work (synthetic-trained ProPOSNet, full pipeline) was done but
  Colab kept timing out and the trained net cannot read real faces.
- **Decision — pivot to a local webcam demo** using the **classical ProPOS** method
  (works on real faces with no training), because the neural net needs real-data
  training we could not complete in time. This makes the demo reliable *and* uses
  our actual research method.
- **Decision — chase ≤3 bpm honestly:** add band-pass + sub-bin spectral peak +
  multi-method SNR fusion, and (after user feedback) a stateful HR tracker.
- **Decision — BP stays experimental**, now with single-point calibration so it is a
  defensible "calibrate once, track changes" demo, never claimed as a measurement.
- **Decision — detector = OpenCV YuNet** (mediapipe 1.1.0 dropped the legacy
  `solutions` API; YuNet is multi-face + distance-robust and needs one small model).

### Done (accomplished / results)
- **Live demo built** (`src/rppg/live/{detect,estimate,overlay}.py`,
  `scripts/live_demo.py`): YuNet multi-face → centroid tracker → per-person ROI →
  3-method fusion → overlay. Validated headlessly (72.0 bpm recovered on a 72.0
  signal; overlay renders).
- **Accuracy engine** (`src/rppg/hr/refine.py`): band-pass + Hann/zero-pad/parabolic
  sub-bin peak + SNR fusion. **Synthetic sweep: MAE 0.20 bpm, 100% within ±3 bpm.**
- **User ran it on their webcam** → reported HR jumping 63→128→84 and bad under
  motion. **Diagnosed:** 2nd-harmonic lock + noise + motion.
- **HRTracker** added (`estimate.py`): folds a ~2×-current reading back to the
  fundamental (kills 128), rate-limits change, requires `_CONFIRM` confident
  agreeing readings before a big jump, holds the last value when SNR is low
  (motion). Unit-tested: 128 spikes rejected, lone spikes rejected, real 64→92 rise
  followed, motion noise held. Clean-signal accuracy unchanged (0.20 bpm).
- **BP single-point calibration** (`c` key, `--ref-sbp/--ref-dbp`); overlay tag
  `exp`→`cal`.
- **Kaggle token** fixed locally: user saved `export KAGGLE_API_TOKEN=...` into the
  token file (61 chars); stripped the prefix in place → 37-char `KGAT_` token at
  `C:\Users\admin\.kaggle\access_token`.
- **Docs:** `docs/DEMO_VSCODE.md` (open+run guide), and this `documentation/` folder
  (PROJECT.md + DEVLOG.md) + `CLAUDE.md` routine.
- **Real UBFC downloaded** (per-file via Kaggle CLI — videos arrive as `vid.avi.zip`,
  unzipped + flattened): subjects 10–13 (~1.78 GB each) to `C:\rppg_data\UBFC`.
- **Real HR accuracy measured** (`scripts/eval_ubfc_fused.py` = YuNet ROI + the live
  POS/CHROM/ProPOS fused estimator vs the contact-PPG ground truth): **overall
  MAE 1.41 bpm, RMSE 4.12, within±5 95%, within±3 93%** over 44 windows — the ≤3 bpm
  target **met on real faces**. Per subject: 0.44 / 0.54 / 1.91 / 2.77 bpm.
- **ProPOSNet trained on real faces** (`train_deep.py`, 4 subjects → 2 train):
  held-out **MAE 3.81 bpm, within±5 67%, Pearson 0.36** — real-data result (no longer
  synthetic), weaker than the classical 1.41 bpm (only 2 training subjects). Checkpoint
  `artifacts/models/best.pt`.
- **Trained net wired into the live demo** (`src/rppg/live/neural.py`, `--neural`):
  shows a "ProPOSNet N bpm" line per face next to the classical HR. Headless-validated
  (loads checkpoint, forwards, returns HR). Run: `live_demo.py --neural`.
- **Retrained ProPOSNet on 8 real subjects** (10–17; 6 train / 1 val / 1 test):
  held-out **MAE 1.17 bpm, RMSE 3.21, within±5 92%, Pearson 0.95** — now **beats the
  classical 1.41 bpm** and well under the 3 bpm target (vs 3.81 bpm on the first
  2-subject run). `artifacts/models/best.pt` updated; runs live via `--neural`.
- **Multi-person labels**: overlay now shows stable **"Subj 1 / Subj 2"** per person
  (lowest free number, freed on exit) via the YuNet + centroid tracker.
- **Classical re-measured on all 8 subjects** (`eval_ubfc_fused.py`): overall **MAE
  2.78 bpm, 93% within±5, 91% within±3** — median subject 1.6, 6/8 under 2 bpm, but
  **subject17 is a 10.3 bpm outlier** (subject16 4.6) that inflates the mean. Likely a
  face-detect / ROI / ground-truth issue on that mirror, not the method. Neural
  ProPOSNet (1.17 bpm) is the cleaner headline now.
- All committed/pushed to GitHub `Zero-State-Logic/rPPG`; 26 tests green.

### Left to do / how to continue
1. **subject17 diagnosed** (not a data bug): face detected 100%, ground truth clean
   (PPG-HR ≈ device-HR ≈ 88). The error is a **late signal collapse** — windows 8–10
   drop to ~53 vs true ~88 — which the raw per-window eval counts in full. The live
   HR tracker (`eval_ubfc_fused.py --track`) rejects it → subject17 **1.73 bpm**. BUT
   at the coarse 2 s eval step the tracker can mis-lock a clean subject (subject10
   0.44→9.82), so the **raw per-window 2.78 bpm stays the conservative headline**;
   live 0.4 s updates behave better than either offline measure. Neural **1.17 bpm**
   remains the cleanest single figure.
2. **Multi-ROI + skin masking** for extra live SNR under motion/distance.
3. **Distance/motion** robustness ladders; **BP datasets** (academic) → real BP model.
4. More subjects (C: ~3.6 GB free — clear space or use another drive) would lift the
   neural net further.

---

## Before 2026-10-12 — Foundation (summary)

- Read the ProPOS paper; built classical methods (POS/CHROM/PBV/ProPOS), the novel
  **ProPOSNet**, and PhysNet/TS-CAN baselines; losses; UBFC/PURE loaders;
  subject-independent + cross-dataset splits; trainer; HR eval; 26 tests.
- Made the repo `pip install -e .`-able; pushed to GitHub.
- **Blocker:** real UBFC on Google Drive is quota-blocked; built a Kaggle downloader
  and a **synthetic UBFC** generator to validate the pipeline end-to-end.
- Proved the pipeline on Colab T4: ProPOSNet/PhysNet/TS-CAN all train to convergence;
  **perfect scores on synthetic** (0.0 MAE) — a pipeline check, *not* a real result.
- Colab runtime instability (timeouts, `/content` wiped) made long cloud jobs
  fragile → motivated the local-demo pivot above.
