# Development Log

Newest entries first. Each work session has a **Plan** block (written *before* the
work: what, why, decisions, what changed) and a **Done** block (written *after*:
accomplished, results, what's left, how to continue). Routine defined in the
repo-root `CLAUDE.md`. Technical/status detail lives in [PROJECT.md](PROJECT.md).

---

## 2026-11-12 — Heart-rate accuracy push on MCD-rPPG (proposals 1–4 + new directions)

### Plan (intent / decisions)
- **Why:** the external validation (2026-11-09/10, outside the repo in
  `D:\rppg_eval`, report `D:\rppg_eval\report\RPPG_Validation_Report.docx`) put the
  shipped live HR at **MAE 17.45 bpm** on MCD-rPPG frontal webcam (40 subjects, 80
  recordings); published methods on MCD reach 2.82–4.86. What-if tests traced most of
  the gap to the front end: the Hampel spike filter turns 10 Hz light flicker into
  in-band junk, the background division adds noise, small ROIs are weak, and there is
  no tracking across windows. User approved implementing proposals 1–4 and asked for
  more ways to cut the error, as far as this CPU-only machine allows.
- **Decision — honest protocol first.** All tuning happens on a *development* set (the
  40 already-used subjects). A **fresh test set of 40 new MCD subjects** (never looked
  at while tuning) is downloaded for the final number, so the reported gain is not
  tuned on the people it is scored on. UBFC is re-checked for regressions.
- **Decision — a fast bench before code changes.** Re-extract richer per-frame
  features (each ROI separately, face box raw/skin-masked/inner, a patch grid, motion,
  small face crops) once, then score many ideas offline in seconds. Only winners go
  into `src/`.
- **Directions to test:** (1) front end — ROI choice, skin masking, SNR-weighted patch
  selection; (2) preprocessing — Hampel off / after band-pass, normalisation, detrend;
  (3) pulse methods — POS/CHROM/ProPOS/LGI/OMIT/PBV/GREEN, POS sub-window length;
  (4) spectrum — periodogram vs Welch, harmonic-sum scoring, band 42–180; (5) temporal
  tracking — causal Viterbi / harmonic folding; (6) learned options feasible on CPU
  (peak selector / small 1-D model) with subject-level CV; (7) the live loop itself
  (tracker, display stabiliser, gate) replayed on the test set.
- **Then implement** proposals 1–4 (+ any proven extras) in `src/rppg/live/` and
  `scripts/live_demo.py` with tests first; keep the parked Phase-0 work untouched.
- Blood pressure is out of scope here (stays experimental).

### Notes while working
- **Bench data:** `D:\rppg_eval\code\bench\extract.py` re-downloads MCD frontal videos
  and stores ~20 colour signals per frame (live patches, whole box raw/skin/masked,
  6×6 grid, landmarks, 36×36 crops). Dev = the 40 validation subjects (seed 0), test =
  40 NEW subjects (seed 1, never used for tuning). Extraction is CPU-bound (YuNet ~55 ms/
  frame under load); first run with 6 workers did ~0.9 recordings/min, restarted as two
  parallel runs (dev 5 + test 3 workers). Dev videos kept (6.6 GB on D:), test videos
  deleted after extraction (disk).
- **Methods bench** (old dev features, whole face box, no Hampel/no division):
  raw per-window peak MAE — POS 7.85 (42–180) / 8.35 (42–240), LGI 9.48, OMIT 9.52,
  ProPOS 9.77, CHROM 10.40, GREEN 16.19, PBV 16.52. With the causal Viterbi tracker
  POS reaches 4.24–4.30 (σ 3–4 bpm per 2 s); spectral fusions of several methods
  (4.75–5.3) and harmonic-sum spectra were worse. Subject-held-out CV: **4.33 bpm**.
  → POS alone + 42–180 bpm + tracker (proposals 3–4 confirmed).
- **Literature scan (sub-agent):** the official MCD code scores one 20 s clip per video
  with a raw periodogram (0.5–3.5 Hz) on both rPPG and PPG — not our 10 s/2 s windows —
  so published 3.80 (POS) is not directly comparable; the final evaluation therefore
  also reports that protocol. Grid/patch quality pruning (Face2PPG) and narrow-band
  refinement were suggested; fusion/top-k patch selection did not beat plain skin
  averaging in our smoke run.
- **Implemented (TDD):** `src/rppg/hr/track.py` (`SpectralTracker`: online Viterbi on a
  42–180 bpm, 1-bpm grid; evidence tempered by dt/2 s and random-walk variance ∝ dt so
  0.4 s live updates don't over-count overlapping windows; sub-bin parabolic refine),
  `PersonSignal` rewritten (no Hampel by default — `spike_filter=True` keeps the legacy
  path; POS → spectrum → tracker; SNR = ±9 bpm around the tracked rate; optional
  `stabilize`), `live_demo.py` `--roi face|face-skin|patches` (default whole face),
  `--illum-norm` opt-in. The parked `tests/test_windows_live.py` mirrors the legacy
  preprocessing, so it now asks for `spike_filter=True` explicitly.
- **Live-loop replay on dev** (new code, whole box): any number on screen MAE **4.67**
  (was 19.19); gate shows a number **62 %** of the time at **2.47 bpm** MAE, only
  **3.7 %** of shown numbers off by >10 bpm (was 43.1 %). Gate thresholds unchanged.
  SNR-threshold trade-off on dev: ≥−3 dB → 68 % coverage / 2.54 MAE; ≥0 dB → 40 % / 1.93.
- **UBFC regression check** (8 local subjects, cache_v2 → bench format): 10 s windows
  legacy 5.12 → new **1.26** bpm; live any 10.38 → **2.60**; committed 89 % at 2.07,
  2.1 % off by >10. No regression on the original dataset.
- **Buffer length / tracker σ** (dev, t = 20–90 s, ref = trailing 10 s): 10 s buffer
  with σ 3 bpm is best (4.77); 8 s 4.94, 12 s 5.03, 15 s 5.72, 20 s 6.14 (longer
  buffers lag the after-exercise HR drop). Keeps the current 10 s / σ 3 defaults.
- **Face-region bench** (dev, all 80 recordings, 462 variants, sub-agent script
  `exp_regions.py`): **skin pixels of the face box + POS + tracker σ 4 = 3.64 bpm**
  (both CV folds pick it; held-out 3.64), raw box 4.48, face minus eyes/mouth 4.68,
  inner box 4.66, legacy forehead+cheek skin pool 6.90, single forehead 10.1, single
  cheeks 12.6–14.5. Paired vs raw box: −0.92 bpm (95 % CI −1.90 to −0.10), better in
  32/40 subjects. Adaptive per-cell spectral fusion (SNR-weighted, softmax, top-k) was
  far worse (~15 bpm): single 6×6 cells are too noisy. → default `--roi face-skin`,
  tracker σ 4.
- **Live replay with the final defaults** (dev, face skin, σ 4): any number on screen
  **4.31** bpm; gate shows a number **72 %** of the time at **2.58** bpm, 3.8 % of
  shown numbers off by >10 bpm (whole raw box was 62 % / 2.47 / 3.7 %).
- **Neural retraining (ProPOSNet on MCD 36×36 face crops, sub-agent script
  `bench/neural/train_mcd.py --final`):** 40 dev subjects, NegPearson + FreqMSE,
  35 CPU-minutes (≈1,400 steps); NegPearson 0.77 → 0.58. Even **in-sample** the HR
  error is 18.89 bpm (bias +12), far behind the classical 3.64 — the small net needs
  much more training (GPU) than this machine allows. Scored on the held-out test set
  below; not promoted to `artifacts/` unless it wins there. `--neural` stays optional.
- **Speed:** extraction switched to YuNet on every 2nd frame (box held between) and
  OpenCV-circle masks after 46 dev / 6 test recordings — ~2× faster. The neural
  model is trained once on all dev subjects and scored only on the held-out test set
  (no separate CV run) to save ~70 CPU-minutes.
- **Code review (sub-agent, no CRITICAL/HIGH):** fixed three MEDIUM edge cases with
  tests — a flat/constant input produced a fake 42 bpm (now "no evidence", not ready),
  dt ≤ 0 re-counted the same window (now held), and a long gap now restarts the buffer
  and the tracker; duplicate band/σ constants removed from `estimate.py`.

### Done
- **Held-out test (40 NEW MCD subjects, 80 frontal recordings, never used for
  tuning; `D:\rppg_eval\bench\out\final_test.json`):**

  | | Before (shipped) | After (new defaults) |
  |---|---|---|
  | HR error per 10 s window (MAE, 95 % CI) | 17.85 (15.23–20.52) | **3.85 (2.30–5.92)** |
  | Within ±5 bpm (windows) | 42.7 % | 88.4 % |
  | Live: number on screen, any tick | 17.69 | **3.92** |
  | Live: share of time a number is shown | 57.7 % | **77.3 %** |
  | Live: error of shown numbers | 14.89 | **2.00** (93.9 % within ±5) |
  | Live: shown numbers off by >10 bpm | 43.3 % | **3.3 %** |

  Paired improvement (per-subject bootstrap): windows 11.4–16.6 bpm, live 10.2–17.3.
  Plain POS on the same face skin without tracking: 5.59. MCD's published 20 s-clip
  protocol (their scoring code): POS on our face skin **2.19 bpm** (published POS
  3.80, best model 2.82) — different subjects and face crop, so indicative only; the
  live tracker's reading after just 20 s scores 4.48 under that protocol.
  Neural ProPOSNet held-out: UBFC checkpoint 32.91 → MCD-retrained 20.62 (still far
  behind; `--neural` not recommended with current checkpoints, nothing promoted).
- **Project changes:** `src/rppg/hr/track.py` (new SpectralTracker), `src/rppg/live/
  estimate.py` (POS + tracker, no Hampel by default, gap reset, flat-input guard),
  `scripts/live_demo.py` (`--roi face-skin` default, `--illum-norm` opt-in),
  `src/rppg/live/select.py` (comment: gate re-validated), tests
  `test_spectral_track.py`, `test_live_estimate.py`, `test_live_demo_frontend.py`
  (+ parked `test_windows_live.py` now asks for the legacy filter explicitly).
  **456 tests pass.** Not committed — waiting for the owner's review.
- **Report:** `D:\rppg_eval\report2\RPPG_Improvement_Report.docx` (built by
  `report2\build_report.js` from the result JSONs; every step, data links, numbers).
- **What is left / how:**
  1. Owner reviews, then commit (`feat: whole-face POS + spectral tracker live HR`),
     keeping the parked Phase-0 files out of that commit.
  2. Field-test the demo on a real webcam: `.venv\Scripts\python.exe
     scripts\live_demo.py` (defaults now face-skin + tracker); compare `--roi patches`.
  3. Certified Abstention on top of the new SNR/tracker (flagship paper).
  4. Neural: GPU-scale retraining on many MCD subjects (`bench\neural\train_mcd.py`),
     or a small learned peak selector on POS spectra.
  5. Motion / side-camera / fast after-exercise HR changes remain the weak cases
     (RMSE ~13 shows a few recordings still fail).

---

## 2026-10-29 — Flagship novelty: Certified Abstention for camera HR (paper-first)

### Plan (intent / decisions)
- User wants a **novelty worthy of an IEEE international publication that also wins a
  hackathon**. Method: multi-agent research — 4 candidate directions, each a prior-art
  search **plus an adversarial skeptic**; a field gap survey; an open brainstorm; and a
  synthesis grounded in our own measured data.
- **Verified landscape:** all four directions survive only *with a narrowed claim*. The
  closest abstention work is Liao et al., *Nature* 2026 (Google; smartphone camera HR,
  entropy cutoff tuned to skin-tone equity rules — no guarantee, no no-pulse
  calibration). Also Cali-rPPG (ICIC 2025), Song et al. (IEEE TIM 2023),
  RF-BayesPhysNet (2025), rPPG-VQA (CVPR 2026), Speth et al. (BIOSIGNALS 2023,
  hallucinated heartbeats), Kateri (PhD thesis, Southampton 2025, conformal rPPG). The
  field survey ranks "no validated way to know when not to report an HR" as open
  problem #1.
- **Decision (user):** flagship = **Certified Abstention for Camera Heart Rate** —
  selection-aware, frequency-conditional no-pulse calibration + a physics-consistency
  score + subject-level risk control, wrapping any estimator. **Breathe-to-Verify**
  (paced-breathing liveness) is a hackathon demo extra only. **Paper-first**; no fixed
  hackathon date.
- **Headline, reproduced independently** (no-pulse input = the 4-corner background
  trace of each cached subject, 237 × 10 s windows, exact live classical pipeline):
  median SNR POS −4.50 dB, CHROM −4.48, ProPOS −2.47, fused −1.96 dB. The live −1 dB gate
  passes **35.0%** of no-pulse windows (POS alone 4.2%). **97.9%** of the hallucinated
  rates fall in 42–180 bpm, **median 56.3 bpm** — the mechanism behind the live "45–60
  bpm that varies". Best-of-candidates selection inflates no-pulse SNR by about
  +2.5 dB (difference of medians, fused − POS).
- **Do NOT claim:** first conformal rPPG (Kateri 2025); first no-pulse evaluation
  (Speth 2023); first model-agnostic confidence wrapper (Cali-rPPG 2025); novelty of
  agreement / ROI / chroma cues; risk-coverage reporting by itself. **Report** that
  ProPOS's 50-basis search drives most of the inflation (a property of
  max-over-projections designs).
- **Constraints (measured 2026-10-29):** E: 0.24 GB free, C: 3.35 GB free. With 7–8
  subjects no useful certificate exists (minimum certifiable α ≈ 0.28 at δ = 0.1);
  ≥30 subjects → α ≈ 0.074. Needs (user decisions/actions) freeing the ~14.4 GB raw
  UBFC videos *after* cache v2 + face crops are extracted, streaming more UBFC subjects,
  and recording ~50–100 no-pulse clips (printed photo / phone screen / mannequin — no
  human subjects). `best.pt` was trained on these same 8 subjects → subject-disjoint
  retraining before any ProPOSNet evaluation.

### Done (accomplished / results)
- **Foundation committed:** `documentation/ARCHITECTURE.md` (dac0b7b); frame cache
  `src/rppg/eval/frame_cache.py` + `windows.py` + `scripts/build_frame_cache.py`, 11 tests
  (c95692d); window-level reliability table `src/rppg/eval/window_table.py` +
  `scripts/build_window_table.py`, 4 tests (0ab03e1). Full suite: **52 passed**.
- Cache for subjects 10–17 at `C:/rppg_data/cache` (100% face detection). Fused HR MAE
  per subject (10 s / 2 s): 0.51, 0.82, 0.88, 0.35, 1.52, **26.85** (s15), 0.52, 1.72 bpm;
  237 windows, 18 failing (>10 bpm), 15 from subject15. subject15 is an estimator
  failure, not bad ground truth (contact PPG and oximeter agree at ~110–121 bpm while
  the estimate locks onto ~48–58 bpm).
- **Pilot (read-only; cues chosen after seeing the data, so optimistic):** AUROC for
  flagging failing windows, pooled (within subject15): SNR 0.958 (0.822), method spread
  0.953 (0.822), ROI spread 0.844 (0.569), ROI-median gap 0.828 (0.844), **2×-harmonic
  ratio 0.853 (0.862)**. Pooled risk-coverage (SNR ranking): MAE 4.19 → 0.95 bpm at 80%
  coverage. A threshold calibrated on 7 subjects fails on the atypical one: subject15
  accepted 83–93% of windows with 40–46% wrong.
- _Next:_ reliability package (cues / null / score / certify / metrics / baselines),
  live-faithful offline preprocessing, cache v2 + crops, a dated pre-registration,
  Phase 0 experiments, null-clip recording, subject expansion, then the live certified
  gate and the Breathe-to-Verify demo.

---

## 2026-10-19 — MediaPipe precision landmarks + hard confidence gate + robustness

### Plan (intent / decisions)
- User wants: (1) switch the live face landmarks to **MediaPipe FaceLandmarker**
  (478-point mesh, highest precision) for precise forehead/cheek ROIs; (2) a **hard
  confidence gate** — show "signal too weak" instead of a misleading number when SNR
  is low (dim / motion / flicker); (3) propose + apply **further robustness
  improvements** ("best in any condition", within physical limits).
- **De-risk done**: MediaPipe 1.1.0 Tasks FaceLandmarker confirmed working here
  (478 landmarks/face, num_faces=3, VIDEO mode) on a real face.
- **Decision — multi-agent research then implement** (user preference). Keep YuNet as
  a fallback behind a `--detector` flag; route the gate through overlay/arbiter without
  touching the Reading dataclass where possible.
- Honest stance: no algorithm gives reliable rPPG in dim + harsh-flicker + motion; the
  confidence gate is the correct answer there (say "weak" rather than fake a number).

### Done (accomplished / results)
- **MediaPipe 478-mesh detector** (`--detector mediapipe`, YuNet fallback): real-face
  check gave 3 valid ROIs with cheeks on opposite sides of the nose (2756b3b).
- **Confidence gate — sub-agent thresholds were wrong and were caught:** `_SNR_ENTER`
  3.0 dB sat near a clean face's *peak* SNR and the gate scored the neural SNR (max
  ~2.1 dB), so it never trusted subject10. Re-derived from measured subject10
  distributions (classical SNR median 1.7 dB; neural/classical agree within ~9 bpm
  p90) → trusts subject10 at 7.5 s on both detectors (2756b3b). Later retuned less
  restrictive for live webcams: enter −1.0 dB / exit −3.5 dB, 1.0 s dwell, edge 42 bpm
  → trusts subject10 at **5.6 s**, HR 99.6 vs ~97 (0baaaea).
- **Display stabilizer** (median-5 → EMA 0.2 → 2.5 bpm/update slew; first number at 3 s,
  lock at 8 s): subject10 HR 97.5, locked std 2.61 bpm, **max consecutive jump 0.85 bpm**
  (was up to the 8 bpm tracker slew) (5ae9e25).
- **FAILED — autonomous motion-robustness build reverted:** polygon ROIs + per-ROI
  fusion + bounded peak tracking + signal-absent detector passed 96 unit tests but on a
  real face regressed HR **97 → 54 bpm** (splitting into small per-ROI samples lost the
  spatial averaging; `roi_agree` was 0 on every window) and the signal-absent detector
  false-fired on **38/38** clean-face windows (flatness/entropy thresholds guessed; real
  face flatness 0.48–0.99, entropy 0.78–0.99). Stashed as `stash@{0}`
  ("wip-s1-s4-broken-integration"); `quality.py` / `oneeuro.py` there are salvageable.
- **Lesson:** every signal-path change must pass a real-face HR-accuracy check, not just
  unit tests.
- Left: honest state-machine overlay; motion work that re-pools good ROIs instead of
  estimating per tiny ROI. Superseded in priority by the 2026-10-29 flagship.

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
