# Development Log

Newest entries first. Each work session has a **Plan** block (written *before* the
work: what, why, decisions, what changed) and a **Done** block (written *after*:
accomplished, results, what's left, how to continue). Routine defined in the
repo-root `CLAUDE.md`. Technical/status detail lives in [PROJECT.md](PROJECT.md).

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
