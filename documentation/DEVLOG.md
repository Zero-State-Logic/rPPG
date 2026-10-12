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
- All committed/pushed to GitHub `Zero-State-Logic/rPPG`; 26 tests green.

### Left to do / how to continue
1. **Train ProPOSNet on the real subjects** (`scripts/train_deep.py --dataset ubfc
   --data-root C:\rppg_data\UBFC --model proposnet`, CPU) → save a checkpoint →
   add a neural HR readout to `live_demo.py` next to ProPOS. (Only 4 subjects now →
   weak split; grab a few more subjects if a stronger net is wanted — C: has ~10 GB.)
2. **Multi-ROI + skin masking** for extra live SNR under motion/distance.
3. **BP datasets** (academic) → real BP model; **distance/motion** roadmap phases.

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
