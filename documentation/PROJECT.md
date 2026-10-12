# rPPG Project — Full Technical Documentation

> Living document. The chronological record of *what happened each work session*
> lives in [DEVLOG.md](DEVLOG.md); the routine for maintaining both is in the
> repo-root `CLAUDE.md`. This file is the end-to-end **technical + status**
> picture: what the system is, how it is built, what works, what is experimental,
> what is blocked, and what remains.

Last updated: 2026-10-12.

---

## 1. Goal & scope

Build a **research-grade, end-to-end remote photoplethysmography (rPPG)** system
that reads a face from an ordinary camera and estimates **heart rate (HR)** and
**blood pressure (BP)**. Targets, in priority order:

1. **HR accuracy ≤ 3 bpm MAE** in good conditions (still, lit, near), validated on
   real data, with a live multi-person webcam demo.
2. **BP** — research track; honest *experimental* output now, validated later once
   BP datasets are obtained.
3. **Robustness ladders** (roadmap): moving subject → multiple subjects →
   near/far distance.

Founded on the **ProPOS** method (Rao et al., *Med. Eng. & Phys.* 138, 2025) and
intended to go **beyond** it (a novel differentiable ProPOSNet, multi-method
fusion, a live system) toward our own paper.

## 2. Current status (truth table)

| Capability | State | Notes |
|---|---|---|
| Live webcam HR, 1 person | **Working** | Classical ProPOS + POS/CHROM fusion + HR tracker. |
| Live webcam HR, multiple people | **Working** | YuNet multi-face + per-person tracker. |
| Near/far, moving | **Working, degrades honestly** | Quality label drops; HR holds through motion. |
| HR accuracy | **Real UBFC: 1.41 bpm MAE** (95% within±5, 93% within±3) | Measured on 4 subjects; synthetic 0.20 bpm. ≤3 bpm target **met**. |
| Harmonic/jump robustness | **Fixed** | HRTracker rejects 2× harmonic + spikes, holds through motion. |
| Blood pressure | **Experimental only** | Heuristic + single-point calibration. NOT validated. |
| Neural model (ProPOSNet) | **Real UBFC: 1.17 bpm MAE** (Pearson 0.95), **live** | Trained on 8 real subjects; runs live via `--neural`; beats classical 1.41. |
| Classical/deep training pipeline | **Working** | Subject-independent + cross-dataset; CPU local, GPU on Colab. |
| Real UBFC data | **4 subjects downloaded** | Via Kaggle per-file to `C:\rppg_data\UBFC`; more available (C: ~10 GB free). |
| BP datasets | **Not obtained** | Academic agreements (weeks); request emails drafted. |

## 3. Repository map

```
src/rppg/
  methods/classical/   pos.py chrom.py pbv.py propos.py   # signal-processing rPPG
  models/              proposnet.py physnet.py tscan.py losses.py  # deep models
  hr/                  metrics.py  refine.py               # HR estimation + accuracy
  data/                datasets.py splits.py               # UBFC/PURE loaders, splits
  train/               trainer.py                          # training loop
  eval/                deep_eval.py                        # HR evaluation
  live/                detect.py estimate.py overlay.py    # live webcam pipeline
scripts/
  live_demo.py                 # the webcam demo entry point
  train_deep.py                # train ProPOSNet/PhysNet/TS-CAN
  download_ubfc.py             # gdown UBFC (Drive-quota blocked)
  download_ubfc_kaggle.py      # Kaggle UBFC mirror (bypasses Drive)
  make_synthetic_ubfc.py       # synthetic UBFC for pipeline validation
  eval_ubfc.py                 # classical eval on a real subject
docs/        DEMO_VSCODE.md (run guide), cloud_training.md, data_requests.md, STATUS.md
documentation/  PROJECT.md (this), DEVLOG.md (session log)
models/      (gitignored) YuNet + MediaPipe face models, auto-downloaded
```

## 4. Methods

**Classical (signal processing, no training — work on any real face):**
- **POS** (plane-orthogonal-to-skin, Wang 2017) — baseline, overlap-add.
- **CHROM** (de Haan 2013) — motion-robust chrominance baseline.
- **PBV** — blood-volume-pulse signature.
- **ProPOS** (Rao 2025) — **our core method**: rotates the POS projection plane into
  a bank of 50 candidate bases, projects, selects the best by SNR vs a POS
  reference HR, refines with a 2-harmonic sinusoidal fit. `propos_pulse(rgb, fps)`.

**Deep models:**
- **ProPOSNet** (ours, novel) — a *differentiable* ProPOS: a learnable projection
  bank (nn.Parameter) initialised from the classical bank; 22,020 params.
- **PhysNet** (3D-CNN), **TS-CAN** (2D two-branch) — baselines for the paper table.
- Losses: NegPearson, FrequencyMSE.

## 5. Live demo pipeline (`src/rppg/live/` + `scripts/live_demo.py`)

```
webcam frame (OpenCV, auto-WB off, optional exposure lock)
  -> YuNet multi-face detection (detect.py)
  -> CentroidTracker: stable per-person id across frames
  -> per face: mid-face ROI (cheeks/upper-lip) mean RGB -> rolling 10 s buffer
  -> every 0.4 s per person (estimate.py):
       resample buffer to uniform fps
       POS + CHROM + ProPOS  ->  band-pass each (refine.bandpass)
       sub-bin spectral peak each (refine.hr_peak_interp)  -> (hr, snr)
       SNR fusion across methods (refine.fuse_estimates)   -> raw hr
       HRTracker: 2nd-harmonic fold + slew-limit + confirm + motion-hold -> hr
       experimental BP from hr + pulse morphology (+ optional calibration)
  -> overlay.py: per-person box, ROI, HR, BP(exp/cal), SNR, pulse sparkline
```

## 6. Accuracy engineering (why ≤3 bpm is real)

The naive periodogram quantises HR to a coarse grid and has no defence against
noise, harmonics or motion. Layers added:
1. **Band-pass** 0.7–4 Hz (Butterworth, zero-phase) — removes out-of-band noise.
2. **Sub-bin peak** — Hann window + 8× zero-pad + parabolic interpolation →
   ~0.3 bpm resolution.
3. **Multi-method SNR fusion** — POS/CHROM/ProPOS; trust the strongest-pulse method
   each instant.
4. **HRTracker** — folds a reading near 2× the current rate back to the fundamental
   (kills the 63→128 doubling), rate-limits change (`_SLEW`), only follows a big
   jump after `_CONFIRM` confident readings agree, and holds the last value when
   SNR is low (motion). Constants in `estimate.py`.

**Measured:** synthetic sweep → MAE 0.20 bpm. **Real UBFC (4 subjects, 44 windows,
`scripts/eval_ubfc_fused.py`): MAE 1.41 bpm, RMSE 4.12, within±5 95%, within±3
93%** — the ≤3 bpm target **met on real faces** (per subject 0.44 / 0.54 / 1.91 /
2.77 bpm). Tracker unit-tested against 128-harmonic spikes, lone spikes, real
rises and motion noise.

## 7. Blood pressure (honest status)

Live BP is an **experimental heuristic**: it maps HR + pulse-harmonic morphology to
a plausible, stable mmHg value that responds to physiology. It is **not trained on
BP data and has no validated accuracy.** A **single-point calibration** (`c` key,
`--ref-sbp/--ref-dbp`) anchors it to one cuff reading and then tracks relative
changes (tag `exp` → `cal`). **Must be presented as experimental / research-stage.**
Real BP needs the BP video datasets (Vital Videos / BP4D+ / MSPM) — academic
agreements, weeks out; best published camera-BP is ~±8–12 mmHg and not clinical.

## 8. Data situation

- **UBFC-rPPG** (real HR ground truth): official Google-Drive share is
  **quota-blocked** ("Too many users…"). Mirror on **Kaggle** (`ashfakyeafi/ubfc-2`,
  ~28 GB) — downloader built (`download_ubfc_kaggle.py`, supports `KGAT_` tokens).
- **Synthetic UBFC** (`make_synthetic_ubfc.py`) — validates the pipeline when real
  data is unavailable; **not a real result**.
- **BP datasets** — not obtained; request emails drafted in `docs/data_requests.md`.

## 9. Validation protocol

- **Subject-independent** splits (no subject in two splits) and **cross-dataset**
  ("uneven data") — `splits.py`.
- Metrics: MAE (bpm), RMSE, within-±5 bpm %, Pearson r, SNR — `hr/metrics.py`,
  `eval/deep_eval.py`. 26 unit tests pass.

## 10. Constraints (hard realities)

- **No local GPU** (Intel UHD) → deep training on Colab/Kaggle GPU, or slow CPU.
- **Colab** free runtime **times out on inactivity** and wipes `/content` → fragile
  for long jobs.
- **Local disk tight**: E: ~6.7 GB free, C: ~19 GB — the full 28 GB UBFC will not
  fit; must fetch a **subset** of subjects.
- **Credentials**: assistant must not type the user's secret token; the user sets
  it, tooling reads it from `~/.kaggle/access_token` (KGAT) or env.
- **Live rPPG physics**: ≤3 bpm is achievable only in good conditions; motion /
  distance degrade accuracy — this is the research frontier, not a bug.

## 11. Roadmap / remaining work

1. **Real UBFC**: download a few subjects via Kaggle (token now set) → measure
   **real HR MAE** (the publishable number).
2. **Train ProPOSNet on real faces** → checkpoint → wire a second neural HR readout
   into the live demo.
3. **Multi-ROI + skin masking** (forehead + both cheeks) for extra live SNR.
4. **Distance** and **motion** robustness ladders (roadmap phases).
5. **BP datasets** → real BP model.
6. **Paper**: ProPOSNet + fusion + live system; benchmark table vs PhysNet/TS-CAN.

## 12. Known gaps & risks

- Neural + classical both validated on real UBFC (1.17 / 1.41 bpm) but on only 8
  subjects — more subjects would strengthen the numbers and the held-out test.
- BP unvalidated — present as experimental only.
- Single mid-face ROI can be noisy; multi-ROI not yet implemented.
- Motion / distance robustness still the research frontier (quality degrades honestly).

## 13. How to run

See [../docs/DEMO_VSCODE.md](../docs/DEMO_VSCODE.md). Quick start (project venv):

```
.venv\Scripts\python.exe scripts\live_demo.py
```
