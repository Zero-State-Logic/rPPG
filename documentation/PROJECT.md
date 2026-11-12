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
| Live webcam HR, 1 person | **Working (rebuilt 2026-11-12)** | Whole face box → POS → 42–180 bpm spectrum → causal **SpectralTracker** (`src/rppg/hr/track.py`); confidence gate unchanged. Neural main via `--neural` + arbiter as before. |
| HR accuracy on an external dataset (MCD-rPPG, frontal webcam) | **Measured** | Before: 17.45 bpm per 10 s window, live number 19.19, 43 % of shown numbers off >10 bpm. After (dev, subject-held-out): 4.33 per window; live 4.67, shown 62 % of the time at 2.47, 3.7 % off >10. **Held-out 40 new subjects: 17.85 → 3.85 bpm per window; live shown numbers 14.89 → 2.00 bpm at 77 % coverage, off >10 bpm 43 % → 3.3 %** (DEVLOG 2026-11-12). |
| Sudden-light / optical-noise robustness | **Reworked** | Hampel spike filter **removed by default** (it turned 10 Hz flicker into in-band junk; `PersonSignal(spike_filter=True)` keeps it); background division now opt-in (`--illum-norm`). The tracker rejects brief outliers instead. |
| Live webcam HR, multiple people | **Working** | YuNet multi-face + per-person tracker. |
| Near/far, moving | **Working, degrades honestly** | Quality label drops; HR holds through motion. |
| HR accuracy | **Neural 1.17 bpm**; classical 8-subj **2.78 bpm, 93% within±5** (median 1.6; subj17 outlier 10.3) | 6/8 classical subjects <2 bpm; neural (held-out) is tighter. |
| Harmonic/jump robustness | **Fixed (tracker replaced 2026-11-12)** | SpectralTracker (online Viterbi, σ 3 bpm/2 s) ignores brief outlier windows and follows sustained changes; flat input → "not ready", never a fake rate. Old HRTracker kept for `eval_ubfc_fused.py`. |
| Blood pressure | **Trained model** (contact PPG): SBP 15.1 MAE (BHS-D), DBP 6.5 (BHS-C) | Live via `--bp-model`; rough on rPPG (domain gap); still experimental. Heuristic fallback. |
| Neural model (ProPOSNet) | **Real UBFC: 1.17 bpm MAE** (Pearson 0.95), **live** | Trained on 8 real subjects; runs live via `--neural`; beats classical 1.41. `best.pt` saw these same 8 subjects → retrain subject-disjoint before using it in the reliability study. |
| Classical/deep training pipeline | **Working** | Subject-independent + cross-dataset; CPU local, GPU on Colab. |
| Real UBFC data | **8 subjects (10–17) local** | Raw videos ~14.4 GB in `C:\rppg_data\UBFC`; C: 3.35 GB / E: 0.24 GB free (2026-10-29). Expansion needs freeing the raw videos after cache v2 + crops. |
| Offline research cache | **Working** | Frame cache + window table (`src/rppg/eval/`) at `C:\rppg_data\cache`: 237 × 10 s windows; per-subject fused MAE 0.35–1.72 bpm except subject15 26.85 (estimator locks at ~48–58 vs true ~110–121). |
| Reliability / abstention (**flagship**) | **Research in progress** | Legacy live gate (−1 dB) accepts **35.0%** of no-pulse windows; the fake rates are 97.9% plausible, median 56.3 bpm. Flagship: *Certified Abstention for Camera HR* (DEVLOG 2026-10-29). |
| BP datasets | **Not obtained** | Academic agreements (weeks); request emails drafted. |

## 3. Repository map

```
src/rppg/
  methods/classical/   pos.py chrom.py pbv.py propos.py   # signal-processing rPPG
  models/              proposnet.py physnet.py tscan.py losses.py  # deep models
  hr/                  metrics.py  refine.py  track.py     # HR estimation, accuracy, tracking
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
  -> per face: whole face-box mean RGB (--roi face; face-skin / legacy patches
     selectable; background division only with --illum-norm) -> rolling 10 s buffer
  -> every 0.4 s per person (estimate.py, since 2026-11-12):
       resample buffer to uniform fps (no Hampel unless spike_filter=True)
       POS pulse -> band-passed, Hann, zero-padded spectrum on a 42-180 bpm grid
       SpectralTracker (hr/track.py): online Viterbi, evidence tempered by dt/2 s,
         random walk sigma 3 bpm per 2 s, sub-bin refine -> hr
       SNR = power within +/-9 bpm of hr vs rest of band (feeds the gate)
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

**2026-11-12 rework (external MCD-rPPG validation).** Layers 3–4 above were replaced:
multi-method fusion and the slew-limited `HRTracker` lost to **POS alone + a causal
spectral Viterbi tracker** (`rppg.hr.track.SpectralTracker`), the band became 42–180
bpm, the Hampel pre-filter and background division were switched off by default and
the whole face box replaced the forehead/cheek patches. Bench and evidence:
`D:\rppg_eval\code\bench\`, report `D:\rppg_eval\report2\RPPG_Improvement_Report.docx`.
`HRTracker` / `DisplayStabilizer` remain for `scripts/eval_ubfc_fused.py` and the
optional `PersonSignal(stabilize=True)`.

**Measured (before the rework):** synthetic sweep → MAE 0.20 bpm. **Real UBFC (4 subjects, 44 windows,
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
  `eval/deep_eval.py`. **52 unit tests pass** (2026-10-29).
- Reliability study (flagship): subjects are the exchangeable unit (leave-subject-out
  now; 200+ random subject splits once ≥40 subjects); risk-coverage curves, AURC,
  selective MAE at fixed coverage, per-subject yield and risk, null false-accept rate
  with Clopper-Pearson CIs. The cue set, α0, α, δ, τ and the bin rule are
  pre-registered in a dated file before any new data.

## 10. Constraints (hard realities)

- **No local GPU** (Intel UHD) → deep training on Colab/Kaggle GPU, or slow CPU.
- **Colab** free runtime **times out on inactivity** and wipes `/content` → fragile
  for long jobs.
- **Local disk very tight (2026-10-29): E: 0.24 GB, C: 3.35 GB free.** The 8 raw UBFC
  videos use ~14.4 GB on C:. Keep data artifacts on C:; free the raw videos only after
  cache v2 + face crops are extracted (user decision), then stream further subjects one
  at a time (download → cache → delete).
- **Credentials**: assistant must not type the user's secret token; the user sets
  it, tooling reads it from `~/.kaggle/access_token` (KGAT) or env.
- **Live rPPG physics**: ≤3 bpm is achievable only in good conditions; motion /
  distance degrade accuracy — this is the research frontier, not a bug.

## 11. Roadmap / remaining work

Done since the original roadmap: real UBFC HR measured (8 subjects); ProPOSNet trained
on real faces and running live; multi-ROI (forehead + cheeks) skin-pooled sampling;
display stabilizer and a recalibrated confidence gate.

**Flagship — Certified Abstention for Camera HR (paper-first, DEVLOG 2026-10-29):**
1. Reliability package `src/rppg/reliability/`: cues, selection-aware
   frequency-conditional null calibration, physics-consistency score, subject-level
   certificate (Learn-then-Test / conformal risk control), metrics and baselines —
   with tests, including simulation checks that the guarantees actually hold.
2. Offline preprocessing identical to the live `PersonSignal` path (resample, Hampel,
   background division), so calibration and deployment match.
3. Cache v2 (separate corners, side and clothing patches, per-ROI sub-patches) + face
   crops, built while the raw videos still exist.
4. Dated pre-registration, then Phase 0 experiments on the 8 subjects (LOSO).
5. Null clips: ~50–100 no-pulse recordings (printed photo / phone screen / mannequin;
   static, flicker, hand motion; several lightings) — a team action, no human subjects.
6. Subject expansion to ≥30–40 UBFC subjects (stream-and-discard), then LGI-PPGI /
   UBFC-Phys / CHILL subsets for distribution shift.
7. Subject-disjoint ProPOSNet retraining; wrap it with the same calibration.
8. Live certified gate with reason codes + a legacy-vs-certified A/B demo; the
   Breathe-to-Verify paced-breathing demo as an extra.
9. Papers: EMBC / IEEE SPL from Phase 0 + negative controls; IEEE TIM / JBHI after the
   expansion and cross-dataset tests.

Other tracks (lower priority): BP datasets → real BP model; distance / motion ladders.

## 12. Known gaps & risks

- **The legacy live gate is biased by selection:** best-of-candidates SNR inflates
  no-pulse windows by ~2.5 dB, so the −1 dB gate accepts 35% of no-pulse windows.
  ProPOS's 50-basis search is the largest contributor (30.8% on its own vs POS 4.2%).
- **Small n:** 8 subjects, 18 failing windows, 15 of them from subject15 — the pilot
  numbers are optimistic until confirmed on pre-registered new subjects, and no useful
  certificate is possible below ~30 subjects.
- **Leakage:** `best.pt` was trained on the same 8 subjects.
- **Neural path lags the classical path on new people** (MCD held-out: UBFC
  `best.pt` 32.9 bpm, MCD-retrained 20.6 vs classical 3.85). `--neural` is not
  recommended until it is retrained at GPU scale.
- **Residual failures:** after the 2026-11-12 rework the MCD held-out RMSE is still
  ~13 bpm — a few recordings fail (fast after-exercise HR changes, weak signal); the
  confidence gate hides most of them (3.3 % of shown numbers off >10 bpm).
- **Protocol note:** published MCD numbers use one 20 s clip per video; our main
  metric is 10 s windows every 2 s, so compare like-for-like (both are reported).
- BP unvalidated — present as experimental only.
- Motion / distance robustness is still the research frontier (quality degrades
  honestly); UBFC subjects sit still, so motion methods cannot be validated on UBFC.

## 13. How to run

See [../docs/DEMO_VSCODE.md](../docs/DEMO_VSCODE.md). Quick start (project venv):

```
.venv\Scripts\python.exe scripts\live_demo.py
```
