# rPPG Vital-Signs Project — Master Plan

**Goal:** A working, trained, end-to-end system that takes **face video** as input and outputs **heart rate (HR)** and **blood pressure (BP: systolic/diastolic)**, built around the *ProPOS* paper (Rao, Fang, Zhao, Bai — *Medical Engineering & Physics* 138, 2025) and the broader rPPG literature.

**Status:** Awaiting your approval to start. **Owner/orchestrator:** Claude (Opus 4.8) as "Central Head".
**Date:** 2026-10-09.

---

## 0. TL;DR — what we're really building and why

- The ProPOS paper is a **classical signal-processing algorithm for heart rate only**, at long distance (3–30 m). It is **not a trained model** and **never estimates blood pressure.** We will **reproduce it faithfully** as a strong classical baseline, then go **well beyond it** with trained deep models and a separate BP pipeline.
- **Heart rate is a solved-enough problem**: deep rPPG models reach MAE ≈ 1–2 bpm on standard datasets → our **>90% target is realistic**.
- **Blood pressure from video is an open research problem**: honest state-of-the-art is **~6–12 mmHg error**, and a 2024–2025 benchmark found **no model reliably meets clinical AAMI/ISO standards.** We will build a **rigorous, honestly-evaluated** BP model competitive with SOTA — **not** a fake "90%". (Your chosen target: *HR strong + BP research-grade*.)

---

## 1. Locked decisions (from your answers)

| Topic | Decision |
|---|---|
| **Compute** | Prototype on **free cloud GPUs** (Colab/Kaggle); scale to paid cloud (Lambda/RunPod/Vast) once the pipeline works. Local machine (Intel UHD, no CUDA) is dev/classical only. |
| **Data** | **Public datasets now** (UBFC-rPPG, PURE for HR; apply for Vital Videos + BP4D+ for BP) **and chase LD-rPPG + BP agreements in parallel** (off the critical path). |
| **Accuracy aim** | **HR clinically strong** (MAE < 3 bpm, >90% of windows within ±5 bpm). **BP best-effort research-grade**, benchmarked honestly against SOTA (~8–12 mmHg), with BHS/AAMI grading and strict subject-independent splits. |

---

## 2. Honest success targets (how we define "accuracy")

### Heart rate (primary, must-hit)
- **MAE < 3 bpm**, **RMSE < 5 bpm**, **Pearson r > 0.95** on held-out *subjects*.
- **"Accuracy" = % of windows within ±5 bpm of ground truth → target > 90%** (the standard rPPG success metric).
- Prove generalization with **cross-dataset** eval (train UBFC → test PURE).

### Blood pressure (research, best-effort)
- Report **MAE ± SD** for SBP and DBP, **Bland–Altman**, **BHS grade (A/B/C)**, **AAMI pass/fail**, all under **subject-independent** splits.
- **Stretch target:** SBP MAE ≤ ~10–12 mmHg, DBP ≤ ~8–9 mmHg (SOTA-competitive).
- **Anti-cheating rules (mandatory):** subject-wise splits only; always compare against a **constant-predictor baseline** and report label standard deviation; no per-subject calibration leakage into test; pre-registered eval protocol. This is how we avoid the over-optimistic "90% BP" claims that plague the field.

---

## 3. System architecture (end-to-end pipeline)

```
face video ─▶ [1] Ingest/decode ─▶ [2] Face detect+track, skin ROI ─▶ [3] Raw RGB traces
          ─▶ [4] rPPG extraction ──┬─ classical: POS / CHROM / PBV / ProPOS(paper)
                                    └─ deep: PhysNet / TS-CAN / EfficientPhys / PhysFormer
          ─▶ [5] BVP pulse waveform ─▶ [6a] HR: PSD peak + post-filter ─▶ HR (bpm)
                                     └▶ [6b] BP: waveform features + deep model + transfer ─▶ SBP/DBP
          ─▶ [7] Unified evaluation (HR + BP metrics, Bland-Altman, BHS/AAMI)
          ─▶ [8] predict(video) → {HR, BP}  API + optional live demo
```

**Design principles:** modular (many small files), reuse battle-tested code (rPPG-Toolbox, neurokit2, MediaPipe) rather than hand-rolling, config-driven experiments, reproducible, immutable data transforms, tests for every signal-processing unit.

---

## 4. Implementing the paper — ProPOS (concrete spec)

We reproduce Algorithm 1 exactly (CPU, numpy/scipy), as a module `src/rppg/methods/classical/propos.py`:

1. **Raw trace** `S_trace[c,t]` = spatial mean of ROI pixels per channel per frame (Eq. 1).
2. **Normalize** `S_n = (S_trace − μ)/σ` (Eq. 2).
3. **Rotated bases**: `R = R1·R2·R3·R4` (Eqs. 3–7), parameterized by `θ` and `γ`. Per the paper's implementation details: **γ ∈ {5,10,15,20,25}°** and **θ ∈ {36,72,…,360}°** → **50 candidate bases** (precomputed into one matrix for speed).
4. **Projection**: `P̃ = P·R` with `P = [[0,1,−1],[−2,1,1]]` (Eqs. 8–9); `S̃ = P̃·S_n` (Eq. 10); candidate pulse `p̃ = S̃₁ + α·S̃₂`, `α = σ(S̃₁)/σ(S̃₂)` (Eq. 11).
5. **Signal selection**: take POS output `p̃(0,0)` → FFT peak → approximate HR `f_r`; compute **SNR** (Eq. 13) over 0.7–5 Hz incl. first harmonic, gate window `f_w = 0.4 Hz`; **pick max-SNR candidate** `p*`.
6. **Signal fitting**: fit `p(t)=a₁·sin(ωt+b₁)+a₂·sin(2ωt+b₂)` via nonlinear least squares (`scipy.optimize.curve_fit`), paper's start vector (Eqs. 15–16).
7. **Windows**: 120 frames (4 s) @ 30 fps, step 30; FFT 2048, fs 30.

**Validation:** unit tests on synthetic sinusoids; sanity that SNR-selection ≥ POS on sample UBFC clips; reproduce the paper's qualitative behavior (ProPOS > POS/CHROM/PBV).

> Note: the paper has a minor text inconsistency (θ/γ ranges in Step 3/4 vs. the sampling in §4.1). We follow the **§4.1 implementation details** (the 50-vector sampling), which is what produced the reported results.

---

## 5. Data plan

| Dataset | Signal | Access | Use |
|---|---|---|---|
| **UBFC-rPPG** | video + PPG (HR) | public download | HR train/test (primary) |
| **PURE** | video + PPG (HR) | request (widely available) | HR cross-dataset test |
| **Vital Videos (Europe)** | video + PPG + **BP** (~900 subj) | data agreement | BP train/test (primary) |
| **BP4D+** | video + continuous **BP** | application (Binghamton) | BP train/test |
| **MMSE-HR / MSPM** | video + HR/**BP** | application | BP aux / eval |
| **PulseDB / UCI Cuffless BP / MIMIC** | **contact** PPG + BP | public | **BP pretraining** (transfer) |
| **LD-rPPG** (paper's) | long-distance video + PPG | email authors | stretch: long-distance HR |

**Unblocking BP immediately:** while video+BP agreements are pending, we **pretrain BP on contact PPG→BP** (PulseDB/UCI) and **transfer to rPPG** — exactly the WACV-2025 "PPG→rPPG conversion" strategy. BP work starts on day one, no waiting.

**Parallel actions:** I'll draft data-request emails for Vital Videos, BP4D+, and LD-rPPG so access is pending while we build.

---

## 6. How we reach the targets

**HR → >90%:**
- Deep models (TS-CAN / PhysFormer) on close-range datasets hit MAE ~1–2 bpm out of the box.
- Solid preprocessing (MediaPipe skin ROI, de-trending, 0.7–4 Hz band-pass), motion handling, temporal post-processing, and classical+deep **ensemble** for robustness.
- Report the ±5 bpm window accuracy; iterate with ablations until >90% on held-out subjects.

**BP → SOTA-competitive, honest:**
- Features: PPG morphology (systolic/diastolic peaks, pulse area, PPG 1st/2nd derivatives / APG), proxy pulse-transit features, HR/HRV context.
- Models: XGBoost baseline → 1D-CNN/LSTM/Transformer on the rPPG waveform → **spatiotemporal facial-map** BP model (strong recent approach).
- **Transfer learning** from contact-PPG→BP; optional **per-subject calibration** variant (reported separately and honestly — calibrated vs. uncalibrated).
- Rigorous eval (Bland-Altman, BHS/AAMI, constant-predictor baseline).

---

## 7. Multi-agent organization — the "Central Head" hierarchy

**One honest clarification:** the subagent system lets me run **many Opus / Sonnet / Haiku agents in parallel**, choosing the model per task — but it does **not** expose version labels like "Opus 4.6 / 4.7 / 5.5" as separately selectable models. The orchestration behavior you want is fully deliverable; the version labels aren't real knobs. For large deterministic fan-out (benchmark/hyperparameter sweeps) I'll use the **Workflow engine** as the formal hierarchical orchestrator.

```
                      ┌─────────────────────────────┐
                      │  CENTRAL HEAD = me (Opus 4.8)│  ← owns plan, task board,
                      │  single writer to main repo  │    integration, final review
                      └──────────────┬──────────────┘
          ┌───────────────┬──────────┼───────────┬────────────────┐
     Research Lead   Architecture   HR Lead     BP Lead        QA / Review Lead
       (Opus)          Lead(Opus)   (Opus)      (Opus)            (Opus)
          │               │            │           │                 │
     workers (Sonnet): coding · data-prep · experiment runners · doc writers
          │
     light (Haiku): formatting · log parsing · quick lookups
```

- **Reporting protocol:** every agent returns a structured report (task · result · artifacts · issues · next step). I log them to `docs/STATUS.md` + the live todo board. Independent tasks are launched **in parallel** in one message.
- **Specialist reviewers** (real agents here): `code-reviewer`, `python-reviewer`, `security-reviewer`, `mle-reviewer`, `tdd-guide`, `pytorch-build-resolver`, `performance-optimizer`. Used per your global code-review + TDD rules.
- **Single-writer rule:** to avoid merge chaos, subagents produce code/plans and I integrate — matches the `multi-execute` pattern.

---

## 8. Tools & "skills" we'll use (vetted, not random downloads)

The highest-value "skills" for this project are **proven code libraries/repos** plus a few in-environment Claude skills. I will **not** blindly install arbitrary GitHub "skills" (supply-chain/injection risk) — each is vetted (maintenance, license, stars) and version-pinned, and I'll show you the list before anything heavy is installed.

**Core libraries / repos:**
- **rPPG-Toolbox** (UbiComp/McJackTang) — POS/CHROM + DeepPhys/PhysNet/TS-CAN/EfficientPhys/PhysFormer + dataset loaders → **the backbone for deep HR**.
- **pyVHR** — alternative rPPG pipeline/methods (cross-check).
- **neurokit2**, **heartpy** — PPG/BVP processing, peak detection, HRV, features.
- **MediaPipe** — face mesh / skin ROI (better than the paper's Viola-Jones+KLT).
- **PyTorch**, **scikit-learn**, **XGBoost** — deep + classical BP models.
- **OpenCV**, **PyAV/ffmpeg**, **numpy/scipy/pandas** — IO + signal math.

**In-environment Claude skills/agents:** `mle-reviewer`, `python-review`, `pytorch-build-resolver`, `test-coverage`, `dataviz`, `pdf`, `multi-plan/multi-execute`, `prp-*`, plus the review agents above.

---

## 9. Phased roadmap, milestones & deliverables

| Phase | What | Key deliverable / milestone |
|---|---|---|
| **0 — Foundation** (day 0–1) | Repo scaffold, Python env, pinned deps, STATUS board, draft data-request emails | Runnable skeleton + `docs/datasets.md` + emails ready |
| **1 — Classical HR + ProPOS** (day 1–4) | Implement POS/CHROM/PBV + **ProPOS (the paper)**; unit tests; run on UBFC/PURE | ProPOS reproduced; classical HR metrics table |
| **2 — Deep HR** (day 3–10, cloud GPU) | Integrate rPPG-Toolbox; preprocess UBFC+PURE; train/eval TS-CAN/PhysNet/PhysFormer; cross-dataset | **HR MAE < 3 bpm, >90% within ±5 bpm** on held-out subjects |
| **3 — BP track** (day 7–20) | PPG→BP pretraining (PulseDB/UCI); rPPG→BP transfer; spatiotemporal-map model; subject-independent eval on Vital Videos/BP4D+/MMSE-HR as access arrives | Honest BP benchmark + Bland-Altman + BHS/AAMI |
| **4 — Integration & demo** (day 18–25) | `predict(video)→{HR,BP}` API; evaluation report; optional live webcam demo; model cards | End-to-end working model + report |
| **5 — Iterate** (ongoing) | Multi-agent Workflow sweeps (architectures/hparams), ablations, push accuracy; then decide deployment hardware | Best models + reproducible results |

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| BP may not reach clinical grade | Research-grade target + honest metrics (already agreed) |
| Data-agreement delays | PPG→BP **transfer** unblocks BP now; LD-rPPG off critical path |
| No local CUDA GPU | Cloud training; cloud-ready scripts; CPU for classical + dev |
| Data leakage / over-optimistic BP | Mandatory subject-wise splits + constant-predictor baseline + pre-registered protocol |
| MATLAB→Python ProPOS fidelity | Unit tests, synthetic-signal checks, compare vs. POS |
| Long-distance (paper) vs close-range (datasets) mismatch | Target close-range for accuracy; long-distance as stretch with LD-rPPG |

---

## 11. Proposed repository structure

```
E:\RPPG\
  README.md  PLAN.md  requirements.txt
  configs/                      # experiment configs (yaml)
  src/rppg/
    io/            # video + dataset loaders
    preprocess/    # face detect/track, skin ROI, normalization
    methods/classical/   # pos, chrom, pbv, propos  ← the paper
    methods/deep/        # wrappers over rPPG-Toolbox
    hr/            # PSD/FFT HR + metrics
    bp/            # features, deep models, transfer, calibration
    eval/          # metrics, bland_altman, bhs_aami, splits
    pipeline/      # end-to-end predict()
  scripts/         # download_data, run_benchmark, train_colab
  notebooks/       # Colab/Kaggle training
  tests/           # unit tests (TDD)
  docs/            # datasets.md, agents.md, STATUS.md, reports/
  third_party/     # rPPG-Toolbox clone
  data/            # (gitignored) datasets
  artifacts/       # trained models, result tables, figures
```

---

## 12. What I do immediately on approval (Phase 0 + start of Phase 1)

1. Spawn parallel agents (Central Head hierarchy): **Research Lead** (deep-dive rPPG-Toolbox + BP SOTA + exact dataset access steps), **Architecture Lead** (finalize module interfaces), **Data Lead** (download UBFC, draft emails).
2. Scaffold the repo + Python venv + pinned `requirements.txt`; set up `docs/STATUS.md` task board.
3. Begin **ProPOS** implementation (TDD) + classical POS/CHROM baseline.
4. Report back a Phase-0 status with the first metrics and the data-access emails ready to send.

> **Nothing is installed or downloaded and no code is written until you approve this plan.**
