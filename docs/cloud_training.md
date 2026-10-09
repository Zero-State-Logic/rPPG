# Cloud Training Playbook — Deep HR on Colab + Kaggle

The local machine (Intel UHD, no CUDA) built and validated the classical core. Deep
models — the path to >90% — train on a CUDA GPU, i.e. the cloud. This guide runs the
**same code and config on both Google Colab and Kaggle**, so you train on one and
switch to the other when a daily GPU limit is hit.

## 0. Accounts (all free; Claude cannot create these for you)
- **GitHub** — https://github.com/signup  (code hosting; recommended — see §1)
- **Google / Colab** — https://accounts.google.com/signup → https://colab.research.google.com
- **Kaggle** — https://www.kaggle.com → Register (verify phone to unlock the GPU)

## 1. Why GitHub (recommended)
Research-grade projects use GitHub: reproducibility + version history, a one-line clone
that works identically on Colab **and** Kaggle, and issues/releases for collaboration.
After signup, create an empty repo `rppg` and send the URL; the code gets pushed there
and every cloud notebook starts by cloning it.

Set `REPO_URL` below to your repo once it exists.

---

## 2. Google Colab — deep HR training
New notebook → **Runtime → Change runtime type → T4 GPU**. Then:

**Cell 1 — setup**
```python
REPO_URL = "https://github.com/<you>/rppg.git"   # <-- your repo
!git clone $REPO_URL && cd rppg && pip install -q -e .
%cd /content/rppg
import torch; print("CUDA:", torch.cuda.is_available())   # expect True
```

**Cell 2 — download a UBFC subject (cloud IP is not quota-blocked)**
```python
!mkdir -p data/ubfc/subject1
!gdown --folder "https://drive.google.com/drive/folders/1q6U7jKRBUtr6N2rVs1vwZ2wuUT2VDoRy" -O data/ubfc/subject1
```

**Cell 3 — real-video ProPOS proof (our validated code)**
```python
!python scripts/eval_ubfc.py --video data/ubfc/subject1/vid.avi \
    --gt data/ubfc/subject1/ground_truth.txt --max-frames 1800
```

**Cell 4 — deep HR to >90% via rPPG-Toolbox**
```python
!git clone https://github.com/ubicomplab/rPPG-Toolbox.git
%cd rPPG-Toolbox
# install deps; mamba-ssm/causal-conv1d are only needed for PhysMamba — safe to skip
!pip install -q -r requirements.txt || echo "skipping optional build failures"
# download UBFC-rPPG + PURE per the toolbox README (gdown works here), then:
!python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
```
TS-CAN / PhysNet reach HR MAE ~1–2 bpm on UBFC — comfortably >90% of windows within ±5 bpm.

---

## 3. Kaggle — switch here when Colab's limit hits
New Notebook → **Settings → Accelerator → GPU T4 x2**. The same four cells run (Kaggle has
git + gdown). To avoid re-downloading each session, save the downloaded data as a **private
Kaggle Dataset** and mount it at `/kaggle/input/...`.

## 4. Switch strategy (maximize free GPU)
- Train on Colab until the daily GPU quota ends.
- Save the checkpoint to Google Drive (`drive.mount`) or a Kaggle Dataset.
- Resume on Kaggle from that checkpoint — identical code (same GitHub clone) and config.
- Alternate as quotas refresh. Keep results/metrics in the repo; keep large checkpoints on Drive/Kaggle.

## 5. What to report back
After Cell 3: the POS/CHROM/PBV/ProPOS MAE on real UBFC video.
After Cell 4: the deep model's MAE / RMSE / %-within-±5 bpm — our Phase-2 gate.
Paste any errors here and I'll fix the notebook.
