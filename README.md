# RPPG - Remote Vital Signs (HR + BP) from Face Video

Research-grade remote photoplethysmography: estimate heart rate and blood pressure from
facial video. Extends the ProPOS long-distance method with trained deep models, a rigorous
blood-pressure pipeline, and robustness to camera distance and subject motion.

- Plan: see PLAN.md
- Live status / task board: docs/STATUS.md

## Dev setup (Windows, Python 3.11)
    py -3.11 -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements-classical.txt

Deep-model training runs on cloud GPUs (Colab/Kaggle); this machine is dev + classical only.
