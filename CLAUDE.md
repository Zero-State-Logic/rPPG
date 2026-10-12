# CLAUDE.md — working rules for this repo

rPPG: estimate **heart rate + blood pressure** from a face camera. Research-grade,
ProPOS-based, with a live multi-person webcam demo. Full technical picture:
[documentation/PROJECT.md](documentation/PROJECT.md). Session history:
[documentation/DEVLOG.md](documentation/DEVLOG.md).

## Documentation routine (DO THIS EVERY SESSION)

1. **At the start of a work session**, append a new dated entry to
   `documentation/DEVLOG.md` with a **Plan** block *before* coding:
   - what we are doing and **why**, which **decisions** were made or **changed**,
     and what we are going with.
2. **While working**, keep notes of what failed, constraints hit, and course
   changes (add to the entry).
3. **When the work is complete**, fill the same entry's **Done** block:
   - what was **accomplished** + results/numbers, **what is left**, and **how to do
     the remaining work** (concrete next commands/steps).
4. **Update `documentation/PROJECT.md`** whenever status, architecture, decisions,
   constraints or gaps change (status truth-table, roadmap, known gaps).
5. Keep both files honest: record what **failed** and what is **experimental/
   unvalidated**, not just wins.

## Environment

- Use the project venv: **`.venv\Scripts\python.exe`** (Python 3.11; has mediapipe,
  opencv, torch-cpu, scipy, and the editable `rppg` package).
- Run the demo: `.venv\Scripts\python.exe scripts\live_demo.py` (guide:
  `docs/DEMO_VSCODE.md`).
- Tests: `.venv\Scripts\python.exe -m pytest -q` (26 should pass).
- Shell is **PowerShell**; avoid here-strings when passing code to `python -c`
  (quotes get mangled) — write a file and run it instead.

## Hard constraints (don't relearn these)

- **No local GPU**; deep training is slow on CPU or uses Colab (which times out and
  wipes `/content`).
- **Disk is tight**: E: ~6.7 GB, C: ~19 GB free — the full 28 GB UBFC will not fit;
  fetch a **subset** of subjects.
- **Never type the user's secret token.** It lives at
  `C:\Users\admin\.kaggle\access_token` (a 37-char `KGAT_...`); tooling reads it.
- **Blood pressure is experimental/unvalidated** — always label it so; never present
  it as a medical measurement.
- **≤3 bpm HR** is only expected in good conditions (still, lit, near); motion and
  distance degrade it honestly.

## Style

- Many small, focused files; PEP 8 + type hints; immutable where practical; handle
  errors explicitly; no hardcoded secrets. Run the tests before claiming success,
  and only report metrics that actually printed.
