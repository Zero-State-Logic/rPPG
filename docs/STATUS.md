# Project Status - Central Head Board

Updated: 2026-10-09

## Multi-agent roster (Phase 0 research)
| Lead | Model | Task | Status |
|------|-------|------|--------|
| Research Lead | Opus | rPPG-Toolbox + deep HR + motion + distance SOTA | running |
| Data Lead | Sonnet | dataset access + request emails | running |
| Signal/Vision Lead | Opus | capture-artifact removal + data cleaning | running |
| BP Lead | Opus | BP methodology + honest eval protocol | running |
| Architecture Lead | Opus | toolchain + repo + config schema | running |

## Phase ladder (user-ordered)
- [~] Phase 0 - foundation and scaffold
- [ ] Phase 1 - classical HR + ProPOS reproduction
- [ ] Phase 2 - deep HR (target >90% windows within +/-5 bpm)
- [ ] Phase 3 - BP track (research-grade, honest eval)
- [ ] Phase 4 - end-to-end integration + demo
- [ ] Phase 5 - DISTANCE ladder (increase camera distance)
- [ ] Phase 6 - MOTION ladder (static -> dynamic subject)
- [ ] Phase 7 - accuracy hardening + ablations

## Environment
- Python 3.11 venv at .venv (3.10/3.11/3.13/3.14 available; 3.11 standard for ML)
- git 2.53, ffmpeg present, 31.7 GB RAM, Intel UHD GPU (no CUDA -> cloud training)
- Disk: E: ~8 GB free -> heavy datasets stay in cloud
