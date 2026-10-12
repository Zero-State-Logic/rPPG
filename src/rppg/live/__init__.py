"""Live webcam rPPG: multi-face detection, per-person HR/BP estimation, overlay.

This package turns a laptop webcam into a real-time remote-PPG demo:

* :mod:`rppg.live.detect`   -- YuNet multi-face detection + forehead/cheek ROI
                               + a centroid tracker that keeps a stable id per
                               person as they move (the "moving / multiple /
                               far-near" demo scenarios).
* :mod:`rppg.live.estimate` -- a per-person rolling RGB buffer turned into a
                               heart rate via the classical ProPOS pulse, with an
                               optional trained ProPOSNet neural estimate and an
                               experimental (uncalibrated) blood-pressure proxy.
* :mod:`rppg.live.overlay`  -- OpenCV drawing of boxes, ROI, HR, BP and quality.

The entry point is ``scripts/live_demo.py``.
"""
