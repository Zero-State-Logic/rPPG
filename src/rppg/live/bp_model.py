"""Load the trained BP regressor and predict SBP/DBP from a live rPPG pulse.

Thin wrapper over the pickle written by ``scripts/train_bp.py``. It extracts the
same morphology feature vector the model was trained on (:mod:`rppg.bp.features`)
and returns clamped mmHg estimates. Optional -- the live demo falls back to the
heuristic when no model is supplied.

Honest scope: the model was trained on CONTACT fingertip PPG, so on camera rPPG
this is a rough, experimental estimate (real domain gap), not a measurement.
"""
from __future__ import annotations

import pickle

import numpy as np

from ..bp.features import features_vector

_SBP_RANGE = (80.0, 200.0)
_DBP_RANGE = (45.0, 120.0)
_MIN_SAMPLES = 32


class BPModel:
    """Predict (SBP, DBP) in mmHg from a pulse segment using the trained regressor."""

    def __init__(self, path: str) -> None:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self._model = blob["model"] if isinstance(blob, dict) else blob

    def predict(self, pulse: np.ndarray, fps: float) -> tuple[float, float] | None:
        """Return clamped (sbp, dbp), or None if the segment is too short/invalid."""
        pulse = np.asarray(pulse, dtype=np.float64)
        if pulse.size < _MIN_SAMPLES:
            return None
        feats = features_vector(pulse, float(fps)).reshape(1, -1)
        try:
            sbp_arr, dbp_arr = self._model.predict(feats)
        except Exception:  # noqa: BLE001 - never let BP break the live loop
            return None
        sbp = float(np.clip(sbp_arr[0], *_SBP_RANGE))
        dbp = float(np.clip(dbp_arr[0], _DBP_RANGE[0], min(_DBP_RANGE[1], sbp - 10.0)))
        return sbp, dbp
