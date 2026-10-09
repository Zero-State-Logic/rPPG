"""Unit tests for BP pulse-morphology feature extraction."""
from __future__ import annotations

import numpy as np

from rppg.bp.features import FEATURE_NAMES, features_vector, pulse_features


def _synthetic_pulse(hr_bpm: float = 72.0, fps: float = 30.0,
                     seconds: float = 12.0) -> np.ndarray:
    t = np.arange(int(fps * seconds)) / fps
    f = hr_bpm / 60.0
    return np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)


def test_feature_keys_present_and_finite():
    feats = pulse_features(_synthetic_pulse(), 30.0)
    assert set(feats) == set(FEATURE_NAMES)
    for name in ("hr_bpm", "amp_std", "vpg_max", "n_beats"):
        assert np.isfinite(feats[name])


def test_hr_feature_matches():
    feats = pulse_features(_synthetic_pulse(72.0), 30.0)
    assert abs(feats["hr_bpm"] - 72.0) <= 5.0
    assert feats["n_beats"] >= 10          # ~14 beats in 12 s at 72 bpm


def test_features_vector_shape():
    vec = features_vector(_synthetic_pulse(), 30.0)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec.dtype == np.float64
