"""Unit tests for the classical rPPG core (POS, ProPOS) and HR metrics.

These validate the signal-processing contract on synthetic signals with known
heart rate, following the AAA (arrange-act-assert) structure.
"""
from __future__ import annotations

import numpy as np
import pytest

from rppg.hr.metrics import (accuracy_within, estimate_hr_fft, mae, pearson_r,
                             rmse, snr_db)
from rppg.methods.classical.pos import pos_pulse
from rppg.methods.classical.propos import build_projection_bank, propos_pulse


@pytest.mark.parametrize("hr", [54.0, 72.0, 96.0, 120.0])
def test_pos_recovers_hr(hr, make_rgb):
    rgb = make_rgb(hr)
    est = estimate_hr_fft(pos_pulse(rgb, 30.0), 30.0)
    assert abs(est - hr) <= 5.0


@pytest.mark.parametrize("hr", [54.0, 72.0, 96.0, 120.0])
def test_propos_recovers_hr(hr, make_rgb):
    rgb = make_rgb(hr)
    est = estimate_hr_fft(propos_pulse(rgb, 30.0), 30.0)
    assert abs(est - hr) <= 5.0


def test_projection_bank_shape():
    assert build_projection_bank().shape == (50, 2, 3)


def test_pos_scale_invariance(make_rgb):
    rgb = make_rgb(72.0)
    hr1 = estimate_hr_fft(pos_pulse(rgb, 30.0), 30.0)
    hr2 = estimate_hr_fft(pos_pulse(rgb * 3.0, 30.0), 30.0)
    assert abs(hr1 - hr2) <= 1.0


def test_metrics_basic():
    pred = np.array([70.0, 80.0])
    true = np.array([72.0, 78.0])
    assert mae(pred, true) == pytest.approx(2.0)
    assert rmse(pred, true) == pytest.approx(2.0)
    assert accuracy_within(pred, true, tol_bpm=5.0) == pytest.approx(100.0)
    assert pearson_r(pred, true) == pytest.approx(1.0)


def test_snr_positive_for_clean_signal(make_rgb):
    pulse = propos_pulse(make_rgb(72.0), 30.0)
    assert snr_db(pulse, 30.0, 72.0 / 60.0) > 0.0
