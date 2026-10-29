"""Windowed heart-rate evaluation helpers shared by offline experiments.

The live estimator and the offline evaluator must measure HR identically, so these
mirror the live classical path exactly: POS + CHROM + ProPOS pulses, each band-passed,
sub-bin spectral peak per method, then SNR fusion. The reference HR from contact PPG
uses the same band-pass + sub-bin peak, so predicted and reference HR are measured
the same way.
"""
from __future__ import annotations

import numpy as np

from ..hr.refine import bandpass, fuse_estimates, hr_peak_interp
from ..methods.classical.chrom import chrom_pulse
from ..methods.classical.pos import pos_pulse
from ..methods.classical.propos import propos_pulse


def method_estimates(trace: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """``[(hr, snr)]`` for POS, CHROM and ProPOS on one ``(3, T)`` RGB window."""
    pulses = (pos_pulse(trace, fps), chrom_pulse(trace, fps),
              propos_pulse(trace, fps, do_fit=False))
    return [hr_peak_interp(bandpass(p, fps), fps) for p in pulses]


def fused_hr(trace: np.ndarray, fps: float) -> tuple[float, float]:
    """``(hr_bpm, snr_db)`` of one ``(3, T)`` window via the live SNR fusion."""
    return fuse_estimates(method_estimates(trace, fps))


def reference_hr(ppg_window: np.ndarray, fps: float) -> float:
    """Reference HR (bpm) of a contact-PPG window, measured like the prediction."""
    return hr_peak_interp(bandpass(ppg_window, fps), fps)[0]


def window_starts(n_frames: int, fps: float, win_sec: float,
                  step_sec: float) -> list[int]:
    """Start indices of all full sliding windows of ``win_sec`` every ``step_sec``."""
    win = int(round(win_sec * fps))
    step = max(1, int(round(step_sec * fps)))
    if win <= 0 or n_frames < win:
        return []
    return list(range(0, n_frames - win + 1, step))
