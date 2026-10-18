"""Per-person heart-rate (and an experimental blood-pressure proxy) from ROI colour.

For every tracked face we keep a rolling buffer of ``(timestamp, mean ROI RGB)``
samples. On demand the buffer is resampled to a uniform time grid and turned into a
pulse with the project's **ProPOS** method (``rppg.methods.classical.propos``), from
which heart rate and SNR follow (``rppg.hr.metrics``). HR is median-smoothed over a
few recent estimates so it stays steady while people move.

Blood pressure here is an **experimental, uncalibrated proxy** -- a transparent
heuristic over heart rate and pulse-harmonic morphology, NOT a validated
measurement. Real cuff-grade BP needs the BP datasets on the project roadmap; this
value exists only so the demo can show the intended output, and must be presented
as experimental. An optional one-shot calibration offset lets it track changes
around a known reference reading.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import periodogram

from ..hr.refine import bandpass, fuse_estimates, hr_peak_interp
from ..methods.classical.chrom import chrom_pulse
from ..methods.classical.pos import pos_pulse
from ..methods.classical.propos import propos_pulse

_WINDOW_SEC = 10.0      # rolling buffer length used for each estimate
_MIN_SEC = 1.0          # show a provisional ("acquiring") reading this early
_LOCK_SEC = 4.0         # readings before this stay provisional, then they lock
_PLOT_SAMPLES = 160     # trailing pulse samples handed to the overlay plot

# HR tracker: reject 2nd-harmonic locks, rate-limit jumps, hold through motion.
_HR_MIN, _HR_MAX = 40.0, 180.0   # plausible live heart-rate range (bpm)
_SLEW = 8.0                      # max smoothed HR change accepted per update (bpm)
_STRONG_SNR = 2.0                # dB above which a reading is treated as confident
_CONFIRM = 3                     # confident agreeing readings needed to jump regime
_ALPHA_STRONG, _ALPHA_WEAK = 0.35, 0.15   # smoothing rates by confidence
_HARMONIC_TOL = 12.0             # bpm window for spotting a 2x-current harmonic
_FIRST_LOCK_SNR = -2.0           # accept the first HR lock once SNR clears this (dB)

# Experimental BP heuristic anchors (see module docstring -- NOT validated).
_SBP0, _DBP0, _HR0 = 118.0, 76.0, 70.0
_SBP_HR_K, _DBP_HR_K = 0.45, 0.28
_STIFF_K = 18.0
_SBP_RANGE, _DBP_RANGE = (90.0, 165.0), (55.0, 105.0)


@dataclass(frozen=True)
class Calibration:
    """One-shot offsets so experimental BP tracks around a known reference."""

    sbp_offset: float = 0.0
    dbp_offset: float = 0.0


@dataclass
class Reading:
    """A per-person estimate; ``ready`` is False until enough signal is buffered."""

    ready: bool
    seconds: float
    hr: float = float("nan")
    bp_sys: float = float("nan")
    bp_dia: float = float("nan")
    snr: float = float("nan")
    provisional: bool = False
    pulse: np.ndarray = field(default_factory=lambda: np.empty(0))


def _dia_ceiling(sbp: float, dbp_range: tuple[float, float]) -> tuple[float, float]:
    """Diastolic clip range, capped at sbp - 15 so it stays below systolic."""
    lo, hi = dbp_range
    return lo, min(hi, sbp - 15.0)


def _harmonic_ratio(pulse: np.ndarray, fps: float, hr_hz: float) -> float:
    """Power ratio of the 2nd to the 1st pulse harmonic (a crude stiffness proxy)."""
    if not np.isfinite(hr_hz) or hr_hz <= 0 or pulse.size < 8:
        return 0.0
    freqs, psd = periodogram(pulse - pulse.mean(), fs=fps)

    def _band(center: float) -> float:
        m = (freqs >= center - 0.2) & (freqs <= center + 0.2)
        return float(psd[m].sum()) if np.any(m) else 0.0

    first = _band(hr_hz)
    return _band(2.0 * hr_hz) / first if first > 1e-9 else 0.0


def _experimental_bp(hr: float, pulse: np.ndarray, fps: float,
                     calib: Calibration) -> tuple[float, float]:
    """Return an experimental (uncalibrated) systolic/diastolic estimate in mmHg."""
    stiffness = _harmonic_ratio(pulse, fps, hr / 60.0)
    sbp = _SBP0 + _SBP_HR_K * (hr - _HR0) + _STIFF_K * stiffness + calib.sbp_offset
    dbp = _DBP0 + _DBP_HR_K * (hr - _HR0) + 0.5 * _STIFF_K * stiffness + calib.dbp_offset
    sbp = float(np.clip(sbp, *_SBP_RANGE))
    dbp = float(np.clip(dbp, *_dia_ceiling(sbp, _DBP_RANGE)))
    return sbp, dbp


class HRTracker:
    """Stateful HR smoother: harmonic correction + slew limit + confirmation.

    Fixes the two common live failures -- the estimate jumping to double the true
    rate (2nd-harmonic lock, e.g. 64 -> 128) and brief spikes while moving -- by
    (1) folding x0.5 / x2 harmonics back toward the current rate, (2) rate-limiting
    how far the shown HR can move per update, and (3) only following a large jump
    once several confident readings agree on the new rate. Low-confidence readings
    just hold the last value instead of showing noise.
    """

    def __init__(self) -> None:
        self.hr: float | None = None
        self._cand: float | None = None
        self._cand_n = 0

    def update(self, hr_raw: float, snr: float) -> float | None:
        """Fold one raw (hr, snr) into the tracked HR; None until a clean lock."""
        if not np.isfinite(hr_raw) or not (_HR_MIN <= hr_raw <= _HR_MAX):
            return self.hr                              # implausible/missing -> hold
        if self.hr is None:
            if snr < _FIRST_LOCK_SNR:
                return None                             # wait for a usable first lock
            self.hr = hr_raw
            return self.hr

        chosen = hr_raw
        if abs(hr_raw - 2.0 * self.hr) < _HARMONIC_TOL and hr_raw / 2.0 >= _HR_MIN:
            chosen = hr_raw / 2.0        # reading ~2x current -> 2nd harmonic, halve
        delta = chosen - self.hr
        strong = snr >= _STRONG_SNR

        if abs(delta) <= _SLEW:                         # gentle, in-range change
            self.hr += (_ALPHA_STRONG if strong else _ALPHA_WEAK) * delta
            self._cand, self._cand_n = None, 0
        else:                                           # big jump: demand agreement
            if self._cand is not None and abs(chosen - self._cand) <= _SLEW:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = chosen, 1
            if strong and self._cand_n >= _CONFIRM:
                self.hr, self._cand, self._cand_n = self._cand, None, 0
        return self.hr


def _hampel(x, half, n_sigma=3.0):
    """Replace point outliers in ``x`` with the local median (Hampel filter).

    Slides a ``2*half+1`` window over the 1-D trace, flags samples that sit more
    than ``n_sigma`` robust std-devs (MAD-scaled) from the window median, and
    swaps each flagged sample for its local median. This removes sudden optical
    noise spikes (lighting flicker, specular glints) without smearing the pulse,
    since in-band samples are left untouched. Buffers too short for a full window
    are returned unchanged.
    """
    n = x.size
    if n < 2 * half + 1:
        return x
    pad = np.pad(x, half, mode="edge")
    win = np.stack([pad[i:i + n] for i in range(2 * half + 1)], axis=0)
    med = np.median(win, axis=0)
    sigma = 1.4826 * np.median(np.abs(win - med), axis=0)
    out = x.copy()
    bad = np.abs(x - med) > (n_sigma * sigma)
    out[bad] = med[bad]
    return out


class PersonSignal:
    """Rolling ROI-colour buffer for one tracked face -> HR/BP readings."""

    def __init__(self, window_sec: float = _WINDOW_SEC) -> None:
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._tracker = HRTracker()
        self._window = window_sec

    def add(self, timestamp: float, rgb: np.ndarray) -> None:
        """Append one mean-ROI RGB sample and drop anything older than the window."""
        self._samples.append((timestamp, np.asarray(rgb, dtype=np.float64)))
        cutoff = timestamp - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _uniform_trace(self) -> tuple[np.ndarray, float] | None:
        """Resample buffered samples to a uniform (3, T) RGB trace + fps."""
        if len(self._samples) < 8:
            return None
        ts = np.array([t for t, _ in self._samples], dtype=np.float64)
        span = ts[-1] - ts[0]
        if span <= 0.5:
            return None
        rgb = np.stack([v for _, v in self._samples], axis=1)  # (3, N)
        n = len(self._samples)
        grid = np.linspace(ts[0], ts[-1], n)
        trace = np.stack([np.interp(grid, ts, rgb[c]) for c in range(3)], axis=0)
        fps = (n - 1) / span
        half = max(2, int(round(0.2 * fps)))
        trace = np.stack([_hampel(trace[c], half) for c in range(3)], axis=0)
        return trace, fps

    def estimate(self, calib: Calibration, bp_model=None) -> Reading:
        """Compute the current reading; ``ready`` False until enough signal exists.

        ``bp_model`` (a ``rppg.live.bp_model.BPModel``), when given, replaces the
        experimental heuristic with the trained regressor's BP estimate.
        """
        seconds = (self._samples[-1][0] - self._samples[0][0]) if self._samples else 0.0
        resampled = self._uniform_trace()
        if resampled is None or seconds < _MIN_SEC:
            return Reading(ready=False, seconds=seconds)
        trace, fps = resampled

        # Run three rPPG methods, band-pass each pulse, and fuse by SNR so the
        # strongest-pulse method wins each instant (robust multi-method estimate).
        pulses = {"pos": pos_pulse(trace, fps),
                  "chrom": chrom_pulse(trace, fps),
                  "propos": propos_pulse(trace, fps, do_fit=False)}
        scored = []
        for name, raw in pulses.items():
            filt = bandpass(raw, fps)
            hr_i, snr_i = hr_peak_interp(filt, fps)
            scored.append((name, filt, hr_i, snr_i))
        hr_raw, snr = fuse_estimates([(s[2], s[3]) for s in scored])
        if not np.isfinite(hr_raw):
            return Reading(ready=False, seconds=seconds)

        tracked = self._tracker.update(hr_raw, snr)
        if tracked is not None:
            hr, provisional = tracked, seconds < _LOCK_SEC
        elif _HR_MIN <= hr_raw <= _HR_MAX:
            hr, provisional = hr_raw, True   # early provisional, before a clean lock
        else:
            return Reading(ready=False, seconds=seconds)
        best = max(scored, key=lambda s: s[3] if np.isfinite(s[3]) else -1e9)
        best_pulse = best[1]
        sbp = dbp = float("nan")
        if bp_model is not None:
            pred = bp_model.predict(best_pulse, fps)
            if pred is not None:
                sbp = pred[0] + calib.sbp_offset
                dbp = pred[1] + calib.dbp_offset
        if not np.isfinite(sbp):
            sbp, dbp = _experimental_bp(hr, best_pulse, fps, calib)
        return Reading(ready=True, seconds=seconds, hr=hr, bp_sys=sbp, bp_dia=dbp,
                       snr=snr, provisional=provisional, pulse=best_pulse[-_PLOT_SAMPLES:])


class SignalBank:
    """One :class:`PersonSignal` per tracked face id, pruned as tracks disappear."""

    def __init__(self) -> None:
        self._people: dict[int, PersonSignal] = {}

    def add(self, track_id: int, timestamp: float, rgb: np.ndarray) -> None:
        """Route a sample to the buffer for ``track_id`` (created on first use)."""
        self._people.setdefault(track_id, PersonSignal()).add(timestamp, rgb)

    def estimate(self, track_id: int, calib: Calibration, bp_model=None) -> Reading:
        """Current reading for ``track_id`` (empty not-ready reading if unknown)."""
        signal = self._people.get(track_id)
        return (signal.estimate(calib, bp_model) if signal
                else Reading(ready=False, seconds=0.0))

    def prune(self, live_ids: set[int]) -> None:
        """Forget buffers whose faces are no longer tracked."""
        for gone in [tid for tid in self._people if tid not in live_ids]:
            del self._people[gone]
