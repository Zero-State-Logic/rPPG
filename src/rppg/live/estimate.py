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

from ..hr.metrics import estimate_hr_fft, snr_db
from ..methods.classical.propos import propos_pulse

_WINDOW_SEC = 10.0      # rolling buffer length used for each estimate
_MIN_SEC = 4.0          # need this many seconds before the first HR is shown
_HR_SMOOTH = 7          # median window (estimates) for a steady HR readout
_PLOT_SAMPLES = 160     # trailing pulse samples handed to the overlay plot

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


class PersonSignal:
    """Rolling ROI-colour buffer for one tracked face -> HR/BP readings."""

    def __init__(self, window_sec: float = _WINDOW_SEC) -> None:
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._hr_history: deque[float] = deque(maxlen=_HR_SMOOTH)
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
        return trace, (n - 1) / span

    def estimate(self, calib: Calibration) -> Reading:
        """Compute the current reading; ``ready`` False until enough signal exists."""
        seconds = (self._samples[-1][0] - self._samples[0][0]) if self._samples else 0.0
        resampled = self._uniform_trace()
        if resampled is None or seconds < _MIN_SEC:
            return Reading(ready=False, seconds=seconds)
        trace, fps = resampled
        pulse = propos_pulse(trace, fps)
        hr_raw = estimate_hr_fft(pulse, fps)
        if not np.isfinite(hr_raw):
            return Reading(ready=False, seconds=seconds)
        self._hr_history.append(hr_raw)
        hr = float(np.median(self._hr_history))
        sbp, dbp = _experimental_bp(hr, pulse, fps, calib)
        tail = pulse[-_PLOT_SAMPLES:]
        return Reading(ready=True, seconds=seconds, hr=hr, bp_sys=sbp, bp_dia=dbp,
                       snr=snr_db(pulse, fps, hr / 60.0), pulse=tail)


class SignalBank:
    """One :class:`PersonSignal` per tracked face id, pruned as tracks disappear."""

    def __init__(self) -> None:
        self._people: dict[int, PersonSignal] = {}

    def add(self, track_id: int, timestamp: float, rgb: np.ndarray) -> None:
        """Route a sample to the buffer for ``track_id`` (created on first use)."""
        self._people.setdefault(track_id, PersonSignal()).add(timestamp, rgb)

    def estimate(self, track_id: int, calib: Calibration) -> Reading:
        """Current reading for ``track_id`` (empty not-ready reading if unknown)."""
        signal = self._people.get(track_id)
        return signal.estimate(calib) if signal else Reading(ready=False, seconds=0.0)

    def prune(self, live_ids: set[int]) -> None:
        """Forget buffers whose faces are no longer tracked."""
        for gone in [tid for tid in self._people if tid not in live_ids]:
            del self._people[gone]
