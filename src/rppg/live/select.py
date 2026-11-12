"""Per-track arbiter that picks the live "main" heart-rate source.

The trained neural net (ProPOSNet) is the default/front source, but on a short
window it can be noisier than the classical ProPOS estimate. This module scores
each source per track from its recent **stability** (low HR jitter), its spectral
**SNR**, and how well the two sources **agree**, then shows the neural reading as
the main HR unless the classical one is clearly and persistently steadier -- in
which case it switches, and switches straight back as soon as neural is competitive
again. Neural is sticky (needs several agreeing votes to be dropped) so a single
noisy classical tick cannot flip the headline number.

Pure state + arithmetic over duck-typed ``Reading`` / ``NeuralReading`` objects
(only ``.hr`` / ``.snr`` / ``.ready`` are read); no I/O and no torch import, so the
demo stays importable without the neural stack.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type-only; avoids importing torch via neural at runtime
    from .estimate import Reading
    from .neural import NeuralReading

_MAXLEN = 6            # recent fresh-HR samples kept per source for jitter
_MARGIN = 0.12         # classical must beat neural confidence by this to earn a vote
_SWITCH_VOTES = 3      # agreeing classical-preferred votes needed to leave neural
_STAB_SCALE = 3.0      # bpm jitter -> stability decay constant
_AGREE_SCALE = 6.0     # bpm disagreement -> agreement decay constant
_SNR_OFFSET, _SNR_SPAN = 2.0, 8.0   # maps SNR(dB) into a 0..1 confidence term
_W_STAB, _W_SNR, _W_AGREE = 0.5, 0.3, 0.2   # confidence blend weights

# Hard confidence gate: a Schmitt-trigger / leaky-dwell state machine that decides
# whether the committed headline HR is trustworthy enough to show, or whether the demo
# should fall back to a "signal too weak" message instead of a misleading rate under
# dim / motion / flicker. Entering trust needs sustained good signal; leaving it needs
# sustained bad signal, so the headline cannot flicker between a number and the warning.
#
# Thresholds are calibrated from a clean, lit reference face (UBFC subject10): its
# classical SNR runs median ~1.7 dB (p90 3.3), neural SNR median ~0.8 (max ~2.1), and
# the two sources agree to within ~9 bpm (p90). So the gate (1) scores signal quality
# on the better-calibrated CLASSICAL SNR, not the compressed-scale neural proxy, and
# (2) leans on cross-source AGREEMENT -- not an absolute SNR -- as the primary artifact
# discriminator, since an artifact lock (the strongest leftover low-freq component when
# there is no real pulse) can itself carry a modest positive SNR but makes the two
# independent methods disagree.
# Re-validated 2026-11-12 after the classical SNR changed to the tracked-rate SNR over
# 42-180 bpm (rppg.live.estimate._spectrum_snr): with these unchanged thresholds the
# replayed live loop on MCD-rPPG dev (40 subjects) shows a number 62 % of the time at
# 2.47 bpm MAE, only 3.7 % of shown numbers off by >10 bpm (was 43 % before the fix).
# A live webcam is noisier than the clean subject10 reference, so the enter floor
# sits below 0 dB and the dwell is short, to LATCH a real face quickly rather than
# withhold it; the exit floor is well below the enter floor (asymmetric Schmitt
# hysteresis) so a locked reading holds through a normal SNR dip instead of flickering
# between a number and the warning.
_SNR_ENTER = -1.0      # dB: classical-SNR floor to START trusting a live face
_SNR_EXIT = -3.5       # dB: SNR this low (while trusted) marks a dead-signal frame
_ENTER_SEC = 1.0       # s of net-good signal (leaky) before the HR is shown
_EXIT_SEC = 1.5        # s of net-bad signal (leaky) before trust is dropped
_AGREE_BPM = 12.0      # bpm: neural/classical must agree within this (clean face p90~9)
_EDGE_HR_MIN = 42.0    # bpm: reject near-edge locks at/below this as implausible
_HR_LO, _HR_HI = 40.0, 180.0   # plausible live heart-rate window (bpm)


@dataclass(frozen=True)
class Decision:
    """Which source leads this tick, plus the losing source for the overlay.

    ``main`` is the source key ``"neural"`` or ``"classical"``; ``hr`` is that
    source's HR. ``other_hr`` / ``other_label`` describe the non-selected source
    (``other_label`` is the display name, ``"ProPOSNet"`` or ``"ProPOS"``), and
    ``conf_main`` / ``conf_other`` are their blended confidences.

    ``trusted`` is the hard confidence gate: ``False`` means the headline HR is an
    unreliable artifact (weak / disagreeing / implausible signal) and the overlay
    should show "signal too weak" rather than a number. ``gate_reason`` names the
    dominant reason it is withheld ("acquiring", "low signal", "sources disagree" or
    "rate implausible"); both default so every existing keyword construction stays
    valid.
    """

    main: str
    hr: float
    other_hr: float
    other_label: str
    conf_main: float
    conf_other: float
    trusted: bool = True
    gate_reason: str = ""


def _gate_reason(trusted: bool, good_frame: bool, classical_ready: bool,
                 hr_plausible: bool, disagree: bool, best_snr: float) -> str:
    """Name the dominant reason the headline HR is withheld (or "" when trusted).

    Ordered so the most actionable cause wins: still warming up or already-good
    signal just building dwell time ("acquiring"), the two sources landing on
    different rates ("sources disagree"), an out-of-range / edge lock ("rate
    implausible"), or a weak spectrum ("low signal").
    """
    if trusted:
        return ""
    if not classical_ready or good_frame:
        return "acquiring"              # warming up, or good signal building dwell time
    if disagree:
        return "sources disagree"
    if not hr_plausible:
        return "rate implausible"
    if not np.isfinite(best_snr) or best_snr < _SNR_ENTER:
        return "low signal"
    return "acquiring"


class SourceArbiter:
    """Per-track-id selector between the neural and classical HR sources."""

    def __init__(self) -> None:
        self._hist: dict[int, dict[str, deque]] = {}
        self._main: dict[int, str] = {}
        self._votes: dict[int, int] = {}
        self._shown: dict[int, bool] = {}   # gate latched on (HR is being shown)
        self._good: dict[int, float] = {}   # s of sustained good signal (pre-trust)
        self._bad: dict[int, float] = {}    # s of sustained bad signal (while shown)

    def _ensure(self, tid: int) -> dict[str, deque]:
        """Create (default) per-track state on first sight of ``tid``."""
        if tid not in self._hist:
            self._hist[tid] = {"neural": deque(maxlen=_MAXLEN),
                               "classical": deque(maxlen=_MAXLEN)}
            self._main[tid] = "neural"
            self._votes[tid] = 0
            self._shown[tid] = False
            self._good[tid] = 0.0
            self._bad[tid] = 0.0
        return self._hist[tid]

    @staticmethod
    def confidence(hr_jitter_deque: deque, snr: float,
                   hr_n: float, hr_c: float) -> float:
        """Blend recent stability, spectral SNR and cross-source agreement -> [0,1].

        ``stability`` rewards a low standard deviation of the source's recent HR
        (0 until at least two samples exist); ``snr_term`` maps dB into 0..1; and
        ``agreement`` rewards the two sources landing on a similar rate.
        """
        if len(hr_jitter_deque) < 2:
            stability = 0.0
        else:
            jitter = float(np.std(np.asarray(hr_jitter_deque, dtype=float)))
            stability = math.exp(-jitter / _STAB_SCALE)
        snr_term = float(np.clip((snr + _SNR_OFFSET) / _SNR_SPAN, 0.0, 1.0))
        agreement = math.exp(-abs(hr_n - hr_c) / _AGREE_SCALE)
        return _W_STAB * stability + _W_SNR * snr_term + _W_AGREE * agreement

    def update(self, tid: int, classical: "Reading",
               neural: "NeuralReading | None",
               fresh_classical: bool, fresh_neural: bool,
               dt: float = 0.0) -> Decision:
        """Fold this tick's readings into the per-track state and pick a source.

        ``fresh_*`` say whether that source produced a new estimate this tick; only
        a fresh, finite HR is pushed into its jitter history. ``dt`` is the real
        time elapsed since the previous frame, used by the hard confidence gate's
        dwell timers (``0.0`` leaves the gate state unchanged). Returns the
        :class:`Decision` describing the selected and the other source, including
        its ``trusted`` / ``gate_reason`` gate verdict.
        """
        hist = self._ensure(tid)
        neural_ok = neural is not None and np.isfinite(neural.hr)
        classical_ok = (classical is not None and classical.ready
                        and np.isfinite(classical.hr))

        if fresh_neural and neural_ok:
            hist["neural"].append(float(neural.hr))
        if fresh_classical and classical_ok:
            hist["classical"].append(float(classical.hr))

        hr_n = float(neural.hr) if neural_ok else float("nan")
        hr_c = float(classical.hr) if classical_ok else float("nan")

        conf_n = conf_c = 0.0
        if not neural_ok:
            main = "classical"
        elif not classical_ok:
            main = "neural"
        else:
            conf_n = self.confidence(hist["neural"], float(neural.snr), hr_n, hr_c)
            conf_c = self.confidence(hist["classical"], float(classical.snr),
                                     hr_n, hr_c)
            prefer_classical = conf_c > conf_n + _MARGIN
            self._votes[tid] = self._votes[tid] + 1 if prefer_classical else 0
            if self._main[tid] == "neural" and self._votes[tid] >= _SWITCH_VOTES:
                self._main[tid] = "classical"
            elif self._main[tid] == "classical" and not prefer_classical:
                self._main[tid] = "neural"
            main = self._main[tid]

        # --- Hard confidence gate (Schmitt trigger + leaky dwell) over signal quality.
        # Score quality on the better-calibrated CLASSICAL SNR (the neural SNR is a
        # learned proxy on a compressed scale), and use cross-source AGREEMENT as the
        # primary artifact discriminator -- a real pulse makes the two independent
        # methods land on the same rate, an artifact lock makes them disagree.
        # Agreement compares each source's most recent HR (not same-tick freshness,
        # which the two update cadences rarely share). The dwell integrators leak
        # instead of hard-resetting, so a scattered bad frame does not restart the
        # count but a sustained bad stretch still crosses the threshold.
        sel_hr = hr_n if main == "neural" else hr_c
        best_snr = float("-inf")
        if classical_ok:
            best_snr = float(classical.snr)
        if neural_ok:
            best_snr = max(best_snr, float(neural.snr))
        recent_n = hist["neural"][-1] if hist["neural"] else float("nan")
        recent_c = hist["classical"][-1] if hist["classical"] else float("nan")
        both_known = np.isfinite(recent_n) and np.isfinite(recent_c)
        disagree = bool(both_known and abs(recent_n - recent_c) > _AGREE_BPM)
        hr_plausible = bool(np.isfinite(sel_hr) and _HR_LO <= sel_hr <= _HR_HI
                            and sel_hr > _EDGE_HR_MIN)
        good_frame = (classical.ready and hr_plausible and not disagree
                      and best_snr >= _SNR_ENTER)

        if not self._shown[tid]:
            self._good[tid] = float(np.clip(
                self._good[tid] + (dt if good_frame else -dt), 0.0, _ENTER_SEC))
            if self._good[tid] >= _ENTER_SEC:
                self._shown[tid] = True
                self._bad[tid] = 0.0
        else:
            bad_frame = (not classical.ready or not hr_plausible or disagree
                         or best_snr < _SNR_EXIT)
            self._bad[tid] = float(np.clip(
                self._bad[tid] + (dt if bad_frame else -dt), 0.0, _EXIT_SEC))
            if self._bad[tid] >= _EXIT_SEC:
                self._shown[tid] = False
                self._good[tid] = 0.0

        trusted = self._shown[tid]
        gate_reason = _gate_reason(trusted, good_frame, classical.ready,
                                   hr_plausible, disagree, best_snr)

        if main == "neural":
            return Decision(main="neural", hr=hr_n, other_hr=hr_c,
                            other_label="ProPOS", conf_main=conf_n, conf_other=conf_c,
                            trusted=trusted, gate_reason=gate_reason)
        return Decision(main="classical", hr=hr_c, other_hr=hr_n,
                        other_label="ProPOSNet", conf_main=conf_c, conf_other=conf_n,
                        trusted=trusted, gate_reason=gate_reason)

    def prune(self, live_ids: set[int]) -> None:
        """Forget per-track state whose faces are no longer tracked."""
        for gone in [tid for tid in self._hist if tid not in live_ids]:
            del self._hist[gone]
            self._main.pop(gone, None)
            self._votes.pop(gone, None)
            self._shown.pop(gone, None)
            self._good.pop(gone, None)
            self._bad.pop(gone, None)
