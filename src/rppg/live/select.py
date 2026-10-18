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


@dataclass(frozen=True)
class Decision:
    """Which source leads this tick, plus the losing source for the overlay.

    ``main`` is the source key ``"neural"`` or ``"classical"``; ``hr`` is that
    source's HR. ``other_hr`` / ``other_label`` describe the non-selected source
    (``other_label`` is the display name, ``"ProPOSNet"`` or ``"ProPOS"``), and
    ``conf_main`` / ``conf_other`` are their blended confidences.
    """

    main: str
    hr: float
    other_hr: float
    other_label: str
    conf_main: float
    conf_other: float


class SourceArbiter:
    """Per-track-id selector between the neural and classical HR sources."""

    def __init__(self) -> None:
        self._hist: dict[int, dict[str, deque]] = {}
        self._main: dict[int, str] = {}
        self._votes: dict[int, int] = {}

    def _ensure(self, tid: int) -> dict[str, deque]:
        """Create (default) per-track state on first sight of ``tid``."""
        if tid not in self._hist:
            self._hist[tid] = {"neural": deque(maxlen=_MAXLEN),
                               "classical": deque(maxlen=_MAXLEN)}
            self._main[tid] = "neural"
            self._votes[tid] = 0
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
               fresh_classical: bool, fresh_neural: bool) -> Decision:
        """Fold this tick's readings into the per-track state and pick a source.

        ``fresh_*`` say whether that source produced a new estimate this tick; only
        a fresh, finite HR is pushed into its jitter history. Returns the
        :class:`Decision` describing the selected and the other source.
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

        if main == "neural":
            return Decision(main="neural", hr=hr_n, other_hr=hr_c,
                            other_label="ProPOS", conf_main=conf_n, conf_other=conf_c)
        return Decision(main="classical", hr=hr_c, other_hr=hr_n,
                        other_label="ProPOSNet", conf_main=conf_c, conf_other=conf_n)

    def prune(self, live_ids: set[int]) -> None:
        """Forget per-track state whose faces are no longer tracked."""
        for gone in [tid for tid in self._hist if tid not in live_ids]:
            del self._hist[gone]
            self._main.pop(gone, None)
            self._votes.pop(gone, None)
