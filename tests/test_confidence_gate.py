"""Unit tests for the hard confidence gate in :class:`rppg.live.select.SourceArbiter`.

The gate must refuse to publish a headline HR when the pulse is unreliable -- a weak
spectrum (dim / motion / flicker) or the two sources disagreeing -- so the demo shows
"signal too weak" instead of a misleading ~55-60 bpm artifact. It is a Schmitt trigger
with dwell timers: trust is only earned after sustained good signal and only dropped
after sustained bad signal, so the verdict cannot flicker.

These drive the arbiter directly with duck-typed ``Reading`` / ``NeuralReading`` stand-ins
(``SimpleNamespace``) and an explicit per-call ``dt`` so a few calls cross the thresholds.
"""
from __future__ import annotations

from types import SimpleNamespace

from rppg.live.select import (
    _AGREE_BPM,
    _ENTER_SEC,
    _EXIT_SEC,
    _SNR_ENTER,
    _SNR_EXIT,
    SourceArbiter,
)

_DT = 0.5   # seconds per simulated frame; a few of these cross the dwell thresholds
_TID = 1


def _classical(hr: float, snr: float, ready: bool = True) -> SimpleNamespace:
    """A duck-typed classical ``Reading`` (only .ready/.hr/.snr are read)."""
    return SimpleNamespace(ready=ready, hr=hr, snr=snr)


def _neural(hr: float, snr: float) -> SimpleNamespace:
    """A duck-typed ``NeuralReading`` (only .hr/.snr are read)."""
    return SimpleNamespace(hr=hr, snr=snr)


def _drive(arb: SourceArbiter, calls: int, hr_c: float, snr_c: float,
           hr_n: float, snr_n: float):
    """Feed ``calls`` identical fresh ticks and return the final Decision."""
    decision = None
    for _ in range(calls):
        decision = arb.update(_TID, _classical(hr_c, snr_c), _neural(hr_n, snr_n),
                              True, True, dt=_DT)
    return decision


def test_sustained_low_snr_never_trusts():
    # Arrange: SNR well below the enter threshold on both sources, agreeing HRs.
    arb = SourceArbiter()
    low = _SNR_ENTER - 2.0

    # Act: run far longer than the enter dwell would need at full strength.
    n = int(_ENTER_SEC / _DT) + 6
    decision = None
    for _ in range(n):
        decision = arb.update(_TID, _classical(72.0, low), _neural(72.0, low),
                              True, True, dt=_DT)
        # Assert (every tick): a weak spectrum must never flip the gate on.
        assert decision.trusted is False

    assert decision is not None
    assert decision.gate_reason == "low signal"


def test_sustained_high_snr_and_agreement_trusts_after_dwell():
    # Arrange: strong SNR and the two sources agreeing within tolerance.
    arb = SourceArbiter()
    high = _SNR_ENTER + 3.0

    # Not yet trusted right at the start (needs _ENTER_SEC of sustained good signal).
    first = _drive(arb, 1, 72.0, high, 72.0, high)
    assert first.trusted is False

    # Act: accumulate more than the enter dwell.
    n = int(_ENTER_SEC / _DT) + 4
    decision = _drive(arb, n, 72.0, high, 72.0, high)

    # Assert: the gate has latched on and the HR is now trusted.
    assert decision.trusted is True
    assert decision.gate_reason == ""


def test_single_low_snr_tick_does_not_drop_trust():
    # Arrange: first earn trust with sustained strong, agreeing signal.
    arb = SourceArbiter()
    high = _SNR_ENTER + 3.0
    established = _drive(arb, int(_ENTER_SEC / _DT) + 4, 72.0, high, 72.0, high)
    assert established.trusted is True

    # Act: a single bad (dead-signal) tick -- less than the exit dwell.
    dead = _SNR_EXIT - 1.0   # clearly below the exit-SNR floor
    one_bad = _drive(arb, 1, 72.0, dead, 72.0, dead)

    # Assert: one bad frame is not enough to drop trust (Schmitt hysteresis holds).
    assert one_bad.trusted is True

    # And: sustaining the bad signal past the exit dwell finally drops it.
    more = int(_EXIT_SEC / _DT) + 1
    dropped = _drive(arb, more, 72.0, dead, 72.0, dead)
    assert dropped.trusted is False


def test_sustained_disagreement_gates_despite_high_snr():
    # Arrange: both sources strong, but their rates differ by more than _AGREE_BPM.
    arb = SourceArbiter()
    high = _SNR_ENTER + 3.0
    hr_n = 72.0
    hr_c = hr_n + _AGREE_BPM + 16.0   # clearly beyond the agreement tolerance

    # Act: run well past the enter dwell.
    n = int(_ENTER_SEC / _DT) + 6
    decision = None
    for _ in range(n):
        decision = arb.update(_TID, _classical(hr_c, high), _neural(hr_n, high),
                              True, True, dt=_DT)
        # Assert (every tick): disagreement must keep the gate shut even at high SNR.
        assert decision.trusted is False

    assert decision is not None
    assert decision.gate_reason == "sources disagree"
