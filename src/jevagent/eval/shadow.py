"""Shadow mode (Stage 3): Phase A runs silently beside a live baseline call.

The recorder feeds every caller transcript to its own controller and logs what Phase A *would*
have done: slot values, clips, handoff moments. It produces no audio and never touches the live
call; failures are logged and swallowed. Caveat: the caller answered the baseline agent's
questions, not ours, so slot agreement is an approximation, not an exact replay.
"""

import asyncio
import time
from pathlib import Path
from typing import Literal, Protocol

import structlog
from pydantic import BaseModel

from jevagent.controller.controller import Controller, PassToPhaseB, PlayClip, StartHandoff

log = structlog.get_logger(__name__)

WouldDo = Literal["clip", "end", "handoff", "phase_b"]
STUCK_TURNS = 3


class ShadowTurn(BaseModel):
    call_id: str
    turn: int  # caller turn number, from 1
    metric_turn: int  # the live session's timer turn, to join latency rows
    utterance: str
    would_do: WouldDo
    clip_id: str | None = None
    handoff_reason: str | None = None
    step: str
    slots: dict[str, str | None]
    jev_ms: float
    stt_ms: float | None = None


class ShadowSink(Protocol):
    def write(self, turn: ShadowTurn) -> None: ...


class MemoryShadow:
    def __init__(self) -> None:
        self.rows: list[ShadowTurn] = []

    def write(self, turn: ShadowTurn) -> None:
        self.rows.append(turn)


class JsonlShadow:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, turn: ShadowTurn) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(turn.model_dump_json() + "\n")


class ShadowRecorder:
    def __init__(self, controller: Controller, sink: ShadowSink, call_id: str) -> None:
        self._controller = controller
        self._sink = sink
        self._call_id = call_id
        self._turn = 0
        self._last: asyncio.Task[None] | None = None
        controller.start()

    def submit(self, text: str, metric_turn: int, stt_ms: float | None = None) -> None:
        """Queue one transcript; turns are processed in order and never block the live call."""
        self._turn += 1
        previous, turn = self._last, self._turn
        self._last = asyncio.create_task(self._run(previous, turn, metric_turn, text, stt_ms))

    async def close(self) -> None:
        if self._last is not None:
            await asyncio.gather(self._last, return_exceptions=True)

    async def _run(
        self,
        previous: "asyncio.Task[None] | None",
        turn: int,
        metric_turn: int,
        text: str,
        stt_ms: float | None,
    ) -> None:
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        try:
            start = time.monotonic()
            action = await self._controller.on_caller_turn(text)
            jev_ms = (time.monotonic() - start) * 1000
        except Exception:  # shadow must never affect the live call
            log.exception("shadow_failed", call_id=self._call_id, turn=turn)
            return
        state = self._controller.state
        clip_id = handoff_reason = None
        would_do: WouldDo
        if isinstance(action, PlayClip):
            would_do, clip_id = ("end" if action.end_call else "clip"), action.clip_id
        elif isinstance(action, StartHandoff):
            would_do, handoff_reason = "handoff", action.packet.handoff_reason
        else:
            assert isinstance(action, PassToPhaseB)
            would_do = "phase_b"
        self._sink.write(
            ShadowTurn(
                call_id=self._call_id,
                turn=turn,
                metric_turn=metric_turn,
                utterance=text,
                would_do=would_do,
                clip_id=clip_id,
                handoff_reason=handoff_reason,
                step=state.step,
                slots={name: s.value for name, s in state.slots.items()},
                jev_ms=round(jev_ms, 1),
                stt_ms=stt_ms,
            )
        )


# -- offline comparison ----------------------------------------------------------------


class GroundTruth(BaseModel):
    """What a good outcome looked like for one call (CRM result and/or a human label)."""

    call_id: str
    slots: dict[str, str] = {}
    should_handoff_turn: int | None = None  # caller turn where handoff was right; None if never


class ShadowReport(BaseModel):
    calls: int
    slot_agreement: float | None
    slot_coverage: float | None
    handoff_recall: float | None
    late_handoffs: int
    false_handoff_rate: float | None
    stuck_calls: int
    mean_latency_saving_ms: float | None


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def compare_shadow(
    turns: list[ShadowTurn],
    truths: list[GroundTruth],
    baseline_ttfa_ms: dict[tuple[str, int], float],
) -> ShadowReport:
    """`baseline_ttfa_ms` maps (call_id, metric_turn) to the live baseline's TTFA."""
    by_call: dict[str, list[ShadowTurn]] = {}
    for t in sorted(turns, key=lambda t: (t.call_id, t.turn)):
        by_call.setdefault(t.call_id, []).append(t)

    agree = filled = total = 0
    hits = late = needing = 0
    false = clean = 0
    for truth in truths:
        calls = by_call.get(truth.call_id, [])
        final = calls[-1].slots if calls else {}
        for slot, expected in truth.slots.items():
            total += 1
            got = final.get(slot)
            if got is not None:
                filled += 1
                agree += got == expected
        handoff_turn = next((t.turn for t in calls if t.would_do == "handoff"), None)
        if truth.should_handoff_turn is None:
            clean += 1
            false += handoff_turn is not None
        else:
            needing += 1
            if handoff_turn is not None and handoff_turn <= truth.should_handoff_turn:
                hits += 1
            elif handoff_turn is not None:
                late += 1

    stuck = sum(_is_stuck(calls) for calls in by_call.values())
    savings = [
        baseline_ttfa_ms[(t.call_id, t.metric_turn)] - (t.jev_ms + (t.stt_ms or 0.0))
        for t in turns
        if t.would_do in ("clip", "end") and (t.call_id, t.metric_turn) in baseline_ttfa_ms
    ]
    return ShadowReport(
        calls=len(by_call),
        slot_agreement=_ratio(agree, filled),
        slot_coverage=_ratio(filled, total),
        handoff_recall=_ratio(hits, needing),
        late_handoffs=late,
        false_handoff_rate=_ratio(false, clean),
        stuck_calls=stuck,
        mean_latency_saving_ms=sum(savings) / len(savings) if savings else None,
    )


def _is_stuck(calls: list[ShadowTurn]) -> bool:
    """Phase A stayed on one step for STUCK_TURNS caller turns without ever handing off."""
    if any(t.would_do == "handoff" for t in calls):
        return False
    run = 1
    for prev, cur in zip(calls, calls[1:], strict=False):
        run = run + 1 if cur.step == prev.step else 1
        if run >= STUCK_TURNS:
            return True
    return False
