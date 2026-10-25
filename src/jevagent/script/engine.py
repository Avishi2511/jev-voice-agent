"""Step/slot state machine. Pure code, no I/O.

The engine has no "move on" input from Jev (CT-3): a step advances only when its
slot fills. Jev never selects clips or words (CT-4); clips come from the script.

Answer keys consumed per turn:
  answered_step          choice yes/no/unclear: did the caller answer the current step
  slot_value:<step_id>   choice over the step's slot options (current and later steps)
  answers_other_step     choice over step ids + "none" (early slot fill, A-7)
"""

import random

from jevagent.models import (
    ESCAPE_OPTIONS,
    Advance,
    CallState,
    Decision,
    Handoff,
    HandoffReason,
    JevAnswer,
    ReAsk,
    Slot,
    SlotStatus,
)
from jevagent.script.schema import Clip, Script, Step

NO_STEP = "none"


class ScriptEngine:
    def __init__(self, script: Script, seed: int | None = None) -> None:
        self.script = script
        self._rng = random.Random(seed)

    def initial_state(self) -> CallState:
        return CallState(
            step=self.script.steps[0].id,
            slots={s.slot: Slot() for s in self.script.steps if s.slot},
        )

    def opening_clip(self, state: CallState) -> str:
        return self._pick(self.script.step(state.step).clips.ask, state)

    def apply_answers(self, state: CallState, answers: dict[str, JevAnswer]) -> Decision:
        step = self.script.step(state.step)
        if step.slot is None:
            raise ValueError(f"step {step.id!r} is terminal and takes no answers")

        self._fill_early_slots(state, step, answers)
        value = answers.get(f"slot_value:{step.id}")
        answered = answers.get("answered_step")

        if self._is_yes(step, answered) and self._valid_value(step, value):
            assert value is not None
            self._set_filled(state, step, value)
            return self._advance(state, step)

        status, low_confidence = self._classify_miss(step, value, answered)
        state.slots[step.slot] = Slot(status=status, confidence=value.confidence if value else 0.0)
        return self._retry_or_handoff(state, step, low_confidence)

    # -- slot filling -------------------------------------------------------

    def _threshold(self, step: Step, key: str) -> float:
        return step.thresholds.get(
            key, self.script.thresholds.get(key, self.script.thresholds["default"])
        )

    def _is_yes(self, step: Step, answered: JevAnswer | None) -> bool:
        return (
            answered is not None
            and answered.selected == "yes"
            and answered.confidence >= self._threshold(step, "answered_step")
        )

    def _valid_value(self, step: Step, value: JevAnswer | None) -> bool:
        return (
            value is not None
            and value.selected is not None
            and value.selected not in ESCAPE_OPTIONS
            and value.selected != "refused"
            and value.confidence >= self._threshold(step, "slot_value")
        )

    def _classify_miss(
        self, step: Step, value: JevAnswer | None, answered: JevAnswer | None
    ) -> tuple[SlotStatus, bool]:
        """Return (slot status, whether the miss was due to low/missing confidence)."""
        if value is not None and value.selected == "refused":
            return SlotStatus.REFUSED, False
        if value is not None and value.selected == "not_stated":
            return SlotStatus.EMPTY, False
        if value is not None and value.selected in ESCAPE_OPTIONS:
            return SlotStatus.UNCLEAR, False
        if (
            answered is not None
            and answered.selected == "no"
            and (answered.confidence >= self._threshold(step, "answered_step"))
        ):
            return SlotStatus.EMPTY, False
        return SlotStatus.UNCLEAR, True

    @staticmethod
    def _set_filled(state: CallState, step: Step, value: JevAnswer) -> None:
        assert step.slot is not None
        state.slots[step.slot] = Slot(
            value=value.selected, confidence=value.confidence, status=SlotStatus.FILLED
        )

    def _fill_early_slots(
        self, state: CallState, current: Step, answers: dict[str, JevAnswer]
    ) -> None:
        """A-7: a caller answering a later step fills that slot too."""
        other = answers.get("answers_other_step")
        if other is None or other.selected in (None, NO_STEP) or other.selected == current.id:
            return
        if other.confidence < self._threshold(current, "answers_other_step"):
            return
        try:
            target = self.script.step(other.selected or "")
        except KeyError:
            return
        value = answers.get(f"slot_value:{target.id}")
        if target.slot and self._valid_value(target, value):
            assert value is not None
            self._set_filled(state, target, value)

    # -- transitions --------------------------------------------------------

    def _advance(self, state: CallState, step: Step) -> Advance:
        nxt = self._next_step(state, step)
        state.step = nxt.id
        clip = self._pick(nxt.clips.ask, state)
        return Advance(step=nxt.id, clip_id=clip, terminal=nxt.slot is None)

    def _next_step(self, state: CallState, step: Step) -> Step:
        if step.on_filled:
            return self.script.step(step.on_filled)
        steps = self.script.steps
        for candidate in steps[steps.index(step) + 1 :]:
            if candidate.slot is None:
                return candidate
            if state.slots[candidate.slot].status is not SlotStatus.FILLED:
                return candidate
        return steps[-1]  # pragma: no cover - loader guarantees a terminal last step

    def _retry_or_handoff(self, state: CallState, step: Step, low_confidence: bool) -> Decision:
        limit = step.max_retries if step.max_retries is not None else self.script.max_retries
        used = state.retries.get(step.id, 0)
        if used >= limit:
            reason = HandoffReason.LOW_CONFIDENCE if low_confidence else HandoffReason.RETRY_LIMIT
            return Handoff(reason=reason)
        state.retries[step.id] = used + 1
        return ReAsk(clip_id=self._pick(step.clips.re_ask, state))

    def _pick(self, variants: list[Clip], state: CallState) -> str:
        """Random variant that differs from the one played last (A-8)."""
        last = state.clips_played[-1] if state.clips_played else None
        pool = [c for c in variants if c.id != last] or variants
        chosen = self._rng.choice(pool).id
        state.clips_played.append(chosen)
        return chosen
