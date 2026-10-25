"""Handoff triggers computed in code (H-1). No single "hand off now" model call.

Triggers 3 (low confidence after one re-ask) and 4 (retry limit) come from the script
engine, which owns retry counters. This module covers 1, 2 and 5.
"""

from jevagent.config import Settings
from jevagent.models import CallState, Handoff, HandoffReason, JevAnswer
from jevagent.script.schema import Script

_NOT_OFF_SCRIPT = {"on_script", "unclear", None}


def uncovered_state(script: Script, state: CallState) -> Handoff | None:
    """Trigger 5: the call is somewhere the script does not handle."""
    try:
        step = script.step(state.step)
    except KeyError:
        return Handoff(reason=HandoffReason.UNCOVERED_STATE, detail=f"unknown_step={state.step}")
    if step.slot is None:  # caller kept talking after the closing clip
        return Handoff(reason=HandoffReason.UNCOVERED_STATE, detail=f"terminal_step={step.id}")
    if "slot_value" not in step.questions:
        return Handoff(reason=HandoffReason.UNCOVERED_STATE, detail=f"no_question={step.id}")
    return None


def answer_triggers(
    script: Script, state: CallState, answers: dict[str, JevAnswer], settings: Settings
) -> Handoff | None:
    """Triggers 1 and 2, from this turn's answers."""
    turn = answers.get("turn_type")
    if (
        turn is not None
        and turn.selected not in _NOT_OFF_SCRIPT
        and turn.confidence >= settings.off_script_threshold
    ):
        return Handoff(reason=HandoffReason.OFF_SCRIPT, detail=f"turn_type={turn.selected}")

    correction = answers.get("corrects_stored")
    threshold = script.step(state.step).thresholds.get(
        "corrects_stored", script.thresholds.get("corrects_stored", script.thresholds["default"])
    )
    # v1 has no automatic correction resolver, so every confident correction is unresolved.
    if (
        correction is not None
        and correction.selected == "yes"
        and correction.confidence >= threshold
    ):
        return Handoff(reason=HandoffReason.UNRESOLVED_CORRECTION, detail="corrects_stored=yes")
    return None
