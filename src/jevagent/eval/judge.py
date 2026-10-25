"""Post-call Jev judge (E-6). Unlike the live path, the judge may read the whole transcript.

Jev scores what only hindsight can: was the handoff at the right turn, did the caller have to
repeat themselves. Slot accuracy and latency are computed in code from CRM truth and metrics.
"""

from pydantic import BaseModel

from jevagent.jev.client import JevClient
from jevagent.models import JevQuestion, QuestionType
from jevagent.observability.report import percentile

TIMING_OPTIONS = ["too_early", "right_time", "too_late", "not_needed", "unclear"]


class JudgeResult(BaseModel):
    call_id: str
    handoff_timing: str | None
    handoff_timing_confidence: float
    caller_repeated: float | None  # P(caller repeated themselves)
    slot_accuracy: float | None
    ttfa_p50_ms: float | None
    judged_ok: bool


def judge_questions() -> list[JevQuestion]:
    return [
        JevQuestion(
            key="handoff_timing",
            type=QuestionType.CHOICE,
            instruction=(
                "The transcript is a phone call where a scripted agent may have handed the call "
                "to a more capable agent at caller turn `handoff_turn` (null if it never did). "
                "Was the handoff at the right moment? too_early: the scripted agent could still "
                "have handled it. too_late: it should have handed off earlier, or kept going "
                "when the caller was clearly off-script. not_needed: no handoff was needed and "
                "none happened."
            ),
            options=TIMING_OPTIONS,
        ),
        JevQuestion(
            key="caller_repeated",
            type=QuestionType.NOUL,
            instruction=(
                "After the handoff, did the caller have to repeat or re-explain something they "
                "had already said before the handoff? False if there was no handoff."
            ),
        ),
    ]


async def judge_call(
    jev: JevClient,
    call_id: str,
    transcript: list[str],
    handoff_turn: int | None,
    slots: dict[str, str | None] | None = None,
    crm_truth: dict[str, str] | None = None,
    ttfa_ms: list[float] | None = None,
) -> JudgeResult:
    answers = await jev.ask(
        {"transcript": transcript, "handoff_turn": handoff_turn}, judge_questions()
    )
    timing = answers["handoff_timing"]
    repeated = answers["caller_repeated"]
    judged_ok = timing.selected is not None and repeated.selected is not None

    slot_accuracy = None
    if slots is not None and crm_truth:
        slot_accuracy = sum(slots.get(k) == v for k, v in crm_truth.items()) / len(crm_truth)

    return JudgeResult(
        call_id=call_id,
        handoff_timing=timing.selected,
        handoff_timing_confidence=timing.confidence,
        caller_repeated=repeated.probabilities.get("yes") if repeated.selected else None,
        slot_accuracy=slot_accuracy,
        ttfa_p50_ms=percentile(ttfa_ms or [], 50),
        judged_ok=judged_ok,
    )
