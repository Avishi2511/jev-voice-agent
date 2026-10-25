"""Standard per-turn question set (A-3), escape options (A-4), minimal state (A-2)."""

import random

from jevagent.jev.candidates import Candidate
from jevagent.models import ESCAPE_OPTIONS, CallState, JevQuestion, JsonDict, QuestionType
from jevagent.script.engine import NO_STEP
from jevagent.script.schema import QuestionSpec, Script, Step

TURN_TYPES = ["on_script", "question", "objection", "complaint", "wants_human", "wrong_person"]


def build_state(script: Script, state: CallState, utterance: str) -> JsonDict:
    """Minimal state only: current step, stored values, last clip, latest utterance."""
    last_clip = state.clips_played[-1] if state.clips_played else None
    return {
        "current_step": state.step,
        "stored_values": {k: s.value for k, s in state.slots.items() if s.value is not None},
        "last_clip": script.clip_text(last_clip) if last_clip else None,
        "utterance": utterance,
    }


def _options(spec: QuestionSpec, rng: random.Random | None) -> list[str]:
    options = list(spec.options)
    if rng is not None:
        rng.shuffle(options)  # testing mode: counter first-option bias
    return options


def _later_unfilled(script: Script, state: CallState, current: Step) -> list[Step]:
    steps = script.steps
    return [
        s
        for s in steps[steps.index(current) + 1 :]
        if s.slot and state.slots[s.slot].value is None and "slot_value" in s.questions
    ]


def build_questions(
    script: Script,
    state: CallState,
    shuffle_seed: int | None = None,
    instructions: dict[str, str] | None = None,
) -> list[JevQuestion]:
    """All questions for one turn, to be sent in a single request.

    `instructions` overrides question wording by key (a question-set version, E-5).
    """
    qs = _standard_questions(script, state, shuffle_seed)
    if not instructions:
        return qs
    return [
        q.model_copy(update={"instruction": instructions.get(q.key, q.instruction)}) for q in qs
    ]


def _standard_questions(
    script: Script, state: CallState, shuffle_seed: int | None
) -> list[JevQuestion]:
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    step = script.step(state.step)
    spec = step.questions["slot_value"]
    later = _later_unfilled(script, state, step)

    qs = [
        JevQuestion(
            key="answered_step",
            type=QuestionType.CHOICE,
            instruction="Did the caller's latest utterance answer the question that the agent "
            "asked in last_clip? Judge only this utterance.",
            options=_shuffled(["yes", "no", "unclear"], rng),
        ),
        JevQuestion(
            key=f"slot_value:{step.id}",
            type=spec.type,
            instruction=spec.instruction,
            options=_options(spec, rng),
        ),
        JevQuestion(
            key="turn_type",
            type=QuestionType.CHOICE,
            instruction="Classify the caller's latest utterance. on_script means a direct answer "
            "or acknowledgement relevant to last_clip.",
            options=_shuffled([*TURN_TYPES, "unclear"], rng),
        ),
    ]
    if any(s.value is not None for s in state.slots.values()):
        qs.append(
            JevQuestion(
                key="corrects_stored",
                type=QuestionType.NOUL,
                instruction="Does the caller's latest utterance correct or contradict any of "
                "stored_values?",
            )
        )
    if later:
        qs.append(
            JevQuestion(
                key="answers_other_step",
                type=QuestionType.CHOICE,
                instruction="Does the caller's latest utterance also answer one of these other "
                "steps? Pick its id, or none.",
                options=_shuffled([s.id for s in later] + [NO_STEP], rng),
            )
        )
        for s in later:
            later_spec = s.questions["slot_value"]
            qs.append(
                JevQuestion(
                    key=f"slot_value:{s.id}",
                    type=later_spec.type,
                    instruction=later_spec.instruction,
                    options=_options(later_spec, rng),
                )
            )
    return qs


def _shuffled(options: list[str], rng: random.Random | None) -> list[str]:
    if rng is not None:
        rng.shuffle(options)
    return options


def candidate_question(
    key: str, instruction: str, candidates: list[Candidate], rng: random.Random | None = None
) -> JevQuestion:
    """A-5: Jev picks among parser candidates; free-form values never come from Jev."""
    options = [c.value for c in candidates]
    if rng is not None:
        rng.shuffle(options)
    escape = next(iter(sorted(ESCAPE_OPTIONS - set(options))), "none")
    return JevQuestion(
        key=key,
        type=QuestionType.CHOICE,
        instruction=instruction,
        options=[*options, escape],
        descriptions={c.value: f"the caller said: {c.text}" for c in candidates},
    )
