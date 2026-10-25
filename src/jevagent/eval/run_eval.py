"""Evaluate a question-set version on labelled turns (E-4).

Overall accuracy is never enough: rare classes (corrections, off-script, handoff moments) are
reported separately, a majority-class baseline is flagged, and option-order shuffles check for
position bias.
"""

import asyncio
from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from jevagent.config import Settings
from jevagent.controller.triggers import answer_triggers, uncovered_state
from jevagent.eval.dataset import LabelledTurn, full_state_for
from jevagent.jev.client import JevClient
from jevagent.jev.questions import build_questions, build_state
from jevagent.models import Handoff, JevAnswer
from jevagent.script.engine import ScriptEngine
from jevagent.script.schema import Script

OFF_SCRIPT_EXCLUDED = {"on_script", "unclear"}


class GroupMetrics(BaseModel):
    n: int
    accuracy: float
    majority_baseline: float
    beats_baseline: bool


class RareMetric(BaseModel):
    positives: int
    recall: float | None


class HandoffMetrics(BaseModel):
    should_handoff: int
    recall: float | None
    false_handoff_rate: float | None


class EvalReport(BaseModel):
    model_version: str
    n_turns: int
    groups: dict[str, GroupMetrics]
    rare: dict[str, RareMetric]
    handoff: HandoffMetrics
    position_stability: float | None
    confidences: dict[str, list[tuple[float, bool]]]  # group -> (confidence, correct), for tuning


def _group(key: str) -> str:
    return key.split(":", 1)[0]


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def simulate_handoff(
    script: Script, turn: LabelledTurn, answers: dict[str, JevAnswer], settings: Settings
) -> bool:
    state = full_state_for(script, turn)
    if uncovered_state(script, state) is not None:
        return True
    if answer_triggers(script, state, answers, settings) is not None:
        return True
    return isinstance(ScriptEngine(script, seed=0).apply_answers(state, answers), Handoff)


async def _ask(
    jev: JevClient,
    script: Script,
    turn: LabelledTurn,
    seed: int | None,
    instructions: dict[str, str] | None,
    gate: asyncio.Semaphore,
) -> dict[str, JevAnswer]:
    state = full_state_for(script, turn)
    questions = build_questions(script, state, seed, instructions)
    async with gate:
        answers = await jev.ask(build_state(script, state, turn.utterance), questions)
    return answers


async def run_eval(
    script: Script,
    turns: Sequence[LabelledTurn],
    jev: JevClient,
    settings: Settings,
    instructions: dict[str, str] | None = None,
    shuffle_seeds: Sequence[int] = (),
    concurrency: int = 8,
) -> EvalReport:
    gate = asyncio.Semaphore(concurrency)
    base = await asyncio.gather(*(_ask(jev, script, t, None, instructions, gate) for t in turns))
    shuffled = [
        await asyncio.gather(*(_ask(jev, script, t, seed, instructions, gate) for t in turns))
        for seed in shuffle_seeds
    ]

    expected: dict[str, list[str]] = defaultdict(list)
    correct: dict[str, list[bool]] = defaultdict(list)
    confidences: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    cases: dict[str, list[tuple[str, str | None]]] = defaultdict(list)  # (expected, predicted)
    for turn, answers in zip(turns, base, strict=True):
        for key, label in turn.labels.items():
            ans = answers.get(key)
            predicted = ans.selected if ans else None
            group = _group(key)
            ok = predicted == label
            expected[group].append(label)
            correct[group].append(ok)
            cases[group].append((label, predicted))
            confidences[group].append((ans.confidence if ans else 0.0, ok))

    groups = {}
    for group, flags in correct.items():
        accuracy = sum(flags) / len(flags)
        majority = Counter(expected[group]).most_common(1)[0][1] / len(flags)
        groups[group] = GroupMetrics(
            n=len(flags),
            accuracy=accuracy,
            majority_baseline=majority,
            beats_baseline=accuracy > majority,
        )

    corr = [(e, p) for e, p in cases["corrects_stored"] if e == "yes"]
    off = [(e, p) for e, p in cases["turn_type"] if e not in OFF_SCRIPT_EXCLUDED]
    rare = {
        "corrects_stored_yes": RareMetric(
            positives=len(corr), recall=_ratio(sum(p == "yes" for _, p in corr), len(corr))
        ),
        "off_script": RareMetric(
            positives=len(off),
            recall=_ratio(sum(p not in OFF_SCRIPT_EXCLUDED | {None} for _, p in off), len(off)),
        ),
    }

    predicted_handoff = [
        simulate_handoff(script, t, a, settings) for t, a in zip(turns, base, strict=True)
    ]
    should = [t.should_handoff for t in turns]
    positives = sum(should)
    negatives = len(turns) - positives
    handoff = HandoffMetrics(
        should_handoff=positives,
        recall=_ratio(
            sum(p and s for p, s in zip(predicted_handoff, should, strict=True)), positives
        ),
        false_handoff_rate=_ratio(
            sum(p and not s for p, s in zip(predicted_handoff, should, strict=True)), negatives
        ),
    )

    return EvalReport(
        model_version=settings.jev_model,
        n_turns=len(turns),
        groups=groups,
        rare=rare,
        handoff=handoff,
        position_stability=_stability(base, shuffled),
        confidences=dict(confidences),
    )


def _stability(
    base: list[dict[str, JevAnswer]], shuffled: list[list[dict[str, JevAnswer]]]
) -> float | None:
    """Fraction of (turn, question) pairs answered identically under every option order."""
    if not shuffled:
        return None
    total = agree = 0
    for i, answers in enumerate(base):
        for key, ans in answers.items():
            picks = {ans.selected} | {run[i][key].selected for run in shuffled if key in run[i]}
            total += 1
            agree += len(picks) == 1
    return _ratio(agree, total)
