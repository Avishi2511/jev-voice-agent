"""Turn-level labelled dataset (E-1) with validation and a freeze step (E-2)."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from jevagent.jev.questions import build_questions
from jevagent.models import CallState, QuestionType, Slot, SlotStatus
from jevagent.script.schema import Script


class DatasetError(ValueError):
    pass


class LabelledTurn(BaseModel):
    call_id: str
    turn: int
    step: str
    utterance: str
    stored: dict[str, str] = Field(default_factory=dict)  # slot -> value stored before the turn
    retries_used: int = 0
    last_clip: str | None = None
    labels: dict[str, str]  # question key -> expected option
    should_handoff: bool


def state_for(turn: LabelledTurn) -> CallState:
    return CallState(
        step=turn.step,
        slots={
            name: Slot(value=value, confidence=1.0, status=SlotStatus.FILLED)
            for name, value in turn.stored.items()
        },
        retries={turn.step: turn.retries_used},
        clips_played=[turn.last_clip] if turn.last_clip else [],
    )


def full_state_for(script: Script, turn: LabelledTurn) -> CallState:
    """Like state_for, with every script slot present so question building works."""
    state = state_for(turn)
    for step in script.steps:
        if step.slot and step.slot not in state.slots:
            state.slots[step.slot] = Slot()
    return state


def validate_turn(script: Script, turn: LabelledTurn) -> None:
    where = f"{turn.call_id}#{turn.turn}"
    try:
        script.step(turn.step)
    except KeyError:
        raise DatasetError(f"{where}: unknown step {turn.step!r}") from None
    questions = {q.key: q for q in build_questions(script, full_state_for(script, turn))}
    for key, label in turn.labels.items():
        q = questions.get(key)
        if q is None:
            raise DatasetError(f"{where}: label for unknown question {key!r}")
        allowed = ["yes", "no"] if q.type is QuestionType.NOUL else q.options
        if label not in allowed:
            raise DatasetError(f"{where}: label {label!r} for {key!r} not in {allowed}")
    for required in ("answered_step", "turn_type"):
        if required not in turn.labels:
            raise DatasetError(f"{where}: missing required label {required!r}")
    if turn.labels["answered_step"] == "yes" and f"slot_value:{turn.step}" not in turn.labels:
        raise DatasetError(f"{where}: answered_step=yes needs a slot_value label")


def load_turns(path: str | Path, script: Script) -> list[LabelledTurn]:
    turns: list[LabelledTurn] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            turn = LabelledTurn.model_validate_json(line)
        except ValueError as exc:
            raise DatasetError(f"{path}:{n}: {exc}") from exc
        validate_turn(script, turn)
        turns.append(turn)
    if not turns:
        raise DatasetError(f"{path}: no turns")
    return turns


def save_turns(path: str | Path, turns: list[LabelledTurn]) -> None:
    Path(path).write_text("".join(t.model_dump_json() + "\n" for t in turns), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path: str | Path) -> Path:
    """Record a hash of the labels; later loads with `verify_frozen` fail if they changed."""
    p = Path(path)
    lock = p.with_suffix(p.suffix + ".lock")
    lock.write_text(_digest(p) + "\n", encoding="utf-8")
    return lock


def verify_frozen(path: str | Path) -> None:
    p = Path(path)
    lock = p.with_suffix(p.suffix + ".lock")
    if not lock.exists():
        raise DatasetError(f"{p} is not frozen; run freeze() before evaluating (E-2)")
    if lock.read_text(encoding="utf-8").strip() != _digest(p):
        raise DatasetError(f"{p} changed after it was frozen")
