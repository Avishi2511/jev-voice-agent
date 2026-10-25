from pathlib import Path

import pytest

from jevagent.models import (
    Advance,
    CallState,
    Handoff,
    HandoffReason,
    JevAnswer,
    ReAsk,
    SlotStatus,
)
from jevagent.script.engine import ScriptEngine
from jevagent.script.loader import ScriptError, load_script
from jevagent.script.schema import Script

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "emi_reminder_en.yaml"


@pytest.fixture
def script() -> Script:
    return load_script(SCRIPT_PATH)


@pytest.fixture
def engine(script: Script) -> ScriptEngine:
    return ScriptEngine(script, seed=1)


def ans(key: str, selected: str | None, conf: float = 0.95) -> JevAnswer:
    return JevAnswer(key=key, selected=selected, confidence=conf)


def turn(
    step: str, value: str | None, conf: float = 0.95, answered: str = "yes"
) -> dict[str, JevAnswer]:
    return {
        "answered_step": ans("answered_step", answered, conf),
        f"slot_value:{step}": ans(f"slot_value:{step}", value, conf),
    }


def clip_ids(script: Script) -> set[str]:
    return {
        c.id
        for s in script.steps
        for g in (s.clips.ask, s.clips.re_ask, s.clips.confirm)
        for c in g
    }


def test_advance_on_fill(engine: ScriptEngine, script: Script) -> None:
    state = engine.initial_state()
    decision = engine.apply_answers(state, turn("identity_confirmed", "confirmed"))
    assert isinstance(decision, Advance)
    assert decision.step == "pay_when" == state.step
    assert decision.clip_id in clip_ids(script)  # CT-4: clips come from the script only
    assert state.slots["identity"].status is SlotStatus.FILLED
    assert state.slots["identity"].value == "confirmed"


def test_full_happy_path_ends_terminal(engine: ScriptEngine) -> None:
    state = engine.initial_state()
    engine.apply_answers(state, turn("identity_confirmed", "confirmed"))
    engine.apply_answers(state, turn("pay_when", "tomorrow"))
    decision = engine.apply_answers(state, turn("payment_mode", "upi"))
    assert isinstance(decision, Advance)
    assert decision.terminal and decision.step == "closing"
    with pytest.raises(ValueError):
        engine.apply_answers(state, {})


@pytest.mark.parametrize("selected", ["unclear", "none", "not_stated"])
def test_escape_does_not_advance(engine: ScriptEngine, selected: str) -> None:
    state = engine.initial_state()
    decision = engine.apply_answers(state, turn("identity_confirmed", selected))
    assert isinstance(decision, ReAsk)
    assert state.step == "identity_confirmed"
    assert state.slots["identity"].status is not SlotStatus.FILLED


def test_answered_step_no_does_not_advance(engine: ScriptEngine) -> None:
    state = engine.initial_state()
    decision = engine.apply_answers(state, turn("identity_confirmed", "confirmed", answered="no"))
    assert isinstance(decision, ReAsk)
    assert state.step == "identity_confirmed"


def test_reask_once_then_handoff_low_confidence(engine: ScriptEngine) -> None:
    state = engine.initial_state()
    low = turn("identity_confirmed", "confirmed", conf=0.3)
    first = engine.apply_answers(state, low)
    assert isinstance(first, ReAsk)
    assert state.retries["identity_confirmed"] == 1
    second = engine.apply_answers(state, low)
    assert second == Handoff(reason=HandoffReason.LOW_CONFIDENCE)


def test_retry_limit_when_confidently_unclear(engine: ScriptEngine) -> None:
    state = engine.initial_state()
    engine.apply_answers(state, turn("identity_confirmed", "unclear"))
    decision = engine.apply_answers(state, turn("identity_confirmed", "unclear"))
    assert decision == Handoff(reason=HandoffReason.RETRY_LIMIT)


def test_missing_answers_treated_as_low_confidence(engine: ScriptEngine) -> None:
    state = engine.initial_state()
    assert isinstance(engine.apply_answers(state, {}), ReAsk)
    assert engine.apply_answers(state, {}) == Handoff(reason=HandoffReason.LOW_CONFIDENCE)


def test_refused_and_not_stated_statuses(engine: ScriptEngine) -> None:
    state = engine.initial_state()
    engine.apply_answers(state, turn("identity_confirmed", "refused"))
    assert state.slots["identity"].status is SlotStatus.REFUSED
    state = engine.initial_state()
    engine.apply_answers(state, turn("identity_confirmed", "not_stated"))
    assert state.slots["identity"].status is SlotStatus.EMPTY


def test_per_step_max_retries(script: Script) -> None:
    script.step("identity_confirmed").max_retries = 0
    engine = ScriptEngine(script, seed=1)
    state = engine.initial_state()
    decision = engine.apply_answers(state, turn("identity_confirmed", "unclear"))
    assert isinstance(decision, Handoff)


def test_early_slot_fill_skips_step(engine: ScriptEngine) -> None:
    state = engine.initial_state()
    answers = turn("identity_confirmed", "confirmed")
    answers["answers_other_step"] = ans("answers_other_step", "payment_mode")
    answers["slot_value:payment_mode"] = ans("slot_value:payment_mode", "upi")
    decision = engine.apply_answers(state, answers)
    assert isinstance(decision, Advance)
    assert decision.step == "pay_when"  # pay_when still unfilled
    engine.apply_answers(state, turn("pay_when", "today"))
    assert state.step == "closing"  # payment_mode skipped


def test_early_fill_ignored_when_low_confidence_or_none(engine: ScriptEngine) -> None:
    for other in (
        ans("answers_other_step", "none"),
        ans("answers_other_step", "payment_mode", 0.2),
        ans("answers_other_step", "bogus_step"),
    ):
        state = engine.initial_state()
        answers = turn("identity_confirmed", "confirmed")
        answers["answers_other_step"] = other
        answers["slot_value:payment_mode"] = ans("slot_value:payment_mode", "upi")
        engine.apply_answers(state, answers)
        assert state.slots["payment_mode"].status is SlotStatus.EMPTY


def test_clip_variants_not_repeated_consecutively(script: Script) -> None:
    engine = ScriptEngine(script, seed=0)
    state = engine.initial_state()
    for _ in range(20):
        state.step = "pay_when"
        state.retries.clear()
        engine.apply_answers(state, turn("pay_when", "unclear"))
    played = state.clips_played
    assert all(a != b for a, b in zip(played, played[1:], strict=False))


def test_deterministic_with_seed(script: Script) -> None:
    def run() -> list[str]:
        eng = ScriptEngine(script, seed=42)
        st = eng.initial_state()
        eng.opening_clip(st)
        eng.apply_answers(st, turn("identity_confirmed", "confirmed"))
        eng.apply_answers(st, turn("pay_when", "today"))
        return st.clips_played

    assert run() == run()


def test_on_filled_override(script: Script) -> None:
    script.step("identity_confirmed").on_filled = "closing"
    engine = ScriptEngine(script, seed=1)
    state = engine.initial_state()
    decision = engine.apply_answers(state, turn("identity_confirmed", "confirmed"))
    assert isinstance(decision, Advance) and decision.step == "closing"


def test_engine_has_no_move_on_input() -> None:
    # CT-3 guard: the only path to Advance is a filled slot.
    state = CallState(step="identity_confirmed")
    engine = ScriptEngine(load_script(SCRIPT_PATH), seed=1)
    state.slots = engine.initial_state().slots
    decision = engine.apply_answers(
        state, {"turn_type": ans("turn_type", "on_script"), "move_on": ans("move_on", "yes")}
    )
    assert not isinstance(decision, Advance)


# -- loader ---------------------------------------------------------------


def test_example_script_loads(script: Script) -> None:
    assert [s.id for s in script.steps][0] == "identity_confirmed"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "s.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def _mutated(tmp_path: Path, old: str, new: str) -> Path:
    return _write(tmp_path, SCRIPT_PATH.read_text(encoding="utf-8").replace(old, new, 1))


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScriptError):
        load_script(tmp_path / "nope.yaml")


def test_missing_default_threshold_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="default"):
        load_script(_mutated(tmp_path, "default: 0.7", "other: 0.7"))


def test_choice_without_escape_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="escape"):
        load_script(_mutated(tmp_path, "[confirmed, denied, unclear]", "[confirmed, denied]"))


def test_duplicate_clip_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="unique"):
        load_script(_mutated(tmp_path, "identity.ask.2", "identity.ask.1"))


def test_step_without_ask_clips_rejected(tmp_path: Path) -> None:
    text = "name: x\nthresholds: {default: 0.7}\nsteps:\n  - id: a\n    clips: {ask: []}\n"
    with pytest.raises(ScriptError, match="ask"):
        load_script(_write(tmp_path, text))


def test_slot_step_without_reask_rejected(tmp_path: Path) -> None:
    text = (
        "name: x\nthresholds: {default: 0.7}\nsteps:\n"
        "  - id: a\n    slot: s\n    clips: {ask: [{id: a1, text: hi}]}\n"
    )
    with pytest.raises(ScriptError, match="re_ask"):
        load_script(_write(tmp_path, text))


def test_unknown_on_filled_rejected(tmp_path: Path) -> None:
    text = (
        "name: x\nthresholds: {default: 0.7}\nsteps:\n"
        "  - id: a\n    on_filled: zzz\n    clips: {ask: [{id: a1, text: hi}]}\n"
    )
    with pytest.raises(ScriptError, match="on_filled"):
        load_script(_write(tmp_path, text))


def test_last_step_with_slot_rejected(tmp_path: Path) -> None:
    text = """
name: x
thresholds: {default: 0.7}
steps:
  - id: a
    slot: s
    clips: {ask: [{id: a1, text: hi}], re_ask: [{id: a2, text: hi}]}
    questions: {slot_value: {type: choice, instruction: q, options: [unclear]}}
"""
    with pytest.raises(ScriptError, match="terminal"):
        load_script(_write(tmp_path, text))
