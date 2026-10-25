import json
from pathlib import Path

import pytest

from jevagent.jev.candidates import (
    Candidate,
    date_matches,
    extract_amounts,
    extract_dates,
    extract_phones,
    hindi_number,
)
from jevagent.jev.fake import RuleBasedJev
from jevagent.jev.questions import build_questions, build_state, candidate_question
from jevagent.models import ESCAPE_OPTIONS, QuestionType, Slot, SlotStatus
from jevagent.script.engine import ScriptEngine
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder.yaml")


def fresh():  # type: ignore[no-untyped-def]
    return ScriptEngine(SCRIPT, seed=1).initial_state()


def test_first_step_question_set() -> None:
    keys = [q.key for q in build_questions(SCRIPT, fresh())]
    assert keys == [
        "answered_step",
        "slot_value:identity_confirmed",
        "turn_type",
        "answers_other_step",
        "slot_value:pay_when",
        "slot_value:payment_mode",
    ]  # no corrects_stored until something is stored


def test_corrects_stored_added_once_a_value_exists() -> None:
    state = fresh()
    state.slots["identity"] = Slot(value="confirmed", status=SlotStatus.FILLED, confidence=0.9)
    state.step = "pay_when"
    qs = {q.key: q for q in build_questions(SCRIPT, state)}
    assert qs["corrects_stored"].type is QuestionType.NOUL
    assert "slot_value:identity_confirmed" not in qs


def test_every_choice_has_an_escape() -> None:
    for q in build_questions(SCRIPT, fresh()):
        if q.type is QuestionType.CHOICE:
            assert ESCAPE_OPTIONS & set(q.options), q.key


def test_shuffle_is_seeded_and_off_by_default() -> None:
    base = [q.options for q in build_questions(SCRIPT, fresh())]
    assert base == [q.options for q in build_questions(SCRIPT, fresh())]
    a = [q.options for q in build_questions(SCRIPT, fresh(), shuffle_seed=3)]
    b = [q.options for q in build_questions(SCRIPT, fresh(), shuffle_seed=3)]
    assert a == b
    assert a != base
    assert all(sorted(x) == sorted(y) for x, y in zip(a, base, strict=True))


def test_state_is_minimal() -> None:
    state = fresh()
    state.recent_turns = ["caller: EARLIER-SECRET-TURN"]
    state.clips_played = ["identity.ask.1", "pay_when.ask.1"]
    state.slots["identity"] = Slot(value="confirmed", status=SlotStatus.FILLED, confidence=0.9)
    built = build_state(SCRIPT, state, "kal kar dunga")
    assert set(built) == {"current_step", "stored_values", "last_clip", "utterance"}
    assert "EARLIER-SECRET-TURN" not in json.dumps(built)
    assert built["stored_values"] == {"identity": "confirmed"}
    assert built["last_clip"] == SCRIPT.clip_text("pay_when.ask.1")
    assert built["utterance"] == "kal kar dunga"


async def test_rule_based_fake_answers_every_question() -> None:
    jev = RuleBasedJev({"identity_confirmed": {"confirmed": ["haan"]}}, {"question": ["kyun"]})
    state = fresh()
    qs = build_questions(SCRIPT, state)
    answers = await jev.ask(build_state(SCRIPT, state, "haan ji"), qs)
    assert set(answers) == {q.key for q in qs}
    assert answers["slot_value:identity_confirmed"].selected == "confirmed"
    assert answers["answered_step"].selected == "yes"
    miss = await jev.ask(build_state(SCRIPT, state, "kyun?"), qs)
    assert miss["turn_type"].selected == "question"
    assert miss["slot_value:identity_confirmed"].selected == "unclear"


# -- candidates -------------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (["paanch", "hazaar"], 5000),
        (["do", "lakh", "pachas", "hazaar"], 250000),
        (["sau"], 100),
        (["teen", "sau"], 300),
        (["ek", "hazaar", "paanch", "sau"], 1500),
        (["bees"], 20),
        ([], None),
        (["paanch", "kela"], None),
    ],
)
def test_hindi_number(words: list[str], expected: int | None) -> None:
    assert hindi_number(words) == expected


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("5000 rupaye", ["5000"]),
        ("Rs. 12,500 bhej dunga", ["12500"]),
        ("₹5k", ["5000"]),
        ("2 lakh", ["200000"]),
        ("paanch hazaar rupaye", ["5000"]),
        ("do sau", ["200"]),
        ("kal", []),
        ("15 tareekh ko", []),
    ],
)
def test_amounts(text: str, values: list[str]) -> None:
    assert [c.value for c in extract_amounts(text)] == values


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("9876543210", ["9876543210"]),
        ("mera number +91 98765-43210 hai", ["9876543210"]),
        ("098765 43210", ["9876543210"]),
        ("nau aath saat chhe paanch char teen do ek zero", ["9876543210"]),
        ("12345", []),
        ("1234567890", []),
        ("nau aath saat", []),
    ],
)
def test_phones(text: str, values: list[str]) -> None:
    assert [c.value for c in extract_phones(text)] == values


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("15/03/2025", "15-03-2025"),
        ("15-3", "15-03-????"),
        ("15th March", "15-03-????"),
        ("15 of march 2026", "15-03-2026"),
        ("march 15", "15-03-????"),
        ("15 tareekh", "15-??-????"),
        ("pandrah tareekh ko", "15-??-????"),
        ("1/2/25", "01-02-2025"),
    ],
)
def test_dates(text: str, value: str) -> None:
    assert value in [c.value for c in extract_dates(text)]


@pytest.mark.parametrize("text", ["32/01", "5/13", "kal dunga", "40 tareekh"])
def test_dates_rejected(text: str) -> None:
    assert extract_dates(text) == []


def test_date_parts_compared_in_code() -> None:
    full = extract_dates("15 march 2026")[0]
    day_only = extract_dates("15 tareekh")[0]
    other = extract_dates("16 march")[0]
    assert full.parts == {"day": 15, "month": 3, "year": 2026}
    assert date_matches(full, day_only)
    assert not date_matches(full, other)


def test_candidate_question_offers_candidates_and_escape() -> None:
    cands: list[Candidate] = extract_dates("15 march ya 16 march")
    q = candidate_question("pay_date", "which date?", cands)
    assert q.options[:-1] == [c.value for c in cands]
    assert q.options[-1] in ESCAPE_OPTIONS
