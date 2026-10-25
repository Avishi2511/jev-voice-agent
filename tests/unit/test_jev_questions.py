import json
from pathlib import Path

import pytest

from jevagent.jev.candidates import (
    Candidate,
    date_matches,
    extract_amounts,
    extract_dates,
    extract_phones,
    number_from_words,
)
from jevagent.jev.fake import RuleBasedJev
from jevagent.jev.questions import build_questions, build_state, candidate_question
from jevagent.models import ESCAPE_OPTIONS, QuestionType, Slot, SlotStatus
from jevagent.script.engine import ScriptEngine
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder_en.yaml")


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
    built = build_state(SCRIPT, state, "i will pay tomorrow")
    assert set(built) == {"current_step", "stored_values", "last_clip", "utterance"}
    assert "EARLIER-SECRET-TURN" not in json.dumps(built)
    assert built["stored_values"] == {"identity": "confirmed"}
    assert built["last_clip"] == SCRIPT.clip_text("pay_when.ask.1")
    assert built["utterance"] == "i will pay tomorrow"


async def test_rule_based_fake_answers_every_question() -> None:
    jev = RuleBasedJev({"identity_confirmed": {"confirmed": ["yes"]}}, {"question": ["why"]})
    state = fresh()
    qs = build_questions(SCRIPT, state)
    answers = await jev.ask(build_state(SCRIPT, state, "yes that is me"), qs)
    assert set(answers) == {q.key for q in qs}
    assert answers["slot_value:identity_confirmed"].selected == "confirmed"
    assert answers["answered_step"].selected == "yes"
    miss = await jev.ask(build_state(SCRIPT, state, "why?"), qs)
    assert miss["turn_type"].selected == "question"
    assert miss["slot_value:identity_confirmed"].selected == "unclear"


# -- candidates -------------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (["five", "thousand"], 5000),
        (["two", "lakh", "fifty", "thousand"], 250000),
        (["hundred"], 100),
        (["three", "hundred"], 300),
        (["one", "thousand", "five", "hundred"], 1500),
        (["twenty", "five"], 25),
        ([], None),
        (["five", "banana"], None),
    ],
)
def test_number_from_words(words: list[str], expected: int | None) -> None:
    assert number_from_words(words) == expected


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("5000 rupees", ["5000"]),
        ("Rs. 12,500 by tomorrow", ["12500"]),
        ("₹5k", ["5000"]),
        ("2 lakh", ["200000"]),
        ("five thousand rupees", ["5000"]),
        ("two hundred", ["200"]),
        ("tomorrow", []),
        ("on the 15th", []),
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
        ("nine eight seven six five four three two one zero", ["9876543210"]),
        ("12345", []),
        ("1234567890", []),
        ("nine eight seven", []),
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
        ("on the 15th", "15-??-????"),
        ("the 3rd of march", "03-03-????"),
        ("1/2/25", "01-02-2025"),
    ],
)
def test_dates(text: str, value: str) -> None:
    assert value in [c.value for c in extract_dates(text)]


@pytest.mark.parametrize("text", ["32/01", "5/13", "I will pay tomorrow", "the 40th"])
def test_dates_rejected(text: str) -> None:
    assert extract_dates(text) == []


def test_date_parts_compared_in_code() -> None:
    full = extract_dates("15 march 2026")[0]
    day_only = extract_dates("on the 15th")[0]
    other = extract_dates("16 march")[0]
    assert full.parts == {"day": 15, "month": 3, "year": 2026}
    assert date_matches(full, day_only)
    assert not date_matches(full, other)


def test_candidate_question_offers_candidates_and_escape() -> None:
    cands: list[Candidate] = extract_dates("15 march ya 16 march")
    q = candidate_question("pay_date", "which date?", cands)
    assert q.options[:-1] == [c.value for c in cands]
    assert q.options[-1] in ESCAPE_OPTIONS
