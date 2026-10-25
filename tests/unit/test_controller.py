import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from jevagent.config import Settings
from jevagent.controller.controller import Controller, PassToPhaseB, PlayClip, StartHandoff
from jevagent.controller.packet import HandoffPacket
from jevagent.controller.triggers import answer_triggers, uncovered_state
from jevagent.jev.client import JevClient
from jevagent.jev.fake import RuleBasedJev, ScriptedJev, answer
from jevagent.models import (
    CallState,
    Handoff,
    HandoffReason,
    JevAnswer,
    JevQuestion,
    JsonDict,
    Mode,
    SlotStatus,
)
from jevagent.observability.decision_log import JsonlDecisionLog, MemoryDecisionLog
from jevagent.script.engine import ScriptEngine
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder_en.yaml")
SETTINGS = Settings(_env_file=None, jev_api_key=SecretStr("k"))  # type: ignore[call-arg]
TOOLS = ["get_loan_account", "explain_charges", "schedule_callback", "transfer_to_human"]


def state_at(step: str = "pay_when") -> CallState:
    st = ScriptEngine(SCRIPT, seed=1).initial_state()
    st.step = step
    return st


# -- triggers -------------------------------------------------------------


@pytest.mark.parametrize(
    ("turn_type", "conf", "fires"),
    [
        ("objection", 0.9, True),
        ("question", 0.55, True),  # early-handoff bias: below slot thresholds, still fires
        ("wants_human", 0.4, False),
        ("wrong_person", 0.9, True),
        ("complaint", 0.9, True),
        ("on_script", 0.99, False),
        ("unclear", 0.99, False),
    ],
)
def test_turn_type_trigger(turn_type: str, conf: float, fires: bool) -> None:
    answers = {"turn_type": answer("turn_type", turn_type, conf)}
    result = answer_triggers(SCRIPT, state_at(), answers, SETTINGS)
    if fires:
        assert result == Handoff(reason=HandoffReason.OFF_SCRIPT, detail=f"turn_type={turn_type}")
    else:
        assert result is None


def test_failed_turn_type_does_not_fire() -> None:
    failed = {"turn_type": JevAnswer(key="turn_type", selected=None, confidence=0.0)}
    assert answer_triggers(SCRIPT, state_at(), failed, SETTINGS) is None
    assert answer_triggers(SCRIPT, state_at(), {}, SETTINGS) is None


@pytest.mark.parametrize(
    ("sel", "conf", "fires"), [("yes", 0.9, True), ("yes", 0.3, False), ("no", 0.9, False)]
)
def test_correction_trigger(sel: str, conf: float, fires: bool) -> None:
    answers = {"corrects_stored": answer("corrects_stored", sel, conf)}
    result = answer_triggers(SCRIPT, state_at(), answers, SETTINGS)
    expected = Handoff(reason=HandoffReason.UNRESOLVED_CORRECTION, detail="corrects_stored=yes")
    assert (result == expected) is fires


def test_uncovered_state() -> None:
    assert uncovered_state(SCRIPT, state_at()) is None
    for step in ("nonexistent", "closing"):
        hand = uncovered_state(SCRIPT, state_at(step))
        assert hand is not None and hand.reason is HandoffReason.UNCOVERED_STATE
    broken = SCRIPT.model_copy(deep=True)
    del broken.step("pay_when").questions["slot_value"]
    hand = uncovered_state(broken, state_at())
    assert hand is not None and hand.detail == "no_question=pay_when"


# -- controller simulations -----------------------------------------------


def make(jev: JevClient, log: MemoryDecisionLog | JsonlDecisionLog | None = None) -> Controller:
    return Controller(
        SCRIPT,
        jev,
        SETTINGS,
        log or MemoryDecisionLog(),
        call_id="c1",
        tools=TOOLS,
        seed=1,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def rule_jev() -> RuleBasedJev:
    return RuleBasedJev(
        slot_keywords={
            "identity_confirmed": {"confirmed": ["yes", "yes"], "denied": ["no"]},
            "pay_when": {"today": ["today"], "tomorrow": ["tomorrow"]},
            "payment_mode": {"upi": ["upi"], "net_banking": ["net banking"]},
        },
        turn_keywords={"objection": ["penalty"], "question": ["why"], "wants_human": ["agent"]},
        correction_words=["wrong", "wrong"],
    )


class CountingJev:
    def __init__(self, inner: RuleBasedJev) -> None:
        self.inner = inner
        self.requests: list[JsonDict] = []

    async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
        self.requests.append(state)
        return await self.inner.ask(state, questions)


def group(clip_id: str) -> str:
    return clip_id.split(".")[0]


async def test_happy_path_clip_sequence() -> None:
    log = MemoryDecisionLog()
    ctl = make(rule_jev(), log)
    seq = [ctl.start()]
    for utt in ["yes that is me", "i will pay tomorrow", "by upi"]:
        action = await ctl.on_caller_turn(utt)
        assert isinstance(action, PlayClip)
        seq.append(action)
    assert [group(a.clip_id) for a in seq] == ["identity", "pay_when", "payment_mode", "closing"]
    assert seq[-1].end_call and not any(a.end_call for a in seq[:-1])
    assert ctl.state.slots["pay_date_bucket"].value == "tomorrow"
    assert len(log.records) == 3


async def test_reask_then_handoff_on_unclear() -> None:
    ctl = make(rule_jev())
    ctl.start()
    a1 = await ctl.on_caller_turn("hmm")
    assert isinstance(a1, PlayClip) and "re_ask" in a1.clip_id
    a2 = await ctl.on_caller_turn("hmm hmm")
    assert isinstance(a2, StartHandoff)
    assert a2.packet.handoff_reason == HandoffReason.RETRY_LIMIT.value
    assert ctl.state.mode is Mode.B


async def test_objection_hands_off_with_packet_matching_prd_shape() -> None:
    ctl = make(rule_jev())
    ctl.start()
    await ctl.on_caller_turn("yes that is me")
    action = await ctl.on_caller_turn("first tell me about the penalty, why was it charged?")
    assert isinstance(action, StartHandoff)
    packet = action.packet
    assert packet.handoff_reason == "turn_type=objection"
    assert packet.slots["identity"].value == "confirmed"
    assert packet.slots["pay_date_bucket"].status is not SlotStatus.FILLED
    assert packet.missing == ["pay_date_bucket", "payment_mode"]
    assert packet.tools == TOOLS
    assert packet.clips_played[0].startswith("identity.ask")
    assert packet.recent_turns[-1].startswith("caller: first tell me about the penalty")
    assert any(t.startswith("agent: ") for t in packet.recent_turns)
    assert HandoffPacket.model_validate(json.loads(packet.model_dump_json())) == packet
    assert set(packet.model_dump()) == {
        "handoff_reason",
        "slots",
        "missing",
        "recent_turns",
        "clips_played",
        "tools",
    }


async def test_correction_hands_off() -> None:
    ctl = make(rule_jev())
    ctl.start()
    await ctl.on_caller_turn("yes that is me")
    action = await ctl.on_caller_turn("wrong, not tomorrow")
    assert isinstance(action, StartHandoff)
    assert action.packet.handoff_reason == "corrects_stored=yes"


async def test_no_way_back_to_phase_a_and_jev_not_called() -> None:
    jev = CountingJev(rule_jev())
    log = MemoryDecisionLog()
    ctl = make(jev, log)
    ctl.start()
    await ctl.on_caller_turn("yes")
    await ctl.on_caller_turn("why?")
    assert ctl.state.mode is Mode.B
    calls, records = len(jev.requests), len(log.records)
    for utt in ["yes", "tomorrow", "upi"]:  # would all fill slots if Phase A were still live
        assert await ctl.on_caller_turn(utt) == PassToPhaseB(utterance=utt)
    assert ctl.state.mode is Mode.B
    assert len(jev.requests) == calls and len(log.records) == records


async def test_early_fill_through_controller() -> None:
    jev = ScriptedJev(
        turns=[
            {
                "answered_step": answer("answered_step", "yes"),
                "slot_value:identity_confirmed": answer(
                    "slot_value:identity_confirmed", "confirmed"
                ),
                "answers_other_step": answer("answers_other_step", "payment_mode"),
                "slot_value:payment_mode": answer("slot_value:payment_mode", "upi"),
                "slot_value:pay_when": answer("slot_value:pay_when", "not_stated"),
                "turn_type": answer("turn_type", "on_script"),
            }
        ]
    )
    ctl = make(jev)
    ctl.start()
    await ctl.on_caller_turn("yes, I will pay by upi")
    assert ctl.state.slots["payment_mode"].value == "upi"
    assert ctl.state.step == "pay_when"


async def test_jev_returning_nothing_reasks_then_hands_off() -> None:
    ctl = make(ScriptedJev(handler=lambda s, q: {}))
    ctl.start()
    assert isinstance(await ctl.on_caller_turn("yes"), PlayClip)
    action = await ctl.on_caller_turn("yes")
    assert isinstance(action, StartHandoff)
    assert action.packet.handoff_reason == HandoffReason.LOW_CONFIDENCE.value


async def test_caller_after_closing_hands_off_without_calling_jev() -> None:
    log = MemoryDecisionLog()
    ctl = make(rule_jev(), log)
    ctl.start()
    for utt in ["yes", "today", "upi"]:
        await ctl.on_caller_turn(utt)
    action = await ctl.on_caller_turn("one minute")
    assert isinstance(action, StartHandoff)
    assert action.packet.handoff_reason == "terminal_step=closing"
    assert log.records[-1].questions == []


async def test_recent_turns_bounded() -> None:
    ctl = make(rule_jev())
    ctl.start()
    await ctl.on_caller_turn("hmm")
    await ctl.on_caller_turn("hmm")
    assert len(ctl.state.recent_turns) <= 6


# -- decision log ----------------------------------------------------------


async def test_decision_log_is_complete(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "d.jsonl"
    ctl = make(rule_jev(), JsonlDecisionLog(path))
    ctl.start()
    await ctl.on_caller_turn("yes that is me")
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["call_id"] == "c1" and rec["turn"] == 1
    assert rec["model_version"] == "jev-1.14.0"
    assert rec["timestamp"].startswith("2026-01-01")
    assert rec["state_sent"]["utterance"] == "yes that is me"
    assert {q["key"] for q in rec["questions"]} == set(rec["answers"])
    assert rec["answers"]["answered_step"]["confidence"] > 0
    assert rec["action"]["type"] == "PlayClip"
    assert rec["action"]["clip_id"].startswith("pay_when")
