from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr

from jevagent.config import Settings
from jevagent.controller.controller import (
    Controller,
    PassToPhaseB,
    PlayClip,
    StartHandoff,
)
from jevagent.controller.packet import HandoffPacket, PacketSlot
from jevagent.jev.fake import RuleBasedJev
from jevagent.models import SlotStatus
from jevagent.observability.decision_log import MemoryDecisionLog
from jevagent.phase_b.agent import (
    LlmEvent,
    Message,
    PhaseBAgent,
    Say,
    TextDelta,
    ToolCall,
    Transferred,
)
from jevagent.phase_b.prompt import build_system_prompt, last_caller_utterance
from jevagent.phase_b.tools import MockCrm, RecordingTransfer, ToolDef, ToolRegistry
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder_en.yaml")
SETTINGS = Settings(_env_file=None, jev_api_key=SecretStr("k"))  # type: ignore[call-arg]
TOOLS = ["get_loan_account", "explain_charges", "schedule_callback", "transfer_to_human"]
TODAY = date(2026, 1, 1)

PACKET = HandoffPacket(
    handoff_reason="turn_type=objection",
    slots={
        "identity": PacketSlot(value="confirmed", confidence=0.97, status=SlotStatus.FILLED),
        "pay_date_bucket": PacketSlot(value=None, confidence=0.0, status=SlotStatus.UNCLEAR),
    },
    missing=["pay_date_bucket"],
    recent_turns=[
        "agent: Your EMI is due. By when will you pay?",
        "caller: First tell me about the penalty, why was it charged?",
    ],
    clips_played=["identity.ask.1", "pay_when.ask.1"],
    tools=TOOLS,
)


class FakeLlm:
    """Plays back one list of events per LLM call; can also raise."""

    def __init__(self, rounds: list[list[LlmEvent] | Exception]) -> None:
        self._rounds = list(rounds)
        self.calls: list[list[Message]] = []

    async def stream(
        self, messages: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]:
        self.calls.append([m.model_copy() for m in messages])
        round_ = self._rounds.pop(0)
        if isinstance(round_, Exception):
            raise round_
        for event in round_:
            yield event


def registry(packet: HandoffPacket = PACKET) -> tuple[ToolRegistry, MockCrm, RecordingTransfer]:
    crm, transfer = MockCrm(), RecordingTransfer()
    reg = ToolRegistry(packet, crm, transfer, allowed_clips={"closing.ask.1"}, today=lambda: TODAY)
    return reg, crm, transfer


def agent(llm: FakeLlm, packet: HandoffPacket = PACKET) -> tuple[PhaseBAgent, RecordingTransfer]:
    reg, _, transfer = registry(packet)
    return PhaseBAgent(llm, reg, packet, SCRIPT, SETTINGS, seed=1), transfer


async def collect(stream: AsyncIterator[object]) -> list[object]:
    return [item async for item in stream]


def call(name: str, **args: object) -> ToolCall:
    return ToolCall(id=f"id-{name}", name=name, arguments=dict(args))


# -- tools -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("explain_charges", {"charge_type": "bribe"}),
        ("explain_charges", {}),
        ("explain_charges", {"charge_type": "penalty", "extra": 1}),
        ("schedule_callback", {"day": "not-a-date", "window": "morning"}),
        ("schedule_callback", {"day": "2026-02-01", "window": "midnight"}),
        ("transfer_to_human", {"reason": ""}),
        ("get_loan_account", {"customer_id": "123"}),
        ("play_clip", {"clip_id": "pay_when.ask.1"}),  # not in allowed set
        ("play_clip", {}),
        ("delete_account", {}),
    ],
)
async def test_invalid_tool_calls_rejected_without_side_effects(
    name: str, args: dict[str, object]
) -> None:
    reg, crm, transfer = registry()
    result = await reg.execute(name, args)
    assert not result.ok and "error" in result.data
    assert crm.callbacks == [] and transfer.transfers == []


async def test_callback_must_be_in_the_future() -> None:
    reg, crm, _ = registry()
    past = await reg.execute("schedule_callback", {"day": "2026-01-01", "window": "morning"})
    assert not past.ok and crm.callbacks == []
    ok = await reg.execute("schedule_callback", {"day": "2026-01-02", "window": "evening"})
    assert ok.ok and crm.callbacks == [(date(2026, 1, 2), "evening")]


async def test_tools_limited_to_packet_allowlist() -> None:
    packet = PACKET.model_copy(update={"tools": ["get_loan_account"]})
    reg, _, _ = registry(packet)
    assert {d.name for d in reg.definitions()} == {"get_loan_account", "play_clip"}
    assert not (await reg.execute("explain_charges", {"charge_type": "penalty"})).ok


async def test_valid_calls_execute_and_transfer_carries_packet() -> None:
    reg, _, transfer = registry()
    assert (await reg.execute("get_loan_account", {})).data["emi_amount"] == 4500
    assert (await reg.execute("explain_charges", {"charge_type": "penalty"})).ok
    assert (await reg.execute("play_clip", {"clip_id": "closing.ask.1"})).ok
    assert (await reg.execute("transfer_to_human", {"reason": "wants human"})).ok
    assert transfer.transfers == [(PACKET, "wants human")]


# -- prompt ----------------------------------------------------------------


def test_prompt_contains_all_packet_content() -> None:
    prompt = build_system_prompt(PACKET)
    assert "turn_type=objection" in prompt
    assert "identity: confirmed" in prompt
    assert "pay_date_bucket (unclear)" in prompt
    assert "First tell me about the penalty" in prompt
    assert "FIRST sentence must acknowledge" in prompt
    assert "do NOT ask for these again" in prompt
    for tool in TOOLS:
        assert tool in prompt
    assert last_caller_utterance(PACKET) == "First tell me about the penalty, why was it charged?"


# -- agent -----------------------------------------------------------------


async def test_bridge_clip_then_first_turn_sees_caller_utterance() -> None:
    llm = FakeLlm([[TextDelta(text="About the penalty, ")]])
    ag, _ = agent(llm)
    bridge = ag.begin()
    assert bridge.clip_id.startswith("bridge.")
    out = await collect(ag.first_turn())
    assert out == [Say(text="About the penalty, ")]
    sent = llm.calls[0]
    assert sent[0].role == "system" and sent[1] == Message(
        role="user", content="First tell me about the penalty, why was it charged?"
    )


def references_prior(first_line: str, packet: HandoffPacket) -> bool:
    caller_words = {w.lower().strip("?,.") for w in last_caller_utterance(packet).split()}
    return any(w in first_line.lower() for w in caller_words if len(w) > 4)


async def test_first_line_references_prior_content() -> None:
    llm = FakeLlm([[TextDelta(text="You asked about the penalty, let me explain.")]])
    ag, _ = agent(llm)
    first = next(o for o in await collect(ag.first_turn()) if isinstance(o, Say))
    assert references_prior(first.text, PACKET)
    assert not references_prior("Hello, may I ask your name?", PACKET)


async def test_tool_loop_runs_tools_then_answers() -> None:
    llm = FakeLlm(
        [
            [call("explain_charges", charge_type="penalty")],
            [TextDelta(text="Penalty 350 rupaye hai.")],
        ]
    )
    ag, _ = agent(llm)
    out = await collect(ag.first_turn())
    assert out == [Say(text="Penalty 350 rupaye hai.")]
    tool_msg = llm.calls[1][-1]
    assert tool_msg.role == "tool" and '"amount": 350' in tool_msg.content


async def test_invalid_tool_args_reported_back_to_llm() -> None:
    llm = FakeLlm([[call("explain_charges", charge_type="bribe")], [TextDelta(text="Sorry.")]])
    ag, _ = agent(llm)
    await collect(ag.first_turn())
    assert "invalid arguments" in llm.calls[1][-1].content


async def test_compliance_line_routed_to_clip() -> None:
    llm = FakeLlm([[call("play_clip", clip_id="closing.ask.1")], []])
    ag, _ = agent(llm)
    assert await collect(ag.first_turn()) == [PlayClip(clip_id="closing.ask.1")]


async def test_llm_failure_falls_back_to_clip_and_human() -> None:
    ag, transfer = agent(FakeLlm([RuntimeError("provider down")]))
    out = await collect(ag.first_turn())
    assert out == [PlayClip(clip_id="fallback.1"), Transferred(reason="llm_failure")]
    assert transfer.transfers == [(PACKET, "llm_failure")]
    assert ag.finished
    assert await collect(ag.on_caller_turn("hello?")) == []


async def test_mid_stream_llm_failure_also_falls_back() -> None:
    class Breaks:
        async def stream(self, m: list[Message], t: list[ToolDef]) -> AsyncIterator[LlmEvent]:
            yield TextDelta(text="partial")
            raise TimeoutError

    reg, _, transfer = registry()
    ag = PhaseBAgent(Breaks(), reg, PACKET, SCRIPT, SETTINGS, seed=1)
    out = await collect(ag.first_turn())
    assert out[0] == Say(text="partial") and isinstance(out[-1], Transferred)
    assert len(transfer.transfers) == 1


async def test_tool_loop_guard() -> None:
    rounds: list[list[LlmEvent] | Exception] = [[call("get_loan_account")]] * 10
    ag, transfer = agent(FakeLlm(rounds))
    out = await collect(ag.first_turn())
    assert out[-1] == Transferred(reason="tool_loop_limit")
    assert len(transfer.transfers) == 1


async def test_transfer_tool_ends_agent() -> None:
    ag, transfer = agent(FakeLlm([[call("transfer_to_human", reason="wants human")]]))
    assert await collect(ag.first_turn()) == [Transferred(reason="wants human")]
    assert ag.finished and len(transfer.transfers) == 1


def test_script_without_shared_clips_rejected() -> None:
    bare = SCRIPT.model_copy(update={"shared": {}})
    reg, _, _ = registry()
    with pytest.raises(ValueError, match="bridge"):
        PhaseBAgent(FakeLlm([]), reg, PACKET, bare, SETTINGS)


# -- end to end (text only) -------------------------------------------------


async def test_end_to_end_phase_a_handoff_phase_b_tool_close() -> None:
    jev = RuleBasedJev(
        slot_keywords={"identity_confirmed": {"confirmed": ["yes"]}},
        turn_keywords={"objection": ["penalty"]},
    )
    ctl = Controller(SCRIPT, jev, SETTINGS, MemoryDecisionLog(), "c1", tools=TOOLS, seed=1)
    ctl.start()
    assert isinstance(await ctl.on_caller_turn("yes that is me"), PlayClip)
    handoff = await ctl.on_caller_turn("first tell me about the penalty")
    assert isinstance(handoff, StartHandoff)

    llm = FakeLlm(
        [
            [call("explain_charges", charge_type="penalty")],
            [TextDelta(text="Aapne penalty poochi; woh 350 rupaye hai.")],
            [call("play_clip", clip_id="closing.ask.1")],
            [],
        ]
    )
    reg, _, _ = registry(handoff.packet)
    ag = PhaseBAgent(llm, reg, handoff.packet, SCRIPT, SETTINGS, seed=1)

    outputs: list[object] = [ag.begin()]
    outputs += await collect(ag.first_turn())
    follow = await ctl.on_caller_turn("okay, I understand")
    assert isinstance(follow, PassToPhaseB)
    outputs += await collect(ag.on_caller_turn(follow.utterance))

    assert outputs[0].clip_id.startswith("bridge.")  # type: ignore[attr-defined]
    assert Say(text="Aapne penalty poochi; woh 350 rupaye hai.") in outputs
    assert outputs[-1] == PlayClip(clip_id="closing.ask.1")
    assert "identity: confirmed" in llm.calls[0][0].content
