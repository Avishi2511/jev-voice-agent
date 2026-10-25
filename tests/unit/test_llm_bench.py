from collections.abc import AsyncIterator
from pathlib import Path

from jevagent.controller.packet import HandoffPacket, PacketSlot
from jevagent.eval.llm_bench import (
    BenchCase,
    benchmark,
    load_cases,
    references_prior,
    run_case,
    to_markdown,
)
from jevagent.models import SlotStatus
from jevagent.phase_b.agent import LlmEvent, Message, TextDelta, ToolCall
from jevagent.phase_b.tools import ToolDef

PACKET = HandoffPacket(
    handoff_reason="turn_type=objection",
    slots={"identity": PacketSlot(value="confirmed", confidence=0.9, status=SlotStatus.FILLED)},
    missing=[],
    recent_turns=["agent: EMI due hai.", "caller: why was I charged a penalty"],
    clips_played=[],
    tools=["explain_charges", "transfer_to_human"],
)
CASE = BenchCase(packet=PACKET, expect_tool="explain_charges")


class Scripted:
    def __init__(self, events: list[LlmEvent] | Exception) -> None:
        self.events = events

    async def stream(
        self, messages: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]:
        if isinstance(self.events, Exception):
            raise self.events
        for e in self.events:
            yield e


def tool(name: str) -> ToolCall:
    return ToolCall(id="1", name=name, arguments={})


async def test_run_case_scores_tool_and_acknowledgement() -> None:
    clock = iter([0.0, 0.2, 0.5, 0.0, 0.1, 0.3]).__next__
    good = await run_case(Scripted([tool("explain_charges")]), "m", CASE, clock)
    assert good.tool_ok and good.tool_called == "explain_charges" and good.acknowledged
    assert good.ttft_ms == 200.0 and good.total_ms == 500.0
    wrong = await run_case(Scripted([tool("transfer_to_human")]), "m", CASE, clock)
    assert not wrong.tool_ok


async def test_no_tool_expected_and_text_acknowledgement() -> None:
    case = BenchCase(packet=PACKET)
    ok = await run_case(Scripted([TextDelta(text="You asked about the penalty.")]), "m", case)
    assert ok.tool_ok and ok.acknowledged
    off = await run_case(Scripted([TextDelta(text="Hello, how are you?")]), "m", case)
    assert off.tool_ok and not off.acknowledged
    stray = await run_case(Scripted([tool("explain_charges")]), "m", case)
    assert not stray.tool_ok  # called a tool when none was expected


async def test_errors_are_results_not_crashes() -> None:
    result = await run_case(Scripted(RuntimeError("quota")), "m", CASE)
    assert result.error and not result.tool_ok and result.ttft_ms is None


async def test_benchmark_table_and_cases_file(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(CASE.model_dump_json() + "\n\n", encoding="utf-8")
    cases = load_cases(path)
    summaries = await benchmark(
        cases,
        {
            "good": Scripted([tool("explain_charges")]),
            "broken": Scripted(RuntimeError("x")),
        },
    )
    good, broken = summaries
    assert good.tool_accuracy == 1.0 and good.ack_rate == 1.0 and good.errors == 0
    assert broken.errors == 1 and broken.tool_accuracy is None and broken.ttft_p50_ms is None
    md = to_markdown(summaries)
    assert "| good | 1 | 0 |" in md and "n/a" in md


def test_references_prior_uses_extra_ack_words() -> None:
    case = BenchCase(packet=PACKET, ack_words=["fee"])
    assert references_prior("that fee is standard", case)
    assert not references_prior("hello there", case)
