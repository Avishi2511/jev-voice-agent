"""Benchmark candidate Phase B LLMs on handoff cases (B-1).

Each case is a handoff packet plus the caller's last utterance. We measure time to the first
token (text or tool call), total time, whether the right tool was chosen, and whether the first
words acknowledge what the caller said (H-5). Needs real transcripts for a meaningful result.
"""

import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from jevagent.controller.packet import HandoffPacket
from jevagent.observability.report import percentile
from jevagent.phase_b.agent import LlmClient, Message, TextDelta, ToolCall
from jevagent.phase_b.prompt import build_system_prompt, last_caller_utterance
from jevagent.phase_b.tools import MockCrm, RecordingTransfer, ToolRegistry

MIN_WORD_LEN = 5


class BenchCase(BaseModel):
    packet: HandoffPacket
    expect_tool: str | None = None  # tool that should be called first, if any
    ack_words: list[str] = []  # extra words that count as acknowledging the caller


class CaseResult(BaseModel):
    model: str
    ttft_ms: float | None
    total_ms: float
    tool_called: str | None
    tool_ok: bool
    acknowledged: bool
    error: str | None = None


class ModelSummary(BaseModel):
    model: str
    cases: int
    errors: int
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    tool_accuracy: float | None
    ack_rate: float | None


def load_cases(path: str | Path) -> list[BenchCase]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [BenchCase.model_validate_json(line) for line in lines if line.strip()]


def references_prior(text: str, case: BenchCase) -> bool:
    """Rule check: does the reply echo a content word of what the caller just said?"""
    spoken = {w.strip("?,.!").lower() for w in last_caller_utterance(case.packet).split()}
    words = {w for w in spoken if len(w) >= MIN_WORD_LEN} | {w.lower() for w in case.ack_words}
    return any(w in text.lower() for w in words)


async def run_case(
    llm: LlmClient, model: str, case: BenchCase, clock: Callable[[], float] = time.monotonic
) -> CaseResult:
    registry = ToolRegistry(case.packet, MockCrm(), RecordingTransfer(), allowed_clips=set())
    messages = [
        Message(role="system", content=build_system_prompt(case.packet)),
        Message(role="user", content=last_caller_utterance(case.packet)),
    ]
    start = clock()
    first: float | None = None
    text = ""
    tool: str | None = None
    try:
        async for event in llm.stream(messages, registry.definitions()):
            if first is None:
                first = clock()
            if isinstance(event, TextDelta):
                text += event.text
            elif isinstance(event, ToolCall) and tool is None:
                tool = event.name
    except Exception as exc:  # a failing candidate is a result, not a crash
        return CaseResult(
            model=model,
            ttft_ms=None,
            total_ms=(clock() - start) * 1000,
            tool_called=None,
            tool_ok=False,
            acknowledged=False,
            error=repr(exc),
        )
    tool_ok = tool == case.expect_tool if case.expect_tool else tool is None
    return CaseResult(
        model=model,
        ttft_ms=None if first is None else (first - start) * 1000,
        total_ms=(clock() - start) * 1000,
        tool_called=tool,
        tool_ok=tool_ok,
        acknowledged=references_prior(text, case) if text else tool is not None,
    )


def summarise(model: str, results: list[CaseResult]) -> ModelSummary:
    ok = [r for r in results if r.error is None]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    return ModelSummary(
        model=model,
        cases=len(results),
        errors=len(results) - len(ok),
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p95_ms=percentile(ttfts, 95),
        tool_accuracy=sum(r.tool_ok for r in ok) / len(ok) if ok else None,
        ack_rate=sum(r.acknowledged for r in ok) / len(ok) if ok else None,
    )


async def benchmark(cases: list[BenchCase], clients: dict[str, LlmClient]) -> list[ModelSummary]:
    out = []
    for model, llm in clients.items():
        results = [await run_case(llm, model, case) for case in cases]
        out.append(summarise(model, results))
    return out


def to_markdown(summaries: list[ModelSummary]) -> str:
    def f(x: float | None, pct: bool = False) -> str:
        return "n/a" if x is None else (f"{x:.0%}" if pct else f"{x:.0f}")

    lines = [
        "| model | cases | errors | TTFT P50 ms | TTFT P95 ms | tool accuracy | acknowledges |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | {s.cases} | {s.errors} | {f(s.ttft_p50_ms)} | {f(s.ttft_p95_ms)} "
            f"| {f(s.tool_accuracy, True)} | {f(s.ack_rate, True)} |"
        )
    return "\n".join(lines) + "\n"
