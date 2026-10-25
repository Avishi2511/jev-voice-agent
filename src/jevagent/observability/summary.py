"""Per-call outcome and usage records, the input to completion / cost reporting (Phase 7)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel


@dataclass
class Usage:
    """Mutable counters shared by the components of one call."""

    jev_requests: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


Outcome = Literal["completed", "transferred", "dropped"]


class CallSummary(BaseModel):
    call_id: str
    mode: Literal["hybrid", "baseline"]
    outcome: Outcome
    handoff_reason: str | None = None
    turns: int = 0
    stt_requests: int = 0
    tts_chars: int = 0
    jev_requests: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


class SummarySink(Protocol):
    def write(self, summary: CallSummary) -> None: ...


class MemorySummaries:
    def __init__(self) -> None:
        self.rows: list[CallSummary] = []

    def write(self, summary: CallSummary) -> None:
        self.rows.append(summary)


class JsonlSummaries:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, summary: CallSummary) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(summary.model_dump_json() + "\n")


def load_summaries(path: str | Path) -> list[CallSummary]:
    return [
        CallSummary.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_metric_rows(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
