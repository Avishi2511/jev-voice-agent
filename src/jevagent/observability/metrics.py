"""Per-turn latency instrumentation: turn end, STT, Jev, first audio out."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from jevagent.jev.client import JevClient
from jevagent.models import JevAnswer, JevQuestion, JsonDict

Clock = Callable[[], float]


class TurnTimings(BaseModel):
    call_id: str
    turn: int
    marks: dict[str, float] = Field(default_factory=dict)

    def ms(self, start: str, end: str) -> float | None:
        if start in self.marks and end in self.marks:
            return round((self.marks[end] - self.marks[start]) * 1000, 1)
        return None

    @property
    def ttfa_ms(self) -> float | None:
        """Time to first audio: caller stops talking -> first audio frame sent."""
        return self.ms("turn_end", "first_audio_out")


class TurnTimer:
    def __init__(self, call_id: str, turn: int, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self.timings = TurnTimings(call_id=call_id, turn=turn)

    def mark(self, name: str, overwrite: bool = False) -> None:
        if overwrite or name not in self.timings.marks:
            self.timings.marks[name] = self._clock()


class MetricsSink(Protocol):
    def write(self, timings: TurnTimings) -> None: ...


class MemoryMetrics:
    def __init__(self) -> None:
        self.rows: list[TurnTimings] = []

    def write(self, timings: TurnTimings) -> None:
        self.rows.append(timings)


class JsonlMetrics:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, timings: TurnTimings) -> None:
        row: JsonDict = {**timings.model_dump(), "ttfa_ms": timings.ttfa_ms}
        row["stt_ms"] = timings.ms("turn_end", "stt_done")
        row["jev_ms"] = timings.ms("jev_sent", "jev_received")
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


class TimedJev:
    """Wraps a JevClient and stamps jev_sent / jev_received on the current turn's timer."""

    def __init__(self, inner: JevClient, current: Callable[[], TurnTimer | None]) -> None:
        self._inner = inner
        self._current = current

    async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
        timer = self._current()
        if timer:
            timer.mark("jev_sent")
        try:
            return await self._inner.ask(state, questions)
        finally:
            if timer:
                timer.mark("jev_received")
