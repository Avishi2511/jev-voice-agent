"""Per-decision audit log (CT-5), also the source of future training labels (E-7)."""

from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from jevagent.models import JevAnswer, JsonDict


class DecisionRecord(BaseModel):
    call_id: str
    turn: int
    timestamp: datetime
    model_version: str
    state_sent: JsonDict
    questions: list[JsonDict]
    answers: dict[str, JevAnswer]
    action: JsonDict


class DecisionLog(Protocol):
    def write(self, record: DecisionRecord) -> None: ...


class MemoryDecisionLog:
    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []

    def write(self, record: DecisionRecord) -> None:
        self.records.append(record)


class JsonlDecisionLog:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: DecisionRecord) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
