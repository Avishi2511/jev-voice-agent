"""Core domain models shared by script engine, Jev layer and controller."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

JsonDict = dict[str, Any]
ESCAPE_OPTIONS: frozenset[str] = frozenset({"none", "unclear", "not_stated"})


class SlotStatus(StrEnum):
    FILLED = "filled"
    REFUSED = "refused"
    UNCLEAR = "unclear"
    EMPTY = "empty"


class Mode(StrEnum):
    A = "A"
    B = "B"


class QuestionType(StrEnum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class HandoffReason(StrEnum):
    OFF_SCRIPT = "off_script"
    UNRESOLVED_CORRECTION = "unresolved_correction"
    LOW_CONFIDENCE = "low_confidence_after_reask"
    RETRY_LIMIT = "retry_limit_exceeded"
    UNCOVERED_STATE = "script_state_not_covered"


class Slot(BaseModel):
    value: str | None = None
    confidence: float = 0.0
    status: SlotStatus = SlotStatus.EMPTY


class CallState(BaseModel):
    step: str
    slots: dict[str, Slot] = Field(default_factory=dict)
    retries: dict[str, int] = Field(default_factory=dict)
    clips_played: list[str] = Field(default_factory=list)
    recent_turns: list[str] = Field(default_factory=list)
    mode: Mode = Mode.A


class JevQuestion(BaseModel):
    key: str
    type: QuestionType
    instruction: str
    options: list[str] = Field(default_factory=list)
    descriptions: dict[str, str] = Field(default_factory=dict)


class JevAnswer(BaseModel):
    """Normalised answer. Noul answers use options yes/no."""

    key: str
    selected: str | None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0


class Advance(BaseModel):
    step: str
    clip_id: str
    terminal: bool = False


class ReAsk(BaseModel):
    clip_id: str


class Handoff(BaseModel):
    reason: HandoffReason
    detail: str | None = None  # e.g. turn_type=objection


Decision = Advance | ReAsk | Handoff
