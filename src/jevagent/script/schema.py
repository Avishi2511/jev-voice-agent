"""Declarative call-script schema (CT-2)."""

from pydantic import BaseModel, Field

from jevagent.models import QuestionType


class Clip(BaseModel):
    id: str
    text: str


class StepClips(BaseModel):
    ask: list[Clip]
    re_ask: list[Clip] = Field(default_factory=list)
    confirm: list[Clip] = Field(default_factory=list)


class QuestionSpec(BaseModel):
    type: QuestionType
    instruction: str
    options: list[str] = Field(default_factory=list)


class Step(BaseModel):
    id: str
    slot: str | None = None
    clips: StepClips
    questions: dict[str, QuestionSpec] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    max_retries: int | None = None
    on_filled: str | None = None  # explicit next step id; default is next unfilled step


class Script(BaseModel):
    name: str
    language: str = "hinglish"
    thresholds: dict[str, float]
    max_retries: int = 1
    steps: list[Step]

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)
