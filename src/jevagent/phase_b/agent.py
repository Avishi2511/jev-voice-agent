"""Phase B agent: provider-agnostic streaming tool loop that continues from a packet."""

import json
import random
from collections.abc import AsyncIterator
from typing import Literal, Protocol

import structlog
from pydantic import BaseModel, Field

from jevagent.config import Settings
from jevagent.controller.controller import PlayClip
from jevagent.controller.packet import HandoffPacket
from jevagent.models import JsonDict
from jevagent.phase_b.prompt import build_system_prompt, last_caller_utterance
from jevagent.phase_b.tools import ToolDef, ToolRegistry
from jevagent.script.schema import Script

log = structlog.get_logger(__name__)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: JsonDict


class TextDelta(BaseModel):
    text: str


LlmEvent = TextDelta | ToolCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for role=tool


class LlmClient(Protocol):
    def stream(self, messages: list[Message], tools: list[ToolDef]) -> AsyncIterator[LlmEvent]: ...


class Say(BaseModel):
    text: str


class Transferred(BaseModel):
    reason: str


AgentOutput = Say | PlayClip | Transferred


class PhaseBAgent:
    def __init__(
        self,
        llm: LlmClient,
        tools: ToolRegistry,
        packet: HandoffPacket,
        script: Script,
        settings: Settings,
        seed: int | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._packet = packet
        self._max_iterations = settings.llm_max_tool_iterations
        rng = random.Random(seed)
        self._bridge = rng.choice(_shared(script, "bridge"))
        self._fallback = _shared(script, "fallback")[0]
        self._messages = [Message(role="system", content=build_system_prompt(packet))]
        self.finished = False

    def begin(self) -> PlayClip:
        """H-3: bridge clip plays while first_turn() streams."""
        return PlayClip(clip_id=self._bridge)

    def first_turn(self) -> AsyncIterator[AgentOutput]:
        return self._turn(last_caller_utterance(self._packet))

    def on_caller_turn(self, utterance: str) -> AsyncIterator[AgentOutput]:
        return self._turn(utterance)

    async def _turn(self, utterance: str) -> AsyncIterator[AgentOutput]:
        if self.finished:
            return
        self._messages.append(Message(role="user", content=utterance))
        for _ in range(self._max_iterations):
            text: list[str] = []
            calls: list[ToolCall] = []
            try:
                async for event in self._llm.stream(self._messages, self._tools.definitions()):
                    if isinstance(event, TextDelta):
                        text.append(event.text)
                        yield Say(text=event.text)
                    else:
                        calls.append(event)
            except Exception:  # provider-agnostic boundary: any LLM failure falls back to a human
                log.exception("llm_failed")
                async for out in self._fail("llm_failure"):
                    yield out
                return
            self._messages.append(
                Message(role="assistant", content="".join(text), tool_calls=calls)
            )
            if not calls:
                return
            for call in calls:
                async for out in self._run_tool(call):
                    yield out
                if self.finished:
                    return
        log.error("tool_loop_limit", limit=self._max_iterations)
        async for out in self._fail("tool_loop_limit"):
            yield out

    async def _run_tool(self, call: ToolCall) -> AsyncIterator[AgentOutput]:
        result = await self._tools.execute(call.name, call.arguments)
        self._messages.append(
            Message(
                role="tool", tool_call_id=call.id, name=call.name, content=json.dumps(result.data)
            )
        )
        if not result.ok:
            return
        if call.name == "play_clip":
            yield PlayClip(clip_id=str(result.data["played"]))  # B-4: scripted lines stay clips
        elif call.name == "transfer_to_human":
            self.finished = True
            yield Transferred(reason=str(call.arguments["reason"]))

    async def _fail(self, reason: str) -> AsyncIterator[AgentOutput]:
        yield PlayClip(clip_id=self._fallback)
        await self._tools.transfer(reason)
        self.finished = True
        yield Transferred(reason=reason)


def _shared(script: Script, kind: str) -> list[str]:
    clips = script.shared.get(kind)
    if not clips:
        raise ValueError(f"script {script.name!r} defines no shared {kind!r} clips")
    return [c.id for c in clips]
