"""Gemini implementation of the Phase B LlmClient, on the Interactions API.

The adapter chains turns with `previous_interaction_id`, so each call sends only the messages
added since the last one (the new user text, or tool results).
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import structlog

from jevagent.phase_b.agent import LlmEvent, Message, TextDelta, ToolCall
from jevagent.phase_b.tools import ToolDef

log = structlog.get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


async def check_models(client: Any, required: dict[str, str]) -> None:
    """Fail at startup if a configured model ID is not served to this key (pin check)."""
    available: set[str] = set()
    async for model in await client.aio.models.list():
        available.add(str(model.name).removeprefix("models/"))
    missing = {role: m for role, m in required.items() if m not in available}
    if missing:
        raise ModelUnavailableError(
            f"configured Gemini models not available to this key: {missing}"
        )


class GeminiLlmClient:
    """One instance per call: it holds the interaction chain."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model
        self._previous_id: str | None = None
        self._sent = 0

    async def stream(
        self, messages: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]:
        system = next((m.content for m in messages if m.role == "system"), None)
        new = [m for m in messages[self._sent :] if m.role in ("user", "tool")]
        self._sent = len(messages)
        request: dict[str, Any] = {
            "model": self._model,
            "input": self._input(new),
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ],
        }
        if system:
            request["system_instruction"] = system
        if self._previous_id:
            request["previous_interaction_id"] = self._previous_id

        call_id = call_name = None
        args = ""
        events = await self._client.aio.interactions.create(**request)
        async for event in events:
            kind = event.event_type
            if kind == "step.start" and event.step.type == "function_call":
                call_id, call_name, args = event.step.id, event.step.name, ""
            elif kind == "step.delta":
                if event.delta.type == "text":
                    yield TextDelta(text=event.delta.text)
                elif event.delta.type == "arguments_delta":
                    args += event.delta.arguments
            elif kind == "step.stop" and call_name is not None:
                yield ToolCall(id=str(call_id), name=call_name, arguments=json.loads(args or "{}"))
                call_id = call_name = None
            elif kind == "interaction.completed":
                self._previous_id = event.interaction.id
            else:
                log.debug("gemini_event_skipped", event_type=kind)

    @staticmethod
    def _input(new: list[Message]) -> Any:
        if len(new) == 1 and new[0].role == "user":
            return new[0].content
        items: list[dict[str, Any]] = []
        for m in new:
            if m.role == "tool":
                items.append(
                    {
                        "type": "function_result",
                        "name": m.name,
                        "call_id": m.tool_call_id,
                        "result": [{"type": "text", "text": m.content}],
                    }
                )
            else:
                items.append({"type": "text", "text": m.content})
        return items
