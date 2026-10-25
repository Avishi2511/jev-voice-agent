"""Jev (TypeSafe System One) client. One batched request per turn (A-1)."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx
import structlog

from jevagent.config import Settings
from jevagent.models import JevAnswer, JevQuestion, JsonDict, QuestionType

log = structlog.get_logger(__name__)

# Question keys contain ":" (e.g. slot_value:pay_when); keep wire names plain.
_WIRE_SEP = "__"
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class JevConfigError(RuntimeError):
    """Non-retryable request failure (bad key, bad request): a bug to fix, not to mask."""


class JevClient(Protocol):
    async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]: ...


def low_confidence_answers(questions: list[JevQuestion]) -> dict[str, JevAnswer]:
    """Failure policy: the engine reads these as 'unclear', so it re-asks then hands off."""
    return {q.key: JevAnswer(key=q.key, selected=None, confidence=0.0) for q in questions}


def _wire(key: str) -> str:
    return key.replace(":", _WIRE_SEP)


def build_request(model: str, state: JsonDict, questions: list[JevQuestion]) -> JsonDict:
    body: dict[str, JsonDict] = {}
    for q in questions:
        entry: JsonDict = {"type": q.type.value, "instructions": q.instruction}
        if q.type is QuestionType.CHOICE:
            entry["criteria"] = {o: q.descriptions.get(o, o) for o in q.options}
        elif q.type is QuestionType.SCORE:
            entry["criteria"] = [q.descriptions.get(o, o) for o in q.options]
        body[_wire(q.key)] = entry
    return {"model": model, "state": state, "questions": body}


def parse_answer(key: str, raw: JsonDict) -> JevAnswer:
    kind = raw["type"]
    if kind == "choice":
        return JevAnswer(
            key=key,
            selected=raw["choice"],
            probabilities={k: float(v) for k, v in raw["probabilities"].items()},
            confidence=float(raw["confidence"]),
        )
    if kind == "noul":
        p = float(raw["noul"])
        return JevAnswer(
            key=key,
            selected="yes" if p >= 0.5 else "no",
            probabilities={"yes": p, "no": 1.0 - p},
            confidence=max(p, 1.0 - p),
        )
    if kind == "score":
        return JevAnswer(
            key=key,
            selected=str(round(float(raw["score"]))),
            probabilities={k: float(v) for k, v in raw["probabilities"].items()},
            confidence=float(raw["confidence"]),
        )
    raise ValueError(f"unsupported answer type {kind!r}")


class HttpJevClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._http = httpx.AsyncClient(
            base_url=settings.jev_base_url,
            timeout=settings.jev_timeout_s,
            headers={"Authorization": f"Bearer {settings.jev_api_key.get_secret_value()}"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
        payload = build_request(self._settings.jev_model, state, questions)
        attempts = self._settings.jev_max_retries + 1
        for attempt in range(attempts):
            try:
                resp = await self._http.post("/v1/systemone", json=payload)
            except httpx.TransportError as exc:  # includes timeouts
                log.warning("jev_transport_error", attempt=attempt, error=repr(exc))
            else:
                if resp.status_code == 200:
                    return self._parse(resp, questions)
                if resp.status_code not in _RETRYABLE_STATUS:
                    raise JevConfigError(f"Jev rejected request: HTTP {resp.status_code}")
                log.warning("jev_http_error", attempt=attempt, status=resp.status_code)
            if attempt + 1 < attempts:
                await self._sleep(random.uniform(0.0, 0.1 * 2**attempt))
        log.error("jev_unavailable", attempts=attempts)
        return low_confidence_answers(questions)

    @staticmethod
    def _parse(resp: httpx.Response, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
        try:
            raw_answers: dict[str, JsonDict] = resp.json()["answers"]
            return {q.key: parse_answer(q.key, raw_answers[_wire(q.key)]) for q in questions}
        except (KeyError, ValueError, TypeError) as exc:
            log.error("jev_malformed_response", error=repr(exc))
            return low_confidence_answers(questions)
