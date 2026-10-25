import json

import httpx
import pytest
from pydantic import SecretStr

from jevagent.config import Settings
from jevagent.jev.client import HttpJevClient, JevConfigError, build_request
from jevagent.models import JevQuestion, QuestionType

QUESTIONS = [
    JevQuestion(
        key="slot_value:pay_when",
        type=QuestionType.CHOICE,
        instruction="when?",
        options=["today", "unclear"],
    ),
    JevQuestion(key="corrects_stored", type=QuestionType.NOUL, instruction="corrects?"),
]
GOOD = {
    "model": "jev-1.14.0",
    "answers": {
        "slot_value__pay_when": {
            "type": "choice",
            "choice": "today",
            "confidence": 0.9,
            "probabilities": {"today": 0.9, "unclear": 0.1},
        },
        "corrects_stored": {"type": "noul", "noul": 0.2},
    },
}


def settings(retries: int = 1) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, jev_api_key=SecretStr("k"), jev_max_retries=retries
    )


async def no_sleep(_: float) -> None:
    return None


def client(handler: httpx.MockTransport, retries: int = 1) -> HttpJevClient:
    return HttpJevClient(settings(retries), transport=handler, sleep=no_sleep)


async def test_single_batched_request_and_parsing() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=GOOD)

    answers = await client(httpx.MockTransport(handler)).ask({"utterance": "today"}, QUESTIONS)

    assert len(seen) == 1  # A-1: all questions in one call
    assert seen[0].url.path == "/v1/systemone"
    assert seen[0].headers["authorization"] == "Bearer k"
    body = json.loads(seen[0].content)
    assert body["model"] == "jev-1.14.0"
    assert set(body["questions"]) == {"slot_value__pay_when", "corrects_stored"}
    assert body["questions"]["slot_value__pay_when"]["criteria"] == {
        "today": "today",
        "unclear": "unclear",
    }
    assert answers["slot_value:pay_when"].selected == "today"
    assert answers["slot_value:pay_when"].confidence == 0.9
    assert answers["corrects_stored"].selected == "no"
    assert answers["corrects_stored"].confidence == pytest.approx(0.8)


async def test_timeout_returns_low_confidence() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    answers = await client(httpx.MockTransport(handler)).ask({}, QUESTIONS)
    assert set(answers) == {q.key for q in QUESTIONS}
    assert all(a.selected is None and a.confidence == 0.0 for a in answers.values())


async def test_retry_then_success() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429) if calls == 1 else httpx.Response(200, json=GOOD)

    answers = await client(httpx.MockTransport(handler)).ask({}, QUESTIONS)
    assert calls == 2
    assert answers["slot_value:pay_when"].selected == "today"


async def test_retries_exhausted_returns_low_confidence() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    answers = await client(httpx.MockTransport(handler), retries=2).ask({}, QUESTIONS)
    assert calls == 3
    assert all(a.confidence == 0.0 for a in answers.values())


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_client_errors_raise(status: int) -> None:
    c = client(httpx.MockTransport(lambda _: httpx.Response(status)))
    with pytest.raises(JevConfigError):
        await c.ask({}, QUESTIONS)


@pytest.mark.parametrize(
    "payload",
    [{"answers": {}}, {"nope": 1}, {"answers": {"corrects_stored": {"type": "box"}}}],
)
async def test_malformed_response_returns_low_confidence(payload: dict[str, object]) -> None:
    c = client(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    answers = await c.ask({}, QUESTIONS)
    assert all(a.confidence == 0.0 for a in answers.values())


def test_build_request_score_criteria_is_ordered_list() -> None:
    q = JevQuestion(key="s", type=QuestionType.SCORE, instruction="i", options=["low", "high"])
    assert build_request("m", {}, [q])["questions"]["s"]["criteria"] == ["low", "high"]
