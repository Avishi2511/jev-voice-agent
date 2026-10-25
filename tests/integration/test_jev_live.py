"""Live Jev smoke test. Skipped unless JEV_API_KEY is set."""

import os

import pytest
from pydantic import SecretStr

from jevagent.config import Settings
from jevagent.jev.client import HttpJevClient
from jevagent.jev.questions import build_questions, build_state
from jevagent.script.engine import ScriptEngine
from jevagent.script.loader import load_script

pytestmark = pytest.mark.skipif(not os.getenv("JEV_API_KEY"), reason="JEV_API_KEY not set")

UTTERANCES = ["yes, speaking", "not sure, maybe tomorrow", "who is this?"]


@pytest.mark.parametrize("utterance", UTTERANCES)
async def test_live_answers_are_well_formed(utterance: str) -> None:
    script = load_script("scripts/emi_reminder_en.yaml")
    state = ScriptEngine(script, seed=1).initial_state()
    questions = build_questions(script, state)
    settings = Settings(jev_api_key=SecretStr(os.environ["JEV_API_KEY"]))  # type: ignore[call-arg]
    client = HttpJevClient(settings)
    try:
        answers = await client.ask(build_state(script, state, utterance), questions)
    finally:
        await client.aclose()
    assert set(answers) == {q.key for q in questions}
    for q in questions:
        a = answers[q.key]
        assert a.selected is not None and 0.0 <= a.confidence <= 1.0
