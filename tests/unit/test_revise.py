from collections.abc import AsyncIterator

import pytest

from jevagent.eval.dataset import LabelledTurn
from jevagent.eval.revise import RevisionRejected, overfits, propose_revision
from jevagent.eval.versions import QuestionVersion
from jevagent.phase_b.agent import LlmEvent, Message, TextDelta
from jevagent.phase_b.tools import ToolDef


class FakeLlm:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def stream(
        self, messages: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]:
        self.prompts.append(messages[0].content)
        for word in self.reply.split(" "):
            yield TextDelta(text=word + " ")


def miss(utt: str) -> tuple[LabelledTurn, str | None]:
    turn = LabelledTurn(
        call_id="c",
        turn=1,
        step="pay_when",
        utterance=utt,
        labels={"turn_type": "objection", "answered_step": "no"},
        should_handoff=True,
    )
    return turn, "on_script"


QV = QuestionVersion(key="turn_type", version=1, instruction="Classify the caller's utterance.")


def test_overfit_detection() -> None:
    assert overfits("treat 'pehle penalty ka bataiye' as objection", ["Pehle penalty ka bataiye"])
    assert overfits("pushback about fees is an objection", ["penalty kyun"]) is None
    assert overfits("say ok is on_script", ["ok"]) is None  # short utterances are not quotable


async def test_general_revision_returns_draft_and_diff() -> None:
    llm = FakeLlm("Classify the utterance; disputes about charges are objections.")
    draft, diff = await propose_revision(
        llm, QV, "objection: caller disputes or pushes back", [miss("wo penalty kyun laga")], 2
    )
    assert draft.version == 2 and draft.status == "draft" and draft.key == "turn_type"
    assert "disputes about charges" in draft.instruction
    assert "-Classify the caller's utterance." in diff and "+Classify the utterance;" in diff
    prompt = llm.prompts[0]
    assert "wo penalty kyun laga | objection | on_script" in prompt
    assert "objection: caller disputes or pushes back" in prompt


async def test_quoting_a_failing_utterance_is_rejected() -> None:
    llm = FakeLlm("If caller says wo penalty kyun laga then objection")
    with pytest.raises(RevisionRejected, match="quotes"):
        await propose_revision(llm, QV, "def", [miss("wo penalty kyun laga")], 2)


async def test_empty_reply_rejected() -> None:
    with pytest.raises(RevisionRejected, match="no wording"):
        await propose_revision(FakeLlm(""), QV, "def", [miss("x y z")], 2)
