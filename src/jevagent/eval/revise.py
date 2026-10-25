"""LLM-assisted question revision (E-3, P1). Drafts only; a human reviews the diff.

The model sees the question's written definition, its current wording and where it failed, and
may only propose a *general* wording fix. A proposal that quotes a failing utterance is rejected:
that would be overfitting the question to the tuning set.
"""

import difflib

from jevagent.eval.dataset import LabelledTurn
from jevagent.eval.versions import QuestionVersion
from jevagent.phase_b.agent import LlmClient, Message, TextDelta
from jevagent.phase_b.tools import ToolDef

PROMPT = """You improve the wording of one classification question used by a decision model.
Question key: {key}
Definition (authoritative, do not change its meaning):
{definition}

Current wording:
{current}

It gave wrong answers on these examples (utterance | expected | got):
{misses}

Write an improved wording that fixes the cause in general terms. Rules:
- Do not mention or quote any example utterance, and do not add per-example special cases.
- Keep the definition's meaning; clarify boundaries and how to treat Hinglish and hedged answers.
- Reply with the new wording only.
"""

MIN_QUOTE_WORDS = 3


class RevisionRejected(ValueError):
    pass


def overfits(proposal: str, utterances: list[str]) -> str | None:
    """Return a failing utterance (3+ words) that the proposal quotes, if any."""
    text = " ".join(proposal.lower().split())
    for u in utterances:
        norm = " ".join(u.lower().split())
        if len(norm.split()) >= MIN_QUOTE_WORDS and norm in text:
            return u
    return None


async def propose_revision(
    llm: LlmClient,
    qv: QuestionVersion,
    definition: str,
    misses: list[tuple[LabelledTurn, str | None]],
    next_version: int,
) -> tuple[QuestionVersion, str]:
    """Returns (draft version, unified diff for review). Raises RevisionRejected on overfit."""
    shown = "\n".join(f"- {t.utterance} | {t.labels.get(qv.key)} | {got}" for t, got in misses)
    prompt = PROMPT.format(
        key=qv.key, definition=definition, current=qv.instruction, misses=shown or "- none"
    )
    tools: list[ToolDef] = []
    text = ""
    async for event in llm.stream([Message(role="user", content=prompt)], tools):
        if isinstance(event, TextDelta):
            text += event.text
    proposal = text.strip()
    if not proposal:
        raise RevisionRejected("model returned no wording")
    quoted = overfits(proposal, [t.utterance for t, _ in misses])
    if quoted:
        raise RevisionRejected(f"proposal quotes a failing utterance: {quoted!r}")

    draft = QuestionVersion(key=qv.key, version=next_version, instruction=proposal)
    diff = "\n".join(
        difflib.unified_diff(
            qv.instruction.splitlines(),
            proposal.splitlines(),
            fromfile=f"{qv.key}@v{qv.version}",
            tofile=f"{qv.key}@v{next_version} (draft)",
            lineterm="",
        )
    )
    return draft, diff
