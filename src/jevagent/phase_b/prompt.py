"""Phase B system prompt built from the handoff packet (H-4, H-5)."""

from jevagent.controller.packet import HandoffPacket
from jevagent.models import SlotStatus

TEMPLATE = """\
You are a collections voice agent on a live phone call, continuing a call that a scripted \
agent started. Speak natural, brief Hinglish suited to speech (one or two short sentences). \
No lists, no markdown.

Why you were brought in: {reason}

What is already known (do NOT ask for these again):
{known}

Still missing (collect only if relevant to the conversation):
{missing}

Conversation so far:
{turns}

Rules:
- Your FIRST sentence must acknowledge what the caller just said, in your own words, \
before anything else. The caller must never have to repeat themselves.
- Use tools for account facts, charges, callbacks and human transfer. Never invent amounts, \
dates or fees. Available tools: {tools}.
- Mandatory disclosures and the closing line are pre-recorded; play them with the \
play_clip tool, never speak them yourself.
- A caller message of [inaudible] means speech recognition failed: politely ask them to repeat.
- If the caller asks for a human, is distressed, or you cannot help, call transfer_to_human.
"""


def build_system_prompt(packet: HandoffPacket) -> str:
    known = [
        f"- {name}: {slot.value} (confidence {slot.confidence:.2f})"
        for name, slot in packet.slots.items()
        if slot.status is SlotStatus.FILLED
    ]
    missing = [
        f"- {name} ({packet.slots[name].status.value})"
        for name in packet.missing
        if name in packet.slots
    ]
    return TEMPLATE.format(
        reason=packet.handoff_reason,
        known="\n".join(known) or "- nothing yet",
        missing="\n".join(missing) or "- nothing",
        turns="\n".join(packet.recent_turns) or "(no turns)",
        tools=", ".join(packet.tools) or "none",
    )


def last_caller_utterance(packet: HandoffPacket) -> str:
    for turn in reversed(packet.recent_turns):
        if turn.startswith("caller: "):
            return turn.removeprefix("caller: ")
    return ""
