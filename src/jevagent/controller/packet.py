"""Context packet handed to Phase B (H-4)."""

from pydantic import BaseModel, Field

from jevagent.models import CallState, Handoff, SlotStatus


class PacketSlot(BaseModel):
    value: str | None
    confidence: float
    status: SlotStatus


class HandoffPacket(BaseModel):
    handoff_reason: str
    slots: dict[str, PacketSlot]
    missing: list[str]
    recent_turns: list[str]
    clips_played: list[str]
    tools: list[str] = Field(default_factory=list)


def build_packet(state: CallState, handoff: Handoff, tools: list[str]) -> HandoffPacket:
    slots = {
        name: PacketSlot(value=s.value, confidence=s.confidence, status=s.status)
        for name, s in state.slots.items()
    }
    return HandoffPacket(
        handoff_reason=handoff.detail or handoff.reason.value,
        slots=slots,
        missing=[n for n, s in state.slots.items() if s.status is not SlotStatus.FILLED],
        recent_turns=list(state.recent_turns),
        clips_played=list(state.clips_played),
        tools=list(tools),
    )
