"""Narrow, typed Phase B tools. Every argument is validated before anything executes (B-3)."""

from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, Literal, Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jevagent.controller.packet import HandoffPacket
from jevagent.models import JsonDict

log = structlog.get_logger(__name__)


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetLoanAccountArgs(_Args):
    pass


class ExplainChargesArgs(_Args):
    charge_type: Literal["penalty", "late_fee", "interest", "processing_fee"]


class ScheduleCallbackArgs(_Args):
    day: date
    window: Literal["morning", "afternoon", "evening"]


class TransferToHumanArgs(_Args):
    reason: str = Field(min_length=1, max_length=200)


class PlayClipArgs(_Args):
    clip_id: str


class Crm(Protocol):
    async def get_loan_account(self) -> JsonDict: ...
    async def explain_charges(self, charge_type: str) -> JsonDict: ...
    async def schedule_callback(self, day: date, window: str) -> JsonDict: ...


class HumanTransfer(Protocol):
    async def transfer(self, packet: HandoffPacket, reason: str) -> None: ...


class MockCrm:
    """Deterministic stand-in until a real CRM adapter exists."""

    def __init__(self) -> None:
        self.callbacks: list[tuple[date, str]] = []

    async def get_loan_account(self) -> JsonDict:
        return {"emi_amount": 4500, "due_date": "2026-02-05", "overdue_days": 12}

    async def explain_charges(self, charge_type: str) -> JsonDict:
        return {"charge_type": charge_type, "amount": 350, "reason": "mock explanation"}

    async def schedule_callback(self, day: date, window: str) -> JsonDict:
        self.callbacks.append((day, window))
        return {"scheduled": day.isoformat(), "window": window}


class RecordingTransfer:
    def __init__(self) -> None:
        self.transfers: list[tuple[HandoffPacket, str]] = []

    async def transfer(self, packet: HandoffPacket, reason: str) -> None:
        self.transfers.append((packet, reason))


class ToolDef(BaseModel):
    name: str
    description: str
    parameters: JsonDict


class ToolResult(BaseModel):
    ok: bool
    data: JsonDict


_DESCRIPTIONS = {
    "get_loan_account": "Fetch the caller's EMI amount, due date and overdue days.",
    "explain_charges": "Explain a charge that appears on the caller's account.",
    "schedule_callback": "Schedule a callback on a future day in a time window.",
    "transfer_to_human": "Hand the call to a human agent with full context.",
}
_ARGS: dict[str, type[_Args]] = {
    "get_loan_account": GetLoanAccountArgs,
    "explain_charges": ExplainChargesArgs,
    "schedule_callback": ScheduleCallbackArgs,
    "transfer_to_human": TransferToHumanArgs,
}


class ToolRegistry:
    def __init__(
        self,
        packet: HandoffPacket,
        crm: Crm,
        transfer: HumanTransfer,
        allowed_clips: set[str],
        today: Callable[[], date] = date.today,
    ) -> None:
        self._packet = packet
        self._crm = crm
        self._transfer = transfer
        self._allowed_clips = allowed_clips
        self._today = today
        self._handlers: dict[str, Callable[[Any], Awaitable[JsonDict]]] = {
            "get_loan_account": lambda a: crm.get_loan_account(),
            "explain_charges": lambda a: crm.explain_charges(a.charge_type),
            "schedule_callback": self._schedule_callback,
            "transfer_to_human": self._transfer_to_human,
        }

    def definitions(self) -> list[ToolDef]:
        defs = [
            ToolDef(
                name=name,
                description=_DESCRIPTIONS[name],
                parameters=_ARGS[name].model_json_schema(),
            )
            for name in self._packet.tools
            if name in _ARGS
        ]
        defs.append(
            ToolDef(
                name="play_clip",
                description="Play a pre-recorded line (mandatory disclosure or closing).",
                parameters=PlayClipArgs.model_json_schema(),
            )
        )
        return defs

    async def execute(self, name: str, raw_args: JsonDict) -> ToolResult:
        if name == "play_clip":
            return self._play_clip(raw_args)
        if name not in self._handlers or name not in self._packet.tools:
            return ToolResult(ok=False, data={"error": f"tool {name!r} is not available"})
        try:
            args = _ARGS[name].model_validate(raw_args)
        except ValidationError as exc:
            log.warning("tool_args_rejected", tool=name, errors=exc.error_count())
            return ToolResult(
                ok=False, data={"error": f"invalid arguments: {exc.errors()[0]['msg']}"}
            )
        data = await self._handlers[name](args)
        return ToolResult(ok="error" not in data, data=data)

    def _play_clip(self, raw_args: JsonDict) -> ToolResult:
        try:
            args = PlayClipArgs.model_validate(raw_args)
        except ValidationError:
            return ToolResult(ok=False, data={"error": "invalid arguments"})
        if args.clip_id not in self._allowed_clips:
            return ToolResult(ok=False, data={"error": f"clip {args.clip_id!r} is not allowed"})
        return ToolResult(ok=True, data={"played": args.clip_id})

    async def _schedule_callback(self, args: ScheduleCallbackArgs) -> JsonDict:
        if args.day <= self._today():
            return {"error": "callback day must be in the future"}
        return await self._crm.schedule_callback(args.day, args.window)

    async def transfer(self, reason: str) -> None:
        """B-6: the packet goes with the call. Also used directly by the failure fallback."""
        await self._transfer.transfer(self._packet, reason)

    async def _transfer_to_human(self, args: TransferToHumanArgs) -> JsonDict:
        await self.transfer(args.reason)
        return {"transferred": True}
