"""Deterministic tuning / validation / held-out split by call, never by turn (E-2)."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog

from jevagent.eval.dataset import LabelledTurn

Split = Literal["tuning", "validation", "heldout"]
log = structlog.get_logger(__name__)

RATIOS: tuple[tuple[Split, float], ...] = (("tuning", 0.6), ("validation", 0.2), ("heldout", 0.2))


class HeldOutAccessError(PermissionError):
    pass


def assign(call_id: str, salt: str = "jevagent-v1") -> Split:
    bucket = int.from_bytes(hashlib.sha256(f"{salt}:{call_id}".encode()).digest()[:8], "big")
    point = bucket / 2**64
    edge = 0.0
    for name, ratio in RATIOS:
        edge += ratio
        if point < edge:
            return name
    return "heldout"  # pragma: no cover - float rounding guard


def split_turns(
    turns: list[LabelledTurn], salt: str = "jevagent-v1"
) -> dict[Split, list[LabelledTurn]]:
    out: dict[Split, list[LabelledTurn]] = {"tuning": [], "validation": [], "heldout": []}
    for t in turns:
        out[assign(t.call_id, salt)].append(t)
    return out


def get_split(
    turns: list[LabelledTurn],
    name: Split,
    salt: str = "jevagent-v1",
    allow_heldout: bool = False,
    audit_log: str | Path = "logs/heldout_access.log",
    reason: str = "",
) -> list[LabelledTurn]:
    """Held-out is only released for promotion and every access is recorded."""
    if name == "heldout":
        if not allow_heldout or not reason:
            raise HeldOutAccessError("held-out access needs allow_heldout=True and a reason")
        path = Path(audit_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(UTC).isoformat()} heldout accessed: {reason}\n")
        log.warning("heldout_access", reason=reason)
    return split_turns(turns, salt)[name]
