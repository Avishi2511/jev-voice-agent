"""Live A/B routing (Stage 4): a configurable share of calls to hybrid, the rest to baseline.

Assignment is a hash of the call id, so it is sticky (the same call always gets the same arm)
and needs no shared state. The kill switch is a file: while it exists every new call goes to
baseline. It is read per call, so it needs no restart; calls already in progress are unaffected.
"""

import hashlib
from pathlib import Path
from typing import Literal

import structlog

from jevagent.config import Settings

Arm = Literal["hybrid", "baseline"]
log = structlog.get_logger(__name__)


class AbRouter:
    def __init__(
        self, hybrid_percent: int, kill_switch_path: str | Path, salt: str = "ab-v1"
    ) -> None:
        if not 0 <= hybrid_percent <= 100:
            raise ValueError(f"hybrid_percent must be 0-100, got {hybrid_percent}")
        self._percent = hybrid_percent
        self._kill = Path(kill_switch_path)
        self._salt = salt

    @property
    def killed(self) -> bool:
        return self._kill.exists()

    def assign(self, call_id: str) -> Arm:
        if self.killed:
            return "baseline"
        digest = hashlib.sha256(f"{self._salt}:{call_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        return "hybrid" if bucket < self._percent else "baseline"


def settings_for_arm(settings: Settings, arm: Arm) -> Settings:
    """Per-call settings: the arm decides baseline_mode; shadow never runs in a live A/B."""
    return settings.model_copy(update={"baseline_mode": arm == "baseline", "shadow_mode": False})
