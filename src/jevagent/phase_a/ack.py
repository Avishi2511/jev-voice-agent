"""Acknowledgement clips and speculative Jev requests (A-10)."""

import asyncio
import random

from jevagent.controller.controller import Controller
from jevagent.models import JevAnswer, Mode
from jevagent.script.schema import Script


class AckPolicy:
    """Short 'Ji' / 'Theek hai' clip at end of turn, never the same one twice in a row."""

    def __init__(self, script: Script, enabled: bool = True, seed: int | None = None) -> None:
        self._clips = [c.id for c in script.shared.get("ack", [])]
        self._enabled = enabled and bool(self._clips)
        self._rng = random.Random(seed)
        self._last: str | None = None

    def pick(self, mode: Mode) -> str | None:
        if not self._enabled or mode is Mode.B:
            return None
        pool = [c for c in self._clips if c != self._last] or self._clips
        self._last = self._rng.choice(pool)
        return self._last


class Speculator:
    """Runs Jev on a stable partial transcript; keeps the result only if the final text matches."""

    def __init__(self, controller: Controller) -> None:
        self._controller = controller
        self._text: str | None = None
        self._task: asyncio.Task[dict[str, JevAnswer]] | None = None

    def on_partial(self, text: str) -> None:
        text = text.strip()
        if not text or text == self._text:
            return
        self.cancel()
        self._task = self._controller.prefetch(text)
        self._text = text if self._task else None

    def on_speech_resumed(self) -> None:
        self.cancel()

    def take(self, final_text: str) -> asyncio.Task[dict[str, JevAnswer]] | None:
        """Hand the speculative task to the controller if it was for exactly this text."""
        task, text = self._task, self._text
        self._task = self._text = None
        if task is not None and text == final_text.strip():
            return task
        if task is not None:
            task.cancel()
        return None

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
        self._task = self._text = None
