"""Mock Vobiz: drives a CallSession in memory with synthetic audio, no phone or quota needed.

Speech is a tone and silence is zeros, so use a FakeTranscriber for the words. The mock also
plays the part of Vobiz's playback queue by answering checkpoints with `playedStream`.
"""

import asyncio
import base64
from collections.abc import Callable
from typing import Any

from jevagent.frontend.audio import frame_bytes, silence, tone
from jevagent.frontend.vobiz_bridge import FRAME_MS, CallSession, Close, Send
from jevagent.models import JsonDict


class MockVobiz:
    def __init__(
        self,
        build: Callable[[Send, Close], CallSession],
        rate: int = 16000,
        encoding: str = "audio/x-l16",
        auto_played: bool = True,
    ) -> None:
        self.sent: list[JsonDict] = []
        self.closed = False
        self.rate, self.encoding = rate, encoding
        self.auto_played = auto_played
        self._tasks: list[asyncio.Future[None]] = []
        self._unplayed = 0
        self.session = build(self._send, self._close)

    async def _send(self, msg: JsonDict) -> None:
        self.sent.append(msg)
        if msg["event"] == "checkpoint" and self.auto_played:
            # Vobiz confirms the checkpoint once the queued audio has played.
            self._unplayed += 1
            asyncio.get_running_loop().call_later(
                0.01, lambda: self._tasks.append(asyncio.ensure_future(self._confirm()))
            )

    async def _close(self) -> None:
        self.closed = True

    # -- caller side -----------------------------------------------------------

    async def start(self) -> None:
        await self.session.on_event(
            {
                "event": "start",
                "start": {
                    "callId": "mock-call",
                    "streamId": "mock-stream",
                    "tracks": ["inbound"],
                    "mediaFormat": {"encoding": self.encoding, "sampleRate": self.rate},
                },
            }
        )

    async def feed(self, pcm: bytes) -> None:
        step = frame_bytes(self.rate, FRAME_MS)
        for i in range(0, len(pcm), step):
            frame = pcm[i : i + step]
            await self.session.on_event(
                {"event": "media", "media": {"payload": base64.b64encode(frame).decode()}}
            )
            await asyncio.sleep(0)

    async def say(self, speech_ms: int = 600, silence_ms: int = 900, wait: bool = True) -> None:
        """Caller speaks, then goes quiet long enough for the turn to end."""
        await self.feed(tone(speech_ms, self.rate))
        await self.feed(silence(silence_ms, self.rate))
        if wait:
            await self.wait_idle()

    async def wait_idle(self, timeout: float = 5.0) -> None:
        async def _wait() -> None:
            while not self.session.idle or self._unplayed:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(_wait(), timeout)
        if self._tasks:
            await asyncio.gather(*self._tasks)

    async def _confirm(self) -> None:
        await self.played()
        self._unplayed -= 1

    async def played(self) -> None:
        await self.session.on_event({"event": "playedStream", "name": "mock"})

    # -- inspection ------------------------------------------------------------

    def events(self, name: str) -> list[JsonDict]:
        return [m for m in self.sent if m["event"] == name]

    def audio_since(self, index: int = 0) -> bytes:
        return b"".join(
            base64.b64decode(m["media"]["payload"])
            for m in self.sent[index:]
            if m["event"] == "playAudio"
        )

    def summary(self) -> list[Any]:
        return [m["event"] for m in self.sent]
