"""Text-to-speech adapters (B-2). Output is always PCM16 mono at 8 kHz, the telephony leg."""

import base64
from array import array
from collections.abc import AsyncIterator
from typing import Any, Protocol

import structlog

from jevagent.frontend.audio import resample_pcm16, wav_to_pcm16

log = structlog.get_logger(__name__)

OUT_RATE = 8000
GEMINI_STREAM_RATE = 24000  # documented default for streaming: headerless audio/l16


class SynthesisError(RuntimeError):
    pass


class Synthesizer(Protocol):
    voice: str
    model: str

    async def synthesize(self, text: str) -> bytes: ...

    def stream(self, text: str) -> AsyncIterator[bytes]: ...


class StreamResampler:
    """Chunk-safe integer-ratio downsampler (carries leftover bytes between chunks)."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self._src, self._dst = src_rate, dst_rate
        self._carry = b""
        self._unit = (src_rate // dst_rate if src_rate % dst_rate == 0 else 1) * 2

    def feed(self, chunk: bytes) -> bytes:
        data = self._carry + chunk
        usable = len(data) // self._unit * self._unit
        self._carry = data[usable:]
        return resample_pcm16(data[:usable], self._src, self._dst)


class GeminiSynthesizer:
    def __init__(self, client: Any, model: str, voice: str) -> None:
        self._client = client
        self.model = model
        self.voice = voice

    def _request(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": text,
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": self.voice}]},
        }

    async def synthesize(self, text: str) -> bytes:
        try:
            interaction = await self._client.aio.interactions.create(**self._request(text))
            pcm, rate = wav_to_pcm16(base64.b64decode(interaction.output_audio.data))
        except Exception as exc:
            log.error("tts_failed", error=repr(exc))
            raise SynthesisError(str(exc)) from exc
        return resample_pcm16(pcm, rate, OUT_RATE)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        resampler = StreamResampler(GEMINI_STREAM_RATE, OUT_RATE)
        try:
            events = await self._client.aio.interactions.create(stream=True, **self._request(text))
            async for event in events:
                if event.event_type == "step.delta" and event.delta.type == "audio":
                    out = resampler.feed(base64.b64decode(event.delta.data))
                    if out:
                        yield out
        except Exception as exc:
            log.error("tts_stream_failed", error=repr(exc))
            raise SynthesisError(str(exc)) from exc


class FakeSynthesizer:
    """Deterministic audio: 10 ms of 8 kHz PCM16 per character of text."""

    def __init__(self, voice: str = "fake", model: str = "fake-tts") -> None:
        self.voice, self.model = voice, model
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        sample = array("h", [1000 + ord(text[0]) if text else 0] * (80 * len(text)))
        return sample.tobytes()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        pcm = await self.synthesize(text)
        for i in range(0, len(pcm), 3200):
            yield pcm[i : i + 3200]
