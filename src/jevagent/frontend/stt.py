"""Speech-to-text adapters (FE-3). Gemini first; per-utterance requests, no streaming."""

import base64
from typing import Any, Protocol

import structlog

from jevagent.frontend.audio import pcm16_to_wav

log = structlog.get_logger(__name__)

PROMPT = (
    "Transcribe this phone-call audio verbatim. The speaker uses Hindi, English or a mix "
    "(Hinglish); write Hindi words in Roman script as they are spoken. Domain words include "
    "EMI, UPI, net banking, penalty, loan, payment, tareekh. Return only the transcript text. "
    "If there is no intelligible speech, return nothing."
)


class TranscriptionError(RuntimeError):
    pass


class Transcriber(Protocol):
    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str: ...


class GeminiTranscriber:
    def __init__(self, client: Any, model: str, prompt: str = PROMPT) -> None:
        self._client = client
        self._model = model
        self._prompt = prompt

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        data = base64.b64encode(pcm16_to_wav(pcm16, sample_rate)).decode("ascii")
        try:
            interaction = await self._client.aio.interactions.create(
                model=self._model,
                input=[
                    {"type": "text", "text": self._prompt},
                    {"type": "audio", "data": data, "mime_type": "audio/wav"},
                ],
            )
            text = interaction.output_text
        except Exception as exc:  # SDK/network failures surface as one typed error
            log.error("stt_failed", error=repr(exc))
            raise TranscriptionError(str(exc)) from exc
        return (text or "").strip()


class FakeTranscriber:
    """Returns scripted transcripts in order; an Exception entry is raised instead."""

    def __init__(self, results: list[str | Exception]) -> None:
        self._results = list(results)
        self.calls: list[tuple[int, int]] = []  # (bytes, sample_rate)

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        self.calls.append((len(pcm16), sample_rate))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
