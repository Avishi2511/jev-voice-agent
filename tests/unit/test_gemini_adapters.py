import base64
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import pytest

from jevagent.frontend import audio
from jevagent.frontend.llm import GeminiLlmClient, ModelUnavailableError, check_models
from jevagent.frontend.stt import FakeTranscriber, GeminiTranscriber, TranscriptionError
from jevagent.frontend.tts import (
    FakeSynthesizer,
    GeminiSynthesizer,
    StreamResampler,
    SynthesisError,
)
from jevagent.phase_b.agent import Message, TextDelta, ToolCall
from jevagent.phase_b.tools import ToolDef


class FakeInteractions:
    """Stands in for client.aio.interactions; records requests, replays queued results."""

    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def client_with(results: list[Any], models: list[str] | None = None) -> Any:
    async def list_models() -> AsyncIterator[Any]:
        for m in models or []:
            yield NS(name=f"models/{m}")

    async def models_list() -> AsyncIterator[Any]:
        return list_models()

    inter = FakeInteractions(results)
    return NS(aio=NS(interactions=inter, models=NS(list=models_list))), inter


async def aiter(items: list[Any]) -> AsyncIterator[Any]:
    for i in items:
        yield i


# -- audio -----------------------------------------------------------------


def test_mulaw_roundtrip_is_close() -> None:
    pcm = audio.tone(100, 8000, amplitude=6000)
    back = audio.mulaw_to_pcm16(audio.pcm16_to_mulaw(pcm))
    assert len(back) == len(pcm)
    assert abs(audio.rms(back) - audio.rms(pcm)) / audio.rms(pcm) < 0.05


def test_mulaw_known_values() -> None:
    assert audio.pcm16_to_mulaw(bytes(2)) == b"\xff"  # silence encodes as 0xFF
    assert audio.mulaw_to_pcm16(b"\xff") == bytes(2)


def test_resample_ratios() -> None:
    pcm = audio.tone(100, 24000)
    assert len(audio.resample_pcm16(pcm, 24000, 8000)) == len(pcm) // 3
    assert len(audio.resample_pcm16(audio.tone(100, 16000), 16000, 8000)) == 1600
    odd = audio.tone(100, 8000)
    assert len(audio.resample_pcm16(odd, 8000, 12000)) == pytest.approx(len(odd) * 1.5, abs=4)
    assert audio.resample_pcm16(odd, 8000, 8000) == odd


def test_parse_content_type_and_wav() -> None:
    assert audio.parse_content_type("audio/x-l16;rate=16000") == ("audio/x-l16", 16000)
    assert audio.parse_content_type("audio/x-l16") == ("audio/x-l16", 8000)
    pcm = audio.tone(20, 8000)
    assert audio.wav_to_pcm16(audio.pcm16_to_wav(pcm, 8000)) == (pcm, 8000)
    assert audio.to_pcm16(b"\xff", "audio/x-mulaw") == bytes(2)
    assert audio.to_pcm16(b"ab", "audio/x-l16") == b"ab"
    assert audio.frame_bytes(16000, 20) == 640
    assert audio.rms(b"") == 0.0


def test_stream_resampler_is_chunk_safe() -> None:
    pcm = audio.tone(200, 24000)
    whole = audio.resample_pcm16(pcm, 24000, 8000)
    rs = StreamResampler(24000, 8000)
    pieces = [rs.feed(pcm[i : i + 1001]) for i in range(0, len(pcm), 1001)]  # odd chunk sizes
    assert b"".join(pieces) == whole


# -- STT -------------------------------------------------------------------


async def test_gemini_transcriber_request_and_result() -> None:
    client, inter = client_with([NS(output_text="  haan ji  ")])
    pcm = audio.tone(100, 16000)
    text = await GeminiTranscriber(client, "gemini-3.5-transcribe").transcribe(pcm, 16000)
    assert text == "haan ji"
    req = inter.requests[0]
    assert req["model"] == "gemini-3.5-transcribe"
    audio_item = req["input"][1]
    assert audio_item["type"] == "audio" and audio_item["mime_type"] == "audio/wav"
    assert audio.wav_to_pcm16(base64.b64decode(audio_item["data"])) == (pcm, 16000)


async def test_transcriber_failure_is_typed_and_empty_is_empty() -> None:
    client, _ = client_with([RuntimeError("quota")])
    with pytest.raises(TranscriptionError):
        await GeminiTranscriber(client, "m").transcribe(b"\0\0", 8000)
    client, _ = client_with([NS(output_text=None)])
    assert await GeminiTranscriber(client, "m").transcribe(b"\0\0", 8000) == ""


async def test_fake_transcriber() -> None:
    fake = FakeTranscriber(["a", RuntimeError("x")])
    assert await fake.transcribe(b"12", 8000) == "a"
    with pytest.raises(RuntimeError):
        await fake.transcribe(b"1", 8000)
    assert fake.calls == [(2, 8000), (1, 8000)]


# -- TTS -------------------------------------------------------------------


async def test_synthesize_converts_24k_wav_to_8k_pcm() -> None:
    wav = audio.pcm16_to_wav(audio.tone(300, 24000), 24000)
    client, inter = client_with([NS(output_audio=NS(data=base64.b64encode(wav).decode()))])
    pcm = await GeminiSynthesizer(client, "gemini-3.8-flash-tts", "Kore").synthesize("Namaste")
    assert len(pcm) == 8000 * 2 * 300 // 1000
    req = inter.requests[0]
    assert req["input"] == "Namaste" and req["response_format"] == {"type": "audio"}
    assert req["generation_config"] == {"speech_config": [{"voice": "Kore"}]}


async def test_stream_yields_8k_chunks_and_ignores_other_events() -> None:
    pcm24 = audio.tone(200, 24000)
    events = [
        NS(event_type="interaction.created"),
        *[
            NS(
                event_type="step.delta",
                delta=NS(type="audio", data=base64.b64encode(pcm24[i : i + 999]).decode()),
            )
            for i in range(0, len(pcm24), 999)
        ],
        NS(event_type="step.delta", delta=NS(type="text", text="ignored")),
    ]
    client, inter = client_with([aiter(events)])
    out = b"".join([c async for c in GeminiSynthesizer(client, "m", "Kore").stream("hi")])
    assert out == audio.resample_pcm16(pcm24[: len(out) * 3], 24000, 8000)
    assert len(out) >= 3200 - 4
    assert inter.requests[0]["stream"] is True


async def test_tts_failure_typed() -> None:
    client, _ = client_with([RuntimeError("503")])
    with pytest.raises(SynthesisError):
        await GeminiSynthesizer(client, "m", "v").synthesize("x")


async def test_fake_synthesizer_stream_matches_synthesize() -> None:
    fake = FakeSynthesizer()
    whole = await fake.synthesize("hello")
    assert b"".join([c async for c in fake.stream("hello")]) == whole


# -- LLM -------------------------------------------------------------------

TOOLS = [ToolDef(name="get_loan_account", description="d", parameters={"type": "object"})]


def llm_events(*parts: Any) -> Any:
    return aiter(list(parts))


async def test_llm_streams_text_and_accumulates_function_args() -> None:
    events = [
        NS(event_type="interaction.created"),
        NS(event_type="step.start", step=NS(type="function_call", id="c1", name="explain_charges")),
        NS(event_type="step.delta", delta=NS(type="arguments_delta", arguments='{"charge_')),
        NS(
            event_type="step.delta", delta=NS(type="arguments_delta", arguments='type": "penalty"}')
        ),
        NS(event_type="step.stop"),
        NS(event_type="step.start", step=NS(type="model_output")),
        NS(event_type="step.delta", delta=NS(type="text", text="Theek hai.")),
        NS(event_type="step.stop"),
        NS(
            event_type="interaction.completed", interaction=NS(id="int-1", status="requires_action")
        ),
    ]
    client, inter = client_with([llm_events(*events)])
    llm = GeminiLlmClient(client, "gemini-3.8-flash")
    messages = [
        Message(role="system", content="SYS"),
        Message(role="user", content="penalty kya hai"),
    ]
    out = [e async for e in llm.stream(messages, TOOLS)]
    assert out == [
        ToolCall(id="c1", name="explain_charges", arguments={"charge_type": "penalty"}),
        TextDelta(text="Theek hai."),
    ]
    req = inter.requests[0]
    assert req["system_instruction"] == "SYS" and req["input"] == "penalty kya hai"
    assert req["tools"][0] == {
        "type": "function",
        "name": "get_loan_account",
        "description": "d",
        "parameters": {"type": "object"},
    }
    assert "previous_interaction_id" not in req


async def test_llm_chains_turns_and_sends_only_new_messages() -> None:
    done = NS(event_type="interaction.completed", interaction=NS(id="int-1", status="completed"))
    client, inter = client_with([llm_events(done), llm_events(NS(**done.__dict__))])
    llm = GeminiLlmClient(client, "m")
    msgs = [Message(role="system", content="SYS"), Message(role="user", content="u1")]
    [e async for e in llm.stream(msgs, TOOLS)]
    msgs += [
        Message(
            role="assistant", content="a1", tool_calls=[ToolCall(id="c1", name="t", arguments={})]
        ),
        Message(role="tool", tool_call_id="c1", name="t", content='{"x": 1}'),
    ]
    [e async for e in llm.stream(msgs, TOOLS)]
    second = inter.requests[1]
    assert second["previous_interaction_id"] == "int-1"
    assert second["input"] == [
        {
            "type": "function_result",
            "name": "t",
            "call_id": "c1",
            "result": [{"type": "text", "text": '{"x": 1}'}],
        }
    ]


async def test_check_models() -> None:
    client, _ = client_with([], models=["gemini-3.8-flash", "gemini-3.8-flash-tts"])
    await check_models(client, {"llm": "gemini-3.8-flash", "tts": "gemini-3.8-flash-tts"})
    with pytest.raises(ModelUnavailableError, match="stt"):
        await check_models(client, {"stt": "gemini-3.5-transcribe"})


def test_arguments_json_roundtrip_guard() -> None:
    assert json.loads("{}") == {}
