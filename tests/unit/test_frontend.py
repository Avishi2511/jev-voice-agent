import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import SecretStr

from jevagent.config import Settings
from jevagent.controller.controller import Controller
from jevagent.frontend.audio import frame_bytes, normalise_rms, rms, silence, tone
from jevagent.frontend.mock_caller import MockVobiz
from jevagent.frontend.stt import FakeTranscriber, TranscriptionError
from jevagent.frontend.tts import FakeSynthesizer
from jevagent.frontend.turn_detect import SpeechStarted, TurnDetector, TurnEnded, silence_ms_for
from jevagent.frontend.vad import EnergyVad, SileroVad
from jevagent.frontend.vobiz_bridge import CallSession, Close, Send, TimerSlot, _split_sentences
from jevagent.jev.fake import RuleBasedJev
from jevagent.observability.decision_log import MemoryDecisionLog
from jevagent.observability.metrics import (
    JsonlMetrics,
    MemoryMetrics,
    TimedJev,
    TurnTimer,
)
from jevagent.phase_a.ack import AckPolicy, Speculator
from jevagent.phase_a.clip_player import ClipError, ClipPlayer
from jevagent.phase_a.clip_render import render_clips, variant_warnings
from jevagent.phase_b.agent import LlmEvent, Message, PhaseBAgent, TextDelta, ToolCall
from jevagent.phase_b.tools import MockCrm, RecordingTransfer, ToolDef, ToolRegistry
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder.yaml")
SETTINGS = Settings(  # type: ignore[call-arg]
    _env_file=None,
    jev_api_key=SecretStr("k"),
    gemini_tts_voice="fake",
    gemini_tts_model="fake-tts",
)
TOOLS = ["get_loan_account", "explain_charges", "schedule_callback", "transfer_to_human"]


# -- audio / vad / turn detection --------------------------------------------


def test_normalise_rms_scales_and_respects_peak() -> None:
    quiet = tone(100, 8000, amplitude=500)
    assert rms(normalise_rms(quiet)) == pytest.approx(3000, rel=0.05)
    loud_peaky = tone(100, 8000, amplitude=30000)
    assert (
        max(
            abs(int.from_bytes(loud_peaky[i : i + 2], "little", signed=True))
            for i in range(0, len(loud_peaky), 2)
        )
        <= 30000
    )
    assert normalise_rms(silence(20, 8000)) == silence(20, 8000)


def test_energy_vad_adapts_to_noise_floor() -> None:
    vad = EnergyVad(threshold=500, floor_margin=3.0)
    noise = tone(20, 16000, amplitude=400)  # below threshold: line noise
    assert not any(vad.is_speech(noise) for _ in range(20))
    assert vad.is_speech(tone(20, 16000, amplitude=8000))
    assert not vad.is_speech(silence(20, 16000))


def test_silero_vad_needs_optional_package() -> None:
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="silero-vad"):
            SileroVad()
    else:  # pragma: no cover - only when the extra is installed
        pytest.skip("silero installed")


def make_detector(**kw: int) -> TurnDetector:
    return TurnDetector(EnergyVad(500), frame_ms=20, **kw)


def feed(det: TurnDetector, pcm: bytes, rate: int = 16000) -> list[object]:
    size = frame_bytes(rate, 20)
    out: list[object] = []
    for i in range(0, len(pcm), size):
        out += det.feed(pcm[i : i + size])
    return out


def test_turn_detector_emits_start_then_end_with_audio() -> None:
    det = make_detector(barge_in_frames=1, min_speech_frames=8, silence_ms=700)
    speech = tone(400, 16000)
    events = feed(det, silence(200, 16000) + speech + silence(800, 16000))
    assert [type(e) for e in events] == [SpeechStarted, TurnEnded]
    ended = events[1]
    assert isinstance(ended, TurnEnded)
    assert len(ended.pcm) >= len(speech)  # includes pre-roll and the speech itself


def test_speech_started_fires_on_first_speech_frame() -> None:
    det = make_detector(barge_in_frames=1)
    assert det.feed(tone(20, 16000)) == [SpeechStarted()]
    assert det.feed(tone(20, 16000)) == []  # once per utterance


def test_short_noise_burst_is_not_a_turn() -> None:
    det = make_detector(min_speech_frames=8)
    events = feed(det, tone(60, 16000) + silence(900, 16000))
    assert all(isinstance(e, SpeechStarted) for e in events)  # barge-in only, no TurnEnded


def test_pause_shorter_than_threshold_keeps_one_turn() -> None:
    det = make_detector(silence_ms=700, min_speech_frames=8)
    audio = tone(300, 16000) + silence(400, 16000) + tone(300, 16000) + silence(800, 16000)
    assert sum(isinstance(e, TurnEnded) for e in feed(det, audio)) == 1


def test_two_turns_back_to_back() -> None:
    det = make_detector()
    audio = (tone(400, 16000) + silence(800, 16000)) * 2
    assert sum(isinstance(e, TurnEnded) for e in feed(det, audio)) == 2


def test_per_language_silence() -> None:
    assert silence_ms_for(SETTINGS, "en") == 600
    assert silence_ms_for(SETTINGS, "hinglish") == 700
    assert silence_ms_for(SETTINGS, "unknown") == 700


# -- clips -------------------------------------------------------------------


async def rendered(tmp_path: Path) -> ClipPlayer:
    await render_clips(SCRIPT, FakeSynthesizer(), tmp_path)
    return ClipPlayer.load(tmp_path)


async def test_render_and_load_clips(tmp_path: Path) -> None:
    player = await rendered(tmp_path)
    player.verify(SCRIPT, SETTINGS)
    assert len(player.manifest.clips) == len(SCRIPT.all_clips())
    chunks = list(player.chunks("ack.1"))
    assert all(len(c) == 320 for c in chunks[:-1]) and b"".join(chunks) == player.pcm("ack.1")
    assert rms(player.pcm("ack.1")) == pytest.approx(3000, rel=0.1)  # normalised


async def test_clip_integrity_and_voice_checks(tmp_path: Path) -> None:
    player = await rendered(tmp_path)
    other = Settings(_env_file=None, gemini_tts_voice="Puck")  # type: ignore[call-arg]
    with pytest.raises(ClipError, match="re-render"):
        player.verify(SCRIPT, other)
    with pytest.raises(ClipError, match="unknown clip"):
        player.pcm("nope.1")
    (tmp_path / "ack.1.pcm").write_bytes(b"tampered")
    with pytest.raises(ClipError, match="hash"):
        ClipPlayer.load(tmp_path)
    (tmp_path / "ack.1.pcm").unlink()
    with pytest.raises(ClipError, match="missing audio"):
        ClipPlayer.load(tmp_path)
    with pytest.raises(ClipError, match="manifest"):
        ClipPlayer.load(tmp_path / "empty")


async def test_verify_reports_missing_script_clips(tmp_path: Path) -> None:
    player = await rendered(tmp_path)
    del player._audio["closing.ask.1"]  # noqa: SLF001
    with pytest.raises(ClipError, match="closing.ask.1"):
        player.verify(SCRIPT, SETTINGS)


def test_variant_warnings() -> None:
    warnings = variant_warnings(SCRIPT)
    assert any("identity_confirmed.ask: 2" in w for w in warnings)
    assert not any("closing" in w for w in warnings)


# -- ack / speculation / metrics --------------------------------------------------


def test_ack_policy_never_repeats_and_respects_flags() -> None:
    policy = AckPolicy(SCRIPT, enabled=True, seed=1)
    from jevagent.models import Mode

    picks = [policy.pick(Mode.A) for _ in range(30)]
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False))
    assert policy.pick(Mode.B) is None
    assert AckPolicy(SCRIPT, enabled=False).pick(Mode.A) is None
    assert AckPolicy(SCRIPT.model_copy(update={"shared": {}})).pick(Mode.A) is None


def rule_jev() -> RuleBasedJev:
    return RuleBasedJev(
        slot_keywords={
            "identity_confirmed": {"confirmed": ["haan"]},
            "pay_when": {"today": ["aaj"], "tomorrow": ["kal"]},
            "payment_mode": {"upi": ["upi"]},
        },
        turn_keywords={"objection": ["penalty"]},
    )


class CountingJev:
    def __init__(self) -> None:
        self.inner = rule_jev()
        self.calls = 0

    async def ask(self, state, questions):  # type: ignore[no-untyped-def]
        self.calls += 1
        return await self.inner.ask(state, questions)


async def test_speculation_commit_and_discard() -> None:
    jev = CountingJev()
    ctl = Controller(SCRIPT, jev, SETTINGS, MemoryDecisionLog(), "c", seed=1)  # type: ignore[arg-type]
    ctl.start()
    spec = Speculator(ctl)

    spec.on_partial("haan")
    spec.on_partial("haan")  # unchanged partial: no second request
    task = spec.take("haan")
    assert task is not None
    action = await ctl.on_caller_turn("haan", task)
    assert jev.calls == 1 and action.clip_id.startswith("pay_when")  # type: ignore[union-attr]

    spec.on_partial("kal")
    spec.on_speech_resumed()  # caller continued: discard
    assert spec.take("kal") is None
    spec.on_partial("kal")
    assert spec.take("kal tak") is None  # final text differs: discard
    await asyncio.sleep(0)
    assert ctl.prefetch("x") is not None
    spec.on_partial("   ")  # blank partial ignored


async def test_prefetch_none_after_handoff() -> None:
    ctl = Controller(SCRIPT, rule_jev(), SETTINGS, MemoryDecisionLog(), "c", seed=1)
    ctl.start()
    await ctl.on_caller_turn("penalty kyun")
    assert ctl.prefetch("anything") is None


def test_turn_timings() -> None:
    clock = iter([10.0, 10.5, 10.9, 11.0])
    timer = TurnTimer("c", 1, clock=lambda: next(clock))
    for name in ("turn_end", "stt_done", "first_audio_out"):
        timer.mark(name)
    timer.mark("stt_done")  # first mark wins
    assert timer.timings.ms("turn_end", "stt_done") == 500.0
    assert timer.timings.ttfa_ms == 900.0
    assert timer.timings.ms("turn_end", "missing") is None


async def test_timed_jev_and_jsonl(tmp_path: Path) -> None:
    clock = iter([1.0, 1.1, 1.2, 1.5, 2.0])
    timer = TurnTimer("c", 1, clock=lambda: next(clock))
    timer.mark("turn_end")
    timer.mark("stt_done")
    jev = TimedJev(rule_jev(), lambda: timer)
    await jev.ask({"utterance": "haan", "current_step": "identity_confirmed"}, [])
    timer.mark("first_audio_out")
    sink = JsonlMetrics(tmp_path / "m.jsonl")
    sink.write(timer.timings)
    row = (tmp_path / "m.jsonl").read_text(encoding="utf-8")
    assert '"jev_ms": 300.0' in row and '"ttfa_ms": 1000.0' in row and '"stt_ms": 100.0' in row
    nobody = TimedJev(rule_jev(), lambda: None)
    await nobody.ask({"utterance": "x", "current_step": "identity_confirmed"}, [])


def test_split_sentences() -> None:
    assert _split_sentences("Ek. Do? Teen") == ["Ek.", "Do?", "Teen"]
    assert _split_sentences("Ek.") == ["Ek."]
    long = " ".join(["word"] * 80)
    parts = _split_sentences(long)
    assert len(parts) == 2 and len(parts[0]) <= 160


# -- full call through the mock caller ----------------------------------------------


class ScriptedLlm:
    def __init__(self, rounds: list[list[LlmEvent]], hang_after_first: bool = False) -> None:
        self.rounds = list(rounds)
        self.hang = hang_after_first
        self.started = asyncio.Event()

    async def stream(
        self, messages: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]:
        for event in self.rounds.pop(0):
            yield event
        if self.hang:
            self.started.set()
            await asyncio.Event().wait()


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        transcripts: list[str | Exception],
        llm_rounds: list[list[LlmEvent]] | None = None,
        hang: bool = False,
        auto_played: bool = True,
    ) -> None:
        self.tmp_path = tmp_path
        self.transcripts = transcripts
        self.jev = CountingJev()
        self.synth = FakeSynthesizer()
        self.stt = FakeTranscriber(list(transcripts))
        self.metrics = MemoryMetrics()
        self.llm = ScriptedLlm(llm_rounds or [], hang_after_first=hang)
        self.transfers = RecordingTransfer()
        self.auto_played = auto_played
        self.mock: MockVobiz
        self.player: ClipPlayer
        self.controller: Controller

    async def setup(self) -> "Harness":
        self.player = await rendered(self.tmp_path)

        def build(send: Send, close: Close) -> CallSession:
            slot = TimerSlot()
            self.controller = Controller(
                SCRIPT,
                TimedJev(self.jev, lambda: slot.current),  # type: ignore[arg-type]
                SETTINGS,
                MemoryDecisionLog(),
                "c1",
                tools=TOOLS,
                seed=1,
            )

            def factory(packet):  # type: ignore[no-untyped-def]
                reg = ToolRegistry(packet, MockCrm(), self.transfers, {"closing.ask.1"})
                return PhaseBAgent(self.llm, reg, packet, SCRIPT, SETTINGS, seed=1)

            return CallSession(
                call_id="c1",
                script=SCRIPT,
                settings=SETTINGS,
                controller=self.controller,
                transcriber=self.stt,
                synth=self.synth,
                clips=self.player,
                agent_factory=factory,
                vad=EnergyVad(500),
                send=send,
                close=close,
                metrics=self.metrics,
                ack=AckPolicy(SCRIPT, enabled=False),
                timer_slot=slot,
            )

        self.mock = MockVobiz(build, auto_played=self.auto_played)
        await self.mock.start()
        await self.mock.wait_idle()
        return self


async def test_greeting_plays_on_start_and_checkpoints(tmp_path: Path) -> None:
    h = await Harness(tmp_path, []).setup()
    m = h.mock
    assert m.summary()[0] == "playAudio" and m.summary()[-1] == "checkpoint"
    first = m.events("playAudio")[0]["media"]
    assert first["contentType"] == "audio/x-l16" and first["sampleRate"] == 8000
    assert m.events("playAudio")[0]["streamId"] == "mock-stream"
    opening = h.controller.state.clips_played[0]
    assert m.audio_since() == h.player.pcm(opening)
    assert all(
        len(__import__("base64").b64decode(e["media"]["payload"])) <= 320
        for e in m.events("playAudio")
    )
    assert h.metrics.rows[0].turn == 0


async def test_happy_path_ends_call_after_closing_clip_plays(tmp_path: Path) -> None:
    h = await Harness(tmp_path, ["haan ji", "kal kar dunga", "upi se"]).setup()
    for _ in range(3):
        await h.mock.say()
    await asyncio.sleep(0.05)
    assert h.controller.state.step == "closing"
    assert h.mock.closed  # closed only after playedStream
    assert h.jev.calls == 3
    assert len(h.mock.events("clearAudio")) == 0
    ttfa = [r.ttfa_ms for r in h.metrics.rows if r.turn > 0]
    assert all(t is not None and t >= 0 for t in ttfa)
    jev_marks = [r.ms("jev_sent", "jev_received") for r in h.metrics.rows if r.turn > 0]
    assert all(j is not None for j in jev_marks)


async def test_barge_in_clears_audio_when_caller_speaks_over_greeting(tmp_path: Path) -> None:
    h = await Harness(tmp_path, ["haan"], auto_played=False).setup()
    assert h.mock.session._audio_pending  # noqa: SLF001 - greeting not yet confirmed played
    await h.mock.feed(tone(40, 16000))  # caller starts talking
    assert len(h.mock.events("clearAudio")) == 1
    await h.mock.feed(tone(40, 16000))
    assert len(h.mock.events("clearAudio")) == 1  # nothing left to clear


async def test_no_clear_when_nothing_is_playing(tmp_path: Path) -> None:
    h = await Harness(tmp_path, ["haan"]).setup()  # greeting confirmed played
    await h.mock.feed(tone(40, 16000))
    assert h.mock.events("clearAudio") == []


async def test_objection_hands_off_and_phase_b_speaks_with_bridge_clip(tmp_path: Path) -> None:
    rounds: list[list[LlmEvent]] = [
        [
            TextDelta(text="Penalty ke baare mein "),
            TextDelta(text="batati hoon. "),
            TextDelta(text="Woh late fee hai."),
        ],
        [TextDelta(text="Theek hai, dhanyavaad.")],
    ]
    h = await Harness(tmp_path, ["penalty kyun laga", "samajh gaya"], rounds).setup()
    await h.mock.say()
    assert h.controller.state.mode.value == "B"
    bridge_ids = [c.id for c in SCRIPT.shared["bridge"]]
    assert any(h.player.pcm(b) in h.mock.audio_since() for b in bridge_ids)
    assert h.synth.texts == ["Penalty ke baare mein batati hoon.", "Woh late fee hai."]
    mark = len(h.mock.sent)
    await h.mock.say()
    assert h.synth.texts[-1] == "Theek hai, dhanyavaad."
    assert h.mock.audio_since(mark)
    assert h.jev.calls == 1  # Jev not consulted after handoff


async def test_transfer_tool_ends_the_call(tmp_path: Path) -> None:
    rounds: list[list[LlmEvent]] = [
        [ToolCall(id="t", name="transfer_to_human", arguments={"reason": "wants human"})]
    ]
    h = await Harness(tmp_path, ["penalty kyun"], rounds).setup()
    await h.mock.say()
    await asyncio.sleep(0.05)
    assert len(h.transfers.transfers) == 1 and h.mock.closed


async def test_barge_in_cancels_phase_b_generation(tmp_path: Path) -> None:
    rounds: list[list[LlmEvent]] = [[TextDelta(text="Pehli baat. "), TextDelta(text="Doosri")]]
    h = await Harness(tmp_path, ["penalty kyun"], rounds, hang=True, auto_played=False).setup()
    await h.mock.say(wait=False)
    await asyncio.wait_for(h.llm.started.wait(), 2)
    await asyncio.sleep(0.05)
    assert h.synth.texts == ["Pehli baat."]
    cleared = len(h.mock.events("clearAudio"))  # the unconfirmed greeting was cleared by `say`
    await h.mock.feed(tone(40, 16000))  # caller interrupts mid-generation
    await asyncio.sleep(0.05)
    assert len(h.mock.events("clearAudio")) == cleared + 1
    await h.mock.wait_idle()
    assert h.synth.texts == ["Pehli baat."]  # the cancelled reply never spoke "Doosri"


async def test_stt_failure_is_reasked_then_hands_off(tmp_path: Path) -> None:
    h = await Harness(
        tmp_path,
        [TranscriptionError("down"), TranscriptionError("down")],
        [[TextDelta(text="Maaf kijiye.")]],
    ).setup()
    await h.mock.say()
    assert h.controller.state.retries.get("identity_confirmed") == 1
    await h.mock.say()
    assert h.controller.state.mode.value == "B"


async def test_empty_transcript_is_ignored(tmp_path: Path) -> None:
    h = await Harness(tmp_path, [""]).setup()
    await h.mock.say()
    assert h.jev.calls == 0 and h.controller.state.step == "identity_confirmed"


async def test_mulaw_inbound_is_decoded(tmp_path: Path) -> None:
    h = Harness(tmp_path, ["haan"])
    h.player = await rendered(tmp_path)
    await h.setup()
    assert h.mock.encoding == "audio/x-l16"
    from jevagent.frontend.audio import pcm16_to_mulaw

    h.mock.encoding, h.mock.rate = "audio/x-mulaw", 8000
    await h.mock.session.on_event(
        {
            "event": "start",
            "start": {
                "streamId": "s2",
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000},
            },
        }
    )
    await h.mock.wait_idle()
    import base64

    async def send_mulaw(pcm: bytes) -> None:
        data = pcm16_to_mulaw(pcm)
        for i in range(0, len(data), 160):
            payload = base64.b64encode(data[i : i + 160]).decode()
            await h.mock.session.on_event({"event": "media", "media": {"payload": payload}})
            await asyncio.sleep(0)

    await send_mulaw(tone(600, 8000))
    await send_mulaw(silence(900, 8000))
    await h.mock.wait_idle()
    assert h.stt.calls and h.stt.calls[-1][1] == 8000
