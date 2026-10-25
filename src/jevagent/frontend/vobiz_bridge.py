"""One phone call: Vobiz stream events in, Vobiz stream messages out (FE-1, FE-2, FE-5).

Flow: media frames -> VAD/turn detector -> (ack clip) -> STT -> Controller -> clip, or handoff to
the Phase B agent whose text is synthesised sentence by sentence. Caller speech cuts playback
(`clearAudio`) as soon as the turn detector reports it; inbound audio is never dropped.

Known v1 limitation: if the caller starts talking again while their previous turn is still in
STT/Jev (sub-second), the new speech does not cancel that turn; its reply may overlap.
"""

import asyncio
import base64
import contextlib
import re
from collections.abc import AsyncIterator, Awaitable, Callable

import structlog

from jevagent.config import Settings
from jevagent.controller.controller import (
    Controller,
    PassToPhaseB,
    PlayClip,
    StartHandoff,
)
from jevagent.controller.packet import HandoffPacket
from jevagent.frontend.audio import frame_bytes, to_pcm16
from jevagent.frontend.stt import Transcriber, TranscriptionError
from jevagent.frontend.tts import OUT_RATE, Synthesizer
from jevagent.frontend.turn_detect import SpeechStarted, TurnDetector, TurnEnded, silence_ms_for
from jevagent.frontend.vad import Vad
from jevagent.models import JsonDict
from jevagent.observability.metrics import MetricsSink, TurnTimer
from jevagent.phase_a.ack import AckPolicy, Speculator
from jevagent.phase_a.clip_player import ClipPlayer
from jevagent.phase_b.agent import AgentOutput, PhaseBAgent, Say, Transferred
from jevagent.script.schema import Script

log = structlog.get_logger(__name__)

FRAME_MS = 20
PLAY_CONTENT_TYPE = "audio/x-l16"
CHECKPOINT_TIMEOUT_S = 10.0
_SENTENCE_END = re.compile(r"(?<=[.?!।])\s+")
MAX_SENTENCE_CHARS = 160

Send = Callable[[JsonDict], Awaitable[None]]
Close = Callable[[], Awaitable[None]]
AgentFactory = Callable[[HandoffPacket], PhaseBAgent]


class CallSession:
    def __init__(
        self,
        *,
        call_id: str,
        script: Script,
        settings: Settings,
        controller: Controller,
        transcriber: Transcriber,
        synth: Synthesizer,
        clips: ClipPlayer,
        agent_factory: AgentFactory,
        vad: Vad,
        send: Send,
        close: Close,
        metrics: MetricsSink,
        ack: AckPolicy | None = None,
        speculator: Speculator | None = None,
        timer_slot: "TimerSlot | None" = None,
    ) -> None:
        self.call_id = call_id
        self._script, self._settings = script, settings
        self._controller = controller
        self._stt, self._synth, self._clips = transcriber, synth, clips
        self._agent_factory = agent_factory
        self._vad = vad
        self._send, self._close = send, close
        self._metrics = metrics
        self._ack = ack or AckPolicy(script, enabled=settings.ack_enabled)
        self._speculator = speculator
        self._timer_slot = timer_slot or TimerSlot()

        self._stream_id = ""
        self._encoding = "audio/x-l16"
        self._rate = settings.inbound_sample_rate
        self._detector: TurnDetector | None = None
        self._inbuf = b""
        self._queue: asyncio.Queue[tuple[bytes, TurnTimer]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._current: asyncio.Task[None] | None = None
        self._agent: PhaseBAgent | None = None
        self._turn = 0
        self._speaking = False  # we are mid-way through sending audio for a reply
        self._audio_pending = False  # sent audio not yet confirmed played by Vobiz
        self._played = asyncio.Event()
        self._closed = False
        self._last_stt_failed = False
        self._inflight = 0

    @property
    def idle(self) -> bool:
        return self._inflight == 0

    # -- inbound events ------------------------------------------------------

    async def on_event(self, msg: JsonDict) -> None:
        event = msg.get("event")
        if event == "start":
            await self._on_start(msg["start"])
        elif event == "media":
            await self._on_media(msg["media"]["payload"])
        elif event == "playedStream":
            self._audio_pending = False
            self._played.set()
        elif event == "stop":
            await self.on_stop()
        elif event in ("clearedAudio", "dtmf"):
            log.debug("vobiz_event", event=event)
        else:
            log.debug("vobiz_event_ignored", event=event)

    async def _on_start(self, start: JsonDict) -> None:
        self._stream_id = start["streamId"]
        fmt = start.get("mediaFormat", {})
        self._encoding = str(fmt.get("encoding", self._encoding))
        self._rate = int(fmt.get("sampleRate", self._rate))
        log.info("call_start", call_id=self.call_id, encoding=self._encoding, rate=self._rate)
        s = self._settings
        self._detector = TurnDetector(
            self._vad,
            frame_ms=FRAME_MS,
            barge_in_frames=s.barge_in_frames,
            min_speech_frames=s.min_speech_frames,
            silence_ms=silence_ms_for(s, self._script.language),
        )
        self._worker = asyncio.create_task(self._work())
        greeting = self._begin_turn_timer()  # turn 0
        greeting.mark("turn_end")
        await self._play_clip(self._controller.start().clip_id, greeting)
        await self._finish_turn()
        self._metrics.write(greeting.timings)

    async def _on_media(self, payload_b64: str) -> None:
        if self._detector is None:
            return
        pcm = to_pcm16(base64.b64decode(payload_b64), self._encoding)
        self._inbuf += pcm
        size = frame_bytes(self._rate, FRAME_MS)
        while len(self._inbuf) >= size:
            frame, self._inbuf = self._inbuf[:size], self._inbuf[size:]
            for ev in self._detector.feed(frame):
                if isinstance(ev, SpeechStarted):
                    await self._barge_in()
                elif isinstance(ev, TurnEnded):
                    timer = self._begin_turn_timer()
                    timer.mark("turn_end")
                    self._inflight += 1
                    self._queue.put_nowait((ev.pcm, timer))

    async def on_stop(self) -> None:
        self._closed = True
        if self._speculator:
            self._speculator.cancel()
        for task in (self._current, self._worker):
            if task and not task.done():
                task.cancel()
        log.info("call_stop", call_id=self.call_id)

    # -- barge-in ---------------------------------------------------------------

    async def _barge_in(self) -> None:
        if not (self._speaking or self._audio_pending):
            return
        if self._speculator:
            self._speculator.on_speech_resumed()
        if self._speaking and self._current and not self._current.done():
            self._current.cancel()
        await self._send({"event": "clearAudio", "streamId": self._stream_id})
        self._audio_pending = False
        self._speaking = False
        log.info("barge_in", call_id=self.call_id)

    # -- turn processing ----------------------------------------------------------

    def _begin_turn_timer(self) -> TurnTimer:
        timer = TurnTimer(self.call_id, self._turn)
        self._turn += 1
        self._timer_slot.current = timer
        return timer

    async def _work(self) -> None:
        while not self._closed:
            pcm, timer = await self._queue.get()
            self._current = asyncio.create_task(self._handle(pcm, timer))
            try:
                await self._current
            except asyncio.CancelledError:
                if self._closed:
                    raise
                log.info("turn_cancelled_by_barge_in", call_id=self.call_id)
            except Exception:
                log.exception("turn_failed", call_id=self.call_id)
            finally:
                self._metrics.write(timer.timings)
                self._inflight -= 1

    async def _handle(self, pcm: bytes, timer: TurnTimer) -> None:
        self._timer_slot.current = timer
        self._speaking = False
        ack = self._ack.pick(self._controller.state.mode)
        if ack:
            await self._play_clip(ack, timer)
        text = await self._transcribe(pcm)
        timer.mark("stt_done")
        if not text and not self._last_stt_failed:
            log.info("empty_transcript_ignored", call_id=self.call_id)
            return
        action = await self._controller.on_caller_turn(
            text, self._speculator.take(text) if self._speculator else None
        )
        if isinstance(action, PlayClip):
            await self._play_clip(action.clip_id, timer)
            await self._finish_turn()
            if action.end_call:
                await self._end_call()
        elif isinstance(action, StartHandoff):
            self._agent = self._agent_factory(action.packet)
            await self._play_clip(self._agent.begin().clip_id, timer)
            await self._speak(self._agent.first_turn(), timer)
        elif isinstance(action, PassToPhaseB) and self._agent is not None:
            await self._speak(self._agent.on_caller_turn(action.utterance), timer)

    async def _transcribe(self, pcm: bytes) -> str:
        self._last_stt_failed = False
        try:
            return await self._stt.transcribe(pcm, self._rate)
        except TranscriptionError:
            self._last_stt_failed = True
            return ""  # Jev sees an empty utterance -> unclear -> re-ask, then handoff

    # -- Phase B output -----------------------------------------------------------

    async def _speak(self, outputs: AsyncIterator[AgentOutput], timer: TurnTimer) -> None:
        buffer = ""
        transferred = False
        async for out in outputs:
            if isinstance(out, Say):
                buffer += out.text
                *sentences, buffer = _split_sentences(buffer)
                for sentence in sentences:
                    await self._say(sentence, timer)
            elif isinstance(out, PlayClip):
                if buffer.strip():
                    await self._say(buffer, timer)
                    buffer = ""
                await self._play_clip(out.clip_id, timer)
            elif isinstance(out, Transferred):
                transferred = True
                log.info("transfer_to_human", call_id=self.call_id, reason=out.reason)
        if buffer.strip():
            await self._say(buffer, timer)
        await self._finish_turn()
        if transferred:
            await self._end_call()

    async def _say(self, text: str, timer: TurnTimer) -> None:
        async for chunk in self._synth.stream(text.strip()):
            await self._send_pcm(chunk, timer)

    # -- outbound audio -------------------------------------------------------------

    async def _play_clip(self, clip_id: str, timer: TurnTimer | None = None) -> None:
        await self._send_pcm(self._clips.pcm(clip_id), timer or self._timer_slot.current)

    async def _send_pcm(self, pcm: bytes, timer: TurnTimer | None) -> None:
        step = frame_bytes(OUT_RATE, FRAME_MS)
        for i in range(0, len(pcm), step):
            if timer:
                timer.mark("first_audio_out")
            self._speaking = True
            await self._send(
                {
                    "event": "playAudio",
                    "streamId": self._stream_id,
                    "media": {
                        "contentType": PLAY_CONTENT_TYPE,
                        "sampleRate": OUT_RATE,
                        "payload": base64.b64encode(pcm[i : i + step]).decode("ascii"),
                    },
                }
            )
            await asyncio.sleep(0)  # cancellation point for barge-in

    async def _finish_turn(self) -> None:
        if not self._speaking:
            return
        self._audio_pending = True
        self._played.clear()
        await self._send(
            {"event": "checkpoint", "streamId": self._stream_id, "name": f"turn-{self._turn}"}
        )
        self._speaking = False

    async def _end_call(self) -> None:
        """Let queued audio finish playing, then close the stream so Vobiz hangs up."""
        if self._audio_pending:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._played.wait(), CHECKPOINT_TIMEOUT_S)
        self._closed = True
        await self._close()


class TimerSlot:
    """Shared pointer to the current turn's timer, read by TimedJev."""

    def __init__(self) -> None:
        self.current: TurnTimer | None = None


def _split_sentences(text: str) -> list[str]:
    """Complete sentences first, then the unfinished tail (possibly empty) as the last item."""
    parts = _SENTENCE_END.split(text)
    out: list[str] = []
    for part in parts[:-1]:
        out.append(part)
    tail = parts[-1]
    if len(tail) > MAX_SENTENCE_CHARS:
        cut = tail.rfind(" ", 0, MAX_SENTENCE_CHARS)
        if cut > 0:
            out.append(tail[:cut])
            tail = tail[cut:].lstrip()
    out.append(tail)
    return out
