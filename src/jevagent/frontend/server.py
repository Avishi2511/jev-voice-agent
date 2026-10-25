"""FastAPI app for Vobiz: /answer returns the Stream XML, /ws carries the call audio."""

import json
import uuid
from collections.abc import Callable
from datetime import date

import structlog
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from google import genai

from jevagent.ab import AbRouter, settings_for_arm
from jevagent.config import Settings
from jevagent.controller.controller import Controller
from jevagent.eval.shadow import JsonlShadow, ShadowRecorder
from jevagent.frontend.llm import GeminiLlmClient, check_models
from jevagent.frontend.stt import GeminiTranscriber
from jevagent.frontend.tts import GeminiSynthesizer
from jevagent.frontend.vad import EnergyVad
from jevagent.frontend.vobiz_bridge import CallSession, Close, Send, TimerSlot
from jevagent.jev.client import HttpJevClient
from jevagent.observability.decision_log import JsonlDecisionLog
from jevagent.observability.logging import configure_logging
from jevagent.observability.metrics import JsonlMetrics, TimedJev
from jevagent.observability.summary import JsonlSummaries, JsonlTranscripts, Usage
from jevagent.phase_a.ack import AckPolicy
from jevagent.phase_a.clip_player import ClipPlayer
from jevagent.phase_b.agent import PhaseBAgent
from jevagent.phase_b.tools import MockCrm, RecordingTransfer, ToolRegistry
from jevagent.script.loader import load_script

log = structlog.get_logger(__name__)

SessionBuilder = Callable[[str, Send, Close], CallSession]
TOOLS = ["get_loan_account", "explain_charges", "schedule_callback", "transfer_to_human"]


def answer_xml(public_url: str, inbound_rate: int) -> str:
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    return (
        "<Response>"
        f'<Stream bidirectional="true" audioTrack="inbound" keepCallAlive="true" '
        f'contentType="audio/x-l16;rate={inbound_rate}" '
        f'statusCallbackUrl="{public_url.rstrip("/")}/stream-status" statusCallbackMethod="POST">'
        f"{ws_url}/ws</Stream><Hangup/></Response>"
    )


def create_app(builder: SessionBuilder, public_url: str, inbound_rate: int = 16000) -> FastAPI:
    app = FastAPI(title="jev-voice-agent")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/answer", methods=["GET", "POST"])
    async def answer() -> Response:
        return Response(answer_xml(public_url, inbound_rate), media_type="application/xml")

    @app.post("/stream-status")
    async def stream_status(request: Request) -> dict[str, str]:
        log.info("stream_status", body=(await request.body()).decode("utf-8", "replace")[:500])
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        session = builder(str(uuid.uuid4()), socket.send_json, socket.close)
        try:
            while True:
                await session.on_event(json.loads(await socket.receive_text()))
        except WebSocketDisconnect:
            pass
        finally:
            await session.on_stop()

    return app


def build_real_builder(settings: Settings) -> SessionBuilder:
    script = load_script(settings.script_path)
    clips = ClipPlayer.load(f"{settings.clips_dir}/{script.name}")
    clips.verify(script, settings)
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    transcriber = GeminiTranscriber(client, settings.gemini_stt_model)
    synth = GeminiSynthesizer(client, settings.gemini_tts_model, settings.gemini_tts_voice)
    disclosure = {c.id for c in script.shared.get("disclosure", [])}
    closing = {c.id for s in script.steps if s.slot is None for c in s.clips.ask}
    metrics = JsonlMetrics(settings.metrics_path)
    summaries = JsonlSummaries(settings.summaries_path)
    transcripts = JsonlTranscripts(settings.transcripts_path) if settings.save_transcripts else None
    http_jev = HttpJevClient(settings)  # one pooled HTTP client shared by all calls

    router = (
        AbRouter(settings.ab_hybrid_percent, settings.kill_switch_path)
        if settings.ab_hybrid_percent is not None
        else None
    )

    def build(call_id: str, send: Send, close: Close) -> CallSession:
        cfg = settings_for_arm(settings, router.assign(call_id)) if router else settings
        slot = TimerSlot()
        usage = Usage()
        jev = TimedJev(http_jev, lambda: slot.current, usage)
        controller = Controller(
            script,
            jev,
            cfg,
            JsonlDecisionLog(settings.decision_log_path),
            call_id,
            tools=TOOLS,
        )

        def agent_factory(packet):  # type: ignore[no-untyped-def]
            registry = ToolRegistry(
                packet,
                MockCrm(),
                RecordingTransfer(),
                allowed_clips=disclosure | closing,
                today=date.today,
            )
            return PhaseBAgent(
                GeminiLlmClient(client, settings.gemini_llm_model, usage),
                registry,
                packet,
                script,
                cfg,
            )

        shadow = None
        if cfg.shadow_mode:
            shadow_controller = Controller(
                script,
                http_jev,
                settings,
                JsonlDecisionLog(settings.shadow_decisions_path),
                call_id,
                tools=TOOLS,
            )
            shadow = ShadowRecorder(shadow_controller, JsonlShadow(settings.shadow_path), call_id)

        return CallSession(
            call_id=call_id,
            script=script,
            settings=cfg,
            controller=controller,
            transcriber=transcriber,
            synth=synth,
            clips=clips,
            agent_factory=agent_factory,
            vad=EnergyVad(settings.vad_energy_threshold),
            send=send,
            close=close,
            metrics=metrics,
            ack=AckPolicy(script, cfg.ack_enabled),
            timer_slot=slot,
            summaries=summaries,
            usage=usage,
            tools=TOOLS,
            shadow=shadow,
            transcripts=transcripts,
        )

    return build


async def startup_checks(settings: Settings) -> None:
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    await check_models(
        client,
        {
            "stt": settings.gemini_stt_model,
            "llm": settings.gemini_llm_model,
            "tts": settings.gemini_tts_model,
        },
    )


def main() -> None:
    import asyncio

    import uvicorn

    settings = Settings()
    configure_logging(settings.log_level)
    if (
        not settings.gemini_api_key.get_secret_value()
        or not settings.jev_api_key.get_secret_value()
    ):
        raise SystemExit("GEMINI_API_KEY and JEV_API_KEY must be set (see .env.example)")
    if not settings.public_url:
        raise SystemExit("PUBLIC_URL must be set to the https URL Vobiz can reach")
    asyncio.run(startup_checks(settings))
    app = create_app(
        build_real_builder(settings), settings.public_url, settings.inbound_sample_rate
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104 - must accept Vobiz connections


if __name__ == "__main__":
    main()
