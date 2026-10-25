import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from jevagent.config import Settings
from jevagent.controller.controller import Controller
from jevagent.frontend.server import answer_xml, create_app
from jevagent.frontend.stt import FakeTranscriber
from jevagent.frontend.tts import FakeSynthesizer
from jevagent.frontend.vad import EnergyVad
from jevagent.frontend.vobiz_bridge import CallSession, Close, Send
from jevagent.jev.fake import RuleBasedJev
from jevagent.observability.decision_log import MemoryDecisionLog
from jevagent.observability.metrics import MemoryMetrics
from jevagent.observability.summary import MemorySummaries
from jevagent.phase_a.clip_player import ClipPlayer
from jevagent.phase_a.clip_render import render_clips
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder.yaml")
SETTINGS = Settings(_env_file=None, jev_api_key=SecretStr("k"))  # type: ignore[call-arg]


def test_answer_xml_matches_documented_shape() -> None:
    xml = answer_xml("https://abc.ngrok.app/", 16000)
    assert xml.startswith('<Response><Stream bidirectional="true"')
    assert 'audioTrack="inbound"' in xml and 'keepCallAlive="true"' in xml
    assert 'contentType="audio/x-l16;rate=16000"' in xml
    assert 'statusCallbackUrl="https://abc.ngrok.app/stream-status"' in xml
    assert ">wss://abc.ngrok.app/ws</Stream><Hangup/></Response>" in xml
    assert "&" not in xml  # bare ampersands break Vobiz's XML parser


async def test_app_endpoints_and_websocket_call(tmp_path: Path) -> None:
    await render_clips(SCRIPT, FakeSynthesizer(), tmp_path)
    clips = ClipPlayer.load(tmp_path)
    summaries = MemorySummaries()

    def builder(call_id: str, send: Send, close: Close) -> CallSession:
        return CallSession(
            call_id=call_id,
            script=SCRIPT,
            settings=SETTINGS,
            controller=Controller(
                SCRIPT, RuleBasedJev({}), SETTINGS, MemoryDecisionLog(), call_id, seed=1
            ),
            transcriber=FakeTranscriber([]),
            synth=FakeSynthesizer(),
            clips=clips,
            agent_factory=lambda packet: (_ for _ in ()).throw(AssertionError("no handoff")),
            vad=EnergyVad(500),
            send=send,
            close=close,
            metrics=MemoryMetrics(),
            summaries=summaries,
        )

    client = TestClient(create_app(builder, "https://host.example"))
    assert client.get("/health").json() == {"status": "ok"}
    for method in (client.get, client.post):
        resp = method("/answer")
        assert resp.status_code == 200 and "wss://host.example/ws" in resp.text
        assert resp.headers["content-type"].startswith("application/xml")
    assert client.post("/stream-status", content=b"StartStream").json() == {"status": "ok"}

    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "event": "start",
                "start": {
                    "streamId": "s1",
                    "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 16000},
                },
            }
        )
        first = ws.receive_json()
        assert first["event"] == "playAudio" and first["streamId"] == "s1"
        events = [first["event"]]
        while events[-1] != "checkpoint":
            events.append(ws.receive_json()["event"])
        assert set(events) == {"playAudio", "checkpoint"}
        ws.send_json({"event": "playedStream"})
    # the client hung up mid-call: the server must still close the call out
    for _ in range(100):
        if summaries.rows:
            break
        await asyncio.sleep(0.02)
    assert [r.outcome for r in summaries.rows] == ["dropped"]
