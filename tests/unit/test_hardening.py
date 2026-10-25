"""Phase 9: routing, failure drills, compliance and the A/B gate."""

import asyncio
from pathlib import Path

import pytest

# Reuse the front-end harness (pytest puts this directory on sys.path).
from test_frontend import SCRIPT, Harness  # type: ignore[import-not-found]

from jevagent.ab import AbRouter, settings_for_arm
from jevagent.compliance import (
    ComplianceConfig,
    Disclosure,
    audit,
    check_call,
    check_script,
    load_config,
)
from jevagent.config import Settings
from jevagent.eval.ab_report import (
    Gates,
    arm_stats,
    evaluate_gates,
    record_decision,
)
from jevagent.frontend.stt import TranscriptionError
from jevagent.frontend.tts import SynthesisError
from jevagent.jev.client import JevConfigError
from jevagent.observability.report import Rates
from jevagent.observability.summary import CallSummary
from jevagent.phase_b.agent import LlmEvent, TextDelta

# -- routing ---------------------------------------------------------------------------


def test_router_is_sticky_and_respects_percentage(tmp_path: Path) -> None:
    router = AbRouter(30, tmp_path / "KILL")
    ids = [f"call-{i}" for i in range(2000)]
    arms = [router.assign(i) for i in ids]
    assert arms == [router.assign(i) for i in ids]  # sticky
    share = arms.count("hybrid") / len(arms)
    assert 0.25 < share < 0.35
    assert {AbRouter(0, tmp_path / "K").assign(i) for i in ids} == {"baseline"}
    assert {AbRouter(100, tmp_path / "K").assign(i) for i in ids} == {"hybrid"}
    with pytest.raises(ValueError):
        AbRouter(101, tmp_path / "K")


def test_kill_switch_forces_baseline_without_restart(tmp_path: Path) -> None:
    kill = tmp_path / "KILL"
    router = AbRouter(100, kill)
    assert router.assign("x") == "hybrid" and not router.killed
    kill.write_text("", encoding="utf-8")
    assert router.killed and router.assign("x") == "baseline"
    kill.unlink()
    assert router.assign("x") == "hybrid"


def test_settings_for_arm() -> None:
    s = Settings(_env_file=None, shadow_mode=True, baseline_mode=True)  # type: ignore[call-arg]
    assert settings_for_arm(s, "hybrid").baseline_mode is False
    assert settings_for_arm(s, "hybrid").shadow_mode is False
    assert settings_for_arm(s, "baseline").baseline_mode is True
    assert s.baseline_mode is True  # original untouched


# -- failure drills --------------------------------------------------------------------


class DownJev:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def ask(self, state, questions):  # type: ignore[no-untyped-def]
        raise self.exc


async def test_drill_jev_outage_hands_off_instead_of_going_silent(tmp_path: Path) -> None:
    rounds: list[list[LlmEvent]] = [[TextDelta(text="Ek minute.")]]
    h = Harness(tmp_path, ["haan ji"], rounds)
    await h.setup()
    h.jev.inner.ask = DownJev(JevConfigError("HTTP 401")).ask  # type: ignore[method-assign]
    mark = len(h.mock.sent)
    await h.mock.say()
    assert h.controller.state.mode.value == "B"
    assert h.synth.texts == ["Ek minute."]  # the LLM took over and spoke
    assert h.mock.audio_since(mark)
    await h.mock.session.on_stop()
    assert h.summaries.rows[0].handoff_reason == "jev_unavailable"


async def test_drill_llm_outage_plays_fallback_and_transfers(tmp_path: Path) -> None:
    h = await Harness(tmp_path, ["penalty kyun"], []).setup()  # no LLM rounds: stream() raises
    await h.mock.say()
    await asyncio.sleep(0.05)
    assert len(h.transfers.transfers) == 1 and h.transfers.transfers[0][1] == "llm_failure"
    assert h.player.pcm("fallback.1") in h.mock.audio_since()
    assert h.mock.closed
    await h.mock.session.on_stop()
    assert h.summaries.rows[0].outcome == "transferred"


async def test_drill_stt_down_reasks_then_hands_off_and_phase_b_asks_to_repeat(
    tmp_path: Path,
) -> None:
    errs: list[str | Exception] = [TranscriptionError("503")] * 3
    rounds: list[list[LlmEvent]] = [[TextDelta(text="Dobara boliye.")], [TextDelta(text="Ji.")]]
    h = await Harness(tmp_path, errs, rounds).setup()
    await h.mock.say()  # Phase A: re-ask
    await h.mock.say()  # Phase A: retry limit -> handoff
    assert h.controller.state.mode.value == "B"
    await h.mock.say()  # Phase B: STT still down -> agent is told [inaudible]
    sent_to_llm = [m.content for call in h.llm.calls for m in call if m.role == "user"]
    assert "[inaudible]" in sent_to_llm


async def test_drill_tts_failure_falls_back_to_human(tmp_path: Path) -> None:
    class BrokenSynth:
        voice, model = "fake", "fake-tts"

        async def synthesize(self, text: str) -> bytes:
            raise SynthesisError("quota")

        async def stream(self, text: str):  # type: ignore[no-untyped-def]
            raise SynthesisError("quota")
            yield b""

    rounds: list[list[LlmEvent]] = [[TextDelta(text="Penalty ke baare mein. Dekhte hain.")]]
    h = Harness(tmp_path, ["penalty kyun"], rounds)
    await h.setup()
    h.mock.session._synth = BrokenSynth()  # type: ignore[assignment]  # noqa: SLF001
    await h.mock.say()
    await asyncio.sleep(0.05)
    assert h.player.pcm("fallback.1") in h.mock.audio_since()
    assert len(h.transfers.transfers) == 1 and h.transfers.transfers[0][1] == "tts_failure"
    assert h.mock.closed


async def test_drill_telephony_disconnect_mid_call(tmp_path: Path) -> None:
    h = await Harness(tmp_path, ["haan ji"], auto_played=False).setup()
    await h.mock.session.on_event({"event": "stop"})  # Vobiz ends the stream
    await h.mock.session.on_event({"event": "stop"})  # duplicate stop is harmless
    assert len(h.summaries.rows) == 1 and h.summaries.rows[0].outcome == "dropped"


# -- compliance --------------------------------------------------------------------------

REQUIRED = "Ye call quality ke liye record ki ja rahi hai."
CONFIG = ComplianceConfig(
    disclosures=[
        Disclosure(id="disclosure.1", text=REQUIRED, required_on="all"),
        Disclosure(id="closing.ask.1", text="Dhanyavaad", required_on="completed"),
    ]
)


def script_with_disclosure(text: str):  # type: ignore[no-untyped-def]
    from jevagent.script.schema import Clip

    s = SCRIPT.model_copy(deep=True)
    s.shared["disclosure"] = [Clip(id="disclosure.1", text=text)]
    return s


def test_script_check_requires_verbatim_wording() -> None:
    assert check_script(script_with_disclosure(f"Namaste. {REQUIRED}"), CONFIG) == []
    bad = check_script(script_with_disclosure("Ye call record ho rahi hai."), CONFIG)
    assert bad and "verbatim" in bad[0]
    missing = check_script(SCRIPT, CONFIG)
    assert any("disclosure.1: no such clip" in p for p in missing)


def test_call_check_flags_missing_disclosure_clips() -> None:
    ok = CallSummary(
        call_id="a",
        mode="hybrid",
        outcome="completed",
        clips_played=["disclosure.1", "closing.ask.1"],
    )
    no_disc = CallSummary(
        call_id="b", mode="hybrid", outcome="completed", clips_played=["closing.ask.1"]
    )
    dropped = CallSummary(
        call_id="c", mode="hybrid", outcome="dropped", clips_played=["disclosure.1"]
    )  # closing only required if completed
    assert check_call(ok, CONFIG) == [] and check_call(dropped, CONFIG) == []
    assert check_call(no_disc, CONFIG) == ["disclosure.1"]
    report = audit(script_with_disclosure(REQUIRED), CONFIG, [ok, no_disc, dropped])
    assert report == {"b": ["disclosure.1"]}
    assert "" in audit(SCRIPT, CONFIG, [ok])  # script problems reported under ""


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "compliance.yaml"
    path.write_text(
        "disclosures:\n  - {id: disclosure.1, text: 'x y', required_on: all}\n", encoding="utf-8"
    )
    assert load_config(path).disclosures[0].text == "x y"


async def test_call_records_which_clips_played(tmp_path: Path) -> None:
    h = await Harness(tmp_path, ["haan ji"]).setup()
    await h.mock.say()
    await h.mock.session.on_stop()
    played = h.summaries.rows[0].clips_played
    assert played[0].startswith("identity.ask") and played[1].startswith("pay_when.ask")


# -- A/B gates ---------------------------------------------------------------------------


def calls(mode: str, n_complete: int, n_total: int, handoffs: int = 0, stuck: int = 0):  # type: ignore[no-untyped-def]
    rows = []
    for i in range(n_total):
        done = i < n_complete
        rows.append(
            CallSummary(
                call_id=f"{mode}{i}",
                mode=mode,  # type: ignore[arg-type]
                outcome="completed" if done else "dropped",
                handoff_reason="x" if i < handoffs else None,
                turns=4 if i >= n_total - stuck else 2,
                tts_chars=1000,
                stt_requests=3,
            )
        )
    return rows


GATES = Gates(
    max_completion_drop=0.05,
    max_repeat_rate=0.1,
    min_ttfa_improvement_pct=20,
    max_cost_ratio=1.0,
    max_stuck_rate=0.1,
)
RATES = Rates(stt_per_request=0.01, tts_per_1k_chars=0.1)


def stats(hy, ba, hy_ttfa=400.0, ba_ttfa=1000.0, repeated=None):  # type: ignore[no-untyped-def]
    metrics = [
        {"call_id": s.call_id, "turn": 1, "ttfa_ms": hy_ttfa if s.mode == "hybrid" else ba_ttfa}
        for s in hy + ba
    ]
    return arm_stats(hy + ba, metrics, RATES, repeated)


def test_gates_pass_when_hybrid_is_better() -> None:
    hy, ba = calls("hybrid", 9, 10, handoffs=3), calls("baseline", 9, 10)
    # make hybrid cheaper: fewer TTS chars
    for s in hy:
        s.tts_chars = 200
    repeated = {s.call_id: False for s in hy if s.handoff_reason}
    arms = stats(hy, ba, repeated=repeated)
    result = evaluate_gates(arms["hybrid"], arms["baseline"], GATES)
    assert result.passed, result
    assert arms["hybrid"].repeat_rate == 0.0 and arms["hybrid"].stuck_rate == 0.0


def test_gates_report_each_failure() -> None:
    hy, ba = calls("hybrid", 5, 10, handoffs=3, stuck=2), calls("baseline", 9, 10)
    for c in hy:
        c.tts_chars = 3000  # costlier than baseline
    repeated = {s.call_id: True for s in hy if s.handoff_reason}
    arms = stats(hy, ba, hy_ttfa=900.0, repeated=repeated)
    result = evaluate_gates(arms["hybrid"], arms["baseline"], GATES)
    text = " | ".join(result.failures)
    assert not result.passed
    for needle in ("completion", "repeat rate", "TTFA", "cost ratio", "stuck rate"):
        assert needle in text, needle


def test_unmeasured_gates_do_not_count_as_a_pass() -> None:
    hy, ba = calls("hybrid", 9, 10, handoffs=3), calls("baseline", 9, 10)
    arms = arm_stats(hy + ba, [], Rates())  # no latency rows, no rates, no judge
    result = evaluate_gates(arms["hybrid"], arms["baseline"], GATES)
    assert not result.passed and not result.failures
    assert {"TTFA", "repeat rate (needs judged handed-off calls)"} <= set(result.unmeasured)
    assert any(u.startswith("cost") for u in result.unmeasured)


def test_decision_is_recorded(tmp_path: Path) -> None:
    hy, ba = calls("hybrid", 5, 10), calls("baseline", 9, 10)
    arms = stats(hy, ba)
    result = evaluate_gates(arms["hybrid"], arms["baseline"], GATES)
    path = tmp_path / "docs" / "decision.md"
    record_decision(path, result, decided_by="product owner", note="rerun after fixes")
    record_decision(path, result, decided_by="product owner")
    text = path.read_text(encoding="utf-8")
    assert text.count("NO-GO (recorded by product owner)") == 2
    assert "- failed: completion" in text and "- note: rerun after fixes" in text


def test_console_live_log_echoes_and_persists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import UTC, datetime

    from jevagent.frontend.console import LiveLog
    from jevagent.models import JevAnswer
    from jevagent.observability.decision_log import DecisionRecord

    record = DecisionRecord(
        call_id="c",
        turn=2,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        model_version="jev-1.14.0",
        state_sent={},
        questions=[],
        answers={
            "answered_step": JevAnswer(key="answered_step", selected="yes", confidence=0.9),
            "turn_type": JevAnswer(key="turn_type", selected="objection", confidence=0.4),
        },
        action={"type": "StartHandoff", "packet": {"handoff_reason": "turn_type=objection"}},
    )
    LiveLog(str(tmp_path / "c.jsonl"), 0.7, echo=True).write(record)
    out = capsys.readouterr().out
    assert "answered_step" in out and "yes" in out
    assert "turn_type" in out and "below threshold" in out.split("turn_type")[1]
    assert "StartHandoff turn_type=objection" in out
    assert (tmp_path / "c.jsonl").read_text(encoding="utf-8").count("\n") == 1

    LiveLog(str(tmp_path / "q.jsonl"), 0.7, echo=False).write(record)
    assert capsys.readouterr().out == ""
    assert (tmp_path / "q.jsonl").exists()
