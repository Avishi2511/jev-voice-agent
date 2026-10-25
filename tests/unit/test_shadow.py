import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from jevagent.config import Settings
from jevagent.controller.controller import Controller
from jevagent.eval.export_labels import export_rows, load_decisions, write_jsonl
from jevagent.eval.judge import judge_call, judge_questions
from jevagent.eval.shadow import (
    GroundTruth,
    JsonlShadow,
    MemoryShadow,
    ShadowRecorder,
    ShadowTurn,
    compare_shadow,
)
from jevagent.jev.fake import RuleBasedJev, ScriptedJev, answer
from jevagent.models import JevAnswer, JevQuestion, JsonDict
from jevagent.observability.decision_log import JsonlDecisionLog, MemoryDecisionLog
from jevagent.observability.summary import CallSummary, JsonlTranscripts, MemoryTranscripts
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder_en.yaml")
SETTINGS = Settings(_env_file=None, jev_api_key=SecretStr("k"))  # type: ignore[call-arg]


def rule_jev() -> RuleBasedJev:
    return RuleBasedJev(
        slot_keywords={
            "identity_confirmed": {"confirmed": ["yes"]},
            "pay_when": {"today": ["today"], "tomorrow": ["tomorrow"]},
            "payment_mode": {"upi": ["upi"]},
        },
        turn_keywords={"objection": ["penalty"]},
    )


def recorder(call_id: str = "c1", jev: object = None) -> tuple[ShadowRecorder, MemoryShadow]:
    sink = MemoryShadow()
    ctl = Controller(SCRIPT, jev or rule_jev(), SETTINGS, MemoryDecisionLog(), call_id, seed=1)  # type: ignore[arg-type]
    return ShadowRecorder(ctl, sink, call_id), sink


async def test_shadow_logs_what_phase_a_would_do() -> None:
    rec, sink = recorder()
    for i, text in enumerate(
        ["yes that is me", "by tomorrow", "why was the penalty charged"], start=1
    ):
        rec.submit(text, metric_turn=i, stt_ms=100.0)
    await rec.close()
    assert [t.would_do for t in sink.rows] == ["clip", "clip", "handoff"]
    assert [t.turn for t in sink.rows] == [1, 2, 3]
    assert sink.rows[0].clip_id and sink.rows[0].step == "pay_when"
    assert sink.rows[1].slots["pay_date_bucket"] == "tomorrow"
    assert sink.rows[2].handoff_reason == "turn_type=objection"
    assert sink.rows[0].stt_ms == 100.0 and sink.rows[0].jev_ms >= 0


async def test_shadow_turns_after_handoff_and_end() -> None:
    rec, sink = recorder()
    for text in ["why the penalty", "yes"]:
        rec.submit(text, metric_turn=1)
    await rec.close()
    assert [t.would_do for t in sink.rows] == ["handoff", "phase_b"]

    rec, sink = recorder("c2")
    for text in ["yes", "today", "upi"]:
        rec.submit(text, metric_turn=1)
    await rec.close()
    assert sink.rows[-1].would_do == "end"


async def test_shadow_failure_never_propagates() -> None:
    def boom(state: JsonDict, qs: list[JevQuestion]) -> dict[str, JevAnswer]:
        raise RuntimeError("jev exploded")

    rec, sink = recorder(jev=ScriptedJev(handler=boom))
    rec.submit("yes", metric_turn=1)
    rec.submit("yes", metric_turn=2)
    await rec.close()  # no exception
    assert sink.rows == []


async def test_close_without_submissions_and_jsonl_sink(tmp_path: Path) -> None:
    rec, _ = recorder()
    await rec.close()
    sink = JsonlShadow(tmp_path / "s" / "shadow.jsonl")
    sink.write(
        ShadowTurn(
            call_id="c",
            turn=1,
            metric_turn=1,
            utterance="x",
            would_do="clip",
            step="s",
            slots={},
            jev_ms=1.0,
        )
    )
    assert '"would_do":"clip"' in (tmp_path / "s" / "shadow.jsonl").read_text(encoding="utf-8")


def st(
    call: str,
    turn: int,
    would: str,
    step: str,
    slots: dict[str, str | None] | None = None,
    jev_ms: float = 200.0,
    stt_ms: float | None = 300.0,
) -> ShadowTurn:
    return ShadowTurn(
        call_id=call,
        turn=turn,
        metric_turn=turn,
        utterance="u",
        would_do=would,  # type: ignore[arg-type]
        step=step,
        slots=slots or {},
        jev_ms=jev_ms,
        stt_ms=stt_ms,
    )


def test_comparison_metrics() -> None:
    turns = [
        # a: clean scripted call, slots right
        st("a", 1, "clip", "pay_when", {"pay": "tomorrow"}),
        st("a", 2, "end", "closing", {"pay": "tomorrow", "mode": "upi"}),
        # b: should hand off at turn 2, did at turn 2
        st("b", 1, "clip", "pay_when", {"pay": None}),
        st("b", 2, "handoff", "pay_when", {"pay": None}),
        # c: should hand off at turn 1, handed off at turn 2 (late)
        st("c", 1, "clip", "pay_when"),
        st("c", 2, "handoff", "pay_when"),
        # d: should NOT hand off, did (false handoff)
        st("d", 1, "handoff", "pay_when"),
        # e: stuck on pay_when for 3 turns without handing off
        st("e", 1, "clip", "pay_when"),
        st("e", 2, "clip", "pay_when"),
        st("e", 3, "clip", "pay_when"),
    ]
    truths = [
        GroundTruth(call_id="a", slots={"pay": "tomorrow", "mode": "cash"}),
        GroundTruth(call_id="b", should_handoff_turn=2),
        GroundTruth(call_id="c", should_handoff_turn=1),
        GroundTruth(call_id="d"),
    ]
    ttfa = {("a", 1): 1500.0, ("a", 2): 1200.0}
    r = compare_shadow(turns, truths, ttfa)
    assert r.calls == 5
    assert r.slot_coverage == 1.0 and r.slot_agreement == 0.5  # pay right, mode wrong
    assert r.handoff_recall == 0.5 and r.late_handoffs == 1
    assert r.false_handoff_rate == 0.5  # a clean, d false -> 1 of 2 clean calls
    assert r.stuck_calls == 1
    assert r.mean_latency_saving_ms == pytest.approx((1500 - 500 + 1200 - 500) / 2)


def test_comparison_handles_empty_inputs() -> None:
    r = compare_shadow([], [], {})
    assert r.calls == 0 and r.slot_agreement is None and r.handoff_recall is None
    assert r.mean_latency_saving_ms is None


# -- judge -----------------------------------------------------------------------------


async def test_judge_combines_jev_and_code_metrics() -> None:
    jev = ScriptedJev(
        turns=[
            {
                "handoff_timing": answer("handoff_timing", "right_time", 0.8),
                "caller_repeated": JevAnswer(
                    key="caller_repeated",
                    selected="no",
                    probabilities={"yes": 0.1, "no": 0.9},
                    confidence=0.9,
                ),
            }
        ]
    )
    result = await judge_call(
        jev,
        "c1",
        ["agent: hi", "caller: hello"],
        handoff_turn=2,
        slots={"pay": "tomorrow", "mode": None},
        crm_truth={"pay": "tomorrow", "mode": "upi"},
        ttfa_ms=[300.0, 900.0, 600.0],
    )
    assert result.handoff_timing == "right_time" and result.caller_repeated == 0.1
    assert result.slot_accuracy == 0.5 and result.ttfa_p50_ms == 600.0 and result.judged_ok
    state, questions = jev.requests[0]
    assert state["transcript"] == ["agent: hi", "caller: hello"] and state["handoff_turn"] == 2
    assert {q.key for q in questions} == {"handoff_timing", "caller_repeated"}


async def test_judge_failure_is_flagged_not_trusted() -> None:
    def failed(state: JsonDict, qs: list[JevQuestion]) -> dict[str, JevAnswer]:
        return {q.key: JevAnswer(key=q.key, selected=None, confidence=0.0) for q in qs}

    result = await judge_call(ScriptedJev(handler=failed), "c", ["x"], None)
    assert not result.judged_ok and result.handoff_timing is None and result.caller_repeated is None
    assert result.slot_accuracy is None
    assert all(len(q.options) > 1 for q in judge_questions() if q.options)


# -- label export ----------------------------------------------------------------------


async def test_export_joins_decisions_with_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    for call in ("known", "unknown"):
        ctl = Controller(SCRIPT, rule_jev(), SETTINGS, JsonlDecisionLog(path), call, seed=1)
        ctl.start()
        await ctl.on_caller_turn("yes that is me")
    rows = export_rows(
        load_decisions(path),
        [CallSummary(call_id="known", mode="hybrid", outcome="completed")],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["call_id"] == "known" and row["call_outcome"] == "completed"
    assert row["label_source"] == "model_decision"
    assert row["answers"]["answered_step"]["confidence"] > 0
    assert row["state"]["utterance"] == "yes that is me" and row["action"]["type"] == "PlayClip"
    out = tmp_path / "export.jsonl"
    write_jsonl(out, rows)
    assert out.read_text(encoding="utf-8").count("\n") == 1


# -- transcripts and settings ---------------------------------------------------------


def test_transcript_sinks(tmp_path: Path) -> None:
    mem = MemoryTranscripts()
    mem.write("c", ["caller: hi"])
    assert mem.rows == {"c": ["caller: hi"]}
    JsonlTranscripts(tmp_path / "t.jsonl").write("c", ["agent: hello", "caller: hi"])
    assert "agent: hello" in (tmp_path / "t.jsonl").read_text(encoding="utf-8")


def test_shadow_mode_requires_baseline_mode() -> None:
    with pytest.raises(ValidationError, match="baseline_mode"):
        Settings(_env_file=None, shadow_mode=True)  # type: ignore[call-arg]
    ok = Settings(_env_file=None, shadow_mode=True, baseline_mode=True)  # type: ignore[call-arg]
    assert ok.shadow_mode and ok.baseline_mode
    _ = asyncio  # event loop import kept for pytest-asyncio auto mode
