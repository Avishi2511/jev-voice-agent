"""A/B go/no-go (Stage 4). Gate values are yours (PRD open question 5); none are assumed."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from jevagent.observability.report import ModeReport, Rates, compare
from jevagent.observability.summary import CallSummary

STUCK_MIN_TURNS = 3


class Gates(BaseModel):
    """Hybrid must be no worse than baseline on quality, and better on speed and cost."""

    max_completion_drop: float  # e.g. 0.02: hybrid completion may be at most 2 points lower
    max_repeat_rate: float  # share of handed-off calls where the caller repeated themselves
    min_ttfa_improvement_pct: float  # P50 TTFA must improve by at least this many percent
    max_cost_ratio: float  # hybrid cost per call / baseline cost per call
    max_stuck_rate: float  # hybrid calls that dropped with no handoff after several turns


class ArmStats(BaseModel):
    report: ModeReport
    stuck_rate: float | None
    repeat_rate: float | None  # among judged handed-off calls


class GateResult(BaseModel):
    passed: bool
    failures: list[str]
    unmeasured: list[str]


def _total_cost(r: ModeReport) -> float:
    return sum(r.cost_per_call.values())


def evaluate_gates(hybrid: ArmStats, baseline: ArmStats, gates: Gates) -> GateResult:
    failures: list[str] = []
    unmeasured: list[str] = []

    h, b = hybrid.report, baseline.report
    if h.completion_rate is None or b.completion_rate is None:
        unmeasured.append("completion")
    elif b.completion_rate - h.completion_rate > gates.max_completion_drop:
        failures.append(
            f"completion {h.completion_rate:.1%} vs baseline {b.completion_rate:.1%} "
            f"(allowed drop {gates.max_completion_drop:.1%})"
        )

    if hybrid.repeat_rate is None:
        unmeasured.append("repeat rate (needs judged handed-off calls)")
    elif hybrid.repeat_rate > gates.max_repeat_rate:
        failures.append(f"repeat rate {hybrid.repeat_rate:.1%} > {gates.max_repeat_rate:.1%}")

    if h.ttfa_p50_ms is None or b.ttfa_p50_ms is None or b.ttfa_p50_ms == 0:
        unmeasured.append("TTFA")
    else:
        gain = (b.ttfa_p50_ms - h.ttfa_p50_ms) / b.ttfa_p50_ms * 100
        if gain < gates.min_ttfa_improvement_pct:
            failures.append(
                f"TTFA P50 improved {gain:.0f}% < {gates.min_ttfa_improvement_pct:.0f}%"
            )

    if _total_cost(b) == 0:
        unmeasured.append("cost (no rates supplied)")
    else:
        ratio = _total_cost(h) / _total_cost(b)
        if ratio > gates.max_cost_ratio:
            failures.append(f"cost ratio {ratio:.2f} > {gates.max_cost_ratio:.2f}")

    if hybrid.stuck_rate is None:
        unmeasured.append("stuck rate")
    elif hybrid.stuck_rate > gates.max_stuck_rate:
        failures.append(f"stuck rate {hybrid.stuck_rate:.1%} > {gates.max_stuck_rate:.1%}")

    # An unmeasured gate is not a pass: the decision needs the data.
    return GateResult(
        passed=not failures and not unmeasured, failures=failures, unmeasured=unmeasured
    )


def record_decision(path: str | Path, result: GateResult, decided_by: str, note: str = "") -> None:
    """Append the go/no-go outcome. `decided_by` is a person; the gates inform the decision."""
    verdict = "GO" if result.passed else "NO-GO"
    lines = [
        f"## {datetime.now(UTC).isoformat(timespec='seconds')} {verdict} "
        f"(recorded by {decided_by})",
        *(f"- failed: {f}" for f in result.failures),
        *(f"- unmeasured: {u}" for u in result.unmeasured),
    ]
    if note:
        lines.append(f"- note: {note}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def arm_stats(
    summaries: list[CallSummary],
    metric_rows: list[dict[str, object]],
    rates: Rates,
    caller_repeated: dict[str, bool] | None = None,
) -> dict[str, ArmStats]:
    """Per-arm report plus the two gates that need more than counters.

    `caller_repeated` maps judged call ids to whether the caller had to repeat themselves
    (from the post-call judge). The stuck rate is a proxy: calls that ended dropped, with no
    handoff, after at least STUCK_MIN_TURNS caller turns.
    """
    reports = compare(summaries, metric_rows, rates)
    repeated = caller_repeated or {}
    out: dict[str, ArmStats] = {}
    for arm, report in reports.items():
        calls = [s for s in summaries if s.mode == arm]
        stuck = sum(
            s.outcome == "dropped" and s.handoff_reason is None and s.turns >= STUCK_MIN_TURNS
            for s in calls
        )
        judged = [
            repeated[s.call_id]
            for s in calls
            if s.handoff_reason is not None and s.call_id in repeated
        ]
        out[arm] = ArmStats(
            report=report,
            stuck_rate=stuck / len(calls) if calls else None,
            repeat_rate=sum(judged) / len(judged) if judged else None,
        )
    return out
