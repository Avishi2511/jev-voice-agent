"""Baseline vs hybrid comparison: completion, TTFA P50/P95, cost per call by component."""

import math
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from jevagent.observability.summary import CallSummary


class Rates(BaseModel):
    """Unit prices you supply (currency per unit). Nothing is assumed."""

    stt_per_request: float = 0.0
    tts_per_1k_chars: float = 0.0
    jev_per_request: float = 0.0
    llm_per_1k_input_tokens: float = 0.0
    llm_per_1k_output_tokens: float = 0.0


class ModeReport(BaseModel):
    calls: int
    completion_rate: float | None
    transfer_rate: float | None
    handoff_rate: float | None
    ttfa_p50_ms: float | None
    ttfa_p95_ms: float | None
    cost_per_call: dict[str, float]


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def cost_components(s: CallSummary, r: Rates) -> dict[str, float]:
    return {
        "stt": s.stt_requests * r.stt_per_request,
        "tts": s.tts_chars / 1000 * r.tts_per_1k_chars,
        "jev": s.jev_requests * r.jev_per_request,
        "llm": s.llm_input_tokens / 1000 * r.llm_per_1k_input_tokens
        + s.llm_output_tokens / 1000 * r.llm_per_1k_output_tokens,
    }


def compare(
    summaries: Sequence[CallSummary], metric_rows: Sequence[dict[str, object]], rates: Rates
) -> dict[str, ModeReport]:
    by_mode: dict[str, list[CallSummary]] = defaultdict(list)
    for summary in summaries:
        by_mode[summary.mode].append(summary)
    mode_of: dict[str, str] = {s.call_id: s.mode for s in summaries}
    ttfa: dict[str, list[float]] = defaultdict(list)
    for row in metric_rows:
        row_mode = mode_of.get(str(row.get("call_id")))
        value = row.get("ttfa_ms")
        if row_mode and isinstance(value, int | float) and row.get("turn", 0) != 0:
            ttfa[row_mode].append(float(value))  # turn 0 is the greeting, not a reply

    out: dict[str, ModeReport] = {}
    for mode_name, calls in by_mode.items():
        n = len(calls)
        totals: dict[str, float] = defaultdict(float)
        for c in calls:
            for k, v in cost_components(c, rates).items():
                totals[k] += v
        out[mode_name] = ModeReport(
            calls=n,
            completion_rate=_rate(sum(c.outcome == "completed" for c in calls), n),
            transfer_rate=_rate(sum(c.outcome == "transferred" for c in calls), n),
            handoff_rate=_rate(sum(c.handoff_reason is not None for c in calls), n),
            ttfa_p50_ms=percentile(ttfa[mode_name], 50),
            ttfa_p95_ms=percentile(ttfa[mode_name], 95),
            cost_per_call={k: v / n for k, v in totals.items()},
        )
    return out


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.0f}"


def to_markdown(report: dict[str, ModeReport]) -> str:
    lines = [
        "| mode | calls | completion | transferred | handed off | TTFA P50 ms | TTFA P95 ms "
        "| cost/call (stt / tts / jev / llm) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, r in sorted(report.items()):
        c = r.cost_per_call
        cost = " / ".join(f"{c.get(k, 0.0):.4f}" for k in ("stt", "tts", "jev", "llm"))
        lines.append(
            f"| {mode} | {r.calls} | {_fmt(r.completion_rate, True)} "
            f"| {_fmt(r.transfer_rate, True)} | {_fmt(r.handoff_rate, True)} "
            f"| {_fmt(r.ttfa_p50_ms)} | {_fmt(r.ttfa_p95_ms)} | {cost} |"
        )
    return "\n".join(lines) + "\n"
