"""Human-readable eval report and the threshold tuner (E-4, E-5)."""

from collections.abc import Sequence

from jevagent.eval.run_eval import EvalReport


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def to_markdown(report: EvalReport) -> str:
    lines = [
        f"# Eval report ({report.model_version}, {report.n_turns} turns)",
        "",
        "| question | n | accuracy | majority baseline | beats baseline |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, g in sorted(report.groups.items()):
        flag = "yes" if g.beats_baseline else "**NO**"
        lines.append(
            f"| {name} | {g.n} | {_pct(g.accuracy)} | {_pct(g.majority_baseline)} | {flag} |"
        )
    lines += ["", "## Rare classes (report separately; overall accuracy hides these)", ""]
    for name, r in report.rare.items():
        lines.append(f"- {name}: recall {_pct(r.recall)} over {r.positives} positives")
    h = report.handoff
    lines += [
        "",
        "## Handoff",
        f"- recall {_pct(h.recall)} over {h.should_handoff} should-handoff turns",
        f"- false handoff rate {_pct(h.false_handoff_rate)}",
        "",
        f"Position stability across option shuffles: {_pct(report.position_stability)}",
    ]
    return "\n".join(lines) + "\n"


def tune_precision_threshold(
    samples: Sequence[tuple[float, bool]],
    target_precision: float,
    grid: Sequence[float] | None = None,
) -> float | None:
    """Lowest threshold whose accepted answers are at least `target_precision` correct.

    Used for gates that fill slots: wrongly filling is worse than re-asking.
    """
    for t in grid or [x / 100 for x in range(50, 100)]:
        accepted = [ok for conf, ok in samples if conf >= t]
        if accepted and sum(accepted) / len(accepted) >= target_precision:
            return t
    return None


def tune_recall_threshold(
    samples: Sequence[tuple[float, bool]], target_recall: float, grid: Sequence[float] | None = None
) -> float | None:
    """Highest threshold that still catches `target_recall` of true positives.

    Used for rare-class triggers (corrections, off-script): bias toward early handoff.
    `samples` holds (confidence, is_true_positive) for turns that truly are positives.
    """
    positives = [conf for conf, is_pos in samples if is_pos]
    if not positives:
        return None
    for t in sorted(grid or [x / 100 for x in range(50, 100)], reverse=True):
        if sum(c >= t for c in positives) / len(positives) >= target_recall:
            return t
    return None
