"""Export logged decisions + call outcomes as training data for a later Laya fine-tune (E-7).

These rows are *model decisions with outcomes*, not gold labels: `label_source` says so. Gold
labels come from the human-labelled set (Phase 5); use these to find disagreements and to
augment, not to replace it.
"""

import json
from pathlib import Path
from typing import Any

from jevagent.observability.decision_log import DecisionRecord
from jevagent.observability.summary import CallSummary


def load_decisions(path: str | Path) -> list[DecisionRecord]:
    return [
        DecisionRecord.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def export_rows(
    decisions: list[DecisionRecord], summaries: list[CallSummary]
) -> list[dict[str, Any]]:
    """One row per decision, only for calls whose outcome is known."""
    outcome = {s.call_id: s for s in summaries}
    rows = []
    for d in decisions:
        summary = outcome.get(d.call_id)
        if summary is None:
            continue
        rows.append(
            {
                "call_id": d.call_id,
                "turn": d.turn,
                "model_version": d.model_version,
                "state": d.state_sent,
                "questions": d.questions,
                "answers": {
                    k: {
                        "selected": a.selected,
                        "probabilities": a.probabilities,
                        "confidence": a.confidence,
                    }
                    for k, a in d.answers.items()
                },
                "action": d.action,
                "call_outcome": summary.outcome,
                "call_handoff_reason": summary.handoff_reason,
                "label_source": "model_decision",
            }
        )
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
