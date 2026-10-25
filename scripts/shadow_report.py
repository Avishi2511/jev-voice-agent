"""Compare shadow Phase A decisions against ground truth and the live baseline's latency.

    uv run python scripts/shadow_report.py truth.jsonl

truth.jsonl: one GroundTruth per line (call_id, slots from the CRM, should_handoff_turn or null).
Reads logs/shadow.jsonl and logs/metrics.jsonl (paths from settings).
"""

import sys
from pathlib import Path

from jevagent.config import Settings
from jevagent.eval.shadow import GroundTruth, ShadowTurn, compare_shadow
from jevagent.observability.summary import load_metric_rows


def main(truth_path: str) -> None:
    settings = Settings()
    read = lambda p: [ln for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]  # noqa: E731
    turns = [ShadowTurn.model_validate_json(ln) for ln in read(settings.shadow_path)]
    truths = [GroundTruth.model_validate_json(ln) for ln in read(truth_path)]
    ttfa = {
        (str(r["call_id"]), int(str(r["turn"]))): float(str(r["ttfa_ms"]))
        for r in load_metric_rows(settings.metrics_path)
        if r.get("ttfa_ms") is not None
    }
    print(compare_shadow(turns, truths, ttfa).model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
