"""A/B report and gate check from live logs, optionally recording the go/no-go.

    uv run python scripts/ab_report.py gates.yaml [rates.yaml] [--judged judged.json]
                                       [--record docs/decision.md --by "name"]

gates.yaml: the numeric gates you chose (fields of eval.ab_report.Gates).
judged.json: {"<call_id>": true|false} from the post-call judge (caller repeated themselves).
Unmeasured gates count as not passed.
"""

import argparse
import json
from pathlib import Path

import yaml

from jevagent.config import Settings
from jevagent.eval.ab_report import Gates, arm_stats, evaluate_gates, record_decision
from jevagent.observability.report import Rates, to_markdown
from jevagent.observability.summary import load_metric_rows, load_summaries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gates")
    ap.add_argument("rates", nargs="?")
    ap.add_argument("--judged")
    ap.add_argument("--record")
    ap.add_argument("--by", default="")
    args = ap.parse_args()

    settings = Settings()
    gates = Gates.model_validate(yaml.safe_load(Path(args.gates).read_text(encoding="utf-8")))
    rates = Rates()
    if args.rates:
        rates = Rates.model_validate(yaml.safe_load(Path(args.rates).read_text(encoding="utf-8")))
    judged = json.loads(Path(args.judged).read_text(encoding="utf-8")) if args.judged else None

    arms = arm_stats(
        load_summaries(settings.summaries_path),
        load_metric_rows(settings.metrics_path),
        rates,
        judged,
    )
    print(to_markdown({k: v.report for k, v in arms.items()}))
    for arm, a in arms.items():
        print(f"{arm}: stuck-call proxy {a.stuck_rate}, repeat rate {a.repeat_rate}")
    if "hybrid" not in arms or "baseline" not in arms:
        raise SystemExit("need calls from both arms to evaluate the gates")
    result = evaluate_gates(arms["hybrid"], arms["baseline"], gates)
    print("PASS" if result.passed else "NOT PASSED", result.model_dump_json(indent=2))
    if args.record:
        if not args.by:
            raise SystemExit("--record needs --by <person making the decision>")
        record_decision(args.record, result, args.by)


if __name__ == "__main__":
    main()
