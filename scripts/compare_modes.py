"""Baseline vs hybrid report from logs/calls.jsonl and logs/metrics.jsonl.

    uv run python scripts/compare_modes.py [rates.yaml]

rates.yaml holds your unit prices (fields of observability.report.Rates); default is 0.
"""

import sys
from pathlib import Path

import yaml

from jevagent.config import Settings
from jevagent.observability.report import Rates, compare, to_markdown
from jevagent.observability.summary import load_metric_rows, load_summaries


def main() -> None:
    settings = Settings()
    rates = Rates()
    if len(sys.argv) > 1:
        rates = Rates.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")))
    report = compare(
        load_summaries(settings.summaries_path), load_metric_rows(settings.metrics_path), rates
    )
    print(to_markdown(report))


if __name__ == "__main__":
    main()
