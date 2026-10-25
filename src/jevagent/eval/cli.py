"""Run an evaluation against live Jev: python -m jevagent.eval.cli --data data/turns.jsonl."""

import argparse
import asyncio
from pathlib import Path

from jevagent.config import Settings
from jevagent.eval.dataset import load_turns, verify_frozen
from jevagent.eval.report import to_markdown
from jevagent.eval.run_eval import run_eval
from jevagent.eval.splits import get_split
from jevagent.eval.versions import load_instructions
from jevagent.jev.client import HttpJevClient
from jevagent.script.loader import load_script


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--script", default="scripts/emi_reminder_en.yaml")
    ap.add_argument("--split", choices=["tuning", "validation", "heldout"], default="validation")
    ap.add_argument("--questions-dir", default="questions")
    ap.add_argument("--drafts", action="store_true", help="include unpromoted question versions")
    ap.add_argument("--shuffles", type=int, default=2, help="option-order shuffles for bias check")
    ap.add_argument("--heldout-reason", default="")
    args = ap.parse_args()

    settings = Settings()
    script = load_script(args.script)
    verify_frozen(args.data)
    turns = get_split(
        load_turns(args.data, script),
        args.split,
        allow_heldout=bool(args.heldout_reason),
        reason=args.heldout_reason,
    )
    instructions = load_instructions(args.questions_dir, script.name, promoted_only=not args.drafts)
    jev = HttpJevClient(settings)
    try:
        report = await run_eval(
            script, turns, jev, settings, instructions, shuffle_seeds=range(1, args.shuffles + 1)
        )
    finally:
        await jev.aclose()
    out = Path("logs") / f"eval_{args.split}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    asyncio.run(main())
