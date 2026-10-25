"""Benchmark Phase B LLM candidates on handoff cases. Spends Gemini quota.

    uv run python scripts/benchmark_llm.py data/llm_cases.jsonl MODEL_A MODEL_B

Each line of the cases file is a BenchCase (see jevagent.eval.llm_bench). Use real Hinglish
transcripts; results on invented cases say little.
"""

import asyncio
import sys
from pathlib import Path

from google import genai

from jevagent.config import Settings
from jevagent.eval.llm_bench import benchmark, load_cases, to_markdown
from jevagent.frontend.llm import GeminiLlmClient


async def main(cases_path: str, models: list[str]) -> None:
    key = Settings().gemini_api_key.get_secret_value()
    if not key:
        sys.exit("GEMINI_API_KEY is not set (put it in .env)")
    client = genai.Client(api_key=key)
    cases = load_cases(cases_path)

    class PerCase:
        """A fresh interaction chain per case: GeminiLlmClient is stateful per call."""

        def __init__(self, model: str) -> None:
            self.model = model

        def stream(self, messages, tools):  # type: ignore[no-untyped-def]
            return GeminiLlmClient(client, self.model).stream(messages, tools)

    summaries = await benchmark(cases, {m: PerCase(m) for m in models})  # type: ignore[misc]
    out = Path("logs/llm_benchmark.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_markdown(summaries), encoding="utf-8")
    print(to_markdown(summaries))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
