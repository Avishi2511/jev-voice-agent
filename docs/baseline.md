# Baseline (Stage 2): plain STT -> LLM -> TTS

**Status: not yet run.** No pilot calls have been placed, so there are no baseline numbers.
Do not treat any figure in this file as measured until the table below is filled from `scripts/compare_modes.py`.

## What the baseline is
`BASELINE_MODE=true` bypasses Phase A. The call opens with the same greeting clip, then every caller turn goes
STT -> Gemini LLM (same system prompt, tools and clips rule as Phase B) -> Gemini TTS. Same front end (Vobiz bridge,
VAD, barge-in), same tools, no Jev, no script engine, no acknowledgement clips.

## How to collect it
1. Fill `.env` (Gemini, Jev not needed for baseline calls, `PUBLIC_URL`), render clips: `uv run python scripts/gen_clips.py scripts/emi_reminder_en.yaml`.
2. `BASELINE_MODE=true uv run python -m jevagent.frontend.server`, place N pilot calls.
3. Same again with `BASELINE_MODE=false` (hybrid) for the A/B comparison later.
4. `uv run python scripts/compare_modes.py rates.yaml` (your unit prices in `rates.yaml`).
5. Choose the Phase B model: `uv run python scripts/benchmark_llm.py data/llm_cases.jsonl <model> <model>` on real transcripts.

## Results (fill from the report)
| mode | calls | completion | transferred | handed off | TTFA P50 ms | TTFA P95 ms | cost/call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | | | | | | | |

Definitions: *completion* = the closing clip was played; *TTFA* = caller stops talking until first audio sent
(acknowledgement clips count as first audio in hybrid mode); turn 0 (greeting) is excluded.
Repeat-after-handoff and slot accuracy vs CRM need the post-call judge (Phase 8).

## Gates for Stages 3 and 4
Set by the product owner after these numbers exist (PRD open question 5): _pending_.
