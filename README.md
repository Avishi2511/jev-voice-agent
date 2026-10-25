# jev-voice-agent

Hybrid voice agent. **Phase A**: a scripted fast path where Jev answers typed extraction questions about each caller
turn and plain code owns the conversation (clips, slots, retries). **Phase B**: a one-way handoff to a Gemini LLM agent
with tools when the call goes off script. Telephony is Vobiz; STT, LLM and TTS are Gemini.
Design: `PRD.md`, `plan.md`. Operations: `docs/runbook.md`.

## Setup
```
uv sync
cp .env.example .env     # JEV_API_KEY, GEMINI_API_KEY, PUBLIC_URL
```

## Checks
```
uv run ruff check && uv run mypy src && uv run pytest
```
The suite uses fakes for Jev, STT, TTS and the LLM, so it needs no keys or network. `tests/integration` calls live Jev
and is skipped without `JEV_API_KEY`.

## Try it
| What | Command | Needs |
| --- | --- | --- |
| Type a call, real Jev (Phase A) and Gemini (Phase B) | `uv run python -m jevagent.frontend.console` | Jev key; Gemini key for Phase B |
| Jev only, prints every answer and confidence | `uv run python demo.py` | Jev key |
| Is Gemini good enough at Hinglish? | `uv run python scripts/hinglish_spike.py` | Gemini key |
| Render clip audio | `uv run python scripts/gen_clips.py scripts/emi_reminder.yaml` | Gemini key |
| Run the phone server | `uv run python -m jevagent.frontend.server` | both keys, clips, `PUBLIC_URL` |

## Layout
```
scripts/            call scripts (YAML) and operator scripts
src/jevagent/
  script/           schema, loader, step/slot engine
  jev/              Jev client, question factory, candidate parsers, fakes
  controller/       call state, handoff triggers, context packet
  phase_a/          clip player/renderer, acknowledgement and speculation
  phase_b/          LLM agent, typed tools, prompt
  frontend/         Vobiz bridge, VAD, turn detection, Gemini STT/TTS/LLM adapters, server, mock caller
  eval/             labelled data, splits, eval runner, tuning, shadow mode, judge, A/B gates
  observability/    decision log, latency metrics, call summaries, comparison report
docs/               runbook, labelling guide, question catalog, adding a script, baseline, compliance
```

## Status
All phases of `plan.md` are implemented and unit-tested offline. Nothing has been run against live Gemini, a phone line,
or labelled calls yet: see "Known limitations" in `docs/runbook.md` and the open items in `docs/compliance.md`.
