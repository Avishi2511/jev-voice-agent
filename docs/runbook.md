# Runbook

## What runs
One FastAPI process (`python -m jevagent.frontend.server`) per host.
- `GET/POST /answer`: returns the Vobiz `<Stream>` XML pointing at `wss://PUBLIC_URL/ws`.
- `WS /ws`: one `CallSession` per call (audio in, audio out, controller, Phase B).
- `POST /stream-status`: Vobiz lifecycle callbacks (logged only). `GET /health`.

External services per call: Jev (TypeSafe) for Phase A decisions, Gemini for STT, Phase B LLM and TTS.

## Environment
Copy `.env.example` to `.env`. Required: `JEV_API_KEY`, `GEMINI_API_KEY`, `PUBLIC_URL`.

| Variable | Meaning |
| --- | --- |
| `APP_ENV` | `dev` / `staging` / `prod`. `jev-latest` is rejected outside dev. |
| `JEV_MODEL` | Pinned Jev version (`jev-1.14.0`). Thresholds are tuned per version: re-run eval before changing. |
| `GEMINI_STT_MODEL`, `GEMINI_LLM_MODEL`, `GEMINI_TTS_MODEL`, `GEMINI_TTS_VOICE` | Pinned at startup against the key's model list; the server refuses to start if one is missing. |
| `SCRIPT_PATH` | Script YAML (default `scripts/emi_reminder_en.yaml`). |
| `BASELINE_MODE` | `true`: plain LLM agent, Phase A bypassed. |
| `SHADOW_MODE` | `true` (needs `BASELINE_MODE`): Phase A runs silently beside the baseline, logging to `logs/shadow.jsonl`. |
| `AB_HYBRID_PERCENT` | 0-100: share of calls routed to hybrid; unset means use `BASELINE_MODE` as set. |
| `KILL_SWITCH_PATH` | Default `KILL_HYBRID`. While this file exists, new calls go to baseline. |
| `SAVE_TRANSCRIPTS` | Default false (PII). See `docs/compliance.md`. |

## Deploy
1. `uv sync`; `uv run ruff check && uv run mypy src && uv run pytest`.
2. Render clips with the production voice: `uv run python scripts/gen_clips.py scripts/<script>.yaml` (spends TTS quota).
   Startup fails if clips were rendered with a different TTS model/voice than Phase B uses (B-2) or any script clip is missing.
3. Expose the port publicly (reverse proxy or ngrok) and set `PUBLIC_URL`. Point the Vobiz number's answer URL at `PUBLIC_URL/answer`.
4. `uv run python -m jevagent.frontend.server`.

## Rollout order (PRD stages)
1. Offline: `uv run pytest`; console check `uv run python -m jevagent.frontend.console`.
2. Speech spike: `uv run python scripts/speech_spike.py` (and `--wav` on real recordings). Do not go further if quality is poor.
3. Label turns, freeze, tune: `docs/labelling_guide.md`, then `python -m jevagent.eval.cli --data data/turns.jsonl`.
4. Baseline calls (`BASELINE_MODE=true`), then `scripts/compare_modes.py`. Record numbers in `docs/baseline.md`.
5. Shadow (`SHADOW_MODE=true BASELINE_MODE=true`), then `scripts/shadow_report.py truth.jsonl`.
6. A/B: set `AB_HYBRID_PERCENT`, start small, then `scripts/ab_report.py gates.yaml --record docs/decision.md --by <name>`.

## On-call checks
- `GET /health` returns ok but says nothing about Jev/Gemini. A real check is a test call.
- Silent calls: look for `phase_a_failed`, `tts_failed_mid_reply`, `turn_failed` in the logs. Jev and TTS failures now hand off / fall back to a human rather than going silent; `turn_failed` means an unhandled bug.
- Many `handoff` logs with reason `jev_unavailable`: Jev is down or the key is wrong (HTTP 4xx raises `JevConfigError`). Calls continue on Phase B.
- High TTFA: `logs/metrics.jsonl` has `stt_ms`, `jev_ms`, `ttfa_ms` per turn. STT is one request per utterance, so it dominates.
- Callers cut off mid-sentence: raise `TURN_SILENCE_MS` for the language. Agent talks over callers: check `barge_in` logs and `VAD_ENERGY_THRESHOLD`.
- Compliance: `compliance.audit` over `logs/calls.jsonl` (see `docs/compliance.md`).

## Rollback
- Immediate, no restart: `touch KILL_HYBRID`. New calls go to baseline; calls in progress finish as they are.
- Full: set `AB_HYBRID_PERCENT=0` or `BASELINE_MODE=true` and restart.
- Jev model change: the pin is in `.env`; thresholds are keyed by model in `thresholds.yaml`, and `thresholds_for` raises if none exist for the new model.

## Known limitations
- Outbound calls are not implemented (no Vobiz REST client); calls are inbound via `/answer`.
- Energy VAD by default; Silero is an optional adapter that has not been run here (`uv pip install silero-vad`).
- If the caller speaks again while their previous turn is still in STT/Jev, the reply may overlap.
- Gemini request shapes are written from the docs and unit-tested with fakes; the first live run is the real test.
- `MockCrm` and `RecordingTransfer` stand in for the CRM and human transfer.
