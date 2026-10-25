# jev-voice-agent

Hybrid voice agent: a scripted fast path (Phase A) where Jev answers typed extraction
questions and code owns the conversation, with a one-way handoff to an LLM agent (Phase B).
See `PRD.md` and `plan.md`.

## Setup
```
uv sync
cp .env.example .env   # fill in JEV_API_KEY
```

## Checks
```
uv run ruff check && uv run mypy src && uv run pytest
```
