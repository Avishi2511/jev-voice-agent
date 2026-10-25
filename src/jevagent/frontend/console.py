"""Text console for the whole stack, no audio: real Jev for Phase A, Gemini LLM for Phase B.

    uv run python -m jevagent.frontend.console

You type as the caller. Needs JEV_API_KEY, and GEMINI_API_KEY once the call reaches Phase B.
"""

import asyncio
import sys
from datetime import date

from google import genai

from jevagent.config import Settings
from jevagent.controller.controller import Controller, PassToPhaseB, PlayClip, StartHandoff
from jevagent.frontend.llm import GeminiLlmClient
from jevagent.jev.client import HttpJevClient
from jevagent.observability.decision_log import JsonlDecisionLog
from jevagent.phase_b.agent import PhaseBAgent, Say, Transferred
from jevagent.phase_b.tools import MockCrm, RecordingTransfer, ToolRegistry
from jevagent.script.loader import load_script

TOOLS = ["get_loan_account", "explain_charges", "schedule_callback", "transfer_to_human"]


async def main() -> None:
    settings = Settings()
    if not settings.jev_api_key.get_secret_value():
        sys.exit("JEV_API_KEY is not set (put it in .env)")
    script = load_script(settings.script_path)
    jev = HttpJevClient(settings)
    controller = Controller(
        script, jev, settings, JsonlDecisionLog("logs/console.jsonl"), "console", tools=TOOLS
    )
    agent: PhaseBAgent | None = None

    def say(text: str | None) -> None:
        print(f"AGENT: {text}")

    async def speak(stream) -> None:  # type: ignore[no-untyped-def]
        async for out in stream:
            if isinstance(out, Say):
                print(out.text, end="", flush=True)
            elif isinstance(out, PlayClip):
                say(f"[clip] {script.clip_text(out.clip_id)}")
            elif isinstance(out, Transferred):
                print(f"\n[transferred to human: {out.reason}]")
        print()

    try:
        say(script.clip_text(controller.start().clip_id))
        while True:
            utterance = input("YOU: ")
            action = await controller.on_caller_turn(utterance)
            if isinstance(action, PlayClip):
                say(script.clip_text(action.clip_id))
                if action.end_call:
                    break
                continue
            if isinstance(action, StartHandoff):
                if not settings.gemini_api_key.get_secret_value():
                    print("HANDOFF ->", action.packet.model_dump_json(indent=2))
                    sys.exit("GEMINI_API_KEY is not set, so Phase B cannot run")
                client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
                registry = ToolRegistry(
                    action.packet,
                    MockCrm(),
                    RecordingTransfer(),
                    allowed_clips=set(),
                    today=date.today,
                )
                agent = PhaseBAgent(
                    GeminiLlmClient(client, settings.gemini_llm_model),
                    registry,
                    action.packet,
                    script,
                    settings,
                )
                say(f"[bridge clip] {script.clip_text(agent.begin().clip_id)}")
                print("AGENT: ", end="")
                await speak(agent.first_turn())
            elif isinstance(action, PassToPhaseB) and agent is not None:
                print("AGENT: ", end="")
                await speak(agent.on_caller_turn(action.utterance))
            if agent is not None and agent.finished:
                break
    finally:
        await jev.aclose()


if __name__ == "__main__":
    asyncio.run(main())
