"""Render script clips with Gemini TTS. Spends TTS quota: run deliberately.

uv run python scripts/gen_clips.py scripts/emi_reminder_en.yaml
"""

import asyncio
import sys

from google import genai

from jevagent.config import Settings
from jevagent.frontend.tts import GeminiSynthesizer
from jevagent.phase_a.clip_render import render_clips, variant_warnings
from jevagent.script.loader import load_script


async def main(script_path: str) -> None:
    settings = Settings()
    if not settings.gemini_api_key.get_secret_value():
        sys.exit("GEMINI_API_KEY is not set (put it in .env)")
    script = load_script(script_path)
    for warning in variant_warnings(script):
        print("warning:", warning)
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    synth = GeminiSynthesizer(client, settings.gemini_tts_model, settings.gemini_tts_voice)
    out = f"{settings.clips_dir}/{script.name}"
    manifest = await render_clips(script, synth, out)
    print(f"rendered {len(manifest.clips)} clips to {out} ({manifest.model}/{manifest.voice})")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
