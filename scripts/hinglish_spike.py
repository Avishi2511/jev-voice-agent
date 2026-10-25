"""Phase 6k spike: does Gemini handle Hinglish for STT and TTS well enough? Spends quota.

Round trip: Hinglish text -> Gemini TTS -> 8 kHz audio -> Gemini STT -> compare with the input.
Also checks the configured models exist for the key, and prints latencies.

    uv run python scripts/hinglish_spike.py
    uv run python scripts/hinglish_spike.py --wav my_recording.wav   # transcribe a real recording

Results go to logs/hinglish_spike.md. Round-tripped synthetic speech is an optimistic test:
also run it on real caller recordings before trusting the numbers.
"""

import argparse
import asyncio
import difflib
import sys
import time
from pathlib import Path

from google import genai

from jevagent.config import Settings
from jevagent.frontend.audio import wav_to_pcm16
from jevagent.frontend.llm import check_models
from jevagent.frontend.stt import GeminiTranscriber
from jevagent.frontend.tts import GeminiSynthesizer

SENTENCES = [
    "haan ji main bol raha hoon",
    "kal tak payment kar dunga",
    "UPI se kar dunga, aaj shaam tak",
    "pehle penalty ka bataiye, wo kyun laga",
    "mera EMI paanch hazaar rupaye hai na",
    "main kal nahi, parso payment karunga",
    "ek second, main net banking check karta hoon",
    "aap kaun bol rahe ho, mujhe agent se baat karni hai",
    "pandrah tareekh ko salary aati hai, uske baad kar dunga",
    "nahi nahi, ye galat number hai",
]


def norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c == " ").split())


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="transcribe this mono 16-bit WAV instead of the TTS round trip")
    args = ap.parse_args()

    settings = Settings()
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        sys.exit("GEMINI_API_KEY is not set (put it in .env)")
    client = genai.Client(api_key=key)
    await check_models(
        client,
        {
            "stt": settings.gemini_stt_model,
            "llm": settings.gemini_llm_model,
            "tts": settings.gemini_tts_model,
        },
    )
    stt = GeminiTranscriber(client, settings.gemini_stt_model)
    lines = ["# Hinglish spike", ""]

    if args.wav:
        pcm, rate = wav_to_pcm16(Path(args.wav).read_bytes())
        t0 = time.monotonic()
        text = await stt.transcribe(pcm, rate)
        lines.append(f"{args.wav}: {text!r} ({time.monotonic() - t0:.2f}s)")
    else:
        synth = GeminiSynthesizer(client, settings.gemini_tts_model, settings.gemini_tts_voice)
        lines += [
            "| input | heard | similarity | tts s | stt s |",
            "| --- | --- | --- | --- | --- |",
        ]
        scores = []
        for sentence in SENTENCES:
            t0 = time.monotonic()
            pcm = await synth.synthesize(sentence)
            t1 = time.monotonic()
            heard = await stt.transcribe(pcm, 8000)
            t2 = time.monotonic()
            score = difflib.SequenceMatcher(None, norm(sentence), norm(heard)).ratio()
            scores.append(score)
            lines.append(f"| {sentence} | {heard} | {score:.2f} | {t1 - t0:.2f} | {t2 - t1:.2f} |")
        lines += [
            "",
            f"mean similarity {sum(scores) / len(scores):.2f} over {len(scores)} sentences",
        ]

    out = Path("logs/hinglish_spike.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
