"""Render every script clip with the configured TTS voice (A-8, A-9). Uses TTS quota."""

from pathlib import Path

from jevagent.frontend.audio import normalise_rms
from jevagent.frontend.tts import OUT_RATE, Synthesizer
from jevagent.phase_a.clip_player import ClipEntry, ClipManifest, save_manifest, sha256
from jevagent.script.schema import Script

MIN_VARIANTS = 3


def variant_warnings(script: Script) -> list[str]:
    """Steps with fewer than 3 ask / re_ask variants sound robotic when repeated."""
    out = []
    for step in script.steps:
        for kind in ("ask", "re_ask"):
            n = len(getattr(step.clips, kind))
            if 0 < n < MIN_VARIANTS and not (kind == "ask" and step.slot is None):
                out.append(f"{step.id}.{kind}: {n} variant(s), want {MIN_VARIANTS}+")
    return out


async def render_clips(script: Script, synth: Synthesizer, out_dir: str | Path) -> ClipManifest:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries: dict[str, ClipEntry] = {}
    for clip in script.all_clips():
        pcm = normalise_rms(await synth.synthesize(clip.text))
        file = f"{clip.id}.pcm"
        (root / file).write_bytes(pcm)
        entries[clip.id] = ClipEntry(file=file, sha256=sha256(pcm), text=clip.text)
    manifest = ClipManifest(
        script=script.name,
        voice=synth.voice,
        model=synth.model,
        sample_rate=OUT_RATE,
        clips=entries,
    )
    save_manifest(root, manifest)
    return manifest
