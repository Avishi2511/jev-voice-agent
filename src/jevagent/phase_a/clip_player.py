"""Pre-rendered clips: preloaded in memory, integrity- and voice-checked at startup (A-9, B-2).

Clips are raw PCM16 mono 8 kHz (`<id>.pcm`), the format played to Vobiz, so playback needs no
conversion. `manifest.yaml` is tracked in git; the audio is not.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import BaseModel

from jevagent.config import Settings
from jevagent.script.schema import Script

SAMPLE_RATE = 8000


class ClipError(RuntimeError):
    pass


class ClipEntry(BaseModel):
    file: str
    sha256: str
    text: str


class ClipManifest(BaseModel):
    script: str
    voice: str
    model: str
    sample_rate: int = SAMPLE_RATE
    clips: dict[str, ClipEntry]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_manifest(directory: str | Path, manifest: ClipManifest) -> Path:
    path = Path(directory) / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(manifest.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


class ClipPlayer:
    def __init__(self, manifest: ClipManifest, audio: dict[str, bytes]) -> None:
        self.manifest = manifest
        self._audio = audio

    @classmethod
    def load(cls, directory: str | Path) -> "ClipPlayer":
        root = Path(directory)
        try:
            manifest = ClipManifest.model_validate(
                yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            raise ClipError(f"cannot read clip manifest in {root}: {exc}") from exc
        audio: dict[str, bytes] = {}
        for clip_id, entry in manifest.clips.items():
            try:
                data = (root / entry.file).read_bytes()
            except OSError as exc:
                raise ClipError(f"clip {clip_id!r}: missing audio file {entry.file}") from exc
            if sha256(data) != entry.sha256:
                raise ClipError(f"clip {clip_id!r}: audio does not match manifest hash")
            audio[clip_id] = data
        return cls(manifest, audio)

    def verify(self, script: Script, settings: Settings) -> None:
        """Startup checks: every script clip exists and the clip voice is the Phase B voice."""
        if (self.manifest.voice, self.manifest.model) != (
            settings.gemini_tts_voice,
            settings.gemini_tts_model,
        ):
            raise ClipError(
                f"clips were rendered with {self.manifest.model}/{self.manifest.voice} "
                f"but Phase B uses {settings.gemini_tts_model}/{settings.gemini_tts_voice} "
                "(B-2); re-render clips"
            )
        missing = [c.id for c in script.all_clips() if c.id not in self._audio]
        if missing:
            raise ClipError(f"clips missing from manifest: {missing}")

    def pcm(self, clip_id: str) -> bytes:
        try:
            return self._audio[clip_id]
        except KeyError:
            raise ClipError(f"unknown clip {clip_id!r}") from None

    def chunks(self, clip_id: str, chunk_ms: int = 20) -> Iterator[bytes]:
        data = self.pcm(clip_id)
        size = SAMPLE_RATE * 2 * chunk_ms // 1000
        for i in range(0, len(data), size):
            yield data[i : i + size]
