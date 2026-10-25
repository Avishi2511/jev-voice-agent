"""Voice activity detection. Energy VAD by default; Silero when the optional extra is installed."""

from typing import Any, Protocol

from jevagent.frontend.audio import rms


class Vad(Protocol):
    def is_speech(self, frame: bytes) -> bool: ...


class EnergyVad:
    """RMS gate with an adaptive noise floor, so steady line noise does not read as speech."""

    def __init__(self, threshold: float = 500.0, floor_margin: float = 3.0) -> None:
        self._threshold = threshold
        self._margin = floor_margin
        self._floor = 0.0

    def is_speech(self, frame: bytes) -> bool:
        level = rms(frame)
        speech = level > max(self._threshold, self._floor * self._margin)
        if not speech:
            self._floor = level if self._floor == 0.0 else 0.95 * self._floor + 0.05 * level
        return speech


class SileroVad:
    """Silero VAD (pip install silero-vad). Frames must be 32 ms at 16 kHz or 32 ms at 8 kHz."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise ImportError(
                "SileroVad needs the optional 'silero-vad' package (and torch): "
                "uv pip install silero-vad"
            ) from exc
        self._torch: Any = torch
        self._model: Any = load_silero_vad()
        self._rate = sample_rate
        self._threshold = threshold

    def is_speech(self, frame: bytes) -> bool:
        import array

        samples = array.array("h")
        samples.frombytes(frame)
        tensor = self._torch.tensor(samples, dtype=self._torch.float32) / 32768.0
        return bool(self._model(tensor, self._rate).item() >= self._threshold)
