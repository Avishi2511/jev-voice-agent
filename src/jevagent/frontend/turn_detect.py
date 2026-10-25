"""Frame-driven turn detection (FE-2, FE-4). Jev is not the turn detector.

Events:
  SpeechStarted  after `barge_in_frames` consecutive speech frames: cut playback now.
  TurnEnded      after `silence_ms` of quiet following at least `min_speech_frames` of speech.
"""

from collections import deque

from pydantic import BaseModel

from jevagent.config import Settings
from jevagent.frontend.vad import Vad


class SpeechStarted(BaseModel):
    pass


class TurnEnded(BaseModel):
    pcm: bytes


TurnEvent = SpeechStarted | TurnEnded


def silence_ms_for(settings: Settings, language: str) -> int:
    table = settings.turn_silence_ms
    return table.get(language, table["default"])


class TurnDetector:
    def __init__(
        self,
        vad: Vad,
        frame_ms: int = 20,
        barge_in_frames: int = 1,
        min_speech_frames: int = 8,
        silence_ms: int = 700,
        preroll_frames: int = 10,
    ) -> None:
        self._vad = vad
        self._frame_ms = frame_ms
        self._barge_in = barge_in_frames
        self._min_speech = min_speech_frames
        self._silence_ms = silence_ms
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._reset()

    def _reset(self) -> None:
        self._active = False
        self._buf: list[bytes] = []
        self._speech_frames = 0
        self._run = 0
        self._silent = 0
        self._announced = False

    def feed(self, frame: bytes) -> list[TurnEvent]:
        speech = self._vad.is_speech(frame)
        events: list[TurnEvent] = []
        if not self._active:
            if not speech:
                self._preroll.append(frame)
                return events
            self._active = True
            self._buf = [*self._preroll, frame]
            self._preroll.clear()
        else:
            self._buf.append(frame)

        if speech:
            self._speech_frames += 1
            self._run += 1
            self._silent = 0
            if self._run >= self._barge_in and not self._announced:
                self._announced = True
                events.append(SpeechStarted())
        else:
            self._run = 0
            self._silent += 1
            if self._silent * self._frame_ms >= self._silence_ms:
                if self._speech_frames >= self._min_speech:
                    events.append(TurnEnded(pcm=b"".join(self._buf)))
                self._reset()
        return events
