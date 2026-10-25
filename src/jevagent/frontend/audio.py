"""Pure-Python telephony audio helpers (no audioop: removed in Python 3.13).

Everything inside the app is PCM16 little-endian mono. Vobiz sends L16 (or mu-law), and we
play back L16 at 8 kHz, so mu-law is only needed to decode inbound audio.
"""

import io
import math
import struct
import wave
from array import array

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


def _mulaw_decode_sample(byte: int) -> int:
    byte = ~byte & 0xFF
    sign = byte & 0x80
    exponent = (byte >> 4) & 0x07
    mantissa = byte & 0x0F
    sample = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return -sample if sign else sample


_DECODE_TABLE = [_mulaw_decode_sample(b) for b in range(256)]


def _mulaw_encode_sample(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    sample = min(abs(sample), _MULAW_CLIP) + _MULAW_BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not sample & mask:
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def mulaw_to_pcm16(data: bytes) -> bytes:
    return array("h", (_DECODE_TABLE[b] for b in data)).tobytes()


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    samples = array("h")
    samples.frombytes(pcm)
    return bytes(_mulaw_encode_sample(s) for s in samples)


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono PCM16. Integer down-ratios average (a crude low-pass); otherwise linear."""
    if src_rate == dst_rate or not pcm:
        return pcm
    src = array("h")
    src.frombytes(pcm)
    if src_rate % dst_rate == 0:
        k = src_rate // dst_rate
        n = len(src) // k
        return array("h", (sum(src[i * k : (i + 1) * k]) // k for i in range(n))).tobytes()
    n_out = int(len(src) * dst_rate / src_rate)
    out = array("h")
    step = src_rate / dst_rate
    for i in range(n_out):
        pos = i * step
        j = int(pos)
        frac = pos - j
        a = src[j]
        b = src[j + 1] if j + 1 < len(src) else a
        out.append(int(a + (b - a) * frac))
    return out.tobytes()


def parse_content_type(content_type: str) -> tuple[str, int]:
    """'audio/x-l16;rate=16000' -> ('audio/x-l16', 16000). Rate defaults to 8000."""
    base, _, params = content_type.partition(";")
    rate = 8000
    for part in params.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "rate" and value.isdigit():
            rate = int(value)
    return base.strip().lower(), rate


def to_pcm16(payload: bytes, encoding: str) -> bytes:
    """Decode an inbound frame according to the stream's declared encoding."""
    enc = encoding.lower()
    if any(tag in enc for tag in ("mulaw", "ulaw", "pcmu")):
        return mulaw_to_pcm16(payload)
    return payload


def frame_bytes(sample_rate: int, ms: int) -> int:
    return sample_rate * 2 * ms // 1000


def rms(pcm: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit WAV")
        return w.readframes(w.getnframes()), w.getframerate()


def tone(ms: int, sample_rate: int, amplitude: int = 8000, freq: float = 300.0) -> bytes:
    """Synthetic 'speech' for tests and the mock caller."""
    n = sample_rate * ms // 1000
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)),
    )


def silence(ms: int, sample_rate: int) -> bytes:
    return bytes(sample_rate * 2 * ms // 1000)


def normalise_rms(pcm: bytes, target_rms: float = 3000.0, peak_limit: int = 30000) -> bytes:
    """Scale to a common loudness so clips and Phase B speech match; never clip past the limit."""
    samples = array("h")
    samples.frombytes(pcm)
    level = rms(pcm)
    if level == 0.0:
        return pcm
    peak = max(abs(s) for s in samples)
    gain = min(target_rms / level, peak_limit / peak)
    return array("h", (int(s * gain) for s in samples)).tobytes()
