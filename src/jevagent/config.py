"""Settings loaded from environment / .env."""

from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "staging", "prod"] = "dev"
    jev_model: str = "jev-1.14.0"
    jev_api_key: SecretStr = SecretStr("")
    jev_base_url: str = "https://api.typesafe.ai"
    jev_timeout_s: float = 2.0
    jev_max_retries: int = 1
    # Early-handoff bias (H-2): confidence below this after one re-ask hands off.
    default_confidence_threshold: float = 0.7
    # Off-script turn_type fires at this confidence, deliberately below slot thresholds.
    off_script_threshold: float = 0.5
    llm_max_tool_iterations: int = 4
    # Gemini (STT, Phase B LLM, TTS). IDs are pinned here and checked against the key's model list.
    gemini_api_key: SecretStr = SecretStr("")
    gemini_stt_model: str = "gemini-3.5-transcribe"
    gemini_llm_model: str = "gemini-3.8-flash"
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    gemini_tts_voice: str = "Kore"
    # Real-time front end
    public_url: str = ""
    script_path: str = "scripts/emi_reminder.yaml"
    clips_dir: str = "clips"
    inbound_sample_rate: int = 16000  # Vobiz -> app; 24 kHz inbound does not work
    vad_energy_threshold: float = 500.0  # RMS, PCM16 units; used by the energy VAD
    barge_in_frames: int = 1  # speech frames before playback is cut (FE-2)
    min_speech_frames: int = 8  # shorter bursts are noise, not a turn
    turn_silence_ms: dict[str, int] = {"default": 700, "hinglish": 700, "en": 600}
    baseline_mode: bool = False  # plain STT->LLM->TTS agent, Phase A bypassed (Stage 2)
    summaries_path: str = "logs/calls.jsonl"
    ack_enabled: bool = True
    speculative_jev: bool = False  # needs streaming partial transcripts (A-10)
    metrics_path: str = "logs/metrics.jsonl"
    decision_log_path: str = "logs/decisions.jsonl"

    @model_validator(mode="after")
    def _pin_model(self) -> "Settings":
        if self.app_env != "dev" and self.jev_model.endswith("-latest"):
            raise ValueError(
                f"jev_model={self.jev_model!r} is not allowed when app_env={self.app_env!r}; "
                "pin an exact version such as jev-1.14.0 (A-11)"
            )
        return self
