from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Dict, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Jimmy. Values come from environment variables
    (prefixed ``JIMMY_``) or a ``.env`` file in the project root."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="JIMMY_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Wake word (Vosk) ---------------------------------------------------
    vosk_model_path: str = "models/vosk-model-small-en-in-0.4"
    # NoDecode keeps pydantic-settings from JSON-parsing the CSV env value —
    # our validator below handles the split. 'Jimmy' is used as the wake
    # phrase because Vosk's small model transcribes it far more reliably
    # than harder-to-parse names on Indian-accented English.
    wake_phrases: Annotated[Tuple[str, ...], NoDecode] = (
        "hey jimmy",
        "hi jimmy",
        "hey jim",
        "hey jimmi",
        "hey jimi",
        "he jimmy",
    )
    wake_fuzzy_threshold: int = 78
    wake_cooldown_seconds: float = 2.0

    # --- Microphone ---------------------------------------------------------
    sample_rate: int = 16000
    mic_device_index: int | None = None

    # --- Command capture (VAD) ---------------------------------------------
    command_max_seconds: float = 12.0
    command_silence_ms: int = 900
    vad_aggressiveness: int = 2  # 0=lenient .. 3=aggressive

    # --- Command STT (faster-whisper) --------------------------------------
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"  # cpu | cuda | auto  (cpu is safest — no CUDA runtime required)
    whisper_language: str = ""    # empty = auto-detect (English/Hindi)
    whisper_model_dir: str = "models/whisper"

    # --- Intent resolution (Ollama) ----------------------------------------
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_timeout_seconds: float = 10.0
    ollama_enabled: bool = True

    # --- TTS ---------------------------------------------------------------
    tts_enabled: bool = True
    tts_voice_en: str = "en-IN-NeerjaNeural"
    tts_voice_hi: str = "hi-IN-SwaraNeural"
    tts_rate: str = "+0%"
    tts_low_latency: bool = False

    # --- WhatsApp automation ----------------------------------------------
    whatsapp_contacts: Annotated[Dict[str, str], NoDecode] = {}
    whatsapp_auto_send: bool = False

    # --- Safety ------------------------------------------------------------
    confirm_destructive: bool = True
    destructive_timeout_seconds: float = 6.0

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"

    # --- Validators --------------------------------------------------------
    @field_validator("wake_phrases", mode="before")
    @classmethod
    def _split_wake_phrases(cls, value: object) -> Tuple[str, ...]:
        if isinstance(value, str):
            parts = [p.strip().lower() for p in value.split(",") if p.strip()]
            return tuple(parts or ["hey jimmy"])
        if isinstance(value, (list, tuple)):
            parts = [str(p).strip().lower() for p in value if str(p).strip()]
            return tuple(parts or ["hey jimmy"])
        return ("hey jimmy",)

    @field_validator("mic_device_index", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("vad_aggressiveness")
    @classmethod
    def _clamp_vad(cls, value: int) -> int:
        return max(0, min(3, value))

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        v = value.strip().upper()
        return v if v in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"

    @field_validator("whatsapp_contacts", mode="before")
    @classmethod
    def _parse_whatsapp_contacts(cls, value: object) -> Dict[str, str]:
        # CSV format: "sid original:+919999999999,mom:+919888888888"
        if isinstance(value, dict):
            out: Dict[str, str] = {}
            for k, v in value.items():
                name = str(k).strip().lower()
                phone = str(v).strip()
                if name and phone:
                    out[name] = phone
            return out
        if isinstance(value, str):
            out: Dict[str, str] = {}
            for chunk in value.split(","):
                part = chunk.strip()
                if not part or ":" not in part:
                    continue
                name, phone = part.split(":", 1)
                n = name.strip().lower()
                p = phone.strip()
                if n and p:
                    out[n] = p
            return out
        return {}

    # --- Convenience -------------------------------------------------------
    @property
    def project_root(self) -> Path:
        # In packaged/frozen runs (e.g. PyInstaller), source files can be
        # extracted to a transient temp directory. Use the executable folder
        # so relative model/cache paths stay stable across launches.
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parents[2]

    def resolve_path(self, relative_or_absolute: str) -> Path:
        p = Path(relative_or_absolute)
        return p if p.is_absolute() else (self.project_root / p)


def load_settings() -> Settings:
    return Settings()
