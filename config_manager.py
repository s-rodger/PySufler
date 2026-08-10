import json
import os
from pathlib import Path

from dotenv import dotenv_values


APP_NAME = "PySufler"
APP_DIR = Path(os.getenv("APPDATA", Path.home())) / APP_NAME
DATA_DIR = APP_DIR / "data"
LOGS_DIR = APP_DIR / "logs"
CONFIG_PATH = APP_DIR / "config.json"


DEFAULT_SETTINGS = {
    "whisper_url": "http://127.0.0.1:8001/v1/audio/transcriptions",
    "llm_url": "http://127.0.0.1:3000/api/chat/completions",
    "llm_api_type": "openwebui",
    "llm_model": "qwen3.5:9b",
    "openai_api_key": "",
    "openwebui_chat_id": "meeting-helper",
    "hotkey": "ctrl+alt+a",
    "chunk_seconds": 8,
    "buffer_minutes": 10,
    "language": "ru",
    "input_device": None,
    "prompt_mode": "last",
    "last_lines": 3,
    "llm_timeout": 90,
    "temperature": 0.2,
    "max_tokens": 300,
    "system_audio_enabled": True,
    "system_audio_device": None,
    "auto_system_loopback": True,
    "mix_mic_and_system_audio": True,
    "live_transcription": True,
    "live_chunk_seconds": 2,
}


ENV_TO_CONFIG = {
    "WHISPER_URL": "whisper_url",
    "LLM_URL": "llm_url",
    "LLM_API_TYPE": "llm_api_type",
    "LLM_MODEL": "llm_model",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENWEBUI_CHAT_ID": "openwebui_chat_id",
    "HOTKEY": "hotkey",
    "CHUNK_SECONDS": "chunk_seconds",
    "BUFFER_MINUTES": "buffer_minutes",
    "LANGUAGE": "language",
    "INPUT_DEVICE": "input_device",
    "PROMPT_MODE": "prompt_mode",
    "LAST_LINES": "last_lines",
    "LLM_TIMEOUT": "llm_timeout",
    "LLM_TEMPERATURE": "temperature",
    "LLM_MAX_TOKENS": "max_tokens",
    "SYSTEM_AUDIO_ENABLED": "system_audio_enabled",
    "SYSTEM_AUDIO_DEVICE": "system_audio_device",
    "AUTO_SYSTEM_LOOPBACK": "auto_system_loopback",
    "MIX_MIC_AND_SYSTEM_AUDIO": "mix_mic_and_system_audio",
    "LIVE_TRANSCRIPTION": "live_transcription",
    "LIVE_CHUNK_SECONDS": "live_chunk_seconds",
}


INT_KEYS = {
    "chunk_seconds",
    "buffer_minutes",
    "input_device",
    "last_lines",
    "llm_timeout",
    "max_tokens",
    "system_audio_device",
    "live_chunk_seconds",
}
FLOAT_KEYS = {"temperature"}
BOOL_KEYS = {
    "system_audio_enabled",
    "auto_system_loopback",
    "mix_mic_and_system_audio",
    "live_transcription",
}


def ensure_app_dirs():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def data_path(filename: str) -> Path:
    ensure_app_dirs()
    return DATA_DIR / filename


def log_path(filename: str) -> Path:
    ensure_app_dirs()
    return LOGS_DIR / filename


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "да")


def _coerce_value(key, value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return None if key in ("input_device", "system_audio_device") else ""
    if key in BOOL_KEYS:
        return _to_bool(value)
    if key in INT_KEYS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return DEFAULT_SETTINGS[key]
    if key in FLOAT_KEYS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return DEFAULT_SETTINGS[key]
    if key == "llm_api_type":
        return str(value).strip().lower()
    return value


def normalize_settings(settings: dict) -> dict:
    normalized = dict(DEFAULT_SETTINGS)
    normalized.update(settings or {})
    for key in list(normalized):
        normalized[key] = _coerce_value(key, normalized[key])
    return normalized


def settings_from_env(env_path=".env") -> dict:
    values = dotenv_values(env_path) if Path(env_path).exists() else {}
    settings = {}
    for env_key, config_key in ENV_TO_CONFIG.items():
        if env_key in values:
            settings[config_key] = _coerce_value(config_key, values[env_key])
    return normalize_settings(settings)


def load_config() -> dict:
    ensure_app_dirs()
    if CONFIG_PATH.exists():
        try:
            return normalize_settings(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return normalize_settings({})

    settings = settings_from_env(".env")
    save_config(settings)
    return settings


def save_config(settings: dict) -> dict:
    ensure_app_dirs()
    normalized = normalize_settings(settings)
    CONFIG_PATH.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized


def reset_config() -> dict:
    return save_config(DEFAULT_SETTINGS)
