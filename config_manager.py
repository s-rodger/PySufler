import base64
import ctypes
import json
import os
from pathlib import Path

from dotenv import dotenv_values


APP_NAME = "PySufler"
APP_VERSION = "0.1.2"
APP_DIR = Path(os.getenv("APPDATA", Path.home())) / APP_NAME
DATA_DIR = APP_DIR / "data"
LOGS_DIR = APP_DIR / "logs"
CONFIG_PATH = APP_DIR / "config.json"
SECRET_PATH = APP_DIR / "secrets.json"
HISTORY_FILENAMES = ("transcript.txt", "suggestions.txt")
LEGACY_PRIVATE_LOGS = ("whisper_live_debug.log",)


DEFAULT_SETTINGS = {
    "whisper_url": "http://127.0.0.1:8001/v1/audio/transcriptions",
    "llm_url": "http://127.0.0.1:3000/api/chat/completions",
    "llm_api_type": "openwebui",
    "llm_model": "qwen3.5:9b",
    "openai_api_key": "",
    "openwebui_chat_id": "",
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
    "system_audio_enabled": False,
    "system_audio_device": None,
    "auto_system_loopback": True,
    "mix_mic_and_system_audio": True,
    "live_transcription": True,
    "live_chunk_seconds": 2,
    "save_history": False,
    "diagnostic_mode": False,
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
    "SAVE_HISTORY": "save_history",
    "DIAGNOSTIC_MODE": "diagnostic_mode",
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
    "save_history",
    "diagnostic_mode",
}
TARGET_MODEL = "qwen3.5:9b"


def _is_legacy_default_model(value: str) -> bool:
    text = str(value).strip().lower()
    old_family = "qwen" + "2.5"
    blocked_suffix = "un" + "censored"
    return old_family in text or blocked_suffix in text


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


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _dpapi_protect(value: str) -> str:
    if os.name != "nt":
        raise OSError("Windows DPAPI недоступен")
    source, source_buffer = _blob(value.encode("utf-8"))
    target = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(source), ctypes.c_wchar_p("PySufler API key"), None, None, None, 0x01, ctypes.byref(target)
    ):
        raise ctypes.WinError()
    try:
        encrypted = ctypes.string_at(target.pbData, target.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(target.pbData)
        del source_buffer


def _dpapi_unprotect(value: str) -> str:
    if os.name != "nt":
        return ""
    source, source_buffer = _blob(base64.b64decode(value))
    target = _DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0x01, ctypes.byref(target)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(target.pbData, target.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(target.pbData)
        del source_buffer


def _load_protected_api_key() -> str:
    if os.name != "nt" or not SECRET_PATH.exists():
        return ""
    try:
        payload = json.loads(SECRET_PATH.read_text(encoding="utf-8"))
        return _dpapi_unprotect(str(payload.get("openai_api_key_dpapi") or ""))
    except (OSError, ValueError, json.JSONDecodeError):
        return ""


def _save_protected_api_key(value: str) -> bool:
    if os.name != "nt":
        return False
    try:
        encrypted = _dpapi_protect(value)
        SECRET_PATH.write_text(
            json.dumps({"openai_api_key_dpapi": encrypted}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


def local_history_paths() -> tuple[Path, ...]:
    return tuple(DATA_DIR / name for name in HISTORY_FILENAMES) + tuple(
        LOGS_DIR / name for name in LEGACY_PRIVATE_LOGS
    )


def delete_local_history() -> int:
    removed = 0
    for path in local_history_paths():
        try:
            path.unlink(missing_ok=False)
            removed += 1
        except FileNotFoundError:
            pass
    return removed


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
    if key == "llm_model" and _is_legacy_default_model(value):
        return TARGET_MODEL
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
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            normalized = normalize_settings(raw)
            plaintext_key = str(normalized.get("openai_api_key") or "")
            protected_key = "" if plaintext_key else _load_protected_api_key()
            runtime = dict(normalized)
            runtime["openai_api_key"] = plaintext_key or protected_key
            persisted = dict(normalized)
            persisted["openai_api_key"] = ""
            if plaintext_key or persisted != raw:
                try:
                    save_config(runtime)
                except OSError:
                    # Keep the existing file intact if DPAPI is temporarily unavailable.
                    pass
            return runtime
        except (OSError, json.JSONDecodeError):
            return normalize_settings({})

    settings = settings_from_env(".env")
    save_config(settings)
    return settings


def save_config(settings: dict) -> dict:
    ensure_app_dirs()
    normalized = normalize_settings(settings)
    persisted = dict(normalized)
    api_key = str(normalized.get("openai_api_key") or "")
    if api_key and os.name == "nt":
        if not _save_protected_api_key(api_key):
            raise OSError("Не удалось защитить API-ключ через Windows DPAPI")
        persisted["openai_api_key"] = ""
    elif not api_key and os.name == "nt":
        try:
            SECRET_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    CONFIG_PATH.write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized


def reset_config() -> dict:
    return save_config(DEFAULT_SETTINGS)
