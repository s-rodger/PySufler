import argparse
import io
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime

import keyboard
import numpy as np
import requests
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly

from config_manager import data_path, load_config

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None


PROMPT = (
    "Ты помощник пользователя на рабочем совещании. По фрагменту разговора дай "
    "короткий готовый ответ от лица пользователя. Если данных недостаточно - скажи, "
    "что нужно уточнить. Не выдумывай факты."
)


def env_int(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        print(f"Некорректное значение {name}={value!r}, используется {default}.")
        return default


def env_optional_int(name):
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"Некорректное значение {name}={value!r}. Укажите числовой индекс устройства.")
        sys.exit(1)


def env_bool(name, default=False):
    value = os.getenv(name, str(default)).strip().lower()
    return value in ("1", "true", "yes", "on", "да")


def load_settings():
    return load_config()


def list_input_devices():
    print("Доступные input audio devices:")
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"Не удалось получить список аудиоустройств: {exc}")
        return

    found = False
    for index, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue

        found = True
        name = device.get("name", "")
        default_samplerate = device.get("default_samplerate", "")
        print(
            f"{index}: {name} | max_input_channels={max_input_channels} "
            f"| default_samplerate={default_samplerate}"
        )

    if not found:
        print("Input audio devices не найдены.")


def list_output_devices():
    print("Доступные output audio devices для WASAPI loopback:")
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"Не удалось получить список аудиоустройств: {exc}")
        return

    found = False
    for index, device in enumerate(devices):
        max_output_channels = int(device.get("max_output_channels", 0))
        if max_output_channels <= 0:
            continue

        found = True
        name = device.get("name", "")
        default_samplerate = device.get("default_samplerate", "")
        print(
            f"{index}: {name} | max_output_channels={max_output_channels} "
            f"| default_samplerate={default_samplerate}"
        )

    if not found:
        print("Output audio devices не найдены.")


def print_sounddevice_devices():
    try:
        print(sd.query_devices())
    except Exception as exc:
        print(f"Не удалось получить список аудиоустройств: {exc}")
    list_loopback_devices()


def validate_input_device(settings):
    input_device = settings["input_device"]
    if input_device is not None:
        try:
            sd.query_devices(input_device, "input")
        except Exception as exc:
            print(f"Не удалось открыть INPUT_DEVICE={input_device}: {exc}")
            sys.exit(1)
        return

    try:
        default_input = sd.default.device[0]
    except Exception:
        default_input = -1

    if default_input == -1:
        print("Не найден микрофон по умолчанию. Укажите INPUT_DEVICE в .env.")
        sys.exit(1)

    try:
        sd.query_devices(default_input, "input")
    except Exception:
        print("Не найден микрофон по умолчанию. Укажите INPUT_DEVICE в .env.")
        sys.exit(1)


class TranscriptBuffer:
    def __init__(self, minutes):
        self.max_age_seconds = minutes * 60
        self.items = deque()
        self.lock = threading.Lock()

    def add(self, text):
        now = time.time()
        with self.lock:
            self.items.append((now, text))
            self._trim(now)

    def get_text(self):
        now = time.time()
        with self.lock:
            self._trim(now)
            return "\n".join(text for _, text in self.items)

    def get_last_lines(self, count):
        now = time.time()
        with self.lock:
            self._trim(now)
            if count <= 0:
                return []
            lines = [text for _, text in self.items if text.strip()]
            return lines[-count:]

    def _trim(self, now):
        cutoff = now - self.max_age_seconds
        while self.items and self.items[0][0] < cutoff:
            self.items.popleft()


class ConversationHistory:
    """Rolling conversation context passed to the LLM with each request."""

    def __init__(self, max_exchanges: int = 5):
        self._msgs: list = []
        self._max = max_exchanges * 2  # user + assistant per exchange

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        self._msgs.append({"role": "user",      "content": user_text})
        self._msgs.append({"role": "assistant", "content": assistant_text})
        if len(self._msgs) > self._max:
            self._msgs = self._msgs[-self._max:]

    def get_messages(self) -> list:
        return list(self._msgs)

    def clear(self) -> None:
        self._msgs = []

    def __len__(self) -> int:
        return len(self._msgs) // 2


def append_line(path, text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target = data_path(path) if not os.path.isabs(str(path)) else path
    with open(target, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {text}\n")


def save_history_entry(settings, path, text):
    if not settings.get("save_history", False):
        return False
    append_line(path, text)
    return True


def audio_to_wav_bytes(audio, sample_rate):
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    buffer.seek(0)
    return buffer


def get_default_output_device():
    try:
        default_output = sd.default.device[1]
    except Exception:
        default_output = -1
    return None if default_output == -1 else default_output


def get_default_loopback_device_info():
    if pyaudio is None:
        return None
    pa = pyaudio.PyAudio()
    try:
        return pa.get_default_wasapi_loopback()
    finally:
        pa.terminate()


def list_loopback_devices():
    if pyaudio is None:
        print("pyaudiowpatch не установлен, system audio недоступен.")
        return
    pa = pyaudio.PyAudio()
    try:
        print("Доступные WASAPI loopback devices:")
        found = False
        for device in pa.get_loopback_device_info_generator():
            found = True
            print(
                f"{device['index']}: {device['name']} | "
                f"max_input_channels={device['maxInputChannels']} "
                f"| default_samplerate={device['defaultSampleRate']}"
            )
        if not found:
            print("WASAPI loopback devices не найдены.")
    finally:
        pa.terminate()


def get_system_audio_device(settings):
    if settings.get("system_audio_device") is not None:
        return settings["system_audio_device"]
    if settings.get("auto_system_loopback", True):
        loopback = get_default_loopback_device_info()
        if loopback is not None:
            return int(loopback["index"])
    return get_default_output_device()


def get_output_channels(device):
    if pyaudio is not None:
        pa = pyaudio.PyAudio()
        try:
            info = pa.get_device_info_by_index(device)
            if info.get("isLoopbackDevice"):
                return max(1, min(2, int(info.get("maxInputChannels", 1))))
        except Exception:
            pass
        finally:
            pa.terminate()
    try:
        info = sd.query_devices(device)
        return max(1, min(2, int(info.get("max_output_channels", 1))))
    except Exception:
        return 2


def normalize_audio(audio):
    audio = np.asarray(audio)
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    elif audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    return audio.astype(np.float32) / 32768.0


def float_to_int16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def resample_audio(audio, source_rate, target_rate):
    audio = np.asarray(audio)
    if source_rate == target_rate or len(audio) == 0:
        return audio
    gcd = np.gcd(int(source_rate), int(target_rate))
    up = int(target_rate // gcd)
    down = int(source_rate // gcd)
    resampled = resample_poly(audio.astype(np.float32), up, down, axis=0)
    if np.issubdtype(audio.dtype, np.integer):
        return np.clip(resampled, -32768, 32767).astype(np.int16)
    return resampled.astype(audio.dtype)


def mix_audio_tracks(tracks):
    if not tracks:
        return np.zeros((0, 1), dtype=np.int16)

    normalized = [normalize_audio(track) for track in tracks if len(track) > 0]
    if not normalized:
        return np.zeros((0, 1), dtype=np.int16)

    max_len = max(track.shape[0] for track in normalized)
    padded = []
    for track in normalized:
        if track.shape[0] < max_len:
            pad = np.zeros((max_len - track.shape[0], 1), dtype=np.float32)
            track = np.vstack([track, pad])
        padded.append(track)

    mixed = np.sum(padded, axis=0) / max(1, len(padded))
    return float_to_int16(mixed)


def record_system_audio_loopback(settings, seconds, sample_rate=16000):
    if pyaudio is None:
        raise RuntimeError("pyaudiowpatch не установлен")

    pa = pyaudio.PyAudio()
    stream = None
    try:
        device_index = settings.get("system_audio_device")
        if device_index is None:
            device_info = pa.get_default_wasapi_loopback()
            device_index = int(device_info["index"])
        else:
            device_info = pa.get_device_info_by_index(device_index)

        channels = max(1, min(2, int(device_info.get("maxInputChannels", 1))))
        source_rate = int(device_info.get("defaultSampleRate", sample_rate))
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=source_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
        )

        chunks = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            data = stream.read(1024, exception_on_overflow=False)
            chunks.append(np.frombuffer(data, dtype=np.int16).reshape(-1, channels))
        audio = np.vstack(chunks) if chunks else np.zeros((0, channels), dtype=np.int16)
        audio = float_to_int16(normalize_audio(audio))
        return resample_audio(audio, source_rate, sample_rate)
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()


def record_stream(seconds, sample_rate, channels, device, extra_settings=None):
    frames = []
    lock = threading.Lock()

    def callback(indata, frame_count, time_info, status):
        with lock:
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=device,
        callback=callback,
        extra_settings=extra_settings,
    )
    with stream:
        time.sleep(seconds)

    with lock:
        if not frames:
            return np.zeros((0, channels), dtype=np.int16)
        return np.concatenate(frames, axis=0)


def record_audio_sources(settings, seconds, sample_rate=16000):
    system_enabled = settings.get("system_audio_enabled", False)
    mix_sources = settings.get("mix_mic_and_system_audio", True)

    if not system_enabled:
        return record_stream(seconds, sample_rate, 1, settings["input_device"])

    system_device = get_system_audio_device(settings)
    if system_device is None:
        print("System audio включён, но output device не найден. Записываю только микрофон.")
        return record_stream(seconds, sample_rate, 1, settings["input_device"])

    if not mix_sources:
        try:
            return record_system_audio_loopback(settings, seconds, sample_rate)
        except Exception as exc:
            print(f"Не удалось записать system audio: {exc}. Записываю только микрофон.")
            return record_stream(seconds, sample_rate, 1, settings["input_device"])

    mic_audio = []
    system_audio = []
    errors = []

    def record_mic():
        try:
            mic_audio.append(record_stream(seconds, sample_rate, 1, settings["input_device"]))
        except Exception as exc:
            errors.append(f"микрофон: {exc}")

    def record_system():
        try:
            system_audio.append(record_system_audio_loopback(settings, seconds, sample_rate))
        except Exception as exc:
            errors.append(f"system audio: {exc}")

    threads = [
        threading.Thread(target=record_mic),
        threading.Thread(target=record_system),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tracks = []
    if mic_audio:
        tracks.append(mic_audio[0])
    if system_audio:
        tracks.append(system_audio[0])

    if errors:
        print("Ошибки записи аудио: " + "; ".join(errors))
    if not tracks:
        raise RuntimeError("Не удалось записать микрофон или system audio.")
    return mix_audio_tracks(tracks)


def transcribe_chunk(settings, audio, sample_rate):
    wav_buffer = audio_to_wav_bytes(audio, sample_rate)
    files = {"file": ("chunk.wav", wav_buffer, "audio/wav")}
    data = {"model": "whisper-1", "language": settings["language"]}

    try:
        response = requests.post(
            settings["whisper_url"], files=files, data=data, timeout=120
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Ошибка подключения к Whisper: {exc}")
        return ""

    try:
        payload = response.json()
    except ValueError:
        print(f"Whisper вернул не JSON: {response.text[:300]}")
        return ""

    text = payload.get("text", "").strip()
    if not text:
        print("Whisper вернул пустую расшифровку.")
    return text


def build_llm_request(settings, transcript_text, history: list | None = None):
    """Build an LLM payload.

    history: list of {"role": ..., "content": ...} dicts from ConversationHistory,
             inserted between the system prompt and the new user message.
    """
    messages = [{"role": "system", "content": PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": transcript_text})

    llm_api_type = settings["llm_api_type"]
    temperature = float(settings.get("temperature", 0.2))
    max_tokens = int(settings.get("max_tokens", 300))

    if llm_api_type in ("openai", "openwebui"):
        payload = {
            "model": settings["llm_model"],
            "messages": messages,
            "temperature": temperature,
        }
        if llm_api_type == "openwebui":
            payload["stream"] = False
            # Standalone completions do not need a persisted Open WebUI chat.
            # The former default was a label, not an existing server chat ID.
            chat_id = str(settings.get("openwebui_chat_id") or "").strip()
            if chat_id and chat_id != "meeting-helper":
                payload["chat_id"] = chat_id
            payload["max_tokens"] = max_tokens
        return payload

    if llm_api_type == "ollama":
        return {
            "model": settings["llm_model"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    if llm_api_type == "ollama_generate":
        # ollama_generate uses a flat prompt — prepend history as a plain transcript
        history_text = ""
        if history:
            for msg in history:
                prefix = "Пользователь" if msg["role"] == "user" else "Ассистент"
                history_text += f"{prefix}: {msg['content']}\n"
            history_text += "\n"
        return {
            "model": settings["llm_model"],
            "prompt": f"{PROMPT}\n\n{history_text}Пользователь: {transcript_text}",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    print("Некорректный LLM_API_TYPE. Укажите openai, openwebui, ollama или ollama_generate.")
    return None


def parse_llm_response(settings, response):
    data = response.json()
    llm_api_type = settings["llm_api_type"]

    if llm_api_type in ("openai", "openwebui"):
        return data["choices"][0]["message"]["content"].strip()
    if llm_api_type == "ollama":
        return data["message"]["content"].strip()
    if llm_api_type == "ollama_generate":
        return data["response"].strip()

    raise ValueError(f"Unknown LLM_API_TYPE: {llm_api_type}")


def request_suggestion(settings, transcript_text):
    if not transcript_text.strip():
        print("Пока нет текста для подсказки.")
        return

    payload = build_llm_request(settings, transcript_text)
    if payload is None:
        return

    headers = {}
    if settings["openai_api_key"]:
        headers["Authorization"] = f"Bearer {settings['openai_api_key']}"

    print(f"LLM_API_TYPE={settings['llm_api_type']}")
    print(f"LLM_URL={settings['llm_url']}")
    print(f"LLM_MODEL={settings['llm_model']}")

    try:
        print("Запрос к LLM отправлен...")
        response = requests.post(
            settings["llm_url"],
            json=payload,
            headers=headers,
            timeout=settings["llm_timeout"],
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        response_text = exc.response.text if exc.response is not None else ""
        print(f"Ошибка подключения к LLM: HTTP {status_code}")
        print("Ответ сервера:")
        print(response_text)
        return
    except requests.RequestException as exc:
        print(f"Ошибка подключения к LLM: {exc}")
        return

    try:
        suggestion = parse_llm_response(settings, response)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"Не удалось разобрать ответ LLM: {exc}")
        print(response.text[:500])
        return

    print("\n=== Подсказка ===")
    print(suggestion)
    print("=================\n")
    if save_history_entry(settings, "suggestions.txt", suggestion):
        print("Подсказка сохранена в suggestions.txt")


def listen_loop(settings, buffer, stop_event):
    sample_rate = 16000
    chunk_seconds = settings["chunk_seconds"]

    print(f"Микрофон активен: куски по {chunk_seconds} сек., {sample_rate} Hz.")
    if settings["system_audio_enabled"]:
        system_device = get_system_audio_device(settings)
        print(f"System audio включён, output device: {system_device}")

    while not stop_event.is_set():
        try:
            audio = record_audio_sources(settings, chunk_seconds, sample_rate)
        except Exception as exc:
            print(f"Ошибка записи с микрофона: {exc}")
            time.sleep(2)
            continue

        if stop_event.is_set():
            break

        audio = np.asarray(audio)
        text = transcribe_chunk(settings, audio, sample_rate)
        if text:
            print(f"> {text}")
            save_history_entry(settings, "transcript.txt", text)
            buffer.add(text)


def main():
    parser = argparse.ArgumentParser(description="Локальный помощник для совещаний.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Вывести список input audio devices и завершиться.",
    )
    parser.add_argument(
        "--test-llm",
        help="Отправить текст в LLM без микрофона, вывести ответ и завершиться.",
    )
    args = parser.parse_args()

    settings = load_settings()

    if args.list_devices:
        print_sounddevice_devices()
        return

    if args.test_llm:
        request_suggestion(settings, args.test_llm)
        return

    list_input_devices()
    list_output_devices()
    validate_input_device(settings)

    stop_event = threading.Event()
    transcript_buffer = TranscriptBuffer(settings["buffer_minutes"])

    def shutdown(*_):
        stop_event.set()

    def on_hotkey():
        prompt_mode = settings["prompt_mode"]
        if prompt_mode == "last":
            lines = transcript_buffer.get_last_lines(settings["last_lines"])
            text = "\n".join(lines)
            print(f"Отправляю в LLM последние {len(lines)} фраз:")
            print(text if text else "(пока нет распознанных фраз)")
        elif prompt_mode == "buffer":
            text = transcript_buffer.get_text()
        else:
            print("Некорректный PROMPT_MODE. Укажите last или buffer.")
            return

        threading.Thread(
            target=request_suggestion,
            args=(settings, text),
            daemon=True,
        ).start()

    signal.signal(signal.SIGINT, shutdown)
    keyboard.add_hotkey(settings["hotkey"], on_hotkey)

    print("PySufler запущен.")
    print(f"Горячая клавиша для подсказки: {settings['hotkey']}")
    print("Для выхода нажмите Ctrl+C.")

    worker = threading.Thread(
        target=listen_loop,
        args=(settings, transcript_buffer, stop_event),
        daemon=True,
    )
    worker.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        stop_event.set()
        keyboard.unhook_all_hotkeys()
        worker.join(timeout=settings["chunk_seconds"] + 2)
        print("PySufler остановлен.")


if __name__ == "__main__":
    main()
