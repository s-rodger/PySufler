import json
import os
import tempfile
import unittest
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import config_manager
import meeting_helper
import meeting_helper_modern_gui
from PySide6.QtWidgets import QApplication


class _FakeMicStream:
    def __init__(self, *args, **kwargs):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass


class _FakeSystemStream:
    def stop_stream(self):
        pass

    def close(self):
        pass


class _FakePyAudio:
    def terminate(self):
        pass


class PrivacySettingsTest(unittest.TestCase):
    def test_wasapi_reader_finishes_before_stream_close(self):
        config_manager.save_config({"live_transcription": False})
        window = meeting_helper_modern_gui.ModernMeetingHelperGui()
        entered = threading.Event()
        stopped = threading.Event()

        class Stream:
            def get_read_available(self):
                return 1024

            def read(self, count, exception_on_overflow=False):
                entered.set()
                time.sleep(0.05)
                return np.zeros(count, dtype=np.int16).tobytes()

            def stop_stream(self):
                if window._system_reader_thread is not None:
                    raise AssertionError("reader must be joined first")
                stopped.set()

            def close(self):
                if not stopped.is_set():
                    raise AssertionError("stop must precede close")

        window.system_stream = Stream()
        window.system_pa = _FakePyAudio()
        window._recording_stop = threading.Event()
        reader = threading.Thread(target=window._system_audio_reader, args=(
            1, 16000, window.system_stream, window._recording_stop
        ))
        window._system_reader_thread = reader
        reader.start()
        self.assertTrue(entered.wait(1))
        window._close_audio_streams()
        self.assertFalse(reader.is_alive())
        self.assertTrue(stopped.is_set())
        self.assertIsNone(window.system_stream)
        self.assertIsNone(window.system_pa)
        window.close()

    def test_openwebui_standalone_chat_id(self):
        settings = config_manager.normalize_settings({"llm_api_type": "openwebui"})
        for chat_id in ("", "meeting-helper"):
            settings["openwebui_chat_id"] = chat_id
            self.assertNotIn("chat_id", meeting_helper.build_llm_request(settings, "test"))
        settings["openwebui_chat_id"] = "existing-chat-id"
        self.assertEqual(meeting_helper.build_llm_request(settings, "test")["chat_id"], "existing-chat-id")

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.path_patches = (
            patch.object(config_manager, "APP_DIR", root),
            patch.object(config_manager, "DATA_DIR", root / "data"),
            patch.object(config_manager, "LOGS_DIR", root / "logs"),
            patch.object(config_manager, "CONFIG_PATH", root / "config.json"),
            patch.object(config_manager, "SECRET_PATH", root / "secrets.json"),
        )
        for item in self.path_patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.path_patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_saved_audio_settings_have_priority(self):
        config_manager.save_config({
            "system_audio_enabled": False,
            "mix_mic_and_system_audio": False,
        })
        loaded = config_manager.load_config()
        self.assertFalse(loaded["system_audio_enabled"])
        self.assertFalse(loaded["mix_mic_and_system_audio"])

    def test_audio_source_combinations(self):
        mic = np.full((20, 1), 1000, dtype=np.int16)
        system = np.full((20, 1), 3000, dtype=np.int16)

        mic_only = SimpleNamespace(settings={
            "system_audio_enabled": False,
            "mix_mic_and_system_audio": False,
        })
        result = meeting_helper_modern_gui.ModernMeetingHelperGui._mix_frames(
            mic_only, [mic], [system]
        )
        self.assertAlmostEqual(float(result.mean()), 1000, delta=2)

        system_only = SimpleNamespace(settings={
            "system_audio_enabled": True,
            "mix_mic_and_system_audio": False,
        })
        result = meeting_helper_modern_gui.ModernMeetingHelperGui._mix_frames(
            system_only, [mic], [system]
        )
        self.assertAlmostEqual(float(result.mean()), 3000, delta=2)

        mixed = SimpleNamespace(settings={
            "system_audio_enabled": True,
            "mix_mic_and_system_audio": True,
        })
        result = meeting_helper_modern_gui.ModernMeetingHelperGui._mix_frames(
            mixed, [mic], [system]
        )
        self.assertAlmostEqual(float(result.mean()), 2000, delta=2)

    def test_recording_respects_system_audio_toggle(self):
        for system_enabled in (False, True):
            with self.subTest(system_enabled=system_enabled):
                config_manager.save_config({
                    "system_audio_enabled": system_enabled,
                    "mix_mic_and_system_audio": True,
                    "live_transcription": False,
                })
                window = meeting_helper_modern_gui.ModernMeetingHelperGui()

                def start_fake_system():
                    window.system_stream = _FakeSystemStream()
                    window.system_pa = _FakePyAudio()

                with patch.object(meeting_helper_modern_gui.sd, "InputStream", _FakeMicStream), patch.object(
                    window, "_start_system_audio_loopback", side_effect=start_fake_system
                ) as system_start:
                    window.start_recording()
                    self.assertTrue(window.is_recording)
                    self.assertEqual(system_start.called, system_enabled)
                    window.stop_recording()
                    self.assertFalse(window.is_recording)
                window.close()

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI test")
    def test_api_key_is_migrated_to_dpapi_storage(self):
        key = "synthetic-test-key"
        config_manager.ensure_app_dirs()
        config_manager.CONFIG_PATH.write_text(
            json.dumps({**config_manager.DEFAULT_SETTINGS, "openai_api_key": key}),
            encoding="utf-8",
        )

        loaded = config_manager.load_config()

        public_config = json.loads(config_manager.CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(loaded["openai_api_key"], key)
        self.assertEqual(public_config["openai_api_key"], "")
        self.assertTrue(config_manager.SECRET_PATH.exists())
        self.assertNotIn(key, config_manager.SECRET_PATH.read_text(encoding="utf-8"))

    def test_dpapi_failure_never_writes_plaintext_key(self):
        original = '{"llm_model": "existing"}'
        config_manager.CONFIG_PATH.write_text(original, encoding="utf-8")

        with patch.object(config_manager.os, "name", "nt"), patch.object(
            config_manager, "_save_protected_api_key", return_value=False
        ):
            with self.assertRaises(OSError):
                config_manager.save_config({"openai_api_key": "secret-value"})

        self.assertEqual(config_manager.CONFIG_PATH.read_text(encoding="utf-8"), original)

    def test_saved_api_key_is_not_rendered_in_settings(self):
        settings = config_manager.normalize_settings({"openai_api_key": "synthetic-test-key"})
        with patch.object(meeting_helper_modern_gui.SettingsDialog, "_fill_audio_devices"):
            dialog = meeting_helper_modern_gui.SettingsDialog(settings)
        self.assertEqual(dialog.api_key.text(), "")
        self.assertIn("Защищённый ключ сохранён", dialog.api_key.placeholderText())
        dialog.close()

    def test_history_is_opt_in(self):
        target = Path(self.temp_dir.name) / "history.txt"
        with patch.object(meeting_helper, "data_path", return_value=target):
            self.assertFalse(meeting_helper.save_history_entry({}, "history.txt", "private text"))
            self.assertFalse(target.exists())
            self.assertTrue(meeting_helper.save_history_entry(
                {"save_history": True}, "history.txt", "private text"
            ))
            self.assertIn("private text", target.read_text(encoding="utf-8"))

    def test_diagnostic_log_never_contains_conversation_content(self):
        target = Path(self.temp_dir.name) / "diagnostics.log"
        with patch.object(meeting_helper_modern_gui, "DIAGNOSTIC_LOG", target):
            meeting_helper_modern_gui.log_whisper_diagnostic(
                False, "disabled", transcript="secret conversation"
            )
            self.assertFalse(target.exists())
            meeting_helper_modern_gui.log_whisper_diagnostic(
                True,
                "whisper_live_chunk",
                transcript="secret conversation",
                raw_bytes=42,
                extracted_chars=12,
            )
        content = target.read_text(encoding="utf-8")
        self.assertIn("raw_bytes=42", content)
        self.assertNotIn("secret conversation", content)

    def test_local_history_is_deleted_only_by_explicit_action(self):
        config_manager.ensure_app_dirs()
        for path in config_manager.local_history_paths():
            path.write_text("synthetic data", encoding="utf-8")
        self.assertEqual(config_manager.delete_local_history(), 3)
        self.assertTrue(all(not path.exists() for path in config_manager.local_history_paths()))


if __name__ == "__main__":
    unittest.main()
