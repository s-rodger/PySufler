# PySufler — Суфлёр совещаний

PySufler — локальный desktop-помощник для совещаний. Он записывает микрофон и, при включённом режиме «Слушать собеседника», системный звук Windows через WASAPI loopback, расшифровывает речь через локальный faster-whisper OpenAI-compatible endpoint и отправляет текст в локальную LLM.

Основной запуск GUI: `meeting_helper_modern_gui.py`.

## Возможности

- ручная запись вопроса;
- live-preview расшифровки во время записи;
- финальная расшифровка полного аудио после остановки;
- захват микрофона + системного звука;
- поддержка OpenWebUI/OpenAI-compatible, Ollama chat и Ollama generate;
- окно настроек без ручного редактирования `.env`;
- сохранение подсказок и расшифровок в `%APPDATA%\PySufler\data`;
- логи в `%APPDATA%\PySufler\logs`.

## Установка из исходников

```powershell
cd path\to\PySufler
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Для первой настройки можно скопировать пример:

```powershell
copy .env.example .env
```

При первом запуске PySufler создаёт конфиг:

```text
%APPDATA%\PySufler\config.json
```

Если `config.json` ещё нет, но есть `.env`, настройки автоматически мигрируют из `.env` в JSON-конфиг. Файл `.env` не удаляется.

## Настройка

Откройте кнопку `⚙` в верхней панели приложения.

В настройках можно изменить:

- URL Whisper;
- тип LLM API: `openwebui`, `openai`, `ollama`, `ollama_generate`;
- URL LLM;
- модель, например `qwen3.5:9b`;
- `OPENAI_API_KEY` для OpenWebUI/OpenAI-compatible endpoint;
- `OPENWEBUI_CHAT_ID`;
- температуру, лимит токенов и timeout;
- микрофон;
- автоматический WASAPI loopback;
- live-распознавание.

Для OpenWebUI создайте API key в интерфейсе OpenWebUI: откройте профиль пользователя, раздел Settings/API Keys, создайте ключ и вставьте его в поле `OPENAI_API_KEY`.

Безопасные публичные значения по умолчанию находятся в `.env.example` и используют `127.0.0.1`.

## Запуск

Через bat-файл:

```powershell
.\Start Meeting Helper.bat
```

Напрямую:

```powershell
.\.venv\Scripts\pythonw.exe meeting_helper_modern_gui.py
```

Перезапуск:

```powershell
.\Restart Meeting Helper.bat
```

Проверка:

```powershell
.\Check Meeting Helper.bat
```

## System audio

Захват системного звука работает на Windows через WASAPI loopback. Включите переключатель `Слушать собеседника`, чтобы записывать микрофон и звук из Teams/Zoom/Meet/браузера.

По умолчанию:

```text
SYSTEM_AUDIO_ENABLED=true
AUTO_SYSTEM_LOOPBACK=true
MIX_MIC_AND_SYSTEM_AUDIO=true
```

Если автоматический loopback недоступен, приложение не падает и продолжает слушать микрофон. Как fallback можно использовать VB-CABLE или похожий virtual audio cable.

## Консольный режим

Список устройств:

```powershell
python meeting_helper.py --list-devices
```

Проверка LLM без микрофона:

```powershell
python meeting_helper.py --test-llm "Ответь кратко: что такое Солнце?"
```

## Сборка portable exe

```powershell
.\build_windows.bat
```

Результат:

```text
dist\PySufler\PySufler.exe
```

Сборка использует PyInstaller onedir, без консоли, с иконкой `meeting_helper.ico`.

## Конфигурация и данные

Конфигурация приложения и API-ключи хранятся локально и не входят в состав репозитория. При публикации собственных изменений не добавляйте в Git файл `.env`, пользовательский `config.json`, логи, записи совещаний и другие персональные данные.
