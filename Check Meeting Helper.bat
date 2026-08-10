@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

echo Checking PySufler Python files...
"%PYTHON_EXE%" -m py_compile meeting_helper.py meeting_helper_modern_gui.py test_transcription_merge.py
if errorlevel 1 goto fail

echo.
echo Running unit tests...
"%PYTHON_EXE%" -m unittest test_transcription_merge.py -v
if errorlevel 1 goto fail

echo.
echo Checking PySufler audio devices...
"%PYTHON_EXE%" meeting_helper.py --list-devices

echo.
echo Checking PySufler LLM...
"%PYTHON_EXE%" meeting_helper.py --test-llm "Answer with one word: test"
if errorlevel 1 goto fail

echo.
echo PySufler check completed.
pause
exit /b 0

:fail
echo.
echo PySufler check failed. See messages above.
pause
exit /b 1
