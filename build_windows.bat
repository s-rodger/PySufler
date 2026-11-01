@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

"%PYTHON_EXE%" -m pip install -r requirements.txt -r requirements-build.txt
if errorlevel 1 goto fail

"%PYTHON_EXE%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --windowed ^
  --name PySufler ^
  --icon meeting_helper.ico ^
  --version-file version_info.txt ^
  --add-data "meeting_helper.ico;." ^
  meeting_helper_modern_gui.py
if errorlevel 1 goto fail

"%PYTHON_EXE%" -c "from pathlib import Path; d=Path(r'dist\PySufler\_internal'); [(d/n).unlink(missing_ok=True) for n in ('icuuc.dll','icudt78.dll')]"
if errorlevel 1 goto fail

echo.
echo Build completed: dist\PySufler\PySufler.exe
exit /b 0

:fail
echo.
echo Build failed.
exit /b 1
