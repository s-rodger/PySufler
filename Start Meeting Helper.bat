@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python.exe"
set "PYTHONW_EXE=pythonw.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if exist ".venv\Scripts\pythonw.exe" set "PYTHONW_EXE=.venv\Scripts\pythonw.exe"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*') -and $_.CommandLine -like '*meeting_helper_modern_gui.py*' }; if ($p) { exit 2 }"
if errorlevel 2 (
  echo PySufler already running.
  powershell -NoProfile -Command "Start-Sleep -Seconds 3"
  exit /b 0
)

"%PYTHON_EXE%" -m py_compile meeting_helper_modern_gui.py
if errorlevel 1 (
  echo.
  echo Failed to validate PySufler GUI: meeting_helper_modern_gui.py
  pause
  exit /b 1
)

start "PySufler" "%PYTHONW_EXE%" "meeting_helper_modern_gui.py"
exit /b 0
