@echo off
setlocal
cd /d "%~dp0"

set "LOG_DIR=%APPDATA%\PySufler\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG=%LOG_DIR%\meeting_helper_restart.log"
echo [%date% %time%] PySufler restart requested>>"%LOG%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$procs=Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*') -and $_.CommandLine -like '*meeting_helper_modern_gui.py*' }; foreach ($p in $procs) { try { Stop-Process -Id $p.ProcessId -Force; Write-Output ('stopped PID=' + $p.ProcessId) } catch { Write-Output $_.Exception.Message } }; for ($i=0; $i -lt 20; $i++) { $left=Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*') -and $_.CommandLine -like '*meeting_helper_modern_gui.py*' }; if (-not $left) { break }; Start-Sleep -Milliseconds 250 }"

powershell -NoProfile -Command "Start-Sleep -Seconds 2"
call "Start Meeting Helper.bat"
echo [%date% %time%] PySufler restart completed>>"%LOG%"
