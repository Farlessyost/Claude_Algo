@echo off
title BTC Perp Trading Console
cd /d "%~dp0"

REM Open the UI after the server has had a moment to boot
start "" /b cmd /c "timeout /t 6 /nobreak >nul && start """" http://127.0.0.1:8787"

REM Launch the server + bot via the existing PowerShell entrypoint
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"

echo.
echo Server stopped. Press any key to close this window.
pause >nul
