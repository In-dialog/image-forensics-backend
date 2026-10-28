@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run Install.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B install\start.py %*
exit /b %ERRORLEVEL%
