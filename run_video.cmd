@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_demo.cmd first.
  exit /b 1
)
if "%~1"=="" (
  ".venv\Scripts\python.exe" -m classroom_ai.video --tui
) else (
  ".venv\Scripts\python.exe" -m classroom_ai.video %*
)
set "video_exit=%errorlevel%"
if not "%video_exit%"=="0" pause
exit /b %video_exit%
