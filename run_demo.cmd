@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_demo.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m classroom_ai.interfaces.web --port 8090 %*
set "demo_exit=%errorlevel%"
if not "%demo_exit%"=="0" pause
exit /b %demo_exit%
