@echo off
setlocal
cd /d "%~dp0"
py -3.11 scripts\setup_demo.py
if errorlevel 1 (
  echo Install Python 3.11 from python.org with the Python launcher enabled, then retry.
  pause
  exit /b 1
)
echo Setup complete. Open run_demo.cmd or run_video.cmd.
pause
