@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup-demo.cmd first. Do not copy .venv from another computer.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run src/app.py --server.address 127.0.0.1 --server.headless false
if errorlevel 1 pause
