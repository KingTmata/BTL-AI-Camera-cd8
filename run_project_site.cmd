@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "docs\project_site\server.py" --host 0.0.0.0 --port 8765 %*
    goto finished
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "docs\project_site\server.py" --host 0.0.0.0 --port 8765 %*
    goto finished
)

where python >nul 2>nul
if not errorlevel 1 (
    python "docs\project_site\server.py" --host 0.0.0.0 --port 8765 %*
    goto finished
)

echo Python 3 khong duoc tim thay. Cai Python 3.11 hoac tao moi truong .venv truoc.
:finished
if errorlevel 1 pause
endlocal
