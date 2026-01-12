@echo off
REM Open-LLM-VTuber Server Start Script for Windows

REM Get the directory where the batch file is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Open-LLM-VTuber Server
echo ========================================
echo.
echo Starting server...
echo Server will be available at: http://localhost:12393
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Check if uv is available
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo Using uv to run server...
    uv run run_server.py
) else (
    echo uv not found, using Python virtual environment...
    REM Try Unix-style venv (from WSL/Linux)
    if exist ".venv\bin\python.exe" (
        .venv\bin\python.exe run_server.py
    ) else if exist ".venv\Scripts\python.exe" (
        REM Windows-style venv
        call .venv\Scripts\activate.bat
        python run_server.py
    ) else (
        echo ERROR: Neither uv nor Python virtual environment found!
        echo.
        echo Please install uv: https://docs.astral.sh/uv/getting-started/installation/
        echo Or run: pip install uv
        pause
        exit /b 1
    )
)

pause
