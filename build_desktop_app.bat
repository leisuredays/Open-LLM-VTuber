@echo off
REM Open-LLM-VTuber Desktop App Build Script for Windows

REM Get the directory where the batch file is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Open-LLM-VTuber Desktop App Builder
echo ========================================
echo.
echo Current directory: %CD%
echo.

REM Check if Node.js is installed
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: npm is not installed!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [1/4] Cloning frontend repository...
if not exist "Open-LLM-VTuber-Web" (
    git clone https://github.com/Open-LLM-VTuber/Open-LLM-VTuber-Web.git
    if %errorlevel% neq 0 (
        echo ERROR: Failed to clone repository
        pause
        exit /b 1
    )
) else (
    echo Frontend repository already exists, updating...
    cd Open-LLM-VTuber-Web
    git pull
    cd ..
)

echo.
echo [2/4] Installing dependencies...
cd Open-LLM-VTuber-Web
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [3/4] Building Windows executable...
call npm run build:win
if %errorlevel% neq 0 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo [4/4] Build completed!
echo ========================================
echo.
echo The Windows executable has been created in:
echo %cd%\dist\
echo.
echo You can now run the .exe file to start the desktop app.
echo Remember to start the backend server first with:
echo   uv run run_server.py
echo.
pause
