@echo off
REM Install uv package manager for Windows

echo ========================================
echo Installing uv Package Manager
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found!
    pause
    exit /b 1
)

echo Installing uv using the official installer...
echo.

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Installation completed!
    echo ========================================
    echo.
    echo Please close this window and open a new Command Prompt
    echo to use uv commands.
    echo.
    echo You can now run: start_server.bat
    echo.
) else (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Alternative installation method:
    echo   pip install uv
    echo.
)

pause
