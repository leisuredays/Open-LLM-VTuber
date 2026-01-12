@echo off
REM Recreate Python virtual environment for Windows

REM Get the directory where the batch file is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Recreating Python Virtual Environment
echo for Windows
echo ========================================
echo.

echo WARNING: This will delete the existing .venv directory
echo and recreate it for Windows.
echo.
set /p CONFIRM="Continue? (y/n): "

if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo [1/3] Removing old virtual environment...

REM Try to remove normally first
rmdir /s /q .venv 2>nul

REM If that fails, try with attrib to remove attributes
if exist .venv (
    echo Removing file attributes...
    attrib -r -s -h .venv\*.* /s /d
    rmdir /s /q .venv 2>nul
)

REM Last resort: remove specific problem files
if exist .venv\lib64 (
    echo Removing lib64 symlink...
    del /f /q .venv\lib64 2>nul
    rmdir .venv\lib64 2>nul
)

if exist .venv (
    echo ERROR: Could not remove .venv directory
    echo Please try running as Administrator or manually delete:
    echo   C:\Open-LLM-VTuber\.venv
    pause
    exit /b 1
)

echo [OK] Old virtual environment removed
echo.

echo [2/3] Creating new virtual environment with uv...

C:\Users\zekiy\.local\bin\uv.exe sync

if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying installation...

C:\Users\zekiy\.local\bin\uv.exe run python --version

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo Virtual environment has been recreated for Windows.
echo You can now run: start_server.bat
echo.

pause
