@echo off
REM Quick fix for lib64 symlink issue

REM Get the directory where the batch file is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Quick Fix for lib64 Issue
echo ========================================
echo.

echo Attempting to remove problematic lib64 symlink...
echo.

REM Try different methods to remove lib64
if exist .venv\lib64 (
    del /f /q .venv\lib64 2>nul
    if exist .venv\lib64 (
        rmdir .venv\lib64 2>nul
    )
    if exist .venv\lib64 (
        echo Failed with normal delete, trying with attrib...
        attrib -r -s -h .venv\lib64
        del /f /q .venv\lib64 2>nul
        rmdir .venv\lib64 2>nul
    )
)

if not exist .venv\lib64 (
    echo [OK] lib64 removed successfully!
    echo.
    echo Now try running: start_server.bat
) else (
    echo [FAILED] Could not remove lib64
    echo.
    echo Please try:
    echo   1. Run this script as Administrator
    echo   2. Or run: recreate_venv_windows.bat
)

echo.
pause
