@echo off
REM Clear electron-builder cache to fix build issues

echo ========================================
echo Clearing electron-builder cache...
echo ========================================
echo.

set CACHE_DIR=%LOCALAPPDATA%\electron-builder\Cache

if exist "%CACHE_DIR%" (
    echo Deleting cache directory: %CACHE_DIR%
    rmdir /s /q "%CACHE_DIR%"
    echo Cache cleared successfully!
) else (
    echo Cache directory not found. Nothing to clear.
)

echo.
echo You can now try building again.
echo.
pause
