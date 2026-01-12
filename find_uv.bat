@echo off
REM Find uv.exe location on your system

echo ========================================
echo Searching for uv.exe...
echo ========================================
echo.

REM Common locations
echo Checking common locations...
echo.

if exist "%USERPROFILE%\.cargo\bin\uv.exe" echo [FOUND] %USERPROFILE%\.cargo\bin\uv.exe
if exist "%LOCALAPPDATA%\Programs\uv\uv.exe" echo [FOUND] %LOCALAPPDATA%\Programs\uv\uv.exe
if exist "%APPDATA%\Python\Scripts\uv.exe" echo [FOUND] %APPDATA%\Python\Scripts\uv.exe

echo.
echo ========================================
echo.
echo If uv was found above, you can add it to PATH by running:
echo   add_uv_to_path.bat
echo.
pause
