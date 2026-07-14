@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=%CD%\venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo ERROR: venv\Scripts\python.exe was not found.
    echo Run install.bat first.
    pause
    exit /b 2
)

rem Public baseline profile: no experimental MONOLITH_* flags are forced here.
rem See docs\CONFIGURATION.md before enabling cognition or network features.
"%VENV_PY%" main.py
set "APP_RC=%ERRORLEVEL%"

if not "%APP_RC%"=="0" (
    echo.
    echo Monolith exited with code %APP_RC%.
    pause
)
exit /b %APP_RC%
