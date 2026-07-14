@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=base"

if /I "%PROFILE%"=="vision" goto :heavy_profile
if /I "%PROFILE%"=="audio" goto :heavy_profile
if /I "%PROFILE%"=="base" goto :profile_ok
if /I "%PROFILE%"=="files" goto :profile_ok
if /I "%PROFILE%"=="local-llm" goto :profile_ok
if /I "%PROFILE%"=="matrix" goto :profile_ok
if /I "%PROFILE%"=="dev" goto :profile_ok

echo ERROR: Unknown profile "%PROFILE%".
echo Valid profiles: base, files, local-llm, matrix, dev, vision, audio
exit /b 2

:heavy_profile
echo ERROR: The %PROFILE% stack is hardware and Torch-build specific.
echo Follow docs\INSTALLATION.md instead of installing it blindly.
exit /b 2

:profile_ok
set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if errorlevel 1 goto :python_missing
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if errorlevel 1 goto :python_wrong_version
    set "PYTHON_CMD=python"
)

echo Using Python:
%PYTHON_CMD% --version
if errorlevel 1 goto :failed

if not exist "venv\Scripts\python.exe" (
    echo Creating venv...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 goto :failed
) else (
    echo Reusing existing venv. Delete the venv directory to rebuild it.
)

set "VENV_PY=%CD%\venv\Scripts\python.exe"
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
    echo ERROR: The existing venv is not Python 3.11. Delete venv and rerun install.bat.
    exit /b 2
)

echo Updating packaging tools...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed

echo Installing profile: %PROFILE%
if /I "%PROFILE%"=="base" (
    "%VENV_PY%" -m pip install --no-build-isolation -e .
) else (
    "%VENV_PY%" -m pip install --no-build-isolation -e ".[%PROFILE%]"
)
if errorlevel 1 goto :failed

echo Running base import smoke check...
set "MONOLITH_ROOT=%TEMP%\monolith-install-smoke-%RANDOM%"
set "MONOLITH_ALLOW_UNANCHORED_ROOT=1"
"%VENV_PY%" -c "import PySide6, yaml, pydantic, markdown, pygments; import core.config; import ui.components.markdown_renderer; import bootstrap; print('Monolith base imports: OK')"
set "SMOKE_RC=%ERRORLEVEL%"
if exist "%MONOLITH_ROOT%" rmdir /s /q "%MONOLITH_ROOT%" >nul 2>&1
set "MONOLITH_ROOT="
set "MONOLITH_ALLOW_UNANCHORED_ROOT="
if not "%SMOKE_RC%"=="0" goto :failed

echo.
echo Installation complete: %PROFILE%
echo Run start.bat, then configure a model in the Config panel.
echo Optional profiles and hardware notes: docs\INSTALLATION.md
exit /b 0

:python_missing
echo ERROR: Python was not found. Install 64-bit Python 3.11 and rerun this script.
exit /b 2

:python_wrong_version
echo ERROR: Monolith's tested base currently requires Python 3.11.x.
echo Install Python 3.11 or use the Windows py launcher with py -3.11.
exit /b 2

:failed
echo.
echo ERROR: Installation failed. Review the command output above.
exit /b 1
