@echo off
setlocal EnableExtensions
rem Monolith dependency setup. Installs the app's base dependencies into the
rem bundled runtime. Uses only the bundled Python, never system Python.
rem Layout: %~dp0 = <root>\install\ ; runtime and app are siblings.

set "ROOT=%~dp0.."
set "PY=%ROOT%\runtime\python\python.exe"
set "APP=%ROOT%\app"
set "LOG=%~dp0installer.log"
set "MARKER=%~dp0env_ok.marker"

if exist "%MARKER%" del "%MARKER%"
echo [ENV-INSTALL] ---- %DATE% %TIME% ---->>"%LOG%"

if not exist "%PY%" (
  echo [ENV-INSTALL] ERROR: missing bundled python at "%PY%">>"%LOG%"
  exit /b 2
)

rem Bootstrap pip if the bundled runtime lacks it.
"%PY%" -m pip --version >nul 2>&1
if not %errorlevel%==0 (
  echo [ENV-INSTALL] pip missing, running ensurepip>>"%LOG%"
  "%PY%" -m ensurepip --upgrade >>"%LOG%" 2>&1
)

"%PY%" -m pip install --upgrade pip setuptools wheel >>"%LOG%" 2>&1
if not %errorlevel%==0 (
  echo [ENV-INSTALL] ERROR: pip toolchain bootstrap failed>>"%LOG%"
  exit /b 3
)

rem Base profile from the app's own pyproject.toml (PySide6, pydantic, yaml, ...).
rem No torch: vision/audio are optional extras the user can add later.
echo [ENV-INSTALL] Installing Monolith base dependencies>>"%LOG%"
"%PY%" -m pip install "%APP%" >>"%LOG%" 2>&1
if not %errorlevel%==0 (
  echo [ENV-INSTALL] ERROR: base dependency install failed>>"%LOG%"
  exit /b 4
)

rem Best-effort local-LLM support (prebuilt wheel availability varies) —
rem failure here must not fail the install; cloud endpoints work without it.
echo [ENV-INSTALL] Installing optional local-llm extra (best effort)>>"%LOG%"
"%PY%" -m pip install "%APP%[local-llm]" >>"%LOG%" 2>&1
if not %errorlevel%==0 (
  echo [ENV-INSTALL] WARN: local-llm extra unavailable; continuing without it>>"%LOG%"
)

rem Smoke-check the startup imports before declaring success.
"%PY%" -c "import PySide6, yaml, pydantic, markdown, pygments; print('imports OK')" >>"%LOG%" 2>&1
if not %errorlevel%==0 (
  echo [ENV-INSTALL] ERROR: import smoke check failed>>"%LOG%"
  exit /b 5
)

echo [ENV-INSTALL] SUCCESS>>"%LOG%"
echo ok>"%MARKER%"
exit /b 0
