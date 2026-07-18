@echo off
setlocal EnableExtensions
rem Monolith launcher. Lives in <root>\app\ next to main.py.
set "APPDIR=%~dp0"
set "ROOT=%APPDIR%.."
set "PYW=%ROOT%\runtime\python\pythonw.exe"
set "PY=%ROOT%\runtime\python\python.exe"

rem If the install-time dependency step never finished (offline install, aborted
rem download), retry it now with a visible console so the user sees progress.
if not exist "%ROOT%\install\env_ok.marker" (
  echo First-run setup: installing Monolith dependencies...
  call "%ROOT%\install\install_env.bat"
  if not exist "%ROOT%\install\env_ok.marker" (
    echo.
    echo Dependency setup failed. Check your internet connection and try again.
    echo Log: %ROOT%\install\installer.log
    pause
    exit /b 1
  )
)

cd /d "%APPDIR%"
start "" "%PYW%" main.py
exit /b 0
