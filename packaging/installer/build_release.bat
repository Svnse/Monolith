@echo off
setlocal EnableExtensions
rem Local build of the Monolith Windows installer (mirror of release.yml).
rem Usage: build_release.bat [vX.Y.Z] [path-to-ISCC.exe]
rem Requires: Inno Setup 6, internet (NuGet python payload), run from a git checkout.

set "HERE=%~dp0"
cd /d "%HERE%..\.."

set "TAG=%~1"
if "%TAG%"=="" for /f "tokens=2 delims== " %%v in ('findstr /r "APP_VERSION" core\version.py') do set "TAG=v%%~v"
set "ISCC=%~2"
if "%ISCC%"=="" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"

echo [BUILD] Tag: %TAG%
if not exist "%ISCC%" ( echo [BUILD] ERROR: ISCC not found: "%ISCC%" & exit /b 10 )

if exist "%HERE%payload" rmdir /s /q "%HERE%payload"
mkdir "%HERE%payload\app"

echo [BUILD] Staging app payload from %TAG%...
git archive %TAG% | tar -xf - -C "%HERE%payload\app"
if not %errorlevel%==0 ( echo [BUILD] ERROR: git archive failed & exit /b 11 )
copy /y "%HERE%launcher\MonolithLauncher.bat" "%HERE%payload\app\" >nul
copy /y "%HERE%monolith.ico" "%HERE%payload\app\" >nul

echo [BUILD] Staging CPython runtime payload...
if not exist "%TEMP%\python-nuget.zip" curl -sL -o "%TEMP%\python-nuget.zip" https://www.nuget.org/api/v2/package/python/3.11.9
tar -xf "%TEMP%\python-nuget.zip" -C "%HERE%payload" tools
move "%HERE%payload\tools" "%HERE%payload\python" >nul
"%HERE%payload\python\python.exe" -c "import ensurepip, ssl, sqlite3; print('payload python OK')"
if not %errorlevel%==0 ( echo [BUILD] ERROR: python payload unhealthy & exit /b 12 )

set "VER=%TAG:v=%"
echo [BUILD] Compiling installer %VER%...
"%ISCC%" "%HERE%MonolithInstaller.iss" /DMonolithVersion=%VER% /O"%HERE%dist"
if not %errorlevel%==0 ( echo [BUILD] ERROR: ISCC failed & exit /b 13 )

echo [BUILD] SUCCESS: %HERE%dist\MonolithInstaller_%VER%.exe
exit /b 0
