; Monolith Windows installer (Inno Setup 6)
; Installs the app + an isolated CPython runtime under %LOCALAPPDATA%\Programs\Monolith.
; User state lives in %APPDATA%\Monolith (created by the app itself) and is never
; touched by install or uninstall.

#define MyAppName "Monolith"
#ifndef MonolithVersion
  #define MonolithVersion "1.0.0"
#endif

[Setup]
AppId={{7AB0D1E0-A2EA-4BC4-9E43-13F3A7E5F7A0}
AppName={#MyAppName}
AppVersion={#MonolithVersion}
AppPublisher=Eryndel
AppPublisherURL=https://github.com/Svnse/Monolith
AppSupportURL=https://github.com/Svnse/Monolith/issues
DefaultDirName={localappdata}\Programs\Monolith
DefaultGroupName=Monolith
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputBaseFilename=MonolithInstaller_{#MonolithVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
SetupLogging=yes
SetupIconFile=monolith.ico
UninstallDisplayIcon={app}\app\monolith.ico

[InstallDelete]
; Always start from a pristine runtime: upgrading over a runtime whose
; site-packages have diverged (e.g. an upgraded pip) leaves mixed-version
; trees that break pip itself. Dependencies are reinstalled by install_env.bat.
Type: filesandordirs; Name: "{app}\runtime\python"

[Dirs]
Name: "{app}\app"
Name: "{app}\runtime"
Name: "{app}\install"

[Files]
; Bundled CPython runtime (full distribution incl. pythonw.exe and ensurepip).
Source: "payload\python\*"; DestDir: "{app}\runtime\python"; Flags: recursesubdirs ignoreversion
; Monolith v1.0.0 source tree (staged by build_release.bat from the release tag).
Source: "payload\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs ignoreversion
Source: "scripts\install_env.bat"; DestDir: "{app}\install"; Flags: ignoreversion

[Run]
Filename: "{app}\install\install_env.bat"; Flags: runhidden waituntilterminated; StatusMsg: "Installing Python dependencies (a few minutes, requires internet)..."
Filename: "{app}\app\MonolithLauncher.bat"; Description: "Launch Monolith now"; Flags: postinstall nowait skipifsilent shellexec

[Icons]
Name: "{group}\Monolith"; Filename: "{app}\app\MonolithLauncher.bat"; IconFilename: "{app}\app\monolith.ico"
Name: "{autodesktop}\Monolith"; Filename: "{app}\app\MonolithLauncher.bat"; IconFilename: "{app}\app\monolith.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[UninstallDelete]
; pip-installed site-packages under runtime are not in the install manifest,
; so clean the whole program tree explicitly. User state (%APPDATA%\Monolith)
; is deliberately never referenced here.
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\install"
Type: filesandordirs; Name: "{app}\app"

[Code]
procedure WriteInstallTrace(const Msg: String);
var
  LogPath: String;
begin
  LogPath := ExpandConstant('{app}\install\installer.log');
  SaveStringToFile(LogPath, Msg + #13#10, True);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    WriteInstallTrace('[SETUP] Post-install completed for v{#MonolithVersion}');
    if not FileExists(ExpandConstant('{app}\install\env_ok.marker')) then
    begin
      WriteInstallTrace('[SETUP] WARNING: dependency install did not complete');
      if not WizardSilent() then
        MsgBox('Monolith is installed, but the Python dependency step did not finish '
          + '(no internet, or a download failed).' + #13#10#13#10
          + 'Launching Monolith will retry it automatically. Details: '
          + ExpandConstant('{app}\install\installer.log'), mbInformation, MB_OK);
    end;
  end;
end;
