[Setup]
AppName=Amazigh-ASR
AppVersion=1.0
DefaultDirName={autopf}\Amazigh-ASR
DefaultGroupName=Amazigh-ASR
OutputDir=Output
OutputBaseFilename=Amazigh-ASR-Setup
Compression=lzma
SolidCompression=yes
SetupIconFile=icon.ico
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\AmazighASR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\Amazigh-ASR"; Filename: "{app}\AmazighASR.exe"
Name: "{autodesktop}\Amazigh-ASR"; Filename: "{app}\AmazighASR.exe"

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installation des composants requis..."; Flags: waituntilterminated
Filename: "{app}\AmazighASR.exe"; Description: "Lancer Amazigh-ASR"; Flags: nowait postinstall skipifsilent
