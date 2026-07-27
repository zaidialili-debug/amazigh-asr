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

[Files]
Source: "dist\AmazighASR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Amazigh-ASR"; Filename: "{app}\AmazighASR.exe"
Name: "{autodesktop}\Amazigh-ASR"; Filename: "{app}\AmazighASR.exe"

[Run]
Filename: "{app}\AmazighASR.exe"; Description: "Lancer Amazigh-ASR"; Flags: nowait postinstall skipifsilent
