[CmdletBinding()]
param(
    [ValidateSet('Original','GSR1')]
    [string]$Mode = 'Original',
    [switch]$CaptureFGInputs,
    [switch]$FrameGeneration,
    [string]$GamePath
)
$ErrorActionPreference = 'Stop'
$binary = (Resolve-Path (Join-Path $PSScriptRoot '../build/release/bin/NXN1X.exe')).Path
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $binary
$start.WorkingDirectory = Split-Path $binary
$start.UseShellExecute = $false
if ($FrameGeneration) {
    $start.EnvironmentVariables['NXN1X_DLSS_FG'] = '1'
}
$start.EnvironmentVariables['NXN1X_DLSS_AUDIT_FRAME'] = '120'
if ($CaptureFGInputs) {
    $capture = New-Item -ItemType Directory -Path (Join-Path $env:LOCALAPPDATA ('NXN1X/FG-' + [guid]::NewGuid().ToString('N'))) -Force
    $start.EnvironmentVariables['NXN1X_FG_CAPTURE_DIR'] = $capture.FullName
    Write-Host "Ctrl+Shift+F11 records two frames in $($capture.FullName)"
}
if ($Mode -eq 'GSR1') {
    $start.EnvironmentVariables['DM_VK_NDSR'] = '0'
} else {
    $start.EnvironmentVariables['DM_VK_NDSR'] = '1'
}
if ($GamePath) {
    $game = (Get-Item -LiteralPath $GamePath -ErrorAction Stop).FullName
    $start.Arguments = '-g "' + $game + '"'
}
[void][Diagnostics.Process]::Start($start)
