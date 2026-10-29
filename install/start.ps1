$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\scripts\windows_process_job.ps1')
Enable-LauncherProcessCleanup
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
& (Join-Path $projectDirectory '.venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'start.py') @args
exit $LASTEXITCODE
