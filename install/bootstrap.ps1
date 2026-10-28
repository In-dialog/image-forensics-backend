$ErrorActionPreference = 'Stop'
$SetupArgs = @($args)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# This branch MUST precede every download, directory creation and uv invocation.
if (($SetupArgs -contains '--dry-run') -or ($SetupArgs -contains '--help') -or ($SetupArgs -contains '-h')) {
    Write-Host 'Inspection only: no bootstrap, installation, downloads or services.'
    $Candidates = @((Join-Path $ProjectRoot '.venv\Scripts\python.exe'), 'python', 'py')
    foreach ($Candidate in $Candidates) {
        $Available = Get-Command $Candidate -ErrorAction SilentlyContinue
        if ((Test-Path -LiteralPath $Candidate) -or ($Available -and $Available.Source -notmatch 'WindowsApps')) {
            & $Candidate -B (Join-Path $PSScriptRoot 'setup.py') @SetupArgs
            exit $LASTEXITCODE
        }
    }
    Write-Host 'Plan: create a private Python 3.11 environment; select NVIDIA CUDA or CPU wheels; install pinned core requirements.'
    Write-Host 'Then guide Ollama installation, Hugging Face SAM3 access and model downloads. Optional --with-forgery and --with-synthetic.'
    Write-Host 'Windows/Linux x64 backend; 12+ GiB VRAM and 32 GiB RAM practical target, about 50 GiB fresh-install disk allowance.'
    Write-Host 'Options: --profile auto|cpu|cuda|cuda-modern --skip-models --with-forgery --with-synthetic --cache-dir PATH --allow-low-disk --yes --dry-run'
    exit 0
}
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    throw 'The backend installer requires Windows x64. The web viewer also supports macOS.'
}
if ([Environment]::OSVersion.Version.Build -lt 19045) { throw 'Windows 10 22H2 or newer is required.' }
$ProjectDrive = (Get-Item -LiteralPath $ProjectRoot).PSDrive
if ($null -ne $ProjectDrive.Free -and $ProjectDrive.Free -lt 1GB) {
    throw 'Not enough free space to prepare Python. Free at least 1 GiB on this drive, or extract the project onto a larger drive. Nothing has been downloaded.'
}
Write-Host 'Image Forensics Analysis setup: installing a private Python environment in this folder.'
$ToolsDir = Join-Path $ProjectRoot '.tools'
$Uv = Join-Path $ToolsDir 'uv.exe'
if (-not (Test-Path -LiteralPath $Uv)) {
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    $Archive = Join-Path $ToolsDir 'uv-0.9.7.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-pc-windows-msvc.zip' -OutFile $Archive
    $ExpectedHash = '5d250c32d3604e28dbe18dc65c668ff628c53e00dde2c642576e831e4a60da64'
    if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) { throw 'uv checksum mismatch. No downloaded program was run.' }
    Expand-Archive -LiteralPath $Archive -DestinationPath $ToolsDir -Force
    if (-not (Test-Path -LiteralPath $Uv)) { throw 'Python bootstrap extraction failed. Check the download and retry.' }
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ToolsDir 'python'
$env:UV_NO_CACHE = '1'
& $Uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 installation failed.' }
$Python = (& $Uv python find --managed-python 3.11).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 was not found after installation.' }
& $Python -B (Join-Path $PSScriptRoot 'setup.py') --uv $Uv @SetupArgs
exit $LASTEXITCODE
