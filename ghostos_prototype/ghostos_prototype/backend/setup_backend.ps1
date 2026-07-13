param([switch]$WithOCR, [switch]$WithVoice)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $project

$command = Get-Command python -ErrorAction SilentlyContinue
if (-not $command) { $command = Get-Command py -ErrorAction SilentlyContinue }
if (-not $command) { throw "Install Python 3.11 or newer first." }

$venv = Join-Path $project ".venv"
if (-not (Test-Path $venv)) {
    & $command.Source -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python virtual environment. Verify that Python 3.11+ includes the venv module."
    }
}
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "The virtual environment is incomplete: $python was not created. Remove .venv and run this setup again."
}
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip in the GhostOS virtual environment. Check your network connection and retry."
}
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the required GhostOS backend packages from requirements.txt."
}
if ($WithOCR) {
    & $python -m pip install -r requirements-ocr.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the optional local OCR Python packages from requirements-ocr.txt."
    }
    Write-Host "OCR Python packages installed. Install the local Tesseract executable separately if diagnostics still reports OCR unavailable."
}
if ($WithVoice) {
    & $python -m pip install -r requirements-voice.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the optional local voice Python packages from requirements-voice.txt."
    }
    Write-Host "Voice packages installed. The Whisper model itself downloads once, on first transcription, then runs fully offline."
}

Write-Host "GhostOS backend setup complete. Start it with .\run_backend.ps1"