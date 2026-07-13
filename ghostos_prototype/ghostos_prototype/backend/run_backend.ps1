$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $project

$venvPython = Join-Path $project ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) { $command = Get-Command py -ErrorAction SilentlyContinue }
    if (-not $command) { throw "Python 3.11+ is required. Install Python, then run .\setup_backend.ps1" }
    $python = $command.Source
}

& $python -c "import flask, requests, pypdf, docx, numpy, watchdog, psutil" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Backend dependencies are missing for this Python. Run .\setup_backend.ps1 once, then retry."
}

Write-Host "Starting GhostOS at http://127.0.0.1:5000 using your configured local Ollama models"
& $python app.py
