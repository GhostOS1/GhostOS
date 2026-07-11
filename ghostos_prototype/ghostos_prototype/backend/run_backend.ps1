$ErrorActionPreference = "Stop"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
    throw "Python 3 is required. Install Python and run: pip install -r requirements.txt"
}

Write-Host "Starting GhostOS with chat model gemma4:e2b"
& $python.Source app.py

