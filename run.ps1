# BTC Perp Trading Console launcher (Windows PowerShell)
# Usage:  .\run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"

Write-Host "Installing dependencies..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# Optional: enable Opus decisions by setting your Anthropic key first, e.g.
#   $env:ANTHROPIC_API_KEY = "sk-ant-..."
if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "NOTE: ANTHROPIC_API_KEY not set -> loop runs on the deterministic strategy (still fully autonomous)." -ForegroundColor Yellow
}

Write-Host "Open http://127.0.0.1:8787 in your browser." -ForegroundColor Green
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8787
