$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) {
    $Uv = Join-Path $ProjectRoot ".venv\Scripts\uv.exe"
}

if (-not (Test-Path -LiteralPath $Uv)) {
    throw "Install UV and run 'uv sync --frozen --extra test'; see README.md"
}

$ActualUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name.Split('\')[-1]
$SafeUser = $ActualUser -replace '[^A-Za-z0-9_.-]', '_'
$env:UV_CACHE_DIR = Join-Path ([System.IO.Path]::GetTempPath()) "optimizer-resurrection-uv-$SafeUser"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& (Join-Path $ProjectRoot "scripts\test.ps1") -Full
& $Uv run --frozen --no-sync python -m optimizer_resurrection.train --track mlp --dataset shapeset --activation logistic --depth 4 --width 64 --optimizer adam_o --steps 10 --batch-size 32 --train-examples 256 --validation-examples 128 --diagnostic-interval 5 --device cuda --stage engineering-smoke --run-group local-gpu-smoke --run-name shapeset-logistic-depth4-adam_o
& $Uv run --frozen --no-sync python -m optimizer_resurrection.train --track rnn --task latch --hidden-size 32 --max-length 50 --optimizer mm --steps 10 --batch-size 16 --validation-examples 128 --diagnostic-interval 5 --device cuda --stage engineering-smoke --run-group local-gpu-smoke --run-name latch-length50-hidden32-mm
