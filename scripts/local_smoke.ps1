$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Create .venv and install the project first; see README.md"
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& (Join-Path $ProjectRoot "scripts\test.ps1") -Full
& $Python -m optimizer_resurrection.train --track mlp --dataset shapeset --activation logistic --depth 4 --width 64 --optimizer adam_o --steps 10 --batch-size 32 --train-examples 256 --validation-examples 128 --diagnostic-interval 5 --device cuda --stage engineering-smoke --run-group local-gpu-smoke --run-name shapeset-logistic-depth4-adam_o
& $Python -m optimizer_resurrection.train --track rnn --task latch --hidden-size 32 --max-length 50 --optimizer mm --steps 10 --batch-size 16 --validation-examples 128 --diagnostic-interval 5 --device cuda --stage engineering-smoke --run-group local-gpu-smoke --run-name latch-length50-hidden32-mm
