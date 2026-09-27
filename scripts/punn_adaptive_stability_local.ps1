param([ValidateRange(1, 20)][int]$Workers = 12)

$ErrorActionPreference = 'Stop'
$taskRepository = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRepository
$taskUv = Join-Path $taskRepository '.venv/Scripts/uv.exe'
$taskWandb = Join-Path $taskRepository 'wandb'
$env:UV_CACHE_DIR = Join-Path $taskRepository '.cache/uv'
$env:WANDB_DIR = $taskWandb
$env:WANDB_DATA_DIR = $taskWandb
$env:WANDB_CONFIG_DIR = $taskWandb
$env:WANDB_CACHE_DIR = $taskWandb
$env:WANDB_SILENT = 'true'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $taskRepository.Replace('\', '/')

# Keep the registered commands and subsequent analyses in one durable process.
# Credential loading and clean scientific provenance checks happen in Python.
$taskCommands = @(
    @('run', '--frozen', '--no-sync', 'python', '-m', 'optimizer_resurrection.experiment',
      'plan', 'configs/product_unit/adaptive_architecture_stability.yaml',
      '--run', '--workers', "$Workers"),
    @('run', '--frozen', '--no-sync', 'python', '-m', 'optimizer_resurrection.punn_architecture_analysis',
      '--config', 'configs/product_unit/adaptive_architecture_stability.yaml'),
    @('run', '--frozen', '--no-sync', 'python', '-m', 'optimizer_resurrection.punn_architecture_comparison',
      '--adaptive-config', 'configs/product_unit/adaptive_architecture_stability.yaml',
      '--baseline-config', 'configs/product_unit/architecture_stability.yaml')
)
foreach ($taskCommand in $taskCommands) {
    Write-Output ('COMMAND: ' + $taskUv + ' ' + ($taskCommand -join ' '))
    & $taskUv @taskCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment pipeline stopped with exit code $LASTEXITCODE. Inspect online W&B before retrying any cell."
    }
}
