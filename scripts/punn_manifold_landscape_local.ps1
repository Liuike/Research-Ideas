param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$ConfigPath = 'configs/product_unit/manifold_landscape_gpu.yaml',

    [Parameter()]
    [ValidateRange(1, 8)]
    [int]$Workers = 4,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$EnvironmentRoot = (Split-Path -Parent $PSScriptRoot),

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$taskRepository = Split-Path -Parent $PSScriptRoot
$taskEnvironmentRoot = (Resolve-Path -LiteralPath $EnvironmentRoot).Path
Set-Location -LiteralPath $taskRepository

$taskConfig = if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    (Resolve-Path -LiteralPath $ConfigPath).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $taskRepository $ConfigPath)).Path
}
$taskUv = Join-Path $taskEnvironmentRoot '.venv/Scripts/uv.exe'
if (-not (Test-Path -LiteralPath $taskUv -PathType Leaf)) {
    throw "Pinned UV executable not found: $taskUv"
}

# Reuse the root's frozen Python environment while importing code from this
# clean worktree. W&B-owned local state stays in the root's ignored `wandb/`.
$taskVenv = Join-Path $taskEnvironmentRoot '.venv'
$taskWandb = Join-Path $taskEnvironmentRoot 'wandb'
$env:VIRTUAL_ENV = $taskVenv
$env:UV_PROJECT_ENVIRONMENT = $taskVenv
$env:UV_CACHE_DIR = Join-Path $taskEnvironmentRoot '.cache/uv'
$env:PYTHONPATH = Join-Path $taskRepository 'src'
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

# Only load the three allowlisted W&B variables, and never replace a value
# already present in the process environment.
$taskCredentialPath = Join-Path $taskEnvironmentRoot '.secrets/env'
$taskAllowedCredentialNames = @('WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT')
if (Test-Path -LiteralPath $taskCredentialPath -PathType Leaf) {
    foreach ($taskCredentialLine in Get-Content -LiteralPath $taskCredentialPath) {
        $taskLine = $taskCredentialLine.Trim()
        if (-not $taskLine -or $taskLine.StartsWith('#') -or -not $taskLine.Contains('=')) {
            continue
        }
        $taskParts = $taskLine.Split('=', 2)
        $taskKey = $taskParts[0].Trim()
        if ($taskKey -notin $taskAllowedCredentialNames -or
            [Environment]::GetEnvironmentVariable($taskKey, 'Process')) {
            continue
        }
        $taskValue = $taskParts[1].Trim()
        if ($taskValue.Length -ge 2 -and
            (($taskValue.StartsWith('"') -and $taskValue.EndsWith('"')) -or
             ($taskValue.StartsWith("'") -and $taskValue.EndsWith("'")))) {
            $taskValue = $taskValue.Substring(1, $taskValue.Length - 2)
        }
        if ($taskValue) {
            [Environment]::SetEnvironmentVariable($taskKey, $taskValue, 'Process')
        }
    }
}

if ($DryRun) {
    # Expand the exact frozen plan for inspection without starting any run.
    $taskPlanArgs = @(
        'run', '--frozen', '--no-sync', 'python',
        '-m', 'optimizer_resurrection.experiment', 'plan', $taskConfig
    )
    if ($Resume) {
        $taskPlanArgs += '--resume'
    }
    Write-Output ('DRY RUN: ' + $taskUv + ' ' + ($taskPlanArgs -join ' '))
    & $taskUv @taskPlanArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen experiment plan expansion failed with exit code $LASTEXITCODE."
    }
    return
}

# Scientific runs must use a committed snapshot, with no dirty or untracked
# source that would make the recorded provenance ambiguous.
$taskGitStatus = @(& git -C $taskRepository status --porcelain --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw 'Could not inspect the source checkout before starting the sweep.'
}
if ($taskGitStatus.Count -gt 0) {
    throw 'The scientific source checkout is not clean. Commit the intended source and config before launching the sweep.'
}

$taskMissingCredentials = @(
    $taskAllowedCredentialNames | Where-Object {
        -not [Environment]::GetEnvironmentVariable($_, 'Process')
    }
)
if ($taskMissingCredentials.Count -gt 0) {
    throw ('Missing W&B environment variables: ' + ($taskMissingCredentials -join ', '))
}

# Honor the registered device; never substitute a fallback.
$taskDevice = & $taskUv run --frozen --no-sync python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["device"])' $taskConfig
if ($LASTEXITCODE -ne 0 -or $taskDevice -notin @('cpu', 'cuda')) {
    throw 'Could not validate the registered training device.'
}
if ($taskDevice -eq 'cuda') {
    $taskCudaDevice = & $taskUv run --frozen --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE -ne 0) {
        throw 'CUDA preflight failed; no experiment run was started.'
    }
    Write-Output ('CUDA device: ' + ($taskCudaDevice -join ' '))
} else {
    Write-Output 'Registered device: CPU, FP32 training and landscapes'
}

# The online controller runs the frozen experiment.plan command and publishes
# its stream in W&B. Resume delegates strict accepted-condition reconciliation
# to the registered planner; it is never implied by restarting this launcher.
$taskControllerArgs = @(
    'run', '--frozen', '--no-sync', 'python',
    '-m', 'optimizer_resurrection.punn_manifold_sweep',
    '--config', $taskConfig,
    '--workers', [string]$Workers,
    '--uv', $taskUv
)
if ($Resume) {
    $taskControllerArgs += '--resume'
}
Write-Output ('COMMAND: ' + $taskUv + ' ' + ($taskControllerArgs -join ' '))
& $taskUv @taskControllerArgs
if ($LASTEXITCODE -ne 0) {
    throw "DA-10 landscape controller exited with code $LASTEXITCODE. Inspect its online W&B record before continuing."
}
