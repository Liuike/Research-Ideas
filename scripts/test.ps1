[CmdletBinding()]
param(
    [switch]$Full,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

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
$BaseTemp = Join-Path ([System.IO.Path]::GetTempPath()) "optimizer-resurrection-pytest-$SafeUser"
$env:UV_CACHE_DIR = Join-Path ([System.IO.Path]::GetTempPath()) "optimizer-resurrection-uv-$SafeUser"
$Selection = if ($Full) { "integration or not integration" } else { "not integration" }

Push-Location $ProjectRoot
try {
    & $Uv run --frozen --no-sync python -m pytest --basetemp $BaseTemp -m $Selection @PytestArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
