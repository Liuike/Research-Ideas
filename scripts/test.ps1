[CmdletBinding()]
param(
    [switch]$Full,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Create .venv and install the project first; see README.md"
}

$ActualUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name.Split('\')[-1]
$SafeUser = $ActualUser -replace '[^A-Za-z0-9_.-]', '_'
$BaseTemp = Join-Path ([System.IO.Path]::GetTempPath()) "optimizer-resurrection-pytest-$SafeUser"
$Selection = if ($Full) { "integration or not integration" } else { "not integration" }

Push-Location $ProjectRoot
try {
    & $Python -m pytest --basetemp $BaseTemp -m $Selection @PytestArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
