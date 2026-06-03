# Start Robot 1 simulation client (Part 5 Windows)
# Usage: .\Webots\scripts\start_client_r1.ps1
# Prerequisite: Webots open with updated_world\worlds\Environmentnewww.wbt -> Reset -> Play

$WebotsHome = "$env:LOCALAPPDATA\Programs\Webots"
$RepoRoot   = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$VenvPy     = Join-Path $RepoRoot "..\host_gpu_system\venv\Scripts\python.exe"
$ClientDir  = $RepoRoot

if (-not (Test-Path $VenvPy)) {
    $VenvPy = Join-Path (Split-Path $RepoRoot -Parent) "host_gpu_system\venv\Scripts\python.exe"
}

$env:WEBOTS_HOME = $WebotsHome
$env:WEBOTS_ROBOT_NAME = "ur3e_robot"
$env:PYTHONIOENCODING = "UTF-8"

$paths = @(
    "$WebotsHome\lib\controller",
    "$WebotsHome\msys64\mingw64\bin",
    "$WebotsHome\msys64\mingw64\bin\cpp"
)
foreach ($p in $paths) {
    if (Test-Path $p) { $env:PATH = "$p;$env:PATH" }
}

Set-Location $ClientDir
Write-Host "WEBOTS_HOME=$env:WEBOTS_HOME"
Write-Host "WEBOTS_ROBOT_NAME=$env:WEBOTS_ROBOT_NAME"
Write-Host "Run probe first if connection fails:"
Write-Host "  & `"$VenvPy`" Webots\scripts\probe_webots_connection.py"

& $VenvPy src\simulation_client.py --mode inference --phase 5 --robot-id 1 @args
