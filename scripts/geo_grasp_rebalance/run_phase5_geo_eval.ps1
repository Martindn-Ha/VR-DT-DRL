# Phase 5 geo-grasp eval -> data/Locator/Post-rebalance/Eval/
# Prerequisite: Webots + gpu_server --geo-grasp

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2)]
    [int]$RobotId,
    [int]$Episodes = 941
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$HostRoot = Join-Path $RepoRoot "host_gpu_system"
$VmRoot   = Join-Path $RepoRoot "vm_simulation_system"
$OutDir   = Join-Path $RepoRoot "data\Locator\Post-rebalance\Eval"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$VenvPy = Join-Path $HostRoot "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { $VenvPy = "python" }

$WebotsHome = "$env:LOCALAPPDATA\Programs\Webots"
$env:WEBOTS_HOME = $WebotsHome
$env:WEBOTS_ROBOT_NAME = if ($RobotId -eq 2) { "ur3e_robot2" } else { "ur3e_robot" }
$env:PYTHONIOENCODING = "UTF-8"

foreach ($p in @(
    "$WebotsHome\lib\controller",
    "$WebotsHome\msys64\mingw64\bin"
)) {
    if (Test-Path $p) { $env:PATH = "$p;$env:PATH" }
}

Write-Host "=== Geo grasp phase 5 eval R$RobotId | episodes=$Episodes ==="
Write-Host "gpu_server: python src\gpu_server.py --geo-grasp --model models\R1_locator.pth --model-r2 models\R2_locator.pth"
Write-Host "Output log: data\episode_log_r${RobotId}_phase5.xlsx (copy to Post-rebalance/Eval after run)"

Set-Location $VmRoot
& $VenvPy src\simulation_client.py --mode inference --use-geo-grasp --phase 5 --robot-id $RobotId --episodes $Episodes

$src = Join-Path $RepoRoot "data\episode_log_r${RobotId}_phase5.xlsx"
$dst = Join-Path $OutDir "episode_log_r${RobotId}_phase5.xlsx"
if (Test-Path $src) {
    Copy-Item $src $dst -Force
    Write-Host "Copied -> $dst"
    $env:PYTHONPATH = Join-Path $VmRoot "src"
    & $VenvPy (Join-Path $RepoRoot "data\analysis\spawn_spatial_report.py") $dst --spawn-phase 5 -o $OutDir
}
