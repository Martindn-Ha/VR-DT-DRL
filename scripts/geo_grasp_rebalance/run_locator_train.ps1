# Locator retrain after weak_regions rebalance (resume checkpoint).
# Prerequisite: Webots open -> Reset -> Play (Environmentnewww.wbt)

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2)]
    [int]$RobotId,
    [int]$Episodes = 850
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$HostRoot = Join-Path $RepoRoot "host_gpu_system"
$VmRoot   = Join-Path $RepoRoot "vm_simulation_system"
$VenvPy   = Join-Path $HostRoot "venv\Scripts\python.exe"
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

$June17R1 = "models\Locator (No heatmap)\June17th\R1_locator.pth"
$June17R2 = "models\Locator (No heatmap)\June17th\R2_locator.pth"

Write-Host "=== Locator train R$RobotId | episodes=$Episodes ==="
Write-Host "Load:  June17th checkpoints (read-only)"
Write-Host "Save:  host_gpu_system\models\R1_locator.pth / R2_locator.pth"
Write-Host "Start gpu_server in another terminal:"
Write-Host "  cd host_gpu_system; .\venv\Scripts\Activate.ps1; python src\gpu_server.py --locator-train --model `"$June17R1`" --model-r2 `"$June17R2`""

Set-Location $VmRoot
& $VenvPy src\simulation_client.py --mode locator_train --robot-id $RobotId --episodes $Episodes

Write-Host ""
Write-Host "Check quadrant balance:"
Write-Host "  python analysis\locator_quadrant_balance.py data\episode_log_r${RobotId}_locator_train.xlsx"
