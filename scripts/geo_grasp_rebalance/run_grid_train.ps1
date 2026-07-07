# Grid train launcher (shaping + DQN)
# Terminal 1: GPU server
# Terminal 2: sim client --mode grid_train

$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location (Join-Path $repo "host_gpu_system")

Write-Host "Start GPU: python src/gpu_server.py --grid-train --model models/R1_locator.pth --model-r2 models/R2_locator.pth"
Write-Host "Start sim: cd vm_simulation_system; python src/simulation_client.py --mode grid_train --robot-id 1"
