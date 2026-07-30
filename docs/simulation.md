# How to run in simulation (Webots)

This guide is for **one Windows PC**. You will run a 3D robot simulator (Webots) and two small programs: one that does the vision/AI work, and one that moves the simulated robot.

**What happens each try:** the camera sees the board → software finds the block → a learned policy picks where to grasp → the arm moves.

Finish [one-time setup](setup.md) first (Webots, Python venv, models).

---

## Before every session

1. Open Webots and load `updated_world/worlds/Environmentnewww.wbt`.
2. Click **Reset Simulation**, then **Play** (the sim must be running).
3. You will use **two PowerShell windows**. Leave both open.

Always start the **AI program** first, then the **robot program**.

---

## Option A — Teach the robot (train)

Use this when you want the policy to learn from practice in simulation.

### Terminal 1 — AI program

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
python src\gpu_server.py --local-bbox-dqn-train
```

Wait until it is listening / ready.

### Terminal 2 — Robot program

Webots must already be **playing**.

```powershell
cd vm_simulation_system
$env:WEBOTS_HOME = "$env:LOCALAPPDATA\Programs\Webots"
$env:WEBOTS_ROBOT_NAME = "ur3e_robot"
python src\simulation_client.py --mode local_bbox_dqn_train --robot-id 1
```

Episodes will run automatically. Stop with Ctrl+Pause (often Ctrl+Fn+Pause on laptops).

---

## Option B — Use a trained policy (inference)

Use this when you already have model files and just want the robot to pick blocks.

### Terminal 1 — AI program

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
python src\gpu_server.py --local-bbox-dqn
```

### Terminal 2 — Robot program

```powershell
cd vm_simulation_system
$env:WEBOTS_HOME = "$env:LOCALAPPDATA\Programs\Webots"
$env:WEBOTS_ROBOT_NAME = "ur3e_robot"
python src\simulation_client.py --mode inference --use-local-bbox-dqn --phase 5 --robot-id 1
```

| Flag | Meaning |
|------|---------|
| `--phase 5` | Blocks can spawn anywhere on the board |
| `--free` | You place the block yourself in Webots (no auto-spawn) |

Example with manual placement: replace `--phase 5` with `--free`.

---

## Option C — Pick a block by typing (language select)

Use this when there are several blocks and you want to say which one (for example “yellow bottom right”).

You need [Ollama](https://ollama.com) on this PC with model `qwen3-vl:8b-instruct`. Run only **one** `gpu_server`. Warm the model first (`ollama run qwen3-vl:8b-instruct "say ok"`).

### Terminal 1 — AI program

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
python src\gpu_server.py --local-bbox-dqn --use-vlm-select --vlm-model qwen3-vl:8b-instruct
```

### Terminal 2 — Robot program

```powershell
cd vm_simulation_system
$env:WEBOTS_HOME = "$env:LOCALAPPDATA\Programs\Webots"
$env:WEBOTS_ROBOT_NAME = "ur3e_robot"
python src\simulation_client.py --mode inference --use-local-bbox-dqn --use-vlm-select --free
```

1. Place blocks in Webots.
2. When prompted, open `host_gpu_system/debug/board_warp_r1_latest.jpg` to see the board view.
3. Type your instruction in Terminal 2 (for example `yellow top left`).

---

## After a run — make a report (optional)

From the repo root, after you have an episode log Excel file:

```powershell
python analysis\spawn_spatial_report.py data\episode_log_r1_phase5.xlsx --spawn-phase 5 -o "data\episode report"
```

Use the log file name you actually produced under `data/`.

---

## If something fails

| What you see | What to try |
|--------------|-------------|
| Robot not connected | Webots must be **Play** before Terminal 2 |
| Client exits right away | Start Terminal 1 (AI) first |
| No block found | Check models are under `host_gpu_system/models/` |
| `vlm_unavailable` / Ollama `HTTP 404` | Stale extra `gpu_server` or wrong model name. Kill all old servers, start one with `--vlm-model qwen3-vl:8b-instruct`, restart the client. `ollama list` must show that model. |
| `vlm_unavailable` / timeout | Cold Ollama load — run `ollama run qwen3-vl:8b-instruct "say ok"` first. |
