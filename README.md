# VR-DT-DRL

Vision-based **UR3e grasping** with a **Webots digital twin** and a **GPU inference server** on Windows.

MIT License · [Aqlanlab](LICENSE)

---

## What runs where (Windows setup)

Everything runs on **one Windows PC**:

| Component | What it does |
|-----------|----------------|
| **Webots** | Simulation world (`updated_world/worlds/Environmentnewww.wbt`) |
| **`gpu_server.py`** | CNN inference on your NVIDIA GPU |
| **`simulation_client.py`** | Robot control, cameras, episodes (one process per arm) |

Clients talk to the GPU server over **`127.0.0.1:8888`**.

```mermaid
flowchart LR
  WB[Webots]
  GS[gpu_server.py]
  R1[simulation_client R1]
  R2[simulation_client R2]
  R1 --> WB
  R2 --> WB
  R1 -->|localhost :8888| GS
  R2 -->|localhost :8888| GS
```

---

## One-time setup

### 1. Install software

| Tool | Version / notes |
|------|-----------------|
| **Python** | **3.9** (required for Webots 2021a controller API) |
| **Webots** | **2021a** ([download](https://github.com/cyberbotics/webots/releases)) |
| **NVIDIA driver + CUDA** | For GPU inference (PyTorch cu118/cu121) |

Default Webots install path on Windows:

`%LOCALAPPDATA%\Programs\Webots`

### 2. Webots world assets

The Webots project lives in **`updated_world/`** at the repo root (create that folder and put the contents of **`Webots.rar`** from Box inside — see [Files on Box](#files-on-box-not-on-github)). Open in Webots:

- `updated_world/worlds/Environmentnewww.wbt`
- `updated_world/protos/` (meshes, textures, UR3e protos)

If you copy a fresh tree from Box and paths still reference Linux (`/home/seth/...`), fix once:

```powershell
cd VR-DT-DRL
python vm_simulation_system\Webots\scripts\fix_vm_paths_for_windows.py
```

The script defaults to `updated_world/`. Domain-randomization texture JPGs are also read from `vm_simulation_system/Webots/protos/textures/Dataset/` when the client runs.

### 3. Python environment (GPU + sim client)

```powershell
cd host_gpu_system
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Confirm CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`openpyxl` is included in `requirements.txt` (episode Excel logs).

### 4. Network config (localhost)

In `vm_simulation_system/config/network_config.yaml`:

```yaml
network:
  host_ip: "127.0.0.1"
  host_port: 8888
```

Leave `host_gpu_system/config/network_config.yaml` as `host_ip: "0.0.0.0"`.

### 5. Models

Place trained weights in `host_gpu_system/models/`:

- `ur3_live_model_r1.pth`
- `ur3_live_model_r2.pth` (dual-arm world)

### 6. GPU server firewall (once)

```powershell
netsh advfirewall firewall add rule name="UR3e GPU Server" dir=in action=allow protocol=TCP localport=8888
```

### 7. Dual-arm barrier (if using one or two robots)

In `host_gpu_system/src/gpu_server.py` (~line 114), set `_barrier_num_robots`:

| Arms running | Value |
|--------------|-------|
| Robot 1 only | `1` |
| Robot 1 + 2 | `2` |

With **two** clients, the GPU server uses setup barriers: Robot 1 sets up the world first, Robot 2 second, then **both** wait until setup is complete before either resumes grasping.

---

## Curriculum phases

Grasp **training** progresses through six spawn phases (0–5). Each phase places the block at increasing distance from the platform centre. The client advances automatically when enough AI grasp attempts hit a success-rate target (training mode only).

| Phase | Spawn region |
|-------|----------------|
| **0** | Centre only (fixed) |
| **1** | 0.5–1.5 cm radius |
| **2** | 1.5–3.5 cm radius |
| **3** | 3.5–7.0 cm radius |
| **4** | 7.0–11.5 cm radius band |
| **5** | Anywhere on usable platform (uniform) |

**Inference** (`--mode inference`) does not advance the curriculum. Pick how spawns behave:

| Flag | Behavior |
|------|----------|
| `--phase N` | Lock spawns to phase **0–5** (e.g. **`--phase 4`** for the outer ring) |
| *(no extra flag)* | Use saved curriculum state from `config/curriculum_state.json` |
| `--cycle N` | Rotate through phases (default **0→5**); add **`--cycle-from`** / **`--cycle-to`** to limit the range |
| `--cycle-from N` / `--cycle-to M` | With **`--cycle`**, only rotate phases **N…M** (e.g. **1–4**) |
| `--free` | No auto-spawn; place the block manually in Webots |

Episode logs for `--phase N` go to `data/episode_log_r1_phaseN.xlsx`.

---

## Run simulation (every session)

**Order matters:** GPU server → Webots **Play** → client(s).

**Stopping a running command:** In PowerShell, press **Ctrl+Pause** to interrupt `gpu_server.py`, `simulation_client.py`, or other long-running processes. On many laptops the **Pause** key is only available via **Fn** — use **Ctrl+Fn+Pause** instead.

### Terminal 1 — GPU server

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
python src\gpu_server.py
```

Wait for: `BC Server listening on 0.0.0.0:8888`

### Webots

1. Open `updated_world\worlds\Environmentnewww.wbt`
2. **Reset Simulation**
3. Press **Play**

### Terminal 2 — Robot 1

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
cd ..\vm_simulation_system

$env:WEBOTS_HOME = "$env:LOCALAPPDATA\Programs\Webots"
$env:WEBOTS_ROBOT_NAME = "ur3e_robot"

python src\simulation_client.py --mode inference --cycle 20 --cycle-from 1 --cycle-to 4 --robot-id 1
```

This rotates **phases 1→2→3→4→1…**, **20 episodes per phase**. For a single phase, use **`--phase N`** (e.g. **`--phase 4`**). **`--phase 5`** is full-board spawns (hardest; use only when you intend to stress-test edges). See [Curriculum phases](#curriculum-phases).

### Terminal 3 — Robot 2 (optional, dual-arm world)

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
cd ..\vm_simulation_system

$env:WEBOTS_HOME = "$env:LOCALAPPDATA\Programs\Webots"
$env:WEBOTS_ROBOT_NAME = "ur3e_robot2"

python src\simulation_client.py --mode inference --cycle 20 --cycle-from 1 --cycle-to 4 --robot-id 2
```

### Good startup signs

- `[WebotsBridge] Connected to robot 'ur3e_robot'`
- `Webots motors bound: 6/6`
- `Connected to GPU server at 127.0.0.1:8888`
- `[AI PREDICTION R1]` when an episode runs

---

## Episode logs

Written under **`VR-DT-DRL/data/`** (repo root), e.g.:

- `data/episode_log_r1.xlsx` (cycle / normal inference)
- `data/episode_log_r1_phase4.xlsx` (when using `--phase 4`)

Column **timestamp_local** uses your Windows timezone in 12-hour format.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Webots not connected` | Webots must be **playing** before starting the client; use **Reset Simulation** then Play |
| `DLL load failed` / wrong python API | Use **Python 3.9** venv; Webots API folder must be `python39`, not `python38` |
| Client exits immediately | Start `gpu_server.py` first; check `host_ip: "127.0.0.1"` in client config |
| Hang between episodes | Match `_barrier_num_robots` to number of running clients (1 or 2) |
| Missing textures / meshes | Ensure `updated_world/protos/` is complete; run path fix script on `updated_world/` |
| `openpyxl` error | `pip install openpyxl` in `host_gpu_system\venv` |

**Connection test** (Webots playing, Robot 1 env set):

```powershell
python vm_simulation_system\Webots\scripts\probe_webots_connection.py
```

Expected: `OK robot=ur3e_robot timestep=16`

---

## Files on Box (not on GitHub)

Clone the repo first, then download these from **Box** and place them as shown.

**`Webots.rar`** on Box contains the **contents** of the Webots project (`worlds/`, `protos/`, etc.) — not a folder named `updated_world/`. After clone:

1. Create `VR-DT-DRL/updated_world/`
2. Extract `Webots.rar` and move everything into that folder (you should see `updated_world/worlds/Environmentnewww.wbt`, `updated_world/protos/`, …)

| Item | Put here |
|------|----------|
| Webots project (from `Webots.rar`) | `VR-DT-DRL/updated_world/` |
| `ur3_live_model_r1.pth` | `host_gpu_system/models/` |
| `ur3_live_model_r2.pth` | `host_gpu_system/models/` (dual-arm only) |

Also copy `updated_world/protos/textures/Dataset/` → `vm_simulation_system/Webots/protos/textures/Dataset/` (domain randomization; sim runs without it, but with colour-only textures).

---

## License

MIT License — Copyright (c) 2026 Aqlanlab. See [LICENSE](LICENSE).
