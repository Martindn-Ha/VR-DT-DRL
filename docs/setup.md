# One-time setup (Windows)

Do this once before training or running in simulation. For the real arm, also finish the network steps in [physical_arms.md](physical_arms.md).

---

## 1. Install software

| Tool | Version / notes |
|------|-----------------|
| **Python** | **3.9** (needed for Webots 2021a) |
| **Webots** | **2021a** ([download](https://github.com/cyberbotics/webots/releases)) |
| **NVIDIA driver + CUDA** | For the GPU AI program (PyTorch cu118/cu121) |

Default Webots install path:

`%LOCALAPPDATA%\Programs\Webots`

---

## 2. Webots world files

These are **not on GitHub**. Download **`Webots.rar`** from **Box**, then:

1. Create folder `updated_world/` at the repo root (if it is not there).
2. Extract **`Webots.rar`** into that folder (you should see `worlds/`, `protos/`, etc.).
3. Open `updated_world/worlds/Environmentnewww.wbt` in Webots.

If the world still points at Linux paths like `/home/seth/...`, run once:

```powershell
cd VR-DT-DRL
python vm_simulation_system\Webots\scripts\fix_vm_paths_for_windows.py
```

---

## 3. Python environment (GPU + tools)

```powershell
cd host_gpu_system
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

You want `True` from that last line if you plan to use the GPU.

---

## 4. Network for simulation (same PC)

Edit these files so the robot program can talk to the AI program on this machine:

| File | Set |
|------|-----|
| `vm_simulation_system/config/network_config.yaml` | `host_ip: "127.0.0.1"`, port `8888` |
| `host_gpu_system/config/network_config.yaml` | `host_ip: "0.0.0.0"`, port `8888` |

For the **real arm**, the VM must point at the Windows PC IP instead — see [physical_arms.md](physical_arms.md).

---

## 5. Model files

These are **not on GitHub**. Download them from **Box** and place them under `host_gpu_system/models/`:

| File | What it is |
|------|------------|
| `yolo26n.pt` | Finds the block in the camera image |
| `R1_local_bbox_dqn.pth` | Grasp policy for robot 1 |
| `R2_local_bbox_dqn.pth` | Grasp policy for robot 2 (dual-arm only) |

Exact subfolders are set in your local config; ask a teammate if you are unsure where to put them.

---

## Next steps

- Simulation: [simulation.md](simulation.md)
- Real arm: [physical_arms.md](physical_arms.md)
