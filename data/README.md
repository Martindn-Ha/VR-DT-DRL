# Episode log analysis

Put `.xlsx` logs in `episode logs/`. PDFs go to `episode report/`.

**Residual RL training history** (archived `TD3 (pre-dist)/`, `RL logs (pre align v2)/`, etc.): see [`docs/rl_residual_training_chronicle.md`](../docs/rl_residual_training_chronicle.md).

Each row is one grasp episode, logged by `vm_simulation_system/src/simulation_client.py`. Spawn and grasp positions use Webots world coordinates (meters). For a longer analysis-oriented dictionary, see [`docs/simulation_episode_data_dictionary.md`](../docs/simulation_episode_data_dictionary.md).

## Log parameters

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `timestamp` | string | — | UTC wall time when the episode row was written (ISO 8601). |
| `timestamp_local` | datetime | — | Excel-only: local time derived from `timestamp` (column B). |
| `robot_id` | int | — | UR3 instance that ran the episode (`1` or `2`). |
| `episode` | int | — | Monotonic episode counter for that robot client (persists across restarts). |
| `session_episode` | int | — | Episode index for this client run only; resets to 1 on each launch. |
| `run_mode` | string | — | Client mode, e.g. `inference` or training-related modes. |
| `inference_mode` | string | — | When `run_mode=inference`: `normal`, `phase`, `cycle`, or `free`. Empty otherwise. |
| `curriculum_phase` | int | — | Active curriculum phase index (controls spawn difficulty / radius). |
| `spawn_phase` | string | — | How spawn was chosen: curriculum phase number, `free`, or cycle label. |
| `spawn_x` | float | m | World X where the target object was placed. |
| `spawn_y` | float | m | World Y height of the spawned object (fixed at 0.461 m). |
| `spawn_z` | float | m | World Z where the target object was placed. |
| `spawn_radius_cm` | float | cm | Horizontal distance from platform center to spawn point. |
| `cam_delta_x_cm` | float | cm | Camera X perturbation vs. cached base pose (domain randomization). |
| `cam_delta_y_cm` | float | cm | Camera Y perturbation (max ±1.5 cm). |
| `cam_delta_z_cm` | float | cm | Camera Z perturbation (max ±0.8 cm). |
| `cam_delta_pitch_deg` | float | deg | Camera pitch perturbation (max ±1.0°). |
| `cam_delta_yaw_deg` | float | deg | Camera yaw perturbation (max ±0.5°). |
| `cam_delta_roll_deg` | float | deg | Camera roll perturbation (max ±0.35°). |
| `grasp_mode` | string | — | Policy branch: `exploit` (network), `explore` (teacher), `nan_abort`, or `unknown`. |
| `ai_pose_0` | float | m | Raw network grasp target X (before workspace clamp). |
| `ai_pose_1` | float | m | Raw network grasp target Y. |
| `ai_pose_2` | float | m | Raw network grasp target Z. |
| `ai_pose_3` | float | rad | Raw network output: rx (roll-like component). |
| `ai_pose_4` | float | rad | Raw network output: ry (pitch component). |
| `ai_pose_5` | float | rad | Raw network output: yaw (approach heading). |
| `clamp_pose_0` | float | m | Clamped grasp X sent to the motion controller. |
| `clamp_pose_1` | float | m | Clamped grasp Y sent to the motion controller. |
| `clamp_pose_2` | float | m | Clamped grasp Z sent to the motion controller. |
| `success` | int | — | `1` if object lift exceeded 0.023 m; else `0`. |
| `lifted_m` | float | m | Vertical object displacement after the grasp (`final_y − initial_y`). |
| `closest_dist_m` | float | m | Minimum 3D gripper–object distance at descent end. Sentinel `9999` on NaN abort. |
| `reward` | float | — | Shaped reward: `1.0` on success; distance-based otherwise. |
| `object_found` | int | — | `1` if the target object was readable in Webots at grasp start; else `0`. |
| `outcome_class` | string | — | Failure taxonomy label, e.g. `success`, `near_miss`, `far_miss`, `weak_lift`. See [`docs/failure_taxonomy.md`](../docs/failure_taxonomy.md). |
| `clamp_limited` | int | — | `1` if executed X/Z was clipped vs. raw `ai_pose_0` / `ai_pose_2`; else `0`. |
| `support_shade_1` | float | — | Domain-randomized grey shade of bed support 1 (`BedSupports_1` base color R channel). |
| `support_shade_2` | float | — | Domain-randomized grey shade of bed support 2 (`BedSupports_2` base color R channel). |

From the repo root:

```powershell
python analysis/spawn_spatial_report.py "data/episode logs/your_log.xlsx" -o "data/episode report"
```

Quote the path — the folder name has a space (`episode logs`).

Console taxonomy only (no PDF):

```powershell
python analysis/spawn_spatial_report.py "data/episode logs/your_log.xlsx" --taxonomy-only
```

Optional: If you have multiple phases in one spreadsheet, add `--spawn-phase #` flag to filter to one curriculum phase. (i.e '--spawn-phase 4' to view only phase 4 in a spreadsheet with multiple phases)

## Locator training (`run_mode=locator_train`)

Uses a **separate** episode schema (coordinate training, not grasp). Files:

| File | Description |
|------|-------------|
| `data/episode_log_r1_locator_train.xlsx` | Per-demo: spawn, labels, CNN prediction error |
| `data/locator_train_steps_r1.csv` | Per GPU step: `aux_loss`, buffers (written by `--locator-train` server) |

```powershell
python analysis/locator_train_report.py data/episode_log_r1_locator_train.xlsx
```

See [`docs/locator_geo_grasp.md`](../docs/locator_geo_grasp.md) for column definitions. Do **not** run grasp failure taxonomy on locator train logs.
