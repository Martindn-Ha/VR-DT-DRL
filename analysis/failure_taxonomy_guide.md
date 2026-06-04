# Failure taxonomy

Each grasp episode gets two related fields in your episode log:

- **`outcome_class`** — what happened (one label per row: success or a failure type)
- **`clamp_limited`** — extra flag: did the system trim the AI’s move because it was outside the safe zone? (`0` = no, `1` = yes)

Rules live in `vm_simulation_system/src/failure_taxonomy.py`. **`lifted_m`** = block world-Y after grasp minus before (m). **`closest_dist_m`** = minimum gripper–block distance during the grasp (m). The first matching rule below wins.

---

## Outcome labels (`outcome_class`)

| Label | Rule (quantitative) |
|-------|---------------------|
| **success** | `success = 1` (`lifted_m > 23 mm`) |
| **sim_nan_abort** | `grasp_mode = nan_abort`, or `closest_dist_m ≥ 9999` |
| **object_not_found** | `object_found = 0` |
| **far_miss** | Failed; `closest_dist_m > 200 mm` |
| **weak_lift** | Failed; `0 < lifted_m ≤ 23 mm` |
| **drop_or_push** | Failed; `lifted_m < 0 mm` (any downward ΔY) |
| **near_miss** | Failed; `closest_dist_m ≤ 100 mm` and `|lifted_m| < 20 mm` |
| **mid_miss** | Failed; `100 mm < closest_dist_m ≤ 200 mm` |
| **other_failure** | Failed; rare edge case (e.g. close + large \|lift\|, bad/missing fields) |

Distance bands partition failures with valid `closest_dist_m`: **≤100 mm** near, **100–200 mm** mid, **>200 mm** far.

---

## Extra flag (`clamp_limited`)

| Value | Rule (quantitative) |
|-------|---------------------|
| **0** | `|ai_pose − clamp_pose|` on X/Z ≤ 1e−4 m (exploit only) |
| **1** | `|ai_pose − clamp_pose|` on X/Z > 1e−4 m (X/Z clipped before move) |

**Why it’s separate:** Any `outcome_class` can have `clamp_limited` 0 or 1.
