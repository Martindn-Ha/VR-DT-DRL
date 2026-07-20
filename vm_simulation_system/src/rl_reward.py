"""Per-robot RL reward and config loading for residual TD3 training."""

import math

from pathlib import Path

from typing import Any, Dict, Optional, Tuple, Union



import yaml



NAN_DIST = 9999.0





def default_rl_train_config_path() -> Path:

    return (

        Path(__file__).resolve().parent.parent.parent

        / "host_gpu_system" / "config" / "rl_train_config.yaml"

    )





def load_rl_train_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:

    cfg_path = Path(path) if path else default_rl_train_config_path()

    with open(cfg_path, 'r', encoding='utf-8') as f:

        return yaml.safe_load(f)





def get_robot_rl_config(cfg: Dict[str, Any], robot_id: int) -> Dict[str, Any]:

    robots = cfg.get('robots', {})

    if robot_id in robots:

        return robots[robot_id]

    key = str(robot_id)

    if key in robots:

        return robots[key]

    raise KeyError(f"No RL config for robot {robot_id} in rl_train_config.yaml")





def calculate_rl_reward(

    robot_id: int,

    success: bool,

    closest_dist: float,

    lifted_m: Optional[float],

    delta_x: float,

    delta_z: float,

    delta_yaw: float,

    clamp_limited: bool,

    object_found: bool,

    reward_cfg: Dict[str, Any],

    max_delta: Optional[Tuple[float, float, float]] = None,

    lateral_improve_m: Optional[float] = None,

) -> float:

    """Shaped reward for residual TD3 — unified distance term per robot config."""

    if not object_found:

        return 0.0



    r = 0.0

    w_success = float(reward_cfg.get('w_success', 1.0))

    w_dist = float(reward_cfg.get('w_dist', 0.0))

    w_lift = float(reward_cfg.get('w_lift', 0.0))

    w_drop = float(reward_cfg.get('w_drop', 0.0))

    w_clamp = float(reward_cfg.get('w_clamp', 0.0))

    w_corr = float(reward_cfg.get('w_corr', 0.1))

    w_miss = float(reward_cfg.get('w_miss', 0.15))

    w_delta_sat = float(reward_cfg.get('w_delta_sat', 0.10))

    delta_sat_ratio = float(reward_cfg.get('delta_sat_ratio', 0.90))

    lift_progress_min = float(reward_cfg.get('lift_progress_min_m', 0.005))

    corr_fail_mult = float(reward_cfg.get('w_corr_fail_mult', 2.0))

    lift_target = float(reward_cfg.get('lift_target_m', 0.023))

    w_align = float(reward_cfg.get('w_align', 0.0))

    align_scale_m = float(reward_cfg.get('align_scale_m', 0.02))

    align_max_dist = float(reward_cfg.get('align_max_closest_dist_m', 0.08))

    align_require_progress = bool(reward_cfg.get('align_require_progress', False))



    lift = lifted_m if lifted_m is not None else 0.0



    if success:

        r += w_success



    has_progress = success or lift > lift_progress_min

    if has_progress and closest_dist < NAN_DIST and w_dist > 0:

        cd_good = float(reward_cfg.get('cd_good_m', 0.080))

        cd_band = float(reward_cfg.get('cd_band_m', 0.025))

        if cd_band > 0:

            r += w_dist * max(0.0, (cd_good - closest_dist) / cd_band)



    if not success and lifted_m is not None:

        if lift < 0.0:

            r -= w_drop * min(1.0, abs(lift) / 0.02)

        elif lift_progress_min < lift <= lift_target:

            r += w_lift * (lift / lift_target)

        elif lift <= lift_progress_min:

            r -= w_miss



    if clamp_limited:

        r -= w_clamp



    if w_align > 0 and lateral_improve_m is not None and align_scale_m > 0:
        apply_align = (
            closest_dist < NAN_DIST
            and closest_dist <= align_max_dist
            and (not align_require_progress or has_progress)
        )
        if apply_align:
            align_norm = max(-1.0, min(1.0, lateral_improve_m / align_scale_m))
            r += w_align * align_norm



    corr = abs(delta_x) + abs(delta_z) + abs(delta_yaw) / math.pi

    corr_mult = corr_fail_mult if not success else 1.0

    r -= w_corr * corr * corr_mult



    if max_delta is not None:

        mx, mz, my = max_delta

        ratios = [

            abs(delta_x) / mx if mx > 0 else 0.0,

            abs(delta_z) / mz if mz > 0 else 0.0,

            abs(delta_yaw) / my if my > 0 else 0.0,

        ]

        peak = max(ratios)

        if peak >= delta_sat_ratio:

            r -= w_delta_sat * peak



    return float(r)


def exploration_noise_scale(td3_cfg: Dict[str, Any], session_episode: int) -> float:
    """Return exploration noise multiplier for this client session episode."""
    schedule = td3_cfg.get('exploration_noise_schedule')
    base = float(td3_cfg.get('exploration_noise', 0.025))
    if not schedule:
        return base
    ep = max(1, int(session_episode))
    for stage in schedule:
        until = int(stage.get('until_session_episode', 0))
        if ep <= until:
            return float(stage.get('scale', base))
    return float(schedule[-1].get('scale', 0.0))

