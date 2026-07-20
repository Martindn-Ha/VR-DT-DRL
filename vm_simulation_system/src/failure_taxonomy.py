"""Grasp failure taxonomy rules used by simulation episode logging."""

from typing import Any, Mapping, Optional, Sequence, Union

REQUIRED_LIFT = 0.023  # legacy analysis threshold; sim success uses pickup-hold, not lift height
MIN_PICKUP_LIFT = 0.001
FAR_MISS_DIST = 0.20
NEAR_MISS_DIST = 0.10
DROP_LIFT = 0.0  # drop_or_push when lifted_m < 0
NEAR_MISS_LIFT = 0.02
NAN_DIST_SENTINEL = 9999.0
CLAMP_EPS = 1e-4

OUTCOME_CLASSES = (
    'success',
    'sim_nan_abort',
    'object_not_found',
    'yolo_detection_failed',
    'far_miss',
    'weak_lift',
    'drop_or_push',
    'near_miss',
    'mid_miss',
    'other_failure',
)


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool_int(value: Any, default: int = 1) -> int:
    if value is None or value == '':
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def classify_outcome(
    *,
    success: Union[bool, int, Any],
    grasp_mode: str = '',
    object_found: Union[bool, int, Any] = 1,
    closest_dist_m: Union[float, Any] = 0.0,
    lifted_m: Union[float, None, Any] = None,
) -> str:
    """Return outcome_class label for one episode."""
    if int(_as_bool_int(success, 0)) == 1:
        return 'success'

    mode = str(grasp_mode or '')
    closest = _as_float(closest_dist_m)
    if closest is None:
        closest = NAN_DIST_SENTINEL

    if mode == 'nan_abort' or closest >= NAN_DIST_SENTINEL:
        return 'sim_nan_abort'

    if _as_bool_int(object_found, 1) == 0:
        return 'object_not_found'

    if closest > FAR_MISS_DIST:
        return 'far_miss'

    lift = _as_float(lifted_m)
    if lift is not None and lift > MIN_PICKUP_LIFT:
        return 'drop_or_push'

    if lift is not None and lift < DROP_LIFT:
        return 'drop_or_push'

    if closest <= NEAR_MISS_DIST and (lift is None or abs(lift) < NEAR_MISS_LIFT):
        return 'near_miss'

    if closest > NEAR_MISS_DIST and closest <= FAR_MISS_DIST:
        return 'mid_miss'

    return 'other_failure'


def classify_outcome_from_row(row: Mapping[str, Any]) -> str:
    """Classify from an episode log row (CSV/XLSX dict)."""
    return classify_outcome(
        success=row.get('success', 0),
        grasp_mode=str(row.get('grasp_mode', '') or ''),
        object_found=row.get('object_found', 1),
        closest_dist_m=row.get('closest_dist_m', NAN_DIST_SENTINEL),
        lifted_m=row.get('lifted_m'),
    )


def is_clamp_limited(
    raw_pose: Optional[Sequence[float]],
    clamp_pose: Optional[Sequence[float]],
    epsilon: float = CLAMP_EPS,
) -> bool:
    """True when executed X/Z pose was clipped vs raw network output."""
    if not raw_pose or not clamp_pose or len(raw_pose) < 3 or len(clamp_pose) < 3:
        return False
    for raw_idx, clamp_idx in ((0, 0), (2, 2)):
        raw_val = _as_float(raw_pose[raw_idx])
        clamp_val = _as_float(clamp_pose[clamp_idx])
        if raw_val is None or clamp_val is None:
            continue
        if abs(raw_val - clamp_val) > epsilon:
            return True
    return False


def is_clamp_limited_from_row(row: Mapping[str, Any]) -> bool:
    raw = [_as_float(row.get(f'ai_pose_{i}')) for i in range(6)]
    clamp = [_as_float(row.get(f'clamp_pose_{i}')) for i in range(3)]
    if any(v is None for v in raw[:3]) or any(v is None for v in clamp):
        return False
    return is_clamp_limited(raw, clamp)
