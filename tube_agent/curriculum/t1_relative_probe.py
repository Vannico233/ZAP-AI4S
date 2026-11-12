"""Optional relative TCP translation for the paired T1 interface diagnostic."""
import numpy as np

from tube_agent.interface import ARMS, FRAME, _vector


GENERAL = """
An additional motor action is available:
translate: {"action":"translate","arm":arm,"axis":axis,"distance_m":d}
Choose arm as "left" or "right". Choose axis as "camera_x", "camera_y",
"camera_z", or "table_normal". The first three are the fixed middle camera
optical axes. Positive table_normal follows table_up_middle; negative is the
opposite direction. For every axis, d is a signed displacement in metres
from the selected arm's measured current TCP position. Its orientation and
gripper opening are preserved. The other actions are also available."""

_CAMERA_AXES = {"camera_x": (1., 0., 0.), "camera_y": (0., 1., 0.), "camera_z": (0., 0., 1.)}


def resolve_relative(action, observation):
    """Convert one translate request into the equivalent middle-frame move_to."""
    if (not isinstance(action, dict) or set(action) != {"action", "arm", "axis", "distance_m"}
            or action["action"] != "translate"):
        raise ValueError("translate requires exactly action, arm, axis and distance_m")
    arm, axis = action["arm"], action["axis"]
    if arm not in ARMS:
        raise ValueError("arm must be left or right")
    if not isinstance(axis, str) or (axis != "table_normal" and axis not in _CAMERA_AXES):
        raise ValueError("axis must be camera_x, camera_y, camera_z or table_normal")
    distance = _vector([action["distance_m"]], 1, "distance_m")[0]
    if not isinstance(observation, dict) or observation.get("coordinate_frame") != FRAME:
        raise ValueError("Expected a middle_optical observation")
    try:
        pose = observation["arms"][arm]["ee_pose"]
        position = _vector(pose["position"], 3, "measured TCP position")
        quaternion = _vector(pose["quaternion_wxyz"], 4, "measured TCP quaternion")
        direction = (observation["table_up_middle"] if axis == "table_normal" else _CAMERA_AXES[axis])
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing measured pose or calibration for selected axis") from exc
    direction = _vector(direction, 3, "axis direction")
    if abs(np.linalg.norm(direction) - 1.) > 1e-5 or abs(np.linalg.norm(quaternion) - 1.) > .01:
        raise ValueError("Expected a unit axis direction and measured unit quaternion")
    with np.errstate(over="ignore"):
        target = position + distance * direction
    if not np.isfinite(target).all():
        raise ValueError("Computed position must be finite")
    return {"action": "move_to", "arm": arm, "position": target.tolist(),
            "quaternion_wxyz": quaternion.tolist()}
