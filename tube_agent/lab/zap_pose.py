"""ZAP Appendix A semantic angles, adapted to the Panda TCP frame.

Reference: https://zero-real.github.io/assets/js/orientation-widget.js,
orient(): yaw about world Z, then tilt about [sin(phi), -cos(phi), 0].
This is NOT an Euler triple. Panda local X is the jaw slot, local Y is
closing, and local Z points toward the fingertips. The project's DROID
body axes differ; physical tool/slot directions, not body matrices, match.
"""
import math
import numpy as np

PROTOCOL = "zap_pose_v1"


def scalar(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} outside [{low}, {high}]")
    return float(value)


def axis_rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


def rotation(slot_yaw, tilt, tilt_direction=0.):
    psi = math.radians(scalar(slot_yaw, "slot_yaw (degrees)", -360, 360))
    tau = math.radians(scalar(tilt, "tilt (degrees)", 0, 180))
    phi = math.radians(scalar(tilt_direction, "tilt_direction (degrees)", -360, 360))
    yaw = axis_rotation([0., 0., 1.], psi)
    lean = axis_rotation([math.sin(phi), -math.cos(phi), 0.], tau)
    return lean @ yaw @ np.diag([1., -1., -1.])


def angles(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
        or not np.allclose(matrix.T@matrix, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix), 1., atol=1e-6)):
        raise ValueError("Expected a proper rotation matrix")
    tool = matrix[:, 2]
    horizontal = math.hypot(tool[0], tool[1])
    tau = math.atan2(horizontal, -tool[2])
    phi = math.atan2(tool[1], tool[0]) if horizontal > 1e-9 else 0.
    untilt = axis_rotation([math.sin(phi), -math.cos(phi), 0.], -tau)
    slot = untilt @ matrix[:, 0]
    psi = math.atan2(slot[1], slot[0])
    return {"slot_yaw": math.degrees(psi), "tilt": math.degrees(tau),
            "tilt_direction": math.degrees(phi)}


def move_action(xyz, matrix):
    return {"action": "move_to", "xyz": list(xyz), **angles(matrix)}
