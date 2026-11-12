"""Panda environment API. Positions are metres in the middle camera's optical frame.

TCP means the midpoint of the fingertip pads. Its local +Y is the closing axis
and +Z points toward the fingertips. A pose maps TCP-local axes into the middle
frame. Quaternions use scalar-first order. Motor and memory actions cost one turn.
"""
from numbers import Real
import numpy as np

PROTOCOL = "bimanual_camera_v1"
FRAME = "middle_optical"
ARMS = ("left", "right")
CAMERA_NAMES = ("middle", "left_wrist", "right_wrist")
GL_FROM_OPTICAL = np.diag([1., -1., -1.])

SYSTEM = """You operate two independent Panda grippers, left and right.
Return one JSON action per turn. Required fields:
move_to: {"action":"move_to","arm":"left","position":[x,y,z],"quaternion_wxyz":[w,x,y,z]}
set_gripper: {"action":"set_gripper","arm":"right","opening":0.0}
change_query: {"action":"change_query","queries":["red test tube"]}
update_todo: {"action":"update_todo","plan":"short remaining plan"}
done: {"action":"done"}
Use left or right on every motor action. Position is the absolute target of the
fingertip midpoint, in metres in the fixed middle camera optical frame: +X right,
+Y down, +Z forward into the image. The unit quaternion [w,x,y,z] gives TCP
orientation in this same frame: TCP +Y is the jaw closing axis and +Z points
toward the fingertips. Copy the observed quaternion to keep orientation.
Opening is a fraction: 0 closes, 1 opens to 0.08 m. One action controls one arm.
Images appear in order middle, left_wrist, right_wrist. All detections are
measured visible surfaces in middle optical coordinates. Check the RGB images,
current poses and measured openings. Gripper closure alone does not verify a grasp.
Submit done when the requested task is complete."""


def make_env(scene, config):
    """Open the physical scene for its registered action protocol."""
    if config["action_protocol"] == PROTOCOL:
        from .curriculum.bimanual import BimanualEnv
        return BimanualEnv(scene, config)
    if config["action_protocol"] == "zap_pose_v1":
        from .curriculum.env import CurriculumEnv
        return CurriculumEnv(scene, config)
    raise ValueError(f"Unsupported action protocol: {config['action_protocol']}")


def open_scene(scene_id="T1_train_000"):
    """Open one registered bimanual scene for interactive physics or camera debugging."""
    from .curriculum.config import BIMANUAL, load, scene_manifest
    config = load(BIMANUAL)
    for scene in scene_manifest(config)["scenes"]:
        if scene["id"] == scene_id:
            return make_env(scene, config)
    raise ValueError(f"Unknown bimanual scene ID: {scene_id}")


def _world_from_optical(env, name):
    cid = env.model.camera(name).id
    R = env.data.cam_xmat[cid].reshape(3, 3) @ GL_FROM_OPTICAL
    return env.data.cam_xpos[cid], R


def view_extrinsics(env, name):
    """Homogeneous transform from a view's optical frame into middle_optical."""
    middle, R = _world_from_optical(env, "middle")
    origin, axes = _world_from_optical(env, name)
    T = np.eye(4)
    T[:3, :3] = R.T @ axes
    T[:3, 3] = R.T @ (origin - middle)
    return T


def world_pose_to_middle(env, position, rotation):
    """Express a world TCP pose from a controller/diagnostic in the public frame."""
    import mujoco
    middle, R = _world_from_optical(env, "middle")
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, (R.T @ rotation).ravel())
    if q[0] < 0:
        q *= -1
    return {"position": (R.T @ (position-middle)).tolist(), "quaternion_wxyz": q.tolist()}


def _vector(value, size, field):
    values = np.asarray(value, dtype=object)
    if values.shape != (size,) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in values):
        raise ValueError(f"{field} requires {size} real numbers")
    try:
        a = values.astype(float)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not np.isfinite(a).all():
        raise ValueError(f"{field} must be finite")
    return a


def middle_pose_to_world(env, position, quaternion_wxyz):
    """Convert a public target into the simulator's world-space IK inputs."""
    import mujoco
    p = _vector(position, 3, "position")
    q = _vector(quaternion_wxyz, 4, "quaternion_wxyz")
    length = np.linalg.norm(q)
    if abs(length-1.) > .01:
        raise ValueError("quaternion_wxyz must be unit length (tolerance 0.01)")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, q/length)
    middle, R = _world_from_optical(env, "middle")
    return middle + R @ p, R @ rotation.reshape(3, 3)


def move_action_from_world(env, arm, position, rotation):
    """Build a public-frame move for a simulator diagnostic or physical reset."""
    return {"action":"move_to", "arm":arm, **world_pose_to_middle(env, position, rotation)}


def ee_pose(env, arm):
    """Read a live TCP pose in middle_optical, including measured orientation."""
    if arm not in ARMS:
        raise ValueError("arm must be left or right")
    return world_pose_to_middle(env, *env.tcp_pose_world(arm))


def proprioception(env):
    state = {"coordinate_frame": FRAME, "units": "metres; unit quaternion wxyz",
             "view_order": list(CAMERA_NAMES), "steps_left": env.max_steps - env.steps,
             "arms": {arm: {"ee_pose": ee_pose(env, arm),
                            "gripper_opening_m": float(env.gripper_opening_m(arm))} for arm in ARMS}}
    if "middle_camera" in env.spec["observation"]:
        # The calibrated table normal is world +Z, expressed in optical axes.
        _, world_from_middle = _world_from_optical(env, "middle")
        state["table_up_middle"] = world_from_middle[2].tolist()
    return state


def move_to(env, arm, position, quaternion_wxyz):
    """Move the selected TCP to an absolute middle-camera pose. Return motion feedback."""
    return env.step({"action":"move_to", "arm":arm, "position":position, "quaternion_wxyz":quaternion_wxyz})[1]


def set_gripper(env, arm, opening):
    """Command a jaw opening fraction between 0 (closed) and 1 (fully open)."""
    return env.step({"action":"set_gripper", "arm":arm, "opening":opening})[1]


def done(env):
    """Submit the episode for physical judging."""
    return env.step({"action":"done"})[1]


def execute_action(env, observer, policy, action):
    """Dispatch a parsed action during policy rollout or replay, including memory tools."""
    kind = action.get("action")
    if kind not in ("change_query", "update_todo"):
        return env.step(action)[1]
    if env.ended:
        raise RuntimeError("Episode ended")
    env.steps += 1
    try:
        if kind == "change_query":
            queries = action.get("queries")
            if (set(action) != {"action", "queries"} or not isinstance(queries, list)
                    or not 1 <= len(queries) <= 8
                    or any(not isinstance(q, str) or not 1 <= len(q) <= 120 for q in queries)):
                raise ValueError("change_query requires 1-8 text queries, each 1-120 characters")
            observer.queries = list(queries)
        else:
            plan = action.get("plan")
            if set(action) != {"action","plan"} or not isinstance(plan, str) or len(plan) > 1200:
                raise ValueError("update_todo requires a text plan up to 1200 characters")
            policy.plan = plan
        feedback = {"accepted": True}
    except ValueError as exc:
        env.invalid_actions += 1
        feedback = {"accepted": False, "reason": str(exc)}
    if env.steps >= env.max_steps:
        env._advance(.5)
        env.ended = True
    return feedback


def execute_motor(env, action):
    """Validate a model motor command and dispatch it to one physical arm."""
    if not isinstance(action, dict):
        raise ValueError("Expected a JSON action object")
    kind, arm = action.get("action"), action.get("arm")
    if arm not in ARMS:
        raise ValueError("Motor actions require arm equal to left or right")
    if kind == "move_to":
        if set(action) != {"action", "arm", "position", "quaternion_wxyz"}:
            raise ValueError("move_to requires arm, position and quaternion_wxyz")
        position, rotation = middle_pose_to_world(env, action["position"], action["quaternion_wxyz"])
        return env.move_pose_world(arm, position, rotation)
    if kind == "set_gripper":
        if set(action) != {"action", "arm", "opening"}:
            raise ValueError("set_gripper requires arm and opening")
        opening = action["opening"]
        if isinstance(opening, bool) or not isinstance(opening, Real) or not 0 <= opening <= 1:
            raise ValueError("opening must be a number in [0, 1]")
        return env.set_arm_gripper(arm, float(opening))
    raise ValueError("Expected move_to or set_gripper")


def capture_rgbd(env):
    """Capture named RGB, metric axial depth and calibration arrays, without SAM.

    For repeated captures, reuse a lab.vision.Cameras(env) instance and close it
    when finished, as the rollout observer does.
    """
    from .lab.vision import Cameras
    cameras = Cameras(env)
    try:
        return {f["calibration"]["name"]: f for f in cameras.capture()}
    finally:
        cameras.close()


def make_observer(env, segmenter):
    """Open a persistent policy observer; observe() returns measurements and ordered RGB overlays."""
    from .lab.vision import Observer
    observer = Observer(env, segmenter)
    observer.queries = ["red test tube"]
    if env.task_id in ("T1", "T4"):
        observer.queries.append("blue test tube rack")
    if env.task_id in ("T2", "T3", "T4"):
        observer.queries.append("yellow test tube rack")
    return observer
