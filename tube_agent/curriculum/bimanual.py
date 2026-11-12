"""Two contact-driven Panda arms in the upright-tube curriculum."""
import copy
import xml.etree.ElementTree as ET

import numpy as np

from tube_agent.env import numbers
from tube_agent.lab.zap_pose import rotation
from .env import CurriculumEnv, diagnostic_commands as original_diagnostic_commands


def _right_copy(element):
    """Duplicate names and joint/body/tendon references, sharing meshes and defaults."""
    clone = copy.deepcopy(element)
    for item in clone.iter():
        for attribute in ("name", "body1", "body2", "joint", "joint1", "joint2", "tendon"):
            if attribute in item.attrib:
                item.set(attribute, "right_" + item.get(attribute))
    return clone


def _camera(parent, name, pos, target, image_right, fovy):
    # MJCF camera local +Z points backwards; optical-frame conversion is at the API.
    back = np.subtract(pos, target, dtype=float)
    back /= np.linalg.norm(back)
    right = image_right - back * np.dot(image_right, back)
    right /= np.linalg.norm(right)
    ET.SubElement(parent, "camera", name=name, pos=numbers(pos), fovy=str(fovy),
                  xyaxes=numbers(np.r_[right, np.cross(back, right)]))


class BimanualEnv(CurriculumEnv):
    VERSION = "bimanual_panda_middle_camera_v1"
    ACTION_PROTOCOL = "bimanual_camera_v1"
    ARM_PREFIXES = {"left": "", "right": "right_"}
    RENDER_CAMERA = "middle"

    def __init__(self, scene, config=None, frame_callback=None):
        from .config import BIMANUAL, load
        config = load(BIMANUAL) if config is None else config
        if config["action_protocol"] != self.ACTION_PROTOCOL:
            raise ValueError("BimanualEnv requires the bimanual camera action protocol")
        self.grasp_arm = "left"
        self.touching = set()
        super().__init__(scene, config, frame_callback)

    def _customize_scene(self, root, world):
        super()._customize_scene(root, world)
        for camera in list(world.findall("camera")):
            world.remove(camera)
        left = world.find("body[@name='link0']")
        right = _right_copy(left)
        left.set("pos", "0 -.30 0")
        right.set("pos", "0 .30 0")
        # Park the idle wrist outside the other elbow's cross-table approach.
        for base, angle in ((left, -.25), (right, .25)):
            base.set("quat", numbers([np.cos(angle/2), 0, 0, np.sin(angle/2)]))
        world.append(right)
        for tag in ("tendon", "actuator", "equality", "contact"):
            section = root.find(tag)
            for item in list(section):
                section.append(_right_copy(item))
        _camera(world, "middle", (1.20, 0, .92), (.32, 0, .32), (0, 1, 0), 66)
        for name, base, prefix in (("left", left, ""), ("right", right, "right_")):
            hand = base.find(f".//body[@name='{prefix}hand']")
            side = 1 if name == "left" else -1
            _camera(hand, name + "_wrist", (.077, side*.041, .060), (0, -side*.044, .18), (0, -1, 0), 90)

    def tcp_pose_world(self, arm):
        site = self.arms[arm].site
        return self.data.site_xpos[site].copy(), self.data.site_xmat[site].reshape(3, 3).copy()

    def gripper_opening_m(self, arm):
        return float(self.data.qpos[self.arms[arm].fingers].sum())

    def move_pose_world(self, arm, xyz, matrix):
        xyz = np.asarray(xyz, dtype=float)
        if (xyz.shape != (3,) or not np.isfinite(xyz).all()
                or np.any(xyz < [.10, -.60, .009]) or np.any(xyz > [.78, .60, .78])):
            return {"accepted": False, "reached": False, "reason": "TCP target outside the calibrated workcell"}
        feedback = self._pose_motion(xyz, np.asarray(matrix), self.arms[arm])
        # The next observation already carries both measured poses in the middle frame.
        feedback.pop("actual_tcp", None)
        return feedback

    def set_arm_gripper(self, arm, opening):
        self.data.ctrl[self.arms[arm].gripper] = opening * 255.
        self._advance(.65)
        return {"accepted": True, "opening_m": self.gripper_opening_m(arm)}

    def arm_speeds(self):
        return self.data.qvel[self.arm_dofs].copy()

    def proprioception(self):
        from tube_agent.interface import proprioception
        return proprioception(self)

    def observe(self):
        return self.proprioception()

    def step(self, action):
        from tube_agent.interface import execute_motor
        if self.ended:
            raise RuntimeError("Episode already ended")
        self.steps += 1
        try:
            if action == {"action": "done"}:
                self._advance(.5)
                self.ended = True
                feedback = {"accepted": True, "submitted": True}
            else:
                feedback = execute_motor(self, action)
        except (ValueError, TypeError, KeyError) as exc:
            feedback = {"accepted": False, "reason": str(exc)}
        if not feedback["accepted"]:
            self.invalid_actions += 1
        if self.steps >= self.max_steps and not self.ended:
            self._advance(.5)
            self.ended = True
        self.ended |= not self.setup_mode and (self.dropped or self.severe_collision)
        return self.observe(), feedback, self.ended

    def _relative_motion(self):
        # Contacts from different robots cannot combine to satisfy a bilateral hold.
        self.touching.clear()
        for contact in self.data.contact:
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if a == self.tube_id:
                self.touching.add(b)
            elif b == self.tube_id:
                self.touching.add(a)
        if not self.arms[self.grasp_arm].pads <= self.touching:
            for name, arm in self.arms.items():
                if arm.pads <= self.touching:
                    self.grasp_arm = name
                    self.grasp_seconds = 0.
                    self._previous_relative = None
                    break
        return super()._relative_motion(self.arms[self.grasp_arm].site)

    def _contacts(self):
        return self.touching & self.arms[self.grasp_arm].pads

    def _released(self, threshold):
        return (self.gripper_opening_m(self.grasp_arm) >= threshold
                and all(arm.pads.isdisjoint(self.touching) for arm in self.arms.values()))

    def _pregrasp_commands(self):
        for command in super()._pregrasp_commands():
            yield _left_diagnostic_action(self, command)

    @property
    def instruction(self):
        return super().instruction.replace("at world z=", "at height above the tabletop of ")

def _left_diagnostic_action(env, action):
    """Express pregrasp/physics checks through the same boundary used by model episodes."""
    from tube_agent.interface import move_action_from_world
    if action["action"] == "move_to":
        return move_action_from_world(env, "left", action["xyz"],
                                      rotation(action["slot_yaw"], action["tilt"], action["tilt_direction"]))
    if action["action"] == "set_gripper":
        return dict(action, arm="left")
    return action


def diagnostic_commands(env):
    for action in original_diagnostic_commands(env):
        yield _left_diagnostic_action(env, action)
