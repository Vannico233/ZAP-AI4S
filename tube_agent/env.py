"""Panda + free rigid test tubes. Grasping is frictional contact, never attachment."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = np.array([0, -0.35, 0, -2.1, 0, 1.75, 0.7854])
COLORS = {"red": "0.88 0.18 0.20 1", "blue": "0.15 0.40 0.90 1", "green": "0.15 0.72 0.42 1"}
SPLIT_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000, "ood": 300_000}


def numbers(values):
    return " ".join(f"{float(x):.7f}" for x in values)


def down_rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.]])


class TubeEnv:
    def __init__(self, task="transport", split="train", seed=0, max_steps=20,
                 frame_callback: Optional[Callable] = None, detailed_feedback=False):
        if task not in ("pick", "transport") or split not in SPLIT_OFFSETS:
            raise ValueError("Unknown task or split")
        if not 0 <= seed < 100_000 or max_steps < 1:
            raise ValueError("seed must be in [0, 100000); max_steps must be positive")
        self.task, self.split, self.seed = task, split, seed
        self.max_steps, self.steps = max_steps, 0
        self.detailed_feedback = detailed_feedback
        self.rng = np.random.default_rng(SPLIT_OFFSETS[split] + seed)
        self.renderer = None
        self.frame_callback = frame_callback
        self.closed = False
        self.ended = False
        self.invalid_actions = 0
        self.lift_seconds = 0.
        self.place_seconds = 0.
        self.lifted_once = False
        self._tick = 0
        self._initializing = True
        self._build()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.tcp_id = self.model.site("tcp").id
        self.tube_id = self.model.body("tube_red").id
        self.finger_ids = {self.model.body(n).id for n in ("left_finger", "right_finger")}
        self.tube_joint = self.model.joint("tube_red_free").id
        self.tube_dof = self.model.jnt_dofadr[self.tube_joint]
        self.data.qpos[:7] = HOME
        self.data.qpos[7:9] = .04
        self.data.ctrl[:7] = HOME
        self.data.ctrl[7] = 255
        mujoco.mj_forward(self.model, self.data)
        self._advance(.4)
        self.initial_z = float(self.data.xpos[self.tube_id, 2])
        self.initial_bottom = self._bottom()
        self._initializing = False
        self.initial_distance = max(float(np.linalg.norm(self.tcp - self.tube_pos)), .01)
        self._capture()

    def _build(self):
        path = ROOT / "assets" / "panda" / "panda.xml"
        if not path.exists():
            raise FileNotFoundError("Panda assets missing. Run: python scripts/fetch_assets.py")
        root = ET.parse(path).getroot()
        root.find("compiler").set("meshdir", str(path.parent / "assets"))
        root.remove(root.find("keyframe"))
        option = root.find("option")
        option.set("timestep", "0.002")
        option.set("iterations", "60")
        option.set("cone", "elliptic")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="640", offheight="480")
        ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.35 0.35 0.35")
        collision = root.find(".//default[@class='collision']/geom")
        collision.set("rgba", "0.78 0.81 0.86 1")
        # Explicit contact parameters, identical for all policies.
        for default in root.findall(".//default"):
            if default.get("class", "").startswith("fingertip_pad"):
                geom = default.find("geom")
                geom.set("friction", "1.2 0.02 0.002")
                geom.set("condim", "6")
        hand = root.find(".//body[@name='hand']")
        ET.SubElement(hand, "site", name="tcp", pos="0 0 0.1034", size="0.003", rgba="1 0.6 0 1", group="4")
        world = root.find("worldbody")
        ET.SubElement(world, "light", pos="0.3 -0.5 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
        ET.SubElement(world, "geom", name="table", type="plane", size="1 1 0.02", rgba="0.18 0.23 0.28 1", friction="0.8 0.01 0.001")
        camera_pos = np.array([1.10, -1.10, 1.00])
        camera_z = camera_pos - [.30, 0, .25]
        camera_z /= np.linalg.norm(camera_z)
        camera_x = np.cross([0, 0, 1], camera_z)
        camera_x /= np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)
        ET.SubElement(world, "camera", name="overview", pos=numbers(camera_pos),
                      xyaxes=numbers(np.r_[camera_x, camera_y]), fovy="45")
        ET.SubElement(world, "camera", name="top", pos="0.43 0 1.1", xyaxes="1 0 0 0 1 0", fovy="45")
        self.tray = np.array([.45, .22, .004])
        self.tray_half = np.array([.10, .09])
        ET.SubElement(world, "geom", name="tray", type="box", pos=numbers(self.tray / [1, 1, 2]),
                      size=numbers([*self.tray_half, .002]), rgba="0.24 0.60 0.62 1", friction="1 0.01 0.001")
        ood = self.split == "ood"
        self.radius = float(self.rng.uniform(.010, .014) if ood else .012)
        self.half_length = float(self.rng.uniform(.04, .052) if ood else .045)
        x = float(self.rng.uniform(.38, .53) if ood else self.rng.uniform(.41, .49))
        y = float(self.rng.uniform(-.24, -.12) if ood else self.rng.uniform(-.20, -.12))
        yaw = float(self.rng.uniform(-.65, .65) if ood else self.rng.uniform(-.25, .25))
        self.start_yaw = yaw
        self._add_tube(world, "red", x, y, yaw, self.radius, self.half_length)
        if ood:
            self._add_tube(world, "blue", .30, .01, -.25, .012, .045)
        self._customize_scene(root, world)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def _customize_scene(self, root, world):
        """Extension hook. The original two-task scene is unchanged."""

    @staticmethod
    def _add_tube(world, color, x, y, yaw, radius, half_length):
        # Local capsule axis is X; yaw rotates it within the table plane.
        body = ET.SubElement(world, "body", name=f"tube_{color}", pos=numbers([x, y, radius + .001]),
                             quat=numbers([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
        ET.SubElement(body, "freejoint", name=f"tube_{color}_free")
        ET.SubElement(body, "geom", name=f"tube_{color}_geom", type="capsule",
                      fromto=numbers([-half_length, 0, 0, half_length, 0, 0]), size=str(radius),
                      mass="0.025", rgba=COLORS[color], friction="1.0 0.02 0.002", condim="6")
        ET.SubElement(body, "geom", type="cylinder", pos=numbers([half_length-.004, 0, 0]),
                      quat="0.70710678 0 0.70710678 0", size=numbers([radius*1.03, .006]),
                      mass="0.002", rgba="0.93 0.94 0.96 1", contype="0", conaffinity="0")

    @property
    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    @property
    def tube_pos(self):
        return self.data.xpos[self.tube_id].copy()

    def _axis(self):
        return self.data.xmat[self.tube_id].reshape(3, 3)[:, 0]

    def _bottom(self):
        return float(self.tube_pos[2] - abs(self._axis()[2])*self.half_length - self.radius)

    def _contacts(self):
        fingers = set()
        for contact in self.data.contact:
            b1, b2 = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if b1 == self.tube_id and b2 in self.finger_ids:
                fingers.add(b2)
            if b2 == self.tube_id and b1 in self.finger_ids:
                fingers.add(b1)
        return fingers

    def _update_progress(self):
        dt = self.model.opt.timestep
        held = len(self._contacts()) == 2
        lifted = held and self._bottom() - self.initial_bottom >= .06
        self.lift_seconds = self.lift_seconds + dt if lifted else 0.
        self.lifted_once |= self.lift_seconds >= .35
        extent = np.abs(self._axis()[:2])*self.half_length + self.radius
        inside = bool(np.all(np.abs(self.tube_pos[:2]-self.tray[:2]) + extent < self.tray_half - .002))
        resting = abs(self._bottom() - self.tray[2]) < .006
        still = np.linalg.norm(self.data.qvel[self.tube_dof:self.tube_dof+6]) < .08
        released = sum(self.data.qpos[7:9]) > .06 and not self._contacts()
        withdrawn = np.linalg.norm(self.tcp-self.tube_pos) > .08
        placed = self.lifted_once and inside and resting and still and released and withdrawn
        self.place_seconds = self.place_seconds + dt if placed else 0.

    def _capture(self):
        if self.frame_callback and not self._initializing:
            self.frame_callback(self.render())

    def _advance(self, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            # Panda joint servos with standard gravity / bias compensation.
            self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
            mujoco.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)):
                raise RuntimeError("Non-finite simulation state")
            if not self._initializing:
                self._update_progress()
            self._tick += 1
            if self._tick % 40 == 0:
                self._capture()

    def _solve_ik(self, xyz, rotation, initial=None):
        d, m = self.ik_data, self.model
        d.qpos[:] = self.data.qpos
        if initial is not None:
            d.qpos[:7] = initial
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(100):
            mujoco.mj_forward(m, d)
            position_error = xyz - d.site_xpos[self.tcp_id]
            delta = rotation @ d.site_xmat[self.tcp_id].reshape(3, 3).T
            quat, angular_error = np.empty(4), np.empty(3)
            mujoco.mju_mat2Quat(quat, delta.ravel())
            if quat[0] < 0:
                quat *= -1
            mujoco.mju_quat2Vel(angular_error, quat, 1.)
            if np.linalg.norm(position_error) < .0006 and np.linalg.norm(angular_error) < .008:
                return d.qpos[:7].copy()
            mujoco.mj_jacSite(m, d, jp, jr, self.tcp_id)
            jac = np.vstack([jp[:, :7], jr[:, :7]])
            error = np.r_[position_error, angular_error]
            step = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*1e-4, error)
            step *= min(1., .15 / max(np.max(np.abs(step)), 1e-9))
            d.qpos[:7] = np.clip(d.qpos[:7] + step, m.jnt_range[:7, 0]+.005, m.jnt_range[:7, 1]-.005)
        return None

    def _move(self, xyz, yaw):
        start = self.tcp
        mat = self.data.site_xmat[self.tcp_id].reshape(3, 3)
        start_yaw = math.atan2(mat[1, 0], mat[0, 0])
        dyaw = (yaw - start_yaw + math.pi) % (2*math.pi) - math.pi
        count = max(1, math.ceil(np.linalg.norm(xyz-start)/.012), math.ceil(abs(dyaw)/.10))
        for i in range(1, count+1):
            fraction = i/count
            q = self._solve_ik(start + fraction*(xyz-start), down_rotation(start_yaw + fraction*dyaw))
            if q is None:
                return {"accepted": True, "reached": False, "reason": "ik_failed_after_partial_motion"}
            self.data.ctrl[:7] = q
            self._advance(.08)
        self._advance(.25)
        error = float(np.linalg.norm(self.tcp-xyz))
        delta = down_rotation(yaw) @ self.data.site_xmat[self.tcp_id].reshape(3, 3).T
        angular = float(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
        return {"accepted": True, "reached": error < .008 and angular < .08,
                "position_error_m": round(error, 5), "orientation_error_rad": round(angular, 5)}

    def step(self, action):
        if self.ended:
            raise RuntimeError("Episode already ended")
        self.steps += 1
        try:
            kind = action["action"]
            if kind == "move":
                if set(action) != {"action", "xyz", "yaw"}:
                    raise ValueError("move requires exactly action, xyz, yaw")
                xyz, yaw = np.asarray(action["xyz"], dtype=float), float(action["yaw"])
                if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or not math.isfinite(yaw):
                    raise ValueError("Pose must be finite, xyz must contain three numbers")
                if np.any(xyz < [.20, -.40, .009]) or np.any(xyz > [.65, .40, .50]) or abs(yaw) > math.pi:
                    if self.detailed_feedback:
                        violations = [
                            f"{axis}={value:g} outside [{lower:g}, {upper:g}] metres"
                            for axis, value, lower, upper in zip(
                                "xyz", xyz, [.20, -.40, .009], [.65, .40, .50])
                            if value < lower or value > upper
                        ]
                        if abs(yaw) > math.pi:
                            violations.append(f"yaw={yaw:g} outside [{-math.pi}, {math.pi}] radians")
                        raise ValueError("Invalid move TCP pose: " + "; ".join(violations))
                    raise ValueError("Pose outside workspace or yaw outside [-pi, pi]")
                feedback = self._move(xyz, yaw)
            elif kind == "gripper":
                if set(action) != {"action", "opening"}:
                    raise ValueError("gripper requires exactly action and opening")
                opening = float(action["opening"])
                if not math.isfinite(opening) or not 0 <= opening <= .08:
                    raise ValueError("opening must be in [0, 0.08] metres")
                self.data.ctrl[7] = 255 * opening / .08
                self._advance(.65)
                feedback = {"accepted": True, "opening_m": round(float(sum(self.data.qpos[7:9])), 4)}
            elif kind == "finish" and set(action) == {"action"}:
                self._advance(.5)
                self.ended = True
                feedback = {"accepted": True, "submitted": True}
            else:
                raise ValueError("Use move, gripper or finish with the documented fields")
        except (KeyError, TypeError, ValueError) as exc:
            self.invalid_actions += 1
            feedback = {"accepted": False, "reason": str(exc)}
        if self.steps >= self.max_steps and not self.ended:
            self._advance(.5)
            self.ended = True
        return self.observe(), feedback, self.ended

    def observe(self):
        objects = []
        for color in COLORS:
            obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"tube_{color}")
            if obj >= 0:
                axis = self.data.xmat[obj].reshape(3, 3)[:, 0]
                objects.append({"name": f"tube_{color}", "center": np.round(self.data.xpos[obj], 4).tolist(),
                                "axis": np.round(axis, 4).tolist(),
                                "radius": self.radius if color == "red" else .012})
        return {"observation_mode": "oracle_object_state", "units": "metres and radians",
                "tcp": np.round(self.tcp, 4).tolist(),
                "gripper_opening": round(float(sum(self.data.qpos[7:9])), 4),
                "objects": objects, "tray_center": self.tray.tolist(),
                "tray_half_size": self.tray_half.tolist(), "steps_left": self.max_steps-self.steps}

    def result(self):
        success = bool(self.lift_seconds >= .35 if self.task == "pick" else self.place_seconds >= .4)
        held = len(self._contacts()) == 2
        progress = .15 * np.clip(1-np.linalg.norm(self.tcp-self.tube_pos)/self.initial_distance, 0, 1)
        if held:
            progress = .2 + .3*np.clip((self._bottom()-self.initial_bottom)/.06, 0, 1)
        if self.lifted_once:
            progress = max(progress, .5)
            if self.task == "transport":
                progress += .35*np.clip(1-np.linalg.norm(self.tube_pos[:2]-self.tray[:2])/.35, 0, 1)
        # One terminal ladder reward; not a sum that can be farmed by repeating actions.
        reward = (5. if success else min(float(progress), .9)) - .01*self.steps - .05*self.invalid_actions
        return {"success": success, "reward": round(reward, 5), "lifted_once": bool(self.lifted_once),
                "lift_height_m": round(self._bottom()-self.initial_bottom, 4),
                "held": held, "stable_place_seconds": round(self.place_seconds, 3),
                "steps": self.steps, "invalid_actions": self.invalid_actions,
                "tube_position": np.round(self.tube_pos, 4).tolist()}

    @property
    def instruction(self):
        if self.task == "pick":
            return "Pick up the RED test tube. Hold its entire body at least 0.06 m above its initial resting height."
        return "Pick up the RED test tube and place it fully inside the teal tray. Release it, withdraw the gripper and leave it resting stably."

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 1  # Menagerie collision meshes are also our visuals.
        self.renderer.update_scene(self.data, camera="overview", scene_option=options)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.closed = True
