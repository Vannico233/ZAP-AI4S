"""Upright-tube tasks with physical pregrasps and object-state success checks.

Uses the existing Panda servos, contact physics and semantic pose controller.
Initialization of T2/T3 is a deterministic physical setup, never a demonstration.
"""
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from tube_agent.env import numbers
from tube_agent.lab.zap_env import ZapLabEnv
from tube_agent.lab.env import LabEnv
from .config import load, digest, TASKS


def move(xyz):
    return {"action": "move_to", "xyz": [float(v) for v in xyz],
            "slot_yaw": 0., "tilt": 0., "tilt_direction": 0.}


class CurriculumEnv(ZapLabEnv):
    VERSION = "upright_tube_curriculum_v1"
    REWARD_VERSION = "upright_terminal_ladder_v1"

    def __init__(self, scene, config=None, frame_callback=None):
        self.spec = copy.deepcopy(config if config is not None else load())
        self.scene = copy.deepcopy(scene)
        check = {k: v for k, v in scene.items() if k != "sha256"}
        if digest(check) != scene["sha256"] or scene["task"] not in TASKS:
            raise ValueError("Invalid immutable scene specification")
        self.task_id = scene["task"]
        self.setup_mode = True
        self._reset_judge()
        super().__init__("insert", "train", scene["seed"] % 100000,
                         max_steps=self.spec["policy"]["max_steps"], frame_callback=None)
        self.scene_start_bottom = self._bottom()
        self.initialization_actions = []
        if self.task_id in ("T2", "T3"):
            for action in self._pregrasp_commands():
                feedback = self.step(action)[1]
                self.initialization_actions.append({"action": action, "feedback": feedback})
                if not feedback["accepted"] or (action["action"] == "move_to" and not feedback["reached"]):
                    raise RuntimeError(f"Physical pregrasp initialization failed: {feedback}")
            if not self._held_now():
                raise RuntimeError("Preheld scene failed bilateral-contact and relative-motion check")
        self.steps = self.invalid_actions = 0
        self.ended = False
        self._reset_judge()
        self.initial_bottom = self._bottom()
        self.scene_start_bottom = self.initial_bottom
        self.initial_distance = max(.01, float(np.linalg.norm(self.tcp - self.tube_pos)))
        self.initial_target_distance = self.initial_distance
        self.setup_mode = False
        self._advance(.2)
        self._reset_completion_timers()
        if self.task_id in ("T2", "T3"):
            if not self.grasp_valid:
                raise RuntimeError("Initial physical hold is not stable")
            self.ever_grasped = True
            self.extracted = True
        self.frame_callback = frame_callback
        self.initial_state = self.snapshot()

    def _reset_completion_timers(self):
        self.lift_seconds = self.place_seconds = self.insert_seconds = self.cap_seconds = 0.
        self.goal_seconds = 0.

    def _reset_judge(self):
        self._reset_completion_timers()
        self.grasp_seconds = self.extraction_seconds = self.collision_seconds = 0.
        self.grasp_valid = self.ever_grasped = self.extracted = False
        self.dropped = self.severe_collision = False
        self.lifted_once = False
        self._previous_relative = None
        self._previous_relative_rotation = None

    def _customize_scene(self, root, world):
        LabEnv._customize_scene(self, root, world)
        world.remove(world.find("body[@name='target_rack']"))
        g = self.spec["geometry"]
        self.radius, self.half_length = g["tube_radius"], g["tube_half_segment"]
        self.rack_center = np.r_[self.scene["target_xy"], 0.]
        self.source_rack = np.r_[self.scene["source_xy"], 0.]
        self._make_rack(world, "target", self.rack_center, g["target_rim_z"], "0.95 0.75 0.12 1")
        self._make_rack(world, "source", self.source_rack, g["source_rim_z"], "0.2 0.6 0.8 1")
        tube = world.find("body[@name='tube_red']")
        tube.set("pos", numbers([*self.source_rack[:2], g["floor_z"]+self.radius+self.half_length]))
        tube.set("quat", "0.7071067812 0 -0.7071067812 0")
        body_geom = tube.find("geom[@name='tube_red_geom']")
        body_geom.set("size", str(self.radius))
        body_geom.set("fromto", numbers([-self.half_length, 0, 0, self.half_length, 0, 0]))

    def _make_rack(self, world, name, center, rim, color):
        g = self.spec["geometry"]
        inner, outer, floor = g["hole_half_width"], g["rack_outer_half_width"], g["floor_z"]
        body = ET.SubElement(world, "body", name=name+"_rack", pos=numbers(center))
        ET.SubElement(body, "geom", name=name+"_floor", type="box", pos=numbers([0, 0, floor/2]),
                      size=numbers([outer, outer, floor/2]), rgba=color)
        for axis in (0, 1):
            for sign in (-1, 1):
                pos = [0., 0., (floor+rim)/2]; pos[axis] = sign*(inner+outer)/2
                size = [outer, outer, (rim-floor)/2]; size[axis] = (outer-inner)/2
                ET.SubElement(body, "geom", name=f"{name}_wall_{axis}_{sign}", type="box",
                    pos=numbers(pos), size=numbers(size), rgba=color,
                    friction=".6 .005 .0001", solref=".004 1", solimp=".99 .99 .001")

    def _pregrasp_commands(self):
        g = self.spec["geometry"]; xy = self.source_rack[:2]
        yield move([*xy, .24])
        yield move([*xy, g["floor_z"]+g["grasp_tcp_above_floor"]])
        yield {"action": "set_gripper", "opening": 0.}
        # The offset is measured only during privileged task initialization.
        offset = self.tcp-self.tube_pos
        target_center_z = g["initial_carry_bottom_z"]+self.radius+self.half_length
        yield move(np.r_[xy, target_center_z]+offset)
        if self.task_id == "T3":
            target_xy = self.rack_center[:2]+self.scene["insert_offset_xy"]
            center = np.r_[target_xy, g["approach_bottom_z"]+self.radius+self.half_length]
            yield move(center+offset)

    def _relative_motion(self):
        rotation = self.data.site_xmat[self.tcp_id].reshape(3, 3)
        relative = rotation.T @ (self.tube_pos-self.tcp)
        relative_rotation = rotation.T @ self.data.xmat[self.tube_id].reshape(3, 3)
        if self._previous_relative is None:
            speed = angular = 0.
        else:
            speed = np.linalg.norm(relative-self._previous_relative)/self.model.opt.timestep
            # Matrix-distance norm is smooth even for sub-microradian changes.
            angular = np.linalg.norm(relative_rotation-self._previous_relative_rotation)/math.sqrt(2)/self.model.opt.timestep
        self._previous_relative = relative.copy()
        self._previous_relative_rotation = relative_rotation.copy()
        return float(speed), float(angular)

    def _held_now(self):
        s = self.spec["success"]
        return (len(self._contacts()) == 2 and
                getattr(self, "relative_speed", 0.) <= s["grasp_relative_speed_m_s"] and
                getattr(self, "relative_angular_speed", 0.) <= s["grasp_relative_angular_speed_rad_s"])

    def geometry_state(self):
        g, s = self.spec["geometry"], self.spec["success"]
        axis = self._axis()
        bottom = self._bottom()
        # Low endpoint including the rounded cap of the capsule.
        bottom_point = self.tube_pos-self.half_length*axis-np.array([0., 0., self.radius])
        horizontal = float(np.linalg.norm(bottom_point[:2]-self.rack_center[:2]))
        tilt = math.degrees(math.acos(float(np.clip(axis[2], -1, 1))))
        zlevels = np.array([g["floor_z"]+self.radius, g["target_rim_z"]])
        crossings = self.tube_pos[:2]+(zlevels[:, None]-self.tube_pos[2])*axis[:2]/max(axis[2], 1e-6)
        inside = bool(np.all(np.abs(crossings-self.rack_center[:2])+self.radius <=
                             g["hole_half_width"]+s["hole_geometry_tolerance_m"]))
        linear = float(np.linalg.norm(self.data.qvel[self.tube_dof:self.tube_dof+3]))
        angular = float(np.linalg.norm(self.data.qvel[self.tube_dof+3:self.tube_dof+6]))
        released = sum(self.data.qpos[7:9]) >= s["released_opening_m"] and not self._contacts()
        withdrawn = float(np.linalg.norm(self.tcp-self.tube_pos))
        depth = g["target_rim_z"]-bottom
        return dict(bottom=bottom, horizontal_error=horizontal, tilt_deg=tilt, inside=inside,
                    depth=depth, released=bool(released), withdrawal=withdrawn,
                    linear_speed=linear, angular_speed=angular,
                    still=linear <= s["linear_speed_m_s"] and angular <= s["angular_speed_rad_s"],
                    seated=inside and tilt <= s["tilt_deg"] and abs(depth-s["target_depth_m"]) <= s["depth_tolerance_m"])

    def _collision_force(self):
        maximum = 0.
        for contact_index, contact in enumerate(self.data.contact):
            ids = (int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2]))
            # Severe robot contact with racks/table; ordinary tube/finger grip excluded.
            names = [self.model.body(i).name for i in ids]
            fixture = any(n in ("world", "source_rack", "target_rack") for n in names)
            arm = any(n in ("hand", "left_finger", "right_finger") or n.startswith("link") for n in names)
            if fixture and arm:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                maximum = max(maximum, float(np.linalg.norm(force[:3])))
        return maximum

    def _update_progress(self):
        if self.setup_mode:
            self.relative_speed, self.relative_angular_speed = self._relative_motion()
            return
        dt = self.model.opt.timestep; s = self.spec["success"]
        self.relative_speed, self.relative_angular_speed = self._relative_motion()
        self.grasp_seconds = self.grasp_seconds+dt if self._held_now() else 0.
        self.grasp_valid = self.grasp_seconds+1e-9 >= s["grasp_contact_hold_s"]
        self.ever_grasped |= self.grasp_valid
        state = self.geometry_state()
        lifted = self.grasp_valid and state["bottom"]-self.scene_start_bottom >= s["lift_m"]
        cleared = self.grasp_valid and state["bottom"] >= self.spec["geometry"]["source_rim_z"]+s["extraction_clearance_m"]
        self.extraction_seconds = self.extraction_seconds+dt if (cleared if self.task_id == "T4" else lifted) else 0.
        self.extracted |= self.extraction_seconds >= s["extraction_hold_s"]
        self.lifted_once = self.extracted
        if self.task_id == "T1":
            good = lifted and state["still"]
        elif self.task_id == "T2":
            good = (self.grasp_valid and state["horizontal_error"] <= s["horizontal_tolerance_m"] and
                    abs(state["bottom"]-self.spec["geometry"]["approach_bottom_z"]) <= s["approach_z_tolerance_m"] and
                    state["tilt_deg"] <= s["tilt_deg"] and state["still"])
        else:
            good = (self.extracted and state["seated"] and state["released"] and
                    state["withdrawal"] >= s["withdraw_m"] and state["still"])
        self.goal_seconds = self.goal_seconds+dt if good else 0.
        force = self._collision_force()
        self.collision_seconds = self.collision_seconds+dt if force >= self.spec["failure"]["collision_force_n"] else 0.
        self.severe_collision |= self.collision_seconds >= self.spec["failure"]["collision_hold_s"]
        self.dropped |= (self.ever_grasped and not self._held_now() and not state["seated"] and
                         (state["bottom"] < self.spec["failure"]["drop_bottom_z"] or
                          state["tilt_deg"] > self.spec["failure"]["drop_tilt_deg"]))

    def step(self, action):
        obs, feedback, _ = super().step(action)
        if not self.setup_mode and (self.dropped or self.severe_collision):
            self.ended = True
        return obs, feedback, self.ended

    def progress(self, st):
        r, g, s = self.spec["reward"], self.spec["geometry"], self.spec["success"]
        clip = lambda x: float(np.clip(x, 0, 1))
        grasp_point = self.tube_pos + np.array([0., 0., g["grasp_tcp_above_floor"]-self.radius-self.half_length])
        approach = clip(1-np.linalg.norm(self.tcp-grasp_point)/r["approach_scale_m"])
        horizontal = clip(1-st["horizontal_error"]/r["horizontal_scale_m"])
        alignment = clip(1-st["horizontal_error"]/r["alignment_scale_m"])
        height = clip(1-abs(st["bottom"]-g["approach_bottom_z"])/r["height_scale_m"])
        upright = clip(1-st["tilt_deg"]/r["tilt_scale_deg"])
        lift = clip((st["bottom"]-self.scene_start_bottom)/s["lift_m"])
        extract_fraction = clip((st["bottom"]-self.scene_start_bottom)/max(.001,
            g["source_rim_z"]+s["extraction_clearance_m"]-self.scene_start_bottom))
        pose_ok = st["tilt_deg"] <= s["tilt_deg"]
        valid_depth = clip(st["depth"]/s["target_depth_m"]) if st["inside"] and pose_ok else 0.
        holding_or_seated = self.grasp_valid or (self.ever_grasped and st["seated"])
        release = self.extracted and st["seated"] and st["released"]
        withdrawal = clip(st["withdrawal"]/s["withdraw_m"]) if release else 0.
        w = r[self.task_id]; terms = {}
        if self.task_id == "T1":
            terms["approach"] = w["approach"]*approach
            terms["grasp_lift"] = w["grasp"]+w["lift"]*lift if self.grasp_valid else 0.
            value = max(terms.values())
        elif self.task_id == "T2":
            terms = {"grasp": w["grasp"] if self.grasp_valid else 0.,
                     "horizontal": w["horizontal"]*horizontal if self.grasp_valid else 0.,
                     "height": w["height"]*height*horizontal if self.grasp_valid else 0.,
                     "upright": w["upright"]*upright*height*horizontal if self.grasp_valid else 0.}
            value = sum(terms.values())
        elif self.task_id == "T3":
            terms = {"align": w["align"]*alignment if holding_or_seated else 0.,
                     "pose": w["pose"]*alignment*upright if holding_or_seated else 0.,
                     "depth": w["depth"]*valid_depth if holding_or_seated else 0.,
                     "release": w["release"] if release else 0., "withdraw": w["withdraw"]*withdrawal}
            value = sum(terms.values())
        else:
            active = holding_or_seated and self.extracted
            terms = {"approach": w["approach"]*approach,
                     "grasp": w["grasp"] if holding_or_seated else 0.,
                     "extract": w["extract"]*(1 if self.extracted else extract_fraction) if holding_or_seated else 0.,
                     "transport": w["transport"]*horizontal*upright if active else 0.,
                     "depth": w["depth"]*valid_depth if active else 0.,
                     "release": w["release"] if release else 0., "withdraw": w["withdraw"]*withdrawal}
            value = max(terms["approach"], sum(v for k, v in terms.items() if k != "approach"))
        return min(r["progress_cap"], max(0., value)), terms

    def result(self):
        st = self.geometry_state(); r = self.spec["reward"]
        duration = self.spec["success"]["hold_s" if self.task_id in ("T1", "T2") else "placement_hold_s"]
        catastrophic = bool(self.dropped or self.severe_collision)
        success = self.goal_seconds+1e-9 >= duration and not catastrophic
        progress, terms = self.progress(st)
        reward = (r["success"] if success else progress)-r["turn_cost"]*self.steps-r["invalid_cost"]*self.invalid_actions-r["failure_cost"]*catastrophic
        return {"success": bool(success), "reward": float(reward), "task": self.task_id,
                "steps": self.steps, "invalid_actions": self.invalid_actions,
                "held": bool(self.grasp_valid), "ever_grasped": bool(self.ever_grasped),
                "extracted": bool(self.extracted), "goal_stable_seconds": self.goal_seconds,
                "tube_position": self.tube_pos.tolist(), "geometry": st,
                "dropped": bool(self.dropped), "severe_collision": bool(self.severe_collision),
                "reward_terms": {**terms, "terminal_progress": progress,
                    "success_bonus": r["success"] if success else 0., "length_penalty": r["turn_cost"]*self.steps,
                    "invalid_penalty": r["invalid_cost"]*self.invalid_actions, "failure_cost": r["failure_cost"]*catastrophic},
                "failure_type": ("none" if success else "severe_collision" if self.severe_collision else
                    "dropped" if self.dropped else "not_grasped" if not self.ever_grasped else
                    "not_extracted" if not self.extracted and self.task_id in ("T1", "T4") else
                    "not_aligned" if st["horizontal_error"] > self.spec["success"]["horizontal_tolerance_m"] else
                    "not_inserted" if self.task_id in ("T3", "T4") and not st["seated"] else
                    "not_released" if self.task_id in ("T3", "T4") and not st["released"] else "goal_not_stable"),
                "environment_version": self.VERSION, "reward_version": self.REWARD_VERSION,
                "action_protocol": self.ACTION_PROTOCOL}

    @property
    def instruction(self):
        g, s = self.spec["geometry"], self.spec["success"]
        end = f"Insert into the yellow rack hole to a depth of {s['target_depth_m']} m below its rim, open the gripper and withdraw at least {s['withdraw_m']} m. Leave the tube upright and stable for {s['placement_hold_s']} seconds."
        return {"T1": f"Grasp the upright red test tube in the shallow blue support. Lift its bottom at least {s['lift_m']} m above its initial support and hold securely for {s['hold_s']} seconds.",
                "T2": f"The red tube is initially held securely. Carry it upright above the yellow rack hole. Keep its bottom within {s['horizontal_tolerance_m']} m horizontally of the hole center, at world z={g['approach_bottom_z']} m within {s['approach_z_tolerance_m']} m, tilt at most {s['tilt_deg']} degrees, for {s['hold_s']} seconds.",
                "T3": "The red tube is initially held above the yellow rack with a small horizontal offset. "+end,
                "T4": "Grasp the upright red test tube in the shallow blue support, extract it, and carry it to the yellow rack. "+end}[self.task_id]

    def snapshot(self):
        return {"physics": np.r_[self.data.qpos, self.data.qvel, self.data.ctrl, self.data.eq_active,
                                 self.data.time].tolist(),
                "judge": [self.grasp_seconds, self.extraction_seconds, self.goal_seconds, self.collision_seconds,
                          self.grasp_valid, self.ever_grasped, self.extracted, self.dropped, self.severe_collision]}


def diagnostic_commands(env):
    """Privileged feasibility check only. These actions must never enter SFT."""
    g = env.spec["geometry"]
    if env.task_id in ("T1", "T4"):
        yield move([*env.source_rack[:2], .24])
        yield move([*env.source_rack[:2], g["floor_z"]+g["grasp_tcp_above_floor"]])
        yield {"action": "set_gripper", "opening": 0.}
        yield move([*env.tcp[:2], env.tcp[2]+.12])
    offset = env.tcp-env.tube_pos
    if env.task_id in ("T2", "T3", "T4"):
        center = np.r_[env.rack_center[:2], g["approach_bottom_z"]+env.radius+env.half_length]
        yield move(center+offset)
    if env.task_id in ("T3", "T4"):
        center = np.r_[env.rack_center[:2], g["floor_z"]+env.radius+env.half_length]
        yield move(center+offset)
        yield {"action": "set_gripper", "opening": 1.}
        yield move([*env.tcp[:2], .27])
    yield {"action": "done"}
