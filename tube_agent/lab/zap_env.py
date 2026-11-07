"""Versioned ZAP pose boundary and final-state reward ladder for the lab.

Legacy RPY experiments remain replayable with LabEnv. This backend retains
MuJoCo/Panda and local laboratory tasks; it is not the released ZAP backend.
"""
import math
import mujoco
import numpy as np
from tube_agent.env import HOME, TubeEnv
from .env import LabEnv
from . import zap_pose


class ZapLabEnv(LabEnv):
    VERSION = "lab_v2_zap"
    ACTION_PROTOCOL = zap_pose.PROTOCOL
    REWARD_VERSION = "zap_terminal_ladder_v1"

    def __init__(self, *args, action_protocol=ACTION_PROTOCOL, **kwargs):
        if action_protocol != self.ACTION_PROTOCOL:
            raise ValueError("ZapLabEnv requires zap_pose_v1")
        super().__init__(*args, action_protocol=action_protocol, **kwargs)
        target = self.rack_center if self.lab_task in ("insert", "rack_transfer") else self.tray
        self.initial_goal_distance = max(.01, float(np.linalg.norm(self.tube_pos[:2]-target[:2])))

    def proprioception(self):
        pose = zap_pose.angles(self.data.site_xmat[self.tcp_id].reshape(3, 3))
        return {"units": "positions in metres; semantic orientation angles in degrees",
                "tcp": np.round(self.tcp, 5).tolist(),
                "tcp_orientation": {k: round(v, 5) for k, v in pose.items()},
                "gripper_opening": round(float(sum(self.data.qpos[7:9])), 5),
                "gripper_opening_fraction": round(float(sum(self.data.qpos[7:9]))/.08, 5),
                "steps_left": self.max_steps-self.steps}

    @property
    def instruction(self):
        return super().instruction.replace("0.35 seconds", "0.5 seconds").replace("0.4 seconds", "0.5 seconds")

    def _pose_motion(self, xyz, target):
        """Validate a complete straight TCP path before any dynamics advance."""
        start = self.tcp.copy()
        current = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        quat = np.empty(4); velocity = np.empty(3)
        mujoco.mju_mat2Quat(quat, (target@current.T).ravel())
        if quat[0] < 0: quat *= -1
        mujoco.mju_quat2Vel(velocity, quat, 1.)
        angle = float(np.linalg.norm(velocity))
        count = max(1, math.ceil(np.linalg.norm(xyz-start)/.008), math.ceil(angle/.07))
        path = []; previous = self.data.qpos[:7].copy()
        for i in range(1, count+1):
            fraction = i/count
            rotation = (zap_pose.axis_rotation(velocity/angle, angle*fraction)@current
                        if angle > 1e-9 else current)
            position = start+fraction*(xyz-start)
            q = self._solve_ik(position, rotation, previous)
            if q is None:
                # Different IK seeds may help; do not execute an unvalidated jump.
                for j1 in (-.8, 0., .8):
                    for j7 in (-1.8, 0., 1.8):
                        initial = HOME.copy(); initial[0] = j1; initial[6] = j7
                        candidate = self._solve_ik(position, rotation, initial)
                        if candidate is not None and np.max(np.abs(candidate-previous)) < .6:
                            q = candidate; break
                    if q is not None: break
            if q is None or np.max(np.abs(q-previous)) >= .6:
                return {"accepted": False, "reached": False, "outcome": "not_attempted",
                        "reason": "No continuous IK path within joint limits; nothing moved",
                        "failed_waypoint": i, "waypoints": count}
            path.append(q.copy()); previous = q
        for q in path:
            previous = self.data.ctrl[:7].copy()
            substeps = max(1, math.ceil(np.max(np.abs(q-previous))/.12))
            for j in range(1, substeps+1):
                self.data.ctrl[:7] = previous+(q-previous)*j/substeps
                self._advance(.08)
        self._advance(.3)
        error = float(np.linalg.norm(self.tcp-xyz))
        angular = float(np.arccos(np.clip((np.trace(target@self.data.site_xmat[self.tcp_id].reshape(3,3).T)-1)/2,-1,1)))
        reached = error < .008 and angular < .08
        return {"accepted": True, "reached": reached, "outcome": "reached" if reached else "short",
                "position_error_m": round(error, 6), "orientation_error_deg": round(math.degrees(angular), 4),
                "actual_tcp": self.proprioception(), "planned_waypoints": count}

    def step(self, action):
        if self.ended: raise RuntimeError("Episode already ended")
        kind = action.get("action")
        if kind == "set_gripper":
            try:
                if set(action) != {"action", "opening"}:
                    raise ValueError("set_gripper requires exactly action and opening")
                opening = zap_pose.scalar(action["opening"], "opening fraction", 0., 1.)
            except ValueError as exc: return self._reject_action(str(exc))
            return TubeEnv.step(self, {"action": "gripper", "opening": opening*.08})
        if kind == "done" and set(action) == {"action"}:
            return TubeEnv.step(self, {"action": "finish"})
        if kind != "move_to":
            return self._reject_action("Use move_to with xyz/slot_yaw/tilt/tilt_direction, set_gripper, change_query, update_todo or done. Legacy RPY is not accepted.")
        try:
            required = {"action", "xyz", "slot_yaw", "tilt"}
            if not required <= set(action) or set(action)-required-{"tilt_direction"}:
                raise ValueError("move_to requires xyz, slot_yaw, tilt and optional tilt_direction")
            xyz = action["xyz"]
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError("xyz must be a list of three world coordinates in metres")
            xyz = np.array([zap_pose.scalar(v, axis, low, high) for axis, v, low, high in
                            zip("xyz", xyz, [.20, -.4, .009], [.65, .4, .5])])
            if action["tilt"] != 0 and "tilt_direction" not in action:
                raise ValueError("Nonzero tilt requires an explicit tilt_direction")
            target = zap_pose.rotation(action["slot_yaw"], action["tilt"], action.get("tilt_direction", 0.))
        except (TypeError, ValueError) as exc: return self._reject_action(str(exc))
        self.steps += 1
        feedback = self._pose_motion(xyz, target)
        if not feedback["accepted"]: self.invalid_actions += 1
        if self.steps >= self.max_steps:
            self._advance(.5); self.ended = True
        return self.observe(), feedback, self.ended

    def result(self):
        result = super().result()
        if self.lab_task == "pick": success = self.lift_seconds >= .5
        elif self.lab_task == "transport": success = self.place_seconds >= .5
        elif self.lab_task == "unscrew": success = self.cap_seconds >= .5
        else: success = self.insert_seconds >= .5
        target = self.data.body("cap").xpos if self.lab_task == "unscrew" else self.tube_pos
        initial_distance = getattr(self, "initial_target_distance", self.initial_distance)
        approach = .2*float(np.clip(1-np.linalg.norm(self.tcp-target)/initial_distance, 0, 1))
        held = bool(result.get("cap_held") if self.lab_task == "unscrew" else result["held"])
        grasp = .2 if held else 0.
        lift = .55*float(np.clip(result["lift_height_m"]/.08, 0, 1)) if held else 0.
        transport = insertion = rotation = separation = 0.
        progress = max(approach, grasp+lift)
        if self.lab_task == "transport" and self.lifted_once:
            distance = float(np.linalg.norm(self.tube_pos[:2]-self.tray[:2]))
            if held or (distance < .04 and abs(self._bottom()-.004) < .015):
                transport = .75+.15*float(np.clip(1-distance/self.initial_goal_distance, 0, 1))
                progress = max(progress, transport)
        elif self.lab_task in ("insert", "rack_transfer") and self.lifted_once:
            alignment = max(0., float(self._axis()[2])); distance = float(np.linalg.norm(self.tube_pos[:2]-self.rack_center[:2]))
            if held or (alignment > .97 and distance < .012):
                insertion = .75+.05*alignment+.05*max(0., 1-distance/self.initial_goal_distance)
                if alignment > .97 and distance < .012:
                    insertion += .05*float(np.clip((.05-self._bottom())/.042, 0, 1))
                progress = max(progress, insertion)
        elif self.lab_task == "unscrew":
            lift = 0.
            rotation = (.2 if held else 0.)+.5*float(np.clip(self._cap_angle()/(2*math.pi), 0, 1))
            separation = (.7+.2*float(np.clip((self.data.body("cap").xpos[2]-self.cap_initial[2])/.045, 0, 1))
                          if self.thread_released and held else 0.)
            progress = max(approach, grasp, rotation, separation)
        progress = min(.9, progress)
        result.update(success=bool(success), action_protocol=self.ACTION_PROTOCOL,
                      reward_version=self.REWARD_VERSION,
                      reward=round((5. if success else progress)-.01*self.steps-.05*self.invalid_actions, 5))
        result["reward_terms"] = {"approach":round(approach,5), "grasp":grasp,
            "lift":round(lift,5), "transport_total":round(transport,5),
            "insertion_total":round(insertion,5), "rotation_total":round(rotation,5),
            "separation_total":round(separation,5), "terminal_progress":round(progress,5),
            "success_bonus":5. if success else 0., "length_penalty":round(.01*self.steps,5),
            "invalid_penalty":round(.05*self.invalid_actions,5)}
        return result


def make_env(task="pick", split="train", seed=0, max_steps=40, action_protocol="rpy_rounding_v2", **kwargs):
    cls = ZapLabEnv if action_protocol == zap_pose.PROTOCOL else LabEnv
    return cls(task, split, seed, max_steps=max_steps, action_protocol=action_protocol, **kwargs)
