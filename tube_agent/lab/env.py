"""Four contact-physics tasks and one held-out composition.

Unscrew uses an explicit fixed-body fixture and a releasable helical joint
constraint, not resolved thread meshes. Nothing attaches an object to a gripper.
"""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from tube_agent.env import TubeEnv, numbers, HOME

TASKS = ("pick", "transport", "insert", "unscrew", "rack_transfer")
TRAIN_TASKS = TASKS[:4]

def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])

def matrix_rpy(m):
    p = math.asin(float(np.clip(-m[2, 0], -1, 1)))
    if abs(math.cos(p)) < 1e-6:
        return [0., p, math.atan2(-m[0, 1], m[1, 1])]
    return [math.atan2(m[2, 1], m[2, 2]), p, math.atan2(m[1, 0], m[0, 0])]

class LabEnv(TubeEnv):
    VERSION = "lab_v1"
    def __init__(self, task="pick", split="train", seed=0, max_steps=40, frame_callback=None):
        if task not in TASKS:
            raise ValueError(f"Unknown lab task: {task}")
        if task == "rack_transfer" and split == "train":
            raise ValueError("rack_transfer is held out; use validation, test or ood")
        self.lab_task = task
        self.insert_seconds = self.cap_seconds = 0.
        self.thread_released = False
        self.thread_pitch = .006
        self.thread_turns = 1.
        self.rack_center = np.array([.45, .18, .0])
        self.source_rack = np.array([.45, -.18, .0])
        super().__init__(task="pick" if task == "pick" else "transport", split=split,
                         seed=seed, max_steps=max_steps, frame_callback=frame_callback)
        target=self.data.body("cap").xpos if task=="unscrew" else self.tube_pos
        self.initial_target_distance=max(.01,float(np.linalg.norm(self.tcp-target)))

    def _customize_scene(self, root, world):
        task = self.lab_task
        root.find("option").set("timestep", ".001")
        root.find("option").set("iterations", "100")
        # Same position servo, with a documented firmer finger spring (2000 N/m).
        # Force still comes only from the two actuated fingers and contacts.
        grip = root.find("actuator/general[@name='actuator8']")
        grip.set("gainprm", ".3137254902 0 0")
        grip.set("biasprm", "0 -2000 -40")
        for geom in root.findall(".//geom"):
            if geom.get("name", "").startswith("tube_") or geom.get("condim") == "6":
                geom.set("solref", ".004 1")
                geom.set("friction", "2 .02 .002")
        # Additional calibrated view; camera extrinsics are sensor calibration,
        # never object-state measurements.
        pos = np.array([.35, .85, .8]); z = pos - [.45, 0, .08]; z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
        ET.SubElement(world, "camera", name="side", pos=numbers(pos), xyaxes=numbers(np.r_[x, np.cross(z, x)]), fovy="45")
        # Oblique overhead avoids the Panda elbow occluding the initial tube.
        top=world.find("camera[@name='top']")
        pos=np.array([.95,-.50,.85]);z=pos-[.45,0,.04];z/=np.linalg.norm(z)
        x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
        top.set("pos",numbers(pos));top.set("xyaxes",numbers(np.r_[x,np.cross(z,x)]));top.set("fovy","45")
        if task in ("insert", "rack_transfer"):
            world.remove(world.find("geom[@name='tray']"))
            self.rack_center[:2] += self.rng.uniform(-.015, .015, 2)
            self._rack(world, "target", self.rack_center, "0.95 0.75 0.12 1")
        if task == "rack_transfer":
            self._rack(world, "source", self.source_rack, "0.2 0.6 0.8 1")
            tube = world.find("body[@name='tube_red']")
            tube.set("pos", numbers([*self.source_rack[:2], self.half_length+self.radius+.008]))
            tube.set("quat", "0.70710678 0 -0.70710678 0")
        if task == "unscrew":
            tube = world.find("body[@name='tube_red']")
            fixture_xy = np.fromstring(tube.get("pos"),sep=" ")[:2]
            tube.set("pos", numbers([*fixture_xy, self.half_length+self.radius+.01]))
            tube.set("quat", "0.70710678 0 -0.70710678 0")
            # Existing decorative cap would overlap the separately simulated cap.
            for geom in list(tube.findall("geom"))[1:]:
                tube.remove(geom)
            eq = root.find("equality")
            if eq is None: eq = ET.SubElement(root, "equality")
            ET.SubElement(eq, "weld", name="tube_fixture", body1="tube_red", solref="0.002 1", solimp=".999 .999 .001")
            ET.SubElement(world, "geom", name="fixture", type="box", pos=numbers([*fixture_xy,.025]),
                          size=".03 .03 .025", rgba=".3 .4 .5 1", contype="0", conaffinity="0")
            self.cap_initial = np.array([*fixture_xy, .13])
            cap = ET.SubElement(world, "body", name="cap", pos=numbers(self.cap_initial))
            for name, kind, axis in (("x","slide","1 0 0"),("y","slide","0 1 0"),("z","slide","0 0 1"),
                                     ("rx","hinge","1 0 0"),("ry","hinge","0 1 0"),("rz","hinge","0 0 1")):
                ET.SubElement(cap, "joint", name="cap_"+name, type=kind, axis=axis,
                              damping=".002", armature=".00001", limited="false")
            ET.SubElement(cap, "geom", name="cap_geom", type="cylinder", size=".018 .012", mass=".012",
                          rgba=".9 .92 .95 1", friction="2 .04 .004", condim="6", solref=".004 1")
            for name in ("x", "y", "rx", "ry"):
                ET.SubElement(eq, "joint", name="thread_"+name, joint1="cap_"+name,
                              polycoef="0 0 0 0 0", solref=".002 1", solimp=".999 .999 .001")
            ET.SubElement(eq, "joint", name="thread_helix", joint1="cap_z", joint2="cap_rz",
                          polycoef=numbers([0, self.thread_pitch/(2*math.pi), 0, 0, 0]), solref=".002 1", solimp=".999 .999 .001")
            contact = root.find("contact")
            if contact is None: contact = ET.SubElement(root, "contact")
            ET.SubElement(contact, "exclude", body1="tube_red", body2="cap")

    @staticmethod
    def _rack(world, name, center, color):
        # Square through-hole bounded by four actual collision walls and a floor.
        inner, outer, bottom, top = .014, .035, .008, .050
        body = ET.SubElement(world, "body", name=name+"_rack", pos=numbers(center))
        ET.SubElement(body, "geom", type="box", pos="0 0 .004", size=numbers([outer,outer,.004]), rgba=color)
        for axis in (0, 1):
            for sign in (-1, 1):
                pos = [0.,0.,(bottom+top)/2]; pos[axis] = sign*(inner+outer)/2
                size = [outer,outer,(top-bottom)/2]; size[axis] = (outer-inner)/2
                ET.SubElement(body, "geom", type="box", pos=numbers(pos), size=numbers(size), rgba=color,
                              friction=".6 .005 .0001", solref=".004 1", solimp=".99 .99 .001")

    def _cap_angle(self):
        return float(self.data.qpos[self.model.joint("cap_rz").qposadr[0]])

    def _body_contacts(self, body_id):
        fingers = set()
        for c in self.data.contact:
            a, b = (int(self.model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            if a == body_id and b in self.finger_ids: fingers.add(b)
            if b == body_id and a in self.finger_ids: fingers.add(a)
        return fingers

    def _update_progress(self):
        super()._update_progress()
        dt = self.model.opt.timestep
        if self.lab_task in ("insert", "rack_transfer"):
            vertical = self._axis()[2] > .97
            # Whole tube cross-section must fit the opening, including tilt.
            # Test the segment inside the rack; the exposed upper tube may lean.
            axis = self._axis()
            zlevels = np.array([.008+self.radius,.050])
            crossings = self.tube_pos[:2] + (zlevels[:,None]-self.tube_pos[2])*axis[:2]/max(axis[2],1e-6)
            inside = np.all(np.abs(crossings-self.rack_center[:2])+self.radius < .015)
            seated = abs(self._bottom()-.008) < .004
            still = np.linalg.norm(self.data.qvel[self.tube_dof:self.tube_dof+6]) < .05
            released = sum(self.data.qpos[7:9]) > .06 and not self._contacts()
            away = np.linalg.norm(self.tcp-self.tube_pos) > .09
            good = self.lifted_once and vertical and inside and seated and still and released and away
            self.insert_seconds = self.insert_seconds+dt if good else 0.
        if self.lab_task == "unscrew":
            angle = self._cap_angle()
            if (not self.thread_released and angle >= self.thread_turns*2*math.pi
                and self.data.body("cap").xpos[2]>=self.cap_initial[2]+.85*self.thread_pitch
                and len(self._body_contacts(self.model.body("cap").id))==2):
                for name in ("x","y","rx","ry","helix"):
                    self.data.eq_active[self.model.equality("thread_"+name).id] = False
                self.thread_released = True
            cap_id = self.model.body("cap").id
            held = len(self._body_contacts(cap_id)) == 2
            separated = self.data.xpos[cap_id, 2] > self.cap_initial[2]+.045
            good = self.thread_released and held and separated
            self.cap_seconds = self.cap_seconds+dt if good else 0.

    def move_pose(self, xyz, rpy):
        start = self.tcp
        current = self.data.site_xmat[self.tcp_id].reshape(3,3).copy()
        target = rpy_matrix(rpy)
        if np.linalg.norm(current[:,2]-[0,0,-1]) < .02 and np.linalg.norm(target[:,2]-[0,0,-1]) < 1e-6:
            return self._move(xyz, rpy[2])
        quat = np.empty(4); velocity = np.empty(3)
        mujoco.mju_mat2Quat(quat, (target@current.T).ravel())
        if quat[0] < 0: quat *= -1
        mujoco.mju_quat2Vel(velocity, quat, 1.)
        angle = np.linalg.norm(velocity)
        count = max(1, math.ceil(np.linalg.norm(xyz-start)/.008), math.ceil(angle/.07))
        for i in range(1,count+1):
            f = i/count
            rotq = np.empty(4); rot = np.empty(9)
            if angle > 1e-9:
                mujoco.mju_axisAngle2Quat(rotq, velocity/angle, angle*f)
                mujoco.mju_quat2Mat(rot, rotq)
                rotation = rot.reshape(3,3)@current
            else: rotation = current
            q = self._solve_ik(start+f*(xyz-start), rotation)
            if q is None:
                for joint1 in (-.8,0.,.8):
                    for joint7 in (-1.8,0.,1.8):
                        initial = HOME.copy(); initial[0] = joint1; initial[6] = joint7
                        q = self._solve_ik(start+f*(xyz-start), rotation, initial)
                        if q is not None: break
                    if q is not None: break
            if q is None: return {"accepted": True, "reached": False, "reason":"ik_failed_after_partial_motion"}
            previous = self.data.ctrl[:7].copy()
            substeps = max(1, math.ceil(np.max(np.abs(q-previous))/.12))
            for j in range(1,substeps+1):
                self.data.ctrl[:7] = previous+(q-previous)*j/substeps
                self._advance(.08)
        self._advance(.3)
        err = float(np.linalg.norm(self.tcp-xyz))
        angular = float(np.arccos(np.clip((np.trace(target@self.data.site_xmat[self.tcp_id].reshape(3,3).T)-1)/2,-1,1)))
        return {"accepted":True,"reached":err<.008 and angular<.08,"position_error_m":round(err,5),"orientation_error_rad":round(angular,5)}

    def step(self, action):
        if action.get("action")=="move" and "rpy" not in action:
            action={"action":"invalid_move_requires_xyz_and_rpy"}
        if action.get("action") != "move" or "rpy" not in action:
            return super().step(action)
        if self.ended: raise RuntimeError("Episode already ended")
        self.steps += 1
        try:
            if set(action) != {"action","xyz","rpy"}: raise ValueError("move requires exactly action, xyz, rpy")
            xyz, rpy = np.asarray(action["xyz"],float), np.asarray(action["rpy"],float)
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz,rpy]).all():
                raise ValueError("xyz and rpy must be finite vectors of length 3")
            if np.any(xyz < [.20,-.4,.009]) or np.any(xyz > [.65,.4,.5]) or np.any(np.abs(rpy)>math.pi+1e-6):
                raise ValueError("TCP outside x[.2,.65],y[-.4,.4],z[.009,.5] or rpy outside [-pi,pi]")
            feedback = self.move_pose(xyz,rpy)
        except (KeyError,TypeError,ValueError) as exc:
            self.invalid_actions += 1; feedback = {"accepted":False,"reason":str(exc)}
        if self.steps >= self.max_steps:
            self._advance(.5); self.ended = True
        return self.observe(), feedback, self.ended

    def proprioception(self):
        return {"tcp":np.round(self.tcp,4).tolist(),
                "tcp_rpy":np.round(matrix_rpy(self.data.site_xmat[self.tcp_id].reshape(3,3)),4).tolist(),
                "gripper_opening":round(float(sum(self.data.qpos[7:9])),4),
                "steps_left":self.max_steps-self.steps}

    @property
    def instruction(self):
        return {
            "pick":"Pick up the red test tube and hold its whole body at least 6 cm above its starting support for 0.35 seconds.",
            "transport":"Pick up the red test tube, carry it into the teal tray, release it and withdraw. Leave it stable for 0.4 seconds.",
            "insert":"Pick up the horizontal red test tube, turn it cap-up and insert it into the yellow rack hole. Release and withdraw; leave it upright and stable for 0.4 seconds.",
            "unscrew":"The red tube body is clamped in a fixture. Grip the white cap, unscrew it counterclockwise viewed from above by one full turn, then lift it at least 4.5 cm and hold for 0.35 seconds. Regrasp if needed. Thread lead is 6 mm per turn.",
            "rack_transfer":"Move the upright red test tube from the blue rack into the yellow rack. Release and withdraw; leave it upright and stable for 0.4 seconds.",
        }[self.lab_task]

    def result(self):
        result = super().result()
        target=self.data.body("cap").xpos if self.lab_task=="unscrew" else self.tube_pos
        approach=.15*np.clip(1-np.linalg.norm(self.tcp-target)/getattr(self,"initial_target_distance",self.initial_distance),0,1)
        held=len(self._body_contacts(self.model.body("cap").id))==2 if self.lab_task=="unscrew" else bool(result["held"])
        progress=float(approach)
        if held:progress=max(progress,.2)
        if self.lab_task!="unscrew" and held:
            progress=max(progress,.2+.35*np.clip((self._bottom()-self.initial_bottom)/.06,0,1))
        if self.lifted_once:progress=max(progress,.55)
        if self.lab_task=="transport" and self.lifted_once:
            progress=max(progress,.55+.35*np.clip(1-np.linalg.norm(self.tube_pos[:2]-self.tray[:2])/.4,0,1))
        success=result["success"]
        if self.lab_task in ("insert","rack_transfer"):
            success = self.insert_seconds >= .4
            if self.lifted_once:
                aligned=max(0.,float(self._axis()[2]))
                distance=float(np.linalg.norm(self.tube_pos[:2]-self.rack_center[:2]))
                progress=max(progress,.55+.1*aligned+.1*max(0.,1-distance/.4))
                if aligned>.97 and distance<.012:
                    progress=max(progress,.75+.15*np.clip((.05-self._bottom())/.042,0,1))
            result.update(success=success, insertion_stable_seconds=round(self.insert_seconds,3))
        elif self.lab_task == "unscrew":
            success = self.cap_seconds >= .35
            progress = max(progress,.2+.5*np.clip(self._cap_angle()/(2*math.pi),0,1) if held else .5*np.clip(self._cap_angle()/(2*math.pi),0,1))
            if self.thread_released: progress = .7+.2*np.clip((self.data.body("cap").xpos[2]-.13)/.045,0,1)
            result.update(success=success, thread_released=self.thread_released,
                          cap_turns=round(self._cap_angle()/(2*math.pi),4),
                          cap_position=np.round(self.data.body("cap").xpos,4).tolist(),
                          cap_held=len(self._body_contacts(self.model.body("cap").id))==2,
                          cap_hold_seconds=round(self.cap_seconds,3))
        result["reward"] = round((5. if success else min(.9,float(progress)))-.01*self.steps-.05*self.invalid_actions,5)
        result["reward_terms"]={"approach":round(float(approach),5),"grasped":held,
                                "terminal_progress":round(min(.9,float(progress)),5),"success_bonus":5. if success else 0.,
                                "length_penalty":round(.01*self.steps,5),"invalid_penalty":round(.05*self.invalid_actions,5)}
        return result
