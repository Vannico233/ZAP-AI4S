"""Image-grounded tools. No demonstrations, oracle state, or hidden fallback."""
import copy
import json
import re
import httpx
from .vision import image_url
from tube_agent import interface
from tube_agent.interface import execute_action as execute

POLICY_VERSION = "visual_action_evidence_v3"


def parse_visual_action(raw):
    """Accept exactly one trailing JSON object; never choose among alternatives."""
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if clean.startswith("```") and clean.endswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean).strip()
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise ValueError("Duplicate JSON key")
            value[key] = item
        return value
    def constant(value): raise ValueError("Non-finite JSON number")
    decoder = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=constant)
    start = clean.find("{")
    if start < 0 or any(c in clean[:start] for c in "{}[]"):
        raise ValueError("Expected one unambiguous JSON action")
    action, end = decoder.raw_decode(clean[start:])
    if not isinstance(action, dict) or clean[start+end:].strip():
        raise ValueError("Multiple actions, trailing prose or truncated JSON")
    return action, {"version": "single_trailing_json_v1", "mode": "json" if start==0 else "leading_prose",
                    "leading_characters": start}

SYSTEM = """You operate a Franka Panda gripper in a laboratory simulation.
Return exactly one JSON action per turn, with no prose or markdown:
{"action":"move","xyz":[x,y,z],"rpy":[roll,pitch,yaw]}
{"action":"gripper","opening":metres}
{"action":"change_query","queries":["object description", "another description"]}
{"action":"update_todo","plan":"short remaining plan"}
{"action":"finish"}
XYZ is the fingertip-pad midpoint (TCP), in absolute world metres; table z=0.
Workspace: x=[.20,.65], y=[-.4,.4], z=[.009,.5]. Angles are radians in [-pi,pi].
RPY means Rz(yaw) @ Ry(pitch) @ Rx(roll). [pi,0,0] points fingers downward,
with closing along world Y. A move preserves the gripper opening.
Opening 0 closes; .08 opens fully. Motion feedback describes the robot, not task success.
Input contains annotated RGB images, depth-derived SAM object measurements and
robot proprioception. Object coordinates are estimates of VISIBLE surfaces, NOT
simulator ground truth. Occlusion can hide objects. Two views may report the same
object. Principal axes have no semantic sign; use RGB to identify the cap end.
Query again if needed. No object is automatically attached to the gripper.
Missing_queries means SAM returned no usable mask in either view; it does not
mean the object is absent. Use RGB and change_query to refine its description.
Segmentation_query records the actual phrase used for a candidate mask, possibly
an appearance-based retry. Check RGB: a mask is not proof of correct identity.
Known apparatus dimensions: tube nominal radius .012 m, overall length .114 m;
square rack hole .028 m wide, floor z=.008 m, rim z=.050 m; tray top z=.004 m.
The rack hole is at the rack's horizontal center. Estimate its position from vision.
The screw task uses a fixed tube fixture and a right-hand, one-turn thread with
6 mm lead. Positive world-Z rotation unscrews; regrasp within arm joint limits.
Use the current instruction, images and measured feedback to choose each action.
Keep fingers downward [3.14159265,0,yaw] for approaching a horizontal tube.
Align the gripper closing direction across the tube, not along its long axis.
Start above the object with open fingers, descend, close, then lift with clearance.
For a horizontal tube, yaw can follow atan2(axis_y,axis_x), modulo pi;
choose the equivalent yaw with smaller absolute value to reduce wrist rotation.
Surface centers are not grasp centers: use both views and the known radius.
For cap-up insertion, change orientation above the rack before descending;
the TCP is not the tube bottom. Account for the held object's offset.
For unscrewing, use several positive yaw increments and regrasp when necessary.
Keep the cap between the fingers; lift only after completing the full turn.
If reached=false, the motion may have stopped part way. Read current TCP and
revise the waypoint or orientation instead of repeating the same failed pose.
Opening changes do not attach an object. Check subsequent images for motion.
Lift with a margin beyond the task's minimum, then verify the object follows.
Submit finish when the requested task is complete. Steps_left is your remaining budget.
Never describe a grasp as established merely because TCP z increased.
The robot starts with OPEN fingers. A move NEVER closes the fingers.
After positioning around the tube, issue gripper opening 0 BEFORE attempting a lift.
If opening is still near .08, do not claim the tube is held. If opening becomes
nearly zero, the fingers may have closed on empty space: inspect both images.
Use the recorded initial visual reference for the starting support height;
do not repeatedly add 6 cm to the current TCP or current object height.
For pick, lift once with clearance, inspect whether the tube moved with the hand,
then finish while keeping the fingers closed. Do not keep raising an empty hand.
Action_history is executed command feedback, not a demonstration. Treat your
previous prose as unverified; current images and measurements take precedence.
"""

class HttpGenerator:
    def __init__(self,base_url="http://127.0.0.1:8000/v1",model="qwen-tube"):
        self.url=base_url.rstrip("/");self.model=model
        self.client=httpx.Client(timeout=240.,trust_env=False)
    def health(self):
        r=self.client.get(self.url.removesuffix("/v1")+"/health");r.raise_for_status()
        if not r.json().get("ready"):raise RuntimeError("Qwen service is not ready")
        return r.json()
    def __call__(self,messages,seed,temperature):
        r=self.client.post(self.url+"/chat/completions",json={"model":self.model,"messages":messages,
                           "max_tokens":256,"temperature":temperature,"seed":seed,
                           "chat_template_kwargs":{"enable_thinking":False}})
        r.raise_for_status();return r.json()["choices"][0]["message"]["content"]
    def close(self):self.client.close()

class VisualPolicy:
    def __init__(self,generator,temperature=.8,seed=0,action_protocol="rpy_rounding_v2"):
        self.generator,self.temperature,self.seed=generator,temperature,seed
        if action_protocol == interface.PROTOCOL:
            self.system, self.version = interface.SYSTEM, interface.PROTOCOL
        elif action_protocol=="zap_pose_v1":
            from .zap_policy import SYSTEM as prompt, POLICY_VERSION as version
            self.system,self.version=prompt,version
        else:self.system,self.version=SYSTEM,POLICY_VERSION
        self.history=[];self.plan=""
        self.action_memory=[];self.initial_visual_reference=None;self.last_parse=None
    def remember(self,action,feedback,step):
        self.action_memory.append({"step":step,"action":copy.deepcopy(action),
                                   "feedback":copy.deepcopy(feedback)})
    def act(self,instruction,observation,images,feedback,step):
        if self.initial_visual_reference is None:
            self.initial_visual_reference=copy.deepcopy(observation.get("detections",[]))
        # Extrinsics are already applied to each view's measured object points.
        policy_observation={k:v for k,v in observation.items() if k!="cameras"}
        def describe(opening):
            return ("wide_open_not_grasp_evidence" if opening>=.06 else
                    "nearly_closed_may_be_empty" if opening<=.003 else "partial_opening_not_proof_of_grasp")
        if observation.get("coordinate_frame") == interface.FRAME:
            fingers = {name: describe(arm["gripper_opening_m"]) for name, arm in observation["arms"].items()}
        else:
            fingers = describe(observation.get("gripper_opening", 0.08))
        content=json.dumps({"instruction":instruction,"observation":policy_observation,
                            "initial_visual_reference":self.initial_visual_reference,
                            "gripper_interpretation":fingers,"action_history":self.action_memory,
                            "last_action_feedback":feedback,"remaining_plan":self.plan})
        messages=[{"role":"system","content":self.system}]+copy.deepcopy(self.history[-4:])
        messages.append({"role":"user","content":[{"type":"text","text":content}]+
                         [{"type":"image_url","image_url":{"url":image_url(i)}} for i in images]})
        raw=self.generator(messages,self.seed+step*100003,self.temperature)
        if not isinstance(raw,str):raise ValueError("Model response must be text")
        try:action,self.last_parse=parse_visual_action(raw)
        except (ValueError,TypeError) as exc:
            action={"action":"invalid_model_output"}
            self.last_parse={"version":"single_trailing_json_v1","mode":"invalid","reason":str(exc)}
        # Keep the original output in raw_response/response_ids for auditing and
        # likelihood training; do not feed hallucinated prose back as fact.
        self.history.extend([{"role":"user","content":json.dumps({"instruction":instruction,"step":step})},
                             {"role":"assistant","content":json.dumps(action)}])
        return action,raw,messages
