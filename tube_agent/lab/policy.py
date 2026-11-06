"""Image-grounded tools. No demonstrations, oracle state, or hidden fallback."""
import copy
import json
import httpx
from tube_agent.policy import parse_action
from .vision import image_url

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
Known apparatus dimensions: tube nominal radius .012 m, overall length .114 m;
square rack hole .028 m wide, floor z=.008 m, rim z=.050 m; tray top z=.004 m.
The rack hole is at the rack's horizontal center. Estimate its position from vision.
The screw task uses a fixed tube fixture and a right-hand, one-turn thread with
6 mm lead. Positive world-Z rotation unscrews; regrasp within arm joint limits.
Use the current instruction, images and measured feedback to choose each action.
Submit finish when the requested task is complete. Steps_left is your remaining budget.
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
    def __init__(self,generator,temperature=.8,seed=0):
        self.generator,self.temperature,self.seed=generator,temperature,seed
        self.history=[];self.plan=""
    def act(self,instruction,observation,images,feedback,step):
        content=json.dumps({"instruction":instruction,"observation":observation,
                            "last_action_feedback":feedback,"remaining_plan":self.plan})
        messages=[{"role":"system","content":SYSTEM}]+copy.deepcopy(self.history[-4:])
        messages.append({"role":"user","content":[{"type":"text","text":content}]+
                         [{"type":"image_url","image_url":{"url":image_url(i)}} for i in images]})
        raw=self.generator(messages,self.seed+step*100003,self.temperature)
        if not isinstance(raw,str):raise ValueError("Model response must be text")
        self.history.extend([{"role":"user","content":content},{"role":"assistant","content":raw}])
        try:action=parse_action(raw)
        except (ValueError,TypeError):action={"action":"invalid_model_output"}
        return action,raw,messages

def execute(env,observer,policy,action):
    kind=action.get("action")
    if kind not in ("change_query","update_todo"):return env.step(action)[1]
    if env.ended:raise RuntimeError("Episode ended")
    env.steps+=1
    try:
        if kind=="change_query":
            q=action.get("queries")
            if set(action)!={"action","queries"} or not isinstance(q,list) or not 1<=len(q)<=8 or any(not isinstance(v,str) or not 1<=len(v)<=120 for v in q):
                raise ValueError("change_query requires 1-8 text queries, each 1-120 characters")
            observer.queries=list(q)
        else:
            plan=action.get("plan")
            if set(action)!={"action","plan"} or not isinstance(plan,str) or len(plan)>1200:
                raise ValueError("update_todo requires a text plan up to 1200 characters")
            policy.plan=plan
        feedback={"accepted":True}
    except ValueError as exc:
        env.invalid_actions+=1;feedback={"accepted":False,"reason":str(exc)}
    if env.steps>=env.max_steps:env._advance(.5);env.ended=True
    return feedback
