"""Visual fixed-seed sampling and deterministic model-prefix tree search."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from tube_agent.run import summarize
from .env import LabEnv,TASKS
from .policy import HttpGenerator,VisualPolicy,execute,SYSTEM,POLICY_VERSION
from .vision import Observer,SamClient,PERCEPTION_VERSION
from .experiment import write_json

def state(env):
    return np.r_[env.data.qpos,env.data.qvel,env.data.ctrl,env.data.eq_active,
                 env.data.time,env.lift_seconds,env.place_seconds,env.insert_seconds,env.cap_seconds,
                 env.lifted_once,env.thread_released].tolist()

def episode(task,split,seed,attempt,directory,generator,segmenter,temperature=.8,max_steps=40,prefix=None,save_frames=True,log_progress=False,action_protocol="rpy_rounding_v2"):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    env=observer=None;started=time.monotonic();records=[];error=None;feedback=None
    policy=VisualPolicy(generator,temperature,seed*1009+attempt,action_protocol)
    try:
        from .zap_env import make_env
        env=make_env(task,split,seed,max_steps,action_protocol);observer=Observer(env,segmenter)
        for saved in prefix or []:
            feedback=execute(env,observer,policy,saved["action"])
            if feedback!=saved["feedback"] or not np.allclose(state(env),saved["state_after"],atol=1e-9,rtol=0):
                raise RuntimeError("Prefix replay diverged; refusing to continue from a different state")
            policy.history=copy.deepcopy(saved["history_after"])
            policy.remember(saved["action"],feedback,saved["step"])
            if "initial_visual_reference" in saved:
                policy.initial_visual_reference=copy.deepcopy(saved["initial_visual_reference"])
            records.append(copy.deepcopy(saved))
        with (directory/"trajectory.jsonl").open("w",encoding="utf-8") as log:
            for row in records:log.write(json.dumps(row)+"\n")
            while not env.ended:
                observation,images=observer.observe(directory/f"view_{env.steps:03d}" if save_frames else None)
                action,raw,messages=policy.act(env.instruction,observation,images,feedback,env.steps)
                feedback=execute(env,observer,policy,action)
                policy.remember(action,feedback,env.steps)
                row={"step":env.steps,"action":action,"raw_response":raw,"messages":messages,
                     "feedback":feedback,"state_after":state(env),"history_after":copy.deepcopy(policy.history),
                     "judge_after":env.result(),"origin_attempt":attempt}
                row["perception"]=copy.deepcopy(observer.last_audit)
                row["response_format"]=copy.deepcopy(policy.last_parse)
                row["initial_visual_reference"]=copy.deepcopy(policy.initial_visual_reference)
                row["camera_calibration"]=copy.deepcopy(observation.get("cameras",[]))
                if hasattr(generator,"last_response_ids"):
                    row["response_ids"]=list(generator.last_response_ids)
                records.append(row);log.write(json.dumps(row)+"\n");log.flush()
                if log_progress:
                    print(json.dumps({"event":"action","task":task,"attempt":attempt,"step":env.steps,
                                      "action":action,"feedback":feedback,"reward":row["judge_after"]["reward"],
                                      "reward_terms":row["judge_after"]["reward_terms"],
                                      "success":row["judge_after"]["success"]}),flush=True)
        result=env.result()
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"
        result=env.result() if env else {"steps":0,"invalid_actions":0}
        result.update(success=False,reward=0.)
    finally:
        if observer:observer.close()
        if env:env.close()
    result.update(task=task,split=split,seed=seed,attempt=attempt,policy="model",model="qwen-tube",
                  observation_mode="rgbd_sam3.1_detector",environment_version=env.VERSION if env else None,
                  action_protocol=action_protocol,
                  perception_version=PERCEPTION_VERSION,
                  policy_version=policy.version,
                  leading_prose_responses=sum(r.get("response_format",{}).get("mode")=="leading_prose" for r in records),
                  parse_failures=sum(r.get("response_format",{}).get("mode")=="invalid" for r in records),
                  missing_detection_observations=sum(bool(r.get("perception",{}).get("missing_queries")) for r in records),
                  prompt_sha256=hashlib.sha256(policy.system.encode()).hexdigest(),demonstration=None,
                  max_steps=max_steps,search_prefix_length=len(prefix or []),error=error,seconds=round(time.monotonic()-started,2))
    result["generator_provenance"]=getattr(generator,"provenance",None)
    write_json(directory/"result.json",result)
    return result,records

class PrefixTree:
    """UCB selection over deterministic prefixes; only model-generated actions."""
    def __init__(self,max_steps=40,max_nodes=128):
        self.nodes={"root":{"prefix":[],"visits":0,"wins":0,"progress":0.}}
        self.round=0;self.max_steps=max_steps;self.max_nodes=max_nodes
        self.selected="root";self.failed_expansions={};self.blocked=set();self.empty_close_cutoffs=0
    @staticmethod
    def key(rows):return hashlib.sha256(json.dumps([r["action"] for r in rows],sort_keys=True).encode()).hexdigest()
    def select(self):
        self.round+=1
        def score(n):return n["wins"]/(n["visits"]+1)+n["progress"]+.45*math.sqrt(math.log(self.round+1)/(n["visits"]+1))
        key=max(self.nodes,key=lambda k:score(self.nodes[k]))
        self.selected=key
        return self.nodes[key]["prefix"]
    def update(self,rows,result):
        if result.get("error"):return
        prefix_length=result.get("search_prefix_length")
        selected=(self.key(rows[:prefix_length]) if prefix_length else "root") if prefix_length is not None else self.selected
        if selected!="root":
            self.failed_expansions[selected]=0 if result["success"] else self.failed_expansions.get(selected,0)+1
            if self.failed_expansions[selected]>=3:
                self.blocked.add(selected)
                for key,node in list(self.nodes.items()):
                    if key!="root" and any(self.key(node["prefix"][:n])==selected for n in range(1,len(node["prefix"])+1)):
                        del self.nodes[key]
        self.nodes["root"]["visits"]+=1
        self.nodes["root"]["wins"]+=int(result["success"])
        for length,row in enumerate(rows,1):
            if row["feedback"].get("accepted") is False:break
            kind=row["action"]["action"];judge=row["judge_after"]
            closing=(kind=="gripper" and row["action"].get("opening",1.)<=.003) or (kind=="set_gripper" and row["action"].get("opening",1.)<=.04)
            # This search decision uses the reward judge, never policy input.
            # Resample before an empty closure, rather than replaying it and
            # the following empty-hand lift. No replacement action is supplied.
            if closing and judge.get("cap_held",judge.get("held")) is False:
                self.empty_close_cutoffs+=1;break
            if row["action"]["action"] not in ("move","gripper","move_to","set_gripper"):continue
            if row["action"]["action"]=="move_to" and not row["feedback"].get("reached"):continue
            if length>=self.max_steps:continue
            prefix=rows[:length];key=self.key(prefix)
            if key in self.blocked:break
            # Immutable recorded rows can share large embedded images across nodes.
            n=self.nodes.setdefault(key,{"prefix":list(prefix),"visits":0,"wins":0,
                                        "progress":min(.9,max(0.,row["judge_after"]["reward"]+.01*length))})
            n["visits"]+=1;n["wins"]+=int(result["success"])
        while len(self.nodes)>self.max_nodes:
            worst=min((k for k in self.nodes if k!="root"),key=lambda k:
                      (self.nodes[k]["wins"]/(1+self.nodes[k]["visits"])+self.nodes[k]["progress"]))
            del self.nodes[worst]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=TASKS,required=True);p.add_argument("--split",choices=("train","validation","test","ood"),default="train")
    p.add_argument("--seed",type=int,default=0);p.add_argument("--attempts",type=int,default=8)
    p.add_argument("--search",action="store_true");p.add_argument("--max-steps",type=int,default=40)
    p.add_argument("--action-protocol",choices=("rpy_rounding_v2","zap_pose_v1"),default="rpy_rounding_v2")
    p.add_argument("--temperature",type=float,default=.8);p.add_argument("--output",type=Path,default=Path("runs/lab"))
    p.add_argument("--base-url",default="http://127.0.0.1:8000/v1");p.add_argument("--sam-url",default="http://127.0.0.1:8001")
    a=p.parse_args()
    if a.attempts<1 or not 0<=a.seed<100000 or a.max_steps<1:p.error("Invalid attempt count, seed or step budget")
    if a.search and a.split!="train":p.error("Search is a training-data mining operation; evaluate without --search")
    if a.task=="rack_transfer" and a.split=="train":p.error("rack_transfer is held out")
    generator=HttpGenerator(a.base_url);segmenter=SamClient(a.sam_url)
    try:
        model_health=generator.health();sam_health=segmenter.health()
        run=a.output/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_"+a.task)
        run.mkdir(parents=True,exist_ok=False)
        (run/"config.json").write_text(json.dumps({**vars(a),"output":str(a.output),"model_service":model_health,"sam_service":sam_health,
                                                  "mode":"prefix_search" if a.search else "fixed_seed_sampling"},indent=2),encoding="utf-8")
        tree=PrefixTree(a.max_steps);results=[]
        for attempt in range(a.attempts):
            prefix=tree.select() if a.search else None
            result,rows=episode(a.task,a.split,a.seed,attempt,run/f"try_{attempt:03d}",generator,segmenter,
                                a.temperature,a.max_steps,prefix,action_protocol=a.action_protocol)
            results.append(result)
            if a.search:tree.update(rows,result)
            summary={**summarize(results),"scene_count":1,"sampling_mode":"prefix_search" if a.search else "independent_rollouts",
                     "perception_version":PERCEPTION_VERSION,
                     "missing_detection_observations":sum(r["missing_detection_observations"] for r in results),
                     "interpretation":"Search yield is not an independent policy success rate" if a.search else "Same scene, different sampling RNG"}
            (run/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
            print(json.dumps(result),flush=True)
            if result["error"]:raise RuntimeError(result["error"]+"; stopped instead of counting more service failures")
        print("Saved: "+str(run));print(json.dumps(summary,indent=2))
    finally:generator.close();segmenter.close()

if __name__=="__main__":main()
