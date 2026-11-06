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
from .policy import HttpGenerator,VisualPolicy,execute,SYSTEM
from .vision import Observer,SamClient,PERCEPTION_VERSION
from .experiment import write_json

def state(env):
    return np.r_[env.data.qpos,env.data.qvel,env.data.ctrl,env.data.eq_active,
                 env.data.time,env.lift_seconds,env.place_seconds,env.insert_seconds,env.cap_seconds,
                 env.lifted_once,env.thread_released].tolist()

def episode(task,split,seed,attempt,directory,generator,segmenter,temperature=.8,max_steps=40,prefix=None,save_frames=True,log_progress=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    env=observer=None;started=time.monotonic();records=[];error=None;feedback=None
    policy=VisualPolicy(generator,temperature,seed*1009+attempt)
    try:
        env=LabEnv(task,split,seed,max_steps);observer=Observer(env,segmenter)
        for saved in prefix or []:
            feedback=execute(env,observer,policy,saved["action"])
            if feedback!=saved["feedback"] or not np.allclose(state(env),saved["state_after"],atol=1e-9,rtol=0):
                raise RuntimeError("Prefix replay diverged; refusing to continue from a different state")
            policy.history=copy.deepcopy(saved["history_after"])
            records.append(copy.deepcopy(saved))
        with (directory/"trajectory.jsonl").open("w",encoding="utf-8") as log:
            for row in records:log.write(json.dumps(row)+"\n")
            while not env.ended:
                observation,images=observer.observe(directory/f"view_{env.steps:03d}" if save_frames else None)
                action,raw,messages=policy.act(env.instruction,observation,images,feedback,env.steps)
                feedback=execute(env,observer,policy,action)
                row={"step":env.steps,"action":action,"raw_response":raw,"messages":messages,
                     "feedback":feedback,"state_after":state(env),"history_after":copy.deepcopy(policy.history),
                     "judge_after":env.result(),"origin_attempt":attempt}
                row["perception"]=copy.deepcopy(observer.last_audit)
                if hasattr(generator,"last_response_ids"):
                    row["response_ids"]=list(generator.last_response_ids)
                records.append(row);log.write(json.dumps(row)+"\n");log.flush()
                if log_progress:
                    print(json.dumps({"event":"action","task":task,"attempt":attempt,"step":env.steps,
                                      "action":action,"feedback":feedback,"reward":row["judge_after"]["reward"],
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
                  observation_mode="rgbd_sam3.1_detector",environment_version=LabEnv.VERSION,
                  action_protocol=LabEnv.ACTION_PROTOCOL,
                  perception_version=PERCEPTION_VERSION,
                  missing_detection_observations=sum(bool(r.get("perception",{}).get("missing_queries")) for r in records),
                  prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),demonstration=None,
                  max_steps=max_steps,search_prefix_length=len(prefix or []),error=error,seconds=round(time.monotonic()-started,2))
    result["generator_provenance"]=getattr(generator,"provenance",None)
    write_json(directory/"result.json",result)
    return result,records

class PrefixTree:
    """UCB selection over deterministic prefixes; only model-generated actions."""
    def __init__(self,max_steps=40,max_nodes=128):
        self.nodes={"root":{"prefix":[],"visits":0,"wins":0,"progress":0.}};self.round=0;self.max_steps=max_steps;self.max_nodes=max_nodes
    @staticmethod
    def key(rows):return hashlib.sha256(json.dumps([r["action"] for r in rows],sort_keys=True).encode()).hexdigest()
    def select(self):
        self.round+=1
        def score(n):return n["wins"]/(n["visits"]+1)+n["progress"]+.45*math.sqrt(math.log(self.round+1)/(n["visits"]+1))
        key=max(self.nodes,key=lambda k:score(self.nodes[k]))
        return self.nodes[key]["prefix"]
    def update(self,rows,result):
        if result.get("error"):return
        self.nodes["root"]["visits"]+=1
        self.nodes["root"]["wins"]+=int(result["success"])
        for length,row in enumerate(rows,1):
            if row["feedback"].get("accepted") is False:break
            if row["action"]["action"] not in ("move","gripper"):continue
            if length>=self.max_steps:continue
            prefix=rows[:length];key=self.key(prefix)
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
                                a.temperature,a.max_steps,prefix)
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
