"""Seeded physical calibration of the semantic pose interface; NOT model data."""
import argparse
import json
from .env import TASKS, rpy_matrix
from .experiment import write_json
from .sanity import commands
from .zap_env import ZapLabEnv
from .zap_pose import move_action
from pathlib import Path


def convert(action):
    if action["action"]=="move":return move_action(action["xyz"],rpy_matrix(action["rpy"]))
    if action["action"]=="gripper":return {"action":"set_gripper","opening":action["opening"]/.08}
    if action["action"]=="finish":return {"action":"done"}
    raise ValueError("Unexpected diagnostic command")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);results=[]
    for task in TASKS:
        env=ZapLabEnv(task,"test" if task=="rack_transfer" else "train",a.seed,60)
        trace=[]
        try:
            for original in commands(env):
                action=convert(original);feedback=env.step(action)[1]
                trace.append({"step":env.steps,"action":action,"feedback":feedback,"judge":env.result()})
            result={"task":task,"seed":a.seed,"policy":"scripted_pose_calibration_NOT_model",
                    "demonstration":"privileged physics diagnostic; never admit to training",**env.result()}
            write_json(a.output/task/"trajectory.json",trace);results.append(result)
            print(json.dumps(result),flush=True)
        finally:env.close()
    passed=all(r["success"] and not r["invalid_actions"] for r in results)
    write_json(a.output/"report.json",{"passed":passed,"tasks":results,"training_eligible":False})
    if not passed:raise SystemExit("Semantic pose physics check failed; model sampling must not start")


if __name__=="__main__":main()
