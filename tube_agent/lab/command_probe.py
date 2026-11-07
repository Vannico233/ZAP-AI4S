"""Execute explicit JSON commands at a fixed scene seed; never training data.

Without --commands, enter one JSON object per line in the terminal. With it,
read a JSON array from a file. Every step saves images, pose, feedback and reward.
"""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image
from .env import TASKS
from .experiment import write_json
from .run import state
from .zap_env import ZapLabEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=TASKS,required=True);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--commands",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--no-render",action="store_true");p.add_argument("--max-steps",type=int,default=60)
    a=p.parse_args()
    if not 0<=a.seed<100000 or a.max_steps<1:p.error("Invalid scene seed or step budget")
    instructions=json.loads(a.commands.read_text(encoding="utf-8")) if a.commands else None
    if instructions is not None and (not isinstance(instructions,list) or any(not isinstance(v,dict) for v in instructions)):
        p.error("--commands must contain a JSON array of action objects")
    a.output.mkdir(parents=True,exist_ok=False)
    split="test" if a.task=="rack_transfer" else "train"
    env=ZapLabEnv(a.task,split,a.seed,a.max_steps);cameras=None
    def observe():
        if cameras:
            directory=a.output/f"view_{env.steps:03d}";directory.mkdir(exist_ok=True)
            for frame in cameras.capture():
                Image.fromarray(frame["rgb"]).save(directory/(frame["calibration"]["name"]+".png"))
        print(json.dumps({"step":env.steps,"robot":env.proprioception(),"judge":env.result()}),flush=True)
    try:
        if not a.no_render:
            from .vision import Cameras
            cameras=Cameras(env)
        observe()
        stream=iter(instructions) if instructions is not None else (json.loads(line) for line in sys.stdin if line.strip())
        with (a.output/"trajectory.jsonl").open("w",encoding="utf-8") as log:
            for action in stream:
                if env.ended:break
                feedback=env.step(action)[1]
                row={"step":env.steps,"action":action,"feedback":feedback,"judge_after":env.result(),
                     "state_after":state(env),"action_origin":"manual_command"}
                log.write(json.dumps(row)+"\n");log.flush()
                print(json.dumps({"action":action,"feedback":feedback}),flush=True);observe()
        write_json(a.output/"result.json",{**env.result(),"task":a.task,"split":split,"seed":a.seed,
            "max_steps":a.max_steps,"environment_version":env.VERSION,"policy":"manual_command_debug",
            "demonstration":{"source":"explicit user/engineer commands"},
            "observation_mode":"manual_camera_debug","training_eligible":False,"error":None})
    finally:
        if cameras:cameras.close()
        env.close()


if __name__=="__main__":main()
