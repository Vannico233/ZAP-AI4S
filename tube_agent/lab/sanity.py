"""Privileged scripted diagnostics. NEVER export these as model training data."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .env import LabEnv, TASKS

def commands(env):
    x,y,z = env.tube_pos
    yaw = math.atan2(env._axis()[1],env._axis()[0])
    if env.lab_task == "insert": yaw = (yaw+2*math.pi)%(2*math.pi)-math.pi
    def move(xyz, rpy=None): return {"action":"move","xyz":list(xyz),"rpy":rpy or [math.pi,0,yaw]}
    close, opening = {"action":"gripper","opening":0.}, {"action":"gripper","opening":.08}
    if env.lab_task == "unscrew":
        pos = env.cap_initial
        yield move([*pos[:2],.25],[math.pi,0,-1.5])
        yield move(pos,[math.pi,0,-1.5]); yield close
        # Two partial turns with one regrasp; geometry commands, no state edits.
        for turn in range(3):
            for a in np.linspace(-1.5,1.5,7)[1:]:
                height = .13 + min((turn*3+a+1.5)/(2*math.pi),1)*.006
                yield move([*pos[:2],height],[math.pi,0,float(a)])
                if env.thread_released: break
            if env.thread_released: break
            if turn < 2:
                yield opening
                yield move([*pos[:2],.13+min((turn+1)*3/(2*math.pi),1)*.006],[math.pi,0,-1.5])
                yield close
        yield move([*pos[:2],.23],[math.pi,0,float(a)])
    elif env.lab_task == "rack_transfer":
        yield move([x,y,.23],[math.pi,0,0])
        yield move([x,y,.1],[math.pi,0,0]); yield close
        yield move([x,y,.25],[math.pi,0,0])
        yield move([*env.rack_center[:2],.25],[math.pi,0,0])
        yield move([*env.rack_center[:2],.1],[math.pi,0,0]); yield opening
        yield move([*env.rack_center[:2],.26],[math.pi,0,0])
    else:
        offset = 0.
        gx,gy = np.array([x,y])+offset*env._axis()[:2]
        yield move([gx,gy,.22]); yield move([gx,gy,z+.003]); yield close
        yield move([gx,gy,.19])
        if env.lab_task == "transport":
            yield move([*env.tray[:2],.19]); yield move([*env.tray[:2],.022]); yield opening
            yield move([*env.tray[:2],.24])
        elif env.lab_task == "insert":
            vertical = [math.pi,math.pi/2,yaw]
            yield move([*env.rack_center[:2],.28])
            yield move([*env.rack_center[:2],.25],vertical)
            displacement = env.tcp-env.tube_pos
            target = np.r_[env.rack_center[:2],.008+env.half_length+env.radius]+displacement
            yield move(target,vertical)
            yield opening
            yield move([env.rack_center[0]-.12,env.rack_center[1],.24],vertical)
    yield {"action":"finish"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks",nargs="+",choices=TASKS,default=list(TASKS))
    p.add_argument("--seed",type=int,default=0); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--video",action="store_true")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for task in a.tasks:
        directory=a.output/task; directory.mkdir()
        writer=None
        if a.video:
            import imageio.v2 as imageio
            writer=imageio.get_writer(str(directory/"episode.mp4"),fps=25,codec="libx264",quality=7)
        env=LabEnv(task,"test" if task=="rack_transfer" else "train",a.seed,60,
                   frame_callback=writer.append_data if writer else None)
        rows=[]
        try:
            for action in commands(env):
                _,feedback,_=env.step(action)
                row={"step":env.steps,"action":action,"feedback":feedback,"judge":env.result()}
                rows.append(row)
                print(json.dumps({"task":task,**row}),flush=True)
            result={"task":task,"policy":"scripted_diagnostic",**env.result()}
            results.append(result)
            (directory/"trajectory.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        finally:
            env.close()
            if writer: writer.close()
    (a.output/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    if not all(r["success"] for r in results): raise SystemExit(1)

if __name__=="__main__": main()
