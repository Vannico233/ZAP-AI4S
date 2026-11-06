"""Render one saved visual model episode, without SAM or Qwen inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .env import LabEnv
from .vision import Cameras
from .policy import execute,VisualPolicy
from .run import state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--episode",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=json.loads((a.episode/"result.json").read_text(encoding="utf-8"))
    rows=[json.loads(s) for s in (a.episode/"trajectory.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    if result.get("environment_version")!=LabEnv.VERSION:raise ValueError("Environment version mismatch")
    if result.get("error") or not rows:raise ValueError("Cannot render an empty or errored episode")
    a.output.mkdir(parents=True,exist_ok=False)
    import imageio.v2 as imageio
    writer=imageio.get_writer(str(a.output/"episode.mp4"),fps=25,codec="libx264",quality=8,macro_block_size=16,
                             ffmpeg_params=["-movflags","+faststart"])
    env=LabEnv(result["task"],result["split"],result["seed"],max_steps=result.get("max_steps",40),
               action_protocol=result.get("action_protocol","strict_rpy_v1"))
    cameras=Cameras(env);stage={"step":0,"action":{"action":"ready"}};last=None
    class Context:queries=[]
    context=Context();policy=VisualPolicy(None);matches=[]
    def capture(_):
        nonlocal last
        views=cameras.capture();canvas=Image.new("RGB",(1280,544),(15,23,35))
        for i,frame in enumerate(views):canvas.paste(Image.fromarray(frame["rgb"]),(640*i,64))
        draw=ImageDraw.Draw(canvas)
        label="Qwen3.5-4B" if result.get("policy")=="model" else "SCRIPTED PHYSICS DIAGNOSTIC (NOT QWEN)"
        if result.get("search_prefix_length",0):label+=" / MODEL PREFIX SEARCH"
        draw.text((12,8),f"{label} / {result['task']} / seed {result['seed']} / LOCAL ACTION REPLAY",fill="white")
        draw.text((12,28),f"Step {stage['step']} | "+json.dumps(stage["action"]),fill="white")
        draw.text((12,46),"Saved actions; no new inference. Judge: "+str(env.result()["success"]),fill="white")
        last=np.array(canvas);writer.append_data(last)
    try:
        env.frame_callback=capture;capture(None)
        for row in rows:
            stage.update(step=row["step"],action=row["action"])
            feedback=execute(env,context,policy,row["action"])
            matches.append(np.allclose(state(env),row["state_after"],atol=1e-5,rtol=0))
            if feedback.get("accepted")!=row["feedback"].get("accepted"):raise RuntimeError("Action acceptance differs on replay")
        capture(None)
        for _ in range(40):writer.append_data(last)
        replay=env.result()
        (a.output/"report.json").write_text(json.dumps({"source":str(a.episode),"original_result":result,"replay_result":replay,
             "state_matches":matches,"new_inference":False},indent=2),encoding="utf-8")
        if not all(matches) or replay["success"]!=result["success"]:raise RuntimeError("Replay diverged; video retained for diagnosis")
        Image.fromarray(last).save(a.output/"poster.jpg")
        print("Saved: "+str(a.output/"episode.mp4"))
    finally:writer.close();cameras.close();env.close()

if __name__=="__main__":main()
