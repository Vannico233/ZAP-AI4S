"""Physics and service checks, with explicit recoverable perception warnings."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
from .env import LabEnv,TASKS
from .sanity import commands

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--physics-only",action="store_true")
    p.add_argument("--services",action="store_true");p.add_argument("--seed",type=int,default=0)
    p.add_argument("--services-only",action="store_true",help="Skip already-checked physical trajectories")
    p.add_argument("--strict-perception",action="store_true",help="Exit nonzero after reporting all tasks if any query remains missing")
    p.add_argument("--sam-url",default="http://127.0.0.1:8001");p.add_argument("--base-url",default="http://127.0.0.1:8000/v1")
    a=p.parse_args()
    if a.services_only and a.physics_only:p.error("--services-only and --physics-only cannot be combined")
    if a.services_only:a.services=True
    a.output.mkdir(parents=True,exist_ok=False)
    report={"python":sys.version,"seed":a.seed,"checks":[],"policy":"scripted_diagnostic_NOT_model",
            "physical_checks":"not_run" if a.services_only else "requested","perception_warnings":[]};panels=[]
    def save_report():
        (a.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    for task in (() if a.services_only else TASKS):
        env=LabEnv(task,"test" if task=="rack_transfer" else "train",a.seed,60)
        cameras=None;visibility=[]
        try:
            if not a.physics_only:
                from .vision import Cameras
                import mujoco
                cameras=Cameras(env)
                target="cap" if task=="unscrew" else "tube_red"
                body_id=env.model.body(target).id
                for frame in cameras.capture():
                    name=frame["calibration"]["name"]
                    Image.fromarray(frame["rgb"]).save(a.output/f"{task}_{name}_initial.png")
                    np.save(a.output/f"{task}_{name}_depth.npy",frame["depth"],allow_pickle=False)
                    cameras.renderer.enable_segmentation_rendering()
                    options=mujoco.MjvOption();options.geomgroup[3]=1
                    cameras.renderer.update_scene(env.data,camera=name,scene_option=options)
                    ids=cameras.renderer.render().copy();cameras.renderer.disable_segmentation_rendering()
                    geom_ids=np.where(env.model.geom_bodyid==body_id)[0]
                    mask=(ids[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(ids[:,:,0],geom_ids)
                    visibility.append({"camera":name,"target":target,"visible_pixels":int(mask.sum()),
                                       "purpose":"QA only; simulator segmentation is NEVER a policy input"})
                    panel=Image.fromarray(frame["rgb"]).resize((480,360));d=ImageDraw.Draw(panel)
                    d.text((5,5),f"{task} / {name} / {int(mask.sum())} target pixels",fill="white")
                    panels.append(panel)
                if max(v["visible_pixels"] for v in visibility)<15:raise RuntimeError(f"{task}: target not visible")
            actions=[]
            for action in commands(env):
                _,feedback,_=env.step(action);actions.append({"action":action,"feedback":feedback,"judge":env.result()})
                if cameras:
                    for frame in cameras.capture():
                        name=frame["calibration"]["name"]
                        stage_dir=a.output/f"{task}_stages";stage_dir.mkdir(exist_ok=True)
                        Image.fromarray(frame["rgb"]).save(stage_dir/f"step_{env.steps:02d}_{name}.png")
            result=env.result()
            if cameras:
                for frame in cameras.capture():Image.fromarray(frame["rgb"]).save(a.output/f"{task}_{frame['calibration']['name']}_final.png")
            report["checks"].append({"task":task,"result":result,"initial_visibility":visibility})
            save_report()
            (a.output/f"{task}_actions.json").write_text(json.dumps(actions,indent=2),encoding="utf-8")
            print(json.dumps({"task":task,"success":result["success"],"visibility":visibility}),flush=True)
        except Exception as exc:
            report["error"]=f"{task}: {type(exc).__name__}: {exc}";save_report();raise
        finally:
            if cameras:cameras.close()
            env.close()
    if panels:
        sheet=Image.new("RGB",(960,360*len(TASKS)))
        for i,panel in enumerate(panels):sheet.paste(panel,((i%2)*480,(i//2)*360))
        sheet.save(a.output/"camera_contact_sheet.jpg",quality=95)
    if a.services:
        from .vision import SamClient,Observer
        from .policy import HttpGenerator,VisualPolicy
        segmenter=SamClient(a.sam_url);generator=HttpGenerator(a.base_url)
        env=observer=None
        try:
            report["sam"]=segmenter.health();report["qwen"]=generator.health()
            report["live_detections"]={};report["perception_audit"]={}
            for task in TASKS:
                env=LabEnv(task,"test" if task=="rack_transfer" else "train",a.seed)
                observer=Observer(env,segmenter)
                obs,images=observer.observe(a.output/"live_perception"/task)
                report["live_detections"][task]=obs["detections"]
                report["perception_audit"][task]=observer.last_audit
                missing=obs["missing_queries"]
                if missing:
                    report["perception_warnings"].append({"task":task,"missing_queries":missing})
                save_report()
                print(json.dumps({"task":task,"perception_status":obs["perception_status"],
                                  "missing_queries":missing,"detections":len(obs["detections"])}),flush=True)
                if task=="pick":
                    action,raw,_=VisualPolicy(generator,0.,0).act(env.instruction,obs,images,None,0)
                    report["vision_model_response"]={"action":action,"raw":raw}
                    if action.get("action")=="invalid_model_output":raise RuntimeError("Vision model returned malformed action")
                observer.close();observer=None;env.close();env=None
            report["services_ready"]=True
            report["perception_complete"]=not report["perception_warnings"]
        except Exception as exc:
            report["error"]=f"{type(exc).__name__}: {exc}";save_report();raise
        finally:
            if observer:observer.close()
            if env:env.close()
            segmenter.close();generator.close()
    if not all(c["result"]["success"] for c in report["checks"]):
        report["status"]="physical_diagnostic_failed";save_report()
        raise SystemExit("Physical diagnostic failed; inspect report before model sampling")
    if not a.services_only:report["physical_checks"]="passed"
    if report["perception_warnings"]:
        report["status"]="services_ready_with_perception_warnings";save_report()
        print("WARNING: Some queries still lack masks. RGB remains available; the policy may change_query. No missing pose was filled in.",flush=True)
        print("READY_WITH_PERCEPTION_WARNINGS: "+str(a.output/"report.json"),flush=True)
        if a.strict_perception:raise SystemExit(2)
    else:
        report["status"]="passed";save_report()
        print("PASS: "+str(a.output/"report.json"),flush=True)

if __name__=="__main__":main()
