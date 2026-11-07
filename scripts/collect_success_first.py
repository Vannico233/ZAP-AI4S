"""Collect a bounded review archive; no inference or training is performed."""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("runs/lab_success_first/fixed_seed0_v2"))
    a=p.parse_args();root=a.run.resolve()
    if not (root/"config.json").is_file():p.error("No pipeline config found in --run")
    target=Path("success_first_review_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".tar.gz")
    files={};skipped=[]
    def add(path):
        path=path.resolve()
        if not path.is_relative_to(root) or not path.is_file():return
        if path.stat().st_size>64*2**20:
            skipped.append(str(path));return
        files[path.relative_to(root).as_posix()]=path.read_bytes()
    for name in ("config.json","status.json","report.json","success_gate.json","gate_replays.json",
                 "recipe_plan.json","recipe_success_gate.json","recipe_gate_replays.json","selection.json",
                 "sam_health.json","acquisition_diagnostics.json","videos/index.json","REPORT.md","data/export.json"):
        add(root/name)
    for stage in ("warmup","sft","rl_fixed","rl_diverse"):
        add(root/stage/"config.json");add(root/stage/"DONE.json")
        for path in (root/stage/"metrics").glob("*.json"):add(path)
        for path in (root/stage).glob("step_*/COMPLETE.json"):add(path)
    launcher=root/"launcher.log"
    if launcher.is_file():
        with launcher.open("rb") as f:
            f.seek(max(0,launcher.stat().st_size-2**20));files["launcher.log"]=f.read()
    for path in root.rglob("summary.json"):add(path)
    # Last 1 MiB of each log preserves errors without embedding all action logs.
    for path in (root/"logs").glob("*.log"):
        with path.open("rb") as f:
            f.seek(max(0,path.stat().st_size-2**20));files["logs/"+path.name]=f.read()
    episodes=set()
    # Work even while sampling is running, before gate/diagnostics are written.
    # Pick an initial independent failure and the best completed episode per
    # task. Do not let a gate with zero successes produce a log-only archive.
    completed={}
    for result_path in sorted(root.rglob("result.json")):
        if any(".incomplete-" in part for part in result_path.parts):continue
        try:result=json.loads(result_path.read_text(encoding="utf-8"))
        except (ValueError,OSError):continue
        if result.get("policy")!="model" or result.get("error"):continue
        completed.setdefault(result.get("task"),[]).append((result_path.parent,result))
    for candidates in completed.values():
        independent=[pair for pair in candidates if not pair[1].get("search_prefix_length",0)]
        if independent:episodes.add(independent[0][0])
        failures=[pair for pair in candidates if not pair[1].get("success")]
        if failures:episodes.add(failures[0][0])
        successful=[pair for pair in candidates if pair[1].get("success")]
        for episode,result in successful[:10]:episodes.add(episode)
        episodes.add(max(candidates,key=lambda pair:(bool(pair[1].get("success")),pair[1].get("reward",-100)))[0])
    diag=root/"acquisition_diagnostics.json"
    if diag.exists():
        for item in json.loads(diag.read_text()).values():
            if item["best"]:episodes.add(Path(item["best"][0]["episode"]))
    for name in ("success_gate.json","recipe_success_gate.json"):
        gate=root/name
        if gate.exists():
            for records in json.loads(gate.read_text())["approved"].values():
                if records:episodes.add(Path(records[0]["result"]).parent)
    video_index=root/"videos/index.json"
    if video_index.exists():
        for record in json.loads(video_index.read_text(encoding="utf-8")).values():
            if record.get("status")!="verified_success":continue
            video=Path(record["video"])
            add(video);add(video.parent/"report.json");add(video.parent/"poster.jpg")
    for episode in sorted(episodes):
        add(episode/"result.json");add(episode/"trajectory.jsonl")
        trajectory=episode/"trajectory.jsonl"
        if not trajectory.is_file():continue
        # At most first, first invalid, first gripper command and final view.
        selected={};last=None
        with trajectory.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():continue
                try:row=json.loads(line)
                except ValueError:continue
                selected.setdefault("first",row);last=row
                if row.get("feedback",{}).get("accepted") is False:selected.setdefault("invalid",row)
                if row.get("action",{}).get("action") in ("gripper","set_gripper"):selected.setdefault("gripper",row)
        if last:selected["last"]=last
        for label,row in selected.items():
            number=0
            for message in row.get("messages",[]):
                content=message.get("content")
                for part in content if isinstance(content,list) else []:
                    if part.get("type")=="image_url":
                        url=part["image_url"]["url"]
                        if url.startswith("data:image/png;base64,") and episode.resolve().is_relative_to(root):
                            files[episode.resolve().relative_to(root).as_posix()+f"/{label}_{number}.png"]=base64.b64decode(url.split(",",1)[1])
                            number+=1
    files["COLLECTION.json"]=json.dumps({"run":str(root),"selected_episodes":len(episodes),
                                       "skipped_large_files":skipped},indent=2).encode()
    with tarfile.open(target,"w:gz") as tar:
        for name,data in sorted(files.items()):
            member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
    print("SAVED:",target.resolve())
    print("FILES:",len(files),"SKIPPED_LARGE_FILES:",len(skipped))


if __name__=="__main__":main()
