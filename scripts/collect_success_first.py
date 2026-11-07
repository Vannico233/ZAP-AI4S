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
                 "sam_health.json","acquisition_diagnostics.json","videos/index.json"):
        add(root/name)
    for path in root.rglob("summary.json"):add(path)
    # Last 1 MiB of each log preserves errors without embedding all action logs.
    for path in (root/"logs").glob("*.log"):
        with path.open("rb") as f:
            f.seek(max(0,path.stat().st_size-2**20));files["logs/"+path.name]=f.read()
    episodes=set()
    diag=root/"acquisition_diagnostics.json"
    if diag.exists():
        for item in json.loads(diag.read_text()).values():
            if item["best"]:episodes.add(Path(item["best"][0]["episode"]))
    gate=root/"success_gate.json"
    if gate.exists():
        for records in json.loads(gate.read_text())["approved"].values():
            if records:episodes.add(Path(records[0]["result"]).parent)
    for episode in sorted(episodes):
        add(episode/"result.json");add(episode/"trajectory.jsonl")
        trajectory=episode/"trajectory.jsonl"
        if not trajectory.is_file():continue
        with trajectory.open(encoding="utf-8") as f:
            first=next((json.loads(line) for line in f if line.strip()),None)
        if first:
            number=0
            for message in first.get("messages",[]):
                content=message.get("content")
                for part in content if isinstance(content,list) else []:
                    if part.get("type")=="image_url":
                        url=part["image_url"]["url"]
                        if url.startswith("data:image/png;base64,") and episode.resolve().is_relative_to(root):
                            files[episode.resolve().relative_to(root).as_posix()+f"/initial_{number}.png"]=base64.b64decode(url.split(",",1)[1])
                            number+=1
    files["COLLECTION.json"]=json.dumps({"run":str(root),"skipped_large_files":skipped},indent=2).encode()
    with tarfile.open(target,"w:gz") as tar:
        for name,data in sorted(files.items()):
            member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
    print("SAVED:",target.resolve())
    print("FILES:",len(files),"SKIPPED_LARGE_FILES:",len(skipped))


if __name__=="__main__":main()
