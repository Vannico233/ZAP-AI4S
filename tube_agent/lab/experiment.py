"""Small, dependency-light helpers for resumable experiments."""
import hashlib
import json
import os
from pathlib import Path

TRAIN_TASK_NAMES=("pick","transport","insert","unscrew")
TASK_NAMES=(*TRAIN_TASK_NAMES,"rack_transfer")

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=True)+"\n",encoding="utf-8")
    temporary.replace(path)

def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def eligible(result):
    return (result.get("policy")=="model" and result.get("split")=="train"
            and result.get("task") in TRAIN_TASK_NAMES and result.get("success") is True
            and not result.get("error") and result.get("invalid_actions")==0
            and result.get("demonstration") is None
            and result.get("observation_mode")=="rgbd_sam3.1_detector")

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

def checkpoint(root):
    """Only atomically committed optimizer checkpoints are resumable."""
    candidates=[p for p in Path(root).glob("step_*") if (p/"COMPLETE.json").is_file()]
    return max(candidates,key=lambda p:read_json(p/"COMPLETE.json")["step"]) if candidates else None

def successful_results(root):
    for path in sorted(Path(root).rglob("result.json")):
        if any(".incomplete-" in s for s in path.parts):continue
        result=read_json(path)
        if result.get("success") and not result.get("error") and result.get("policy")=="model":
            yield path,result
