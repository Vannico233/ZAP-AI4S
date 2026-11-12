"""Archive one crashed GPU 2 source-evaluation shard before changing its renderer."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path, value):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def preserve(src, dst):
    data = src.read_bytes()
    if dst.exists():
        require(dst.read_bytes() == data, "Archived evidence differs: " + dst.name)
    else:
        tmp = dst.with_name(dst.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    return hashlib.sha256(data).hexdigest()


def recover(out, ops, ckpt, revision, source_sha, gpu_uuid):
    archive = ops/"renderer2-node154"
    receipt = archive/"RECEIPT.json"
    source = read(out/"SOURCE.json")
    identity = {"output": str(out), "source_checkpoint": str(ckpt), "source_adapter_sha256": source_sha,
                "revision": revision, "physical_gpu": 2, "gpu_uuid": gpu_uuid,
                "prior_software_render_gpus": "3", "software_render_gpus": "2,3", "policy_gpus": [0,2,3]}
    require(read(out/"provenance.json")["git_revision"] == revision and source["source_checkpoint"] == str(ckpt)
            and source["adapter_file_sha256"] == source_sha, "Source or Git revision mismatch")
    if receipt.exists():
        require(read(receipt)["identity"] == identity, "This renderer recovery belongs to another run, revision or GPU")
        print("Using committed renderer recovery:", receipt)
        return
    status = read(out/"status.json")
    require(status.get("stage") == "failed" and "evaluate_source child exited -6" in status.get("error",""),
            "Expected source evaluation SIGABRT")
    require(not (out/"steps/evaluate_source.json").exists() and not (out/"warmup").exists()
            and not (out/"baselines/source.json").exists(), "Source evaluation has already advanced")
    attempt = Path((ops/"latest").read_text().strip())
    require(attempt.parent == ops and (attempt/"revision").read_text().strip() == revision, "Invalid recovery attempt")
    log = out/"logs/evaluate_source_1.log"
    lines = log.read_text(errors="replace").rsplit("\nCOMMAND: ",1)[-1].splitlines()
    args = json.loads(lines[0])
    require(args[args.index("--shard")+1] == "1" and args[args.index("--node")+1] == "source"
            and args[args.index("--run")+1] == str(out), "Wrong failed evaluation shard")
    events = [json.loads(line) for line in lines if line.startswith("{") and '"egl_binding_selected"' in line]
    require(len(events) == 1 and events[0].get("policy_gpu") == "2" and events[0].get("backend") == "nvidia-egl"
            and events[0].get("uuid") == gpu_uuid, "Failed shard renderer does not identify the current physical GPU 2")
    trace = "\n".join(lines).partition("Fatal Python error: Aborted")[2]
    require(re.search(r'mujoco/renderer\.py", line 243 in render',trace) and "in stable_capture" in trace,
            "Expected MuJoCo framebuffer readback abort")
    scenes = [s["id"] for s in read(out/"scenes.json")["scenes"] if s["split"] == "eval"]
    require(len(scenes) == len(set(scenes)) == 30 and all(re.fullmatch(r"T1_eval_\d{3}",s) for s in scenes),
            "Invalid registered T1 evaluation scenes")
    canonical = out/"evaluation/source"
    shard = scenes[1::3] + ["shard_1_DONE.json"]
    archive.mkdir(exist_ok=True)
    evidence = archive/"evidence"
    moved = archive/"shard1-before-osmesa"
    evidence.mkdir(exist_ok=True)
    moved.mkdir(exist_ok=True)
    for path in (canonical, archive, evidence, moved):
        require(not path.is_symlink(), "Unexpected symlink at " + str(path))
    plan_path = archive/"PLAN.json"
    if not plan_path.exists():
        logs = [out/"logs"/f"evaluate_source_{i}.log" for i in range(3)]
        evidence_files = [out/"status.json", attempt/"pipeline.log", *logs, *sorted((attempt/"bindings").glob("egl-binding-*.json"))]
        hashes = {p.name:preserve(p,evidence/p.name) for p in evidence_files}
        names = [name for name in shard if (canonical/name).exists()]
        record(plan_path, {"identity":identity, "attempt":str(attempt), "host":socket.gethostname(),
                           "failed_stage":status, "evidence_sha256":hashes, "entries":names,
                           "prepared_utc":datetime.now(timezone.utc).isoformat()})
    plan = read(plan_path)
    require(plan["identity"] == identity and set(plan["entries"]) <= set(shard), "Recovery plan identity mismatch")
    for name in plan["entries"]:
        src,dst = canonical/name,moved/name
        require(not src.is_symlink() and not dst.is_symlink() and src.exists() != dst.exists(),
                "Archive collision or missing canonical evidence: " + name)
        if src.exists():
            src.rename(dst)
    record(receipt, {**plan, "committed_utc":datetime.now(timezone.utc).isoformat()})
    print("Renderer recovery committed:", receipt, "archived shard entries:", len(plan["entries"]))


if __name__ == "__main__":
    import fcntl
    out, ops, ckpt = map(Path,sys.argv[1:4])
    with (out/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = subprocess.check_output(["nvidia-smi","-i","2","--query-gpu=uuid","--format=csv,noheader"],text=True)
        gpu_uuid = raw.strip().lower().removeprefix("gpu-").replace("-","")
        require(bool(re.fullmatch("[0-9a-f]{32}",gpu_uuid)), "Could not verify physical GPU 2 UUID")
        recover(out,ops,ckpt,sys.argv[4],sys.argv[5],gpu_uuid)
