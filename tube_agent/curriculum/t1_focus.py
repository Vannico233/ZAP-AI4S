"""A two-update T1 GRPO block, warm-started from a complete four-task adapter."""
import argparse
from collections import Counter
import copy
import hashlib
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys

from tube_agent.lab.experiment import read_json, write_json
from .config import ROOT, digest, file_digest, load, scene_manifest


def immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen input or evidence changed: {path}")
    else:
        write_json(path, value)


def focus_plan(parent, manifest):
    if manifest != scene_manifest(parent):
        raise ValueError("Parent scene definitions differ from the registered manifest")
    c = copy.deepcopy(parent)
    c.update(experiment="t1_focus_v1", tasks=["T1"])
    c["training"].update(base_rounds=2, hard_rounds=0, groups_per_task=4, failed_done_credit_clamp=True)
    scenes = {"version": c["version"], "config_sha256": digest(c),
              "scenes": [s for s in manifest["scenes"] if s["task"] == "T1"]}
    pool = [s for s in scenes["scenes"] if s["split"] == "train"]
    ordered = random.Random(c["seeds"]["manifest_seed"] + 40000000).sample(pool, 8)
    rounds = []
    for i in range(2):
        groups = []
        for j, scene in enumerate(ordered[4*i:4*i+4]):
            groups.append({"id": f"round_{i+1:02d}_T1_group_{j}", "task": "T1",
                "scene_id": scene["id"], "scene_sha256": scene["sha256"],
                "rollouts": [{"sample": k, "policy_seed": 40000000+32*i+8*j+k} for k in range(8)]})
        rounds.append({"round": i+1, "phase": "t1_focus", "groups": groups})
    return c, scenes, rounds


def source_record(parent, ckpt, parent_config):
    if ckpt.parent != parent/"warmup/checkpoints" or not re.fullmatch(r"step_\d{5}", ckpt.name):
        raise ValueError("Expected an atomic warmup step checkpoint in the specified parent")
    index = int(ckpt.name.split("_")[-1])
    complete = read_json(ckpt/"COMPLETE.json")
    if complete["counters"]["round"] != index or complete["config_sha256"] != digest(parent_config):
        raise ValueError("Source checkpoint does not match its round or parent config")
    for name, key in (("adapter_model.safetensors", "adapter_sha256"), ("optimizer_rng.pt", "optimizer_sha256")):
        if file_digest(ckpt/name) != complete[key]:
            raise ValueError(f"Incomplete or modified source checkpoint: {name}")
    metrics = read_json(parent/"warmup/metrics"/f"round_{index:02d}.json")
    if metrics["round"] != index or metrics["optimizer_steps_total"] != complete["counters"]["optimizer_steps"]:
        raise ValueError("Source adapter lacks matching committed training metrics")
    return {"source_checkpoint": str(ckpt), "parent_run": str(parent), "round": index,
        "parent_config_sha256": digest(parent_config), "tensor_sha256": metrics["adapter_after"],
        "adapter_file_sha256": complete["adapter_sha256"], "complete_sha256": file_digest(ckpt/"COMPLETE.json"),
        "optimizer": "fresh_adamw", "inherited_optimizer_steps": 0, "inherited_rounds": 0}


def evaluation_summary(root, node, scenes, c, tensor):
    details = []
    for scene in scenes:
        path = root/"evaluation"/node/scene["id"]/"result.json"
        r = read_json(path)
        p = r["generator_provenance"]
        if (r["scene"] != scene or r["seed"] != scene["seed"] or r["split"] != "eval" or r["task"] != "T1"
                or r["policy_seed"] != 30000000+scene["seed"] or r["origin"] != "evaluation"
                or r["policy_version"] != node or p["policy_version"] != node or r["policy"] != "model"
                or r["error"] is not None or r["demonstration"] is not None or r["replayed_actions"] != 0
                or r["search_prefix_length"] != 0 or r["generated_actions"] != r["steps"]
                or r["config_sha256"] != digest(c) or r["observation_mode"] != "rgbd_sam3.1_detector"
                or p["adapter_sha256"] != tensor or p["temperature"] != 0 or p["deterministic"] is not True
                or p["sampler"] != "same_live_model_object" or type(r["success"]) is not bool
                or not 0 <= r["invalid_actions"] <= r["steps"] <= c["policy"]["max_steps"]):
            raise ValueError(f"Unmatched evaluation provenance or scene: {path}")
        details.append({**{k: r[k] for k in ("policy_seed", "success", "ever_grasped", "extracted",
                                              "steps", "invalid_actions", "termination", "failure_type")},
            "scene_id": scene["id"], "path": str(path), "result_sha256": file_digest(path)})
    if len(details) != 30 or len({row["scene_id"] for row in details}) != 30:
        raise ValueError("Expected 30 distinct held-out T1 scenes")
    counters = {k: sum(row[k] for row in details) for k in ("success", "ever_grasped", "extracted", "steps", "invalid_actions")}
    return {"node": node, "episodes": 30, **counters, "accepted_actions": counters["steps"]-counters["invalid_actions"],
        "reached_step_limit": sum(r["steps"] == c["policy"]["max_steps"] for r in details),
        "termination": dict(Counter(r["termination"] for r in details)),
        "invalid_actions_definition": "sum of environment invalid_actions", "results": details}


def paired(before, after):
    left = {r["scene_id"]: r["success"] for r in before["results"]}
    right = {r["scene_id"]: r["success"] for r in after["results"]}
    if left.keys() != right.keys():
        raise ValueError("Unpaired evaluation scenes")
    return {"gained": [s for s in left if right[s] and not left[s]],
            "lost": [s for s in left if left[s] and not right[s]],
            "both_succeeded": sum(left[s] and right[s] for s in left),
            "both_failed": sum(not left[s] and not right[s] for s in left)}


def run(a, runner):
    from .pipeline import model_reference, source_digest, gpu_check
    root, parent, ckpt = a.output, a.parent_run, a.source_checkpoint
    parent_config = load(parent/"config.json")
    c, manifest, rounds = focus_plan(parent_config, read_json(parent/"scenes.json"))
    provenance = read_json(parent/"provenance.json")
    if provenance["config_sha256"] != digest(parent_config):
        raise ValueError("Parent configuration differs from the frozen provenance")
    original_node = read_json(parent/"checkpoints/original/node.json")
    a.model = str(Path(a.model or provenance["model"]["resolved_path"]).resolve())
    runner.status("verify_source", source_checkpoint=str(ckpt))
    if (model_reference(a.model) != provenance["model"] or original_node["base_model"] != provenance["model"]
            or original_node["adapter"] is not None or original_node["config_sha256"] != digest(parent_config)):
        raise ValueError("Parent original baseline and base model provenance differ")
    source = source_record(parent, ckpt, parent_config)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, value in (("SOURCE.json", source), ("config.json", c), ("scenes.json", manifest), ("schedule.json", rounds),
            ("provenance.json", {"git_revision": revision, "source_sha256": source_digest(),
                                 "config_sha256": digest(c), "model": provenance["model"]})):
        immutable(root/name, value)
    load(root/"config.json", allow_focus=True)
    scenes = [s for s in manifest["scenes"] if s["split"] == "eval"]
    original = evaluation_summary(parent, "original", scenes, parent_config, hashlib.sha256(b"").hexdigest())
    immutable(root/"baselines/original.json", original)
    if (root/"DONE.json").is_file():
        runner.status("completed", report=str(root/"DONE.json"))
        return
    a.policy_gpus = c["execution"]["policy_gpus"]
    a.sam_gpu = c["execution"]["sam_gpu"]
    a.sam_url = c["execution"]["sam_url"]
    a.sam_checkpoint = ROOT/"assets/sam3.1/sam3.1_multiplex.pt"
    a.cpu_threads, a.min_free_gpu_gb = 4, 48
    runner.status("preflight", gpus=gpu_check(a))
    runner.ensure_sam()
    runner.evaluate("source", episodes=30)
    baseline = evaluation_summary(root, "source", scenes, c, source["tensor_sha256"])
    immutable(root/"baselines/source.json", baseline)
    runner.train_stage("warmup")
    warmup = read_json(root/"warmup/DONE.json")
    if warmup["rounds"] != 2 or warmup["episodes"] != 64:
        raise ValueError("Incomplete T1 training block")
    runner.evaluate("after_warmup", episodes=30)
    metrics = [read_json(root/"warmup/metrics"/f"round_{i:02d}.json") for i in (1, 2)]
    final = evaluation_summary(root, "after_warmup", scenes, c, metrics[-1]["adapter_after"])
    write_json(root/"DONE.json", {"status": "completed", "source_checkpoint": str(ckpt), "warmup": warmup,
        "failed_done_credit_by_round": [r["failed_done_credit"] for r in metrics],
        "original": original, "source": baseline, "after_warmup": final, "paired_source_to_after": paired(baseline, final),
        "paired_original_to_source": paired(original, baseline), "next": "review paired results before extension"})
    runner.status("completed", report=str(root/"DONE.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", help="Default: base model pinned by the parent run")
    a = p.parse_args()
    if sys.platform != "linux":
        p.error("Launch on the existing Linux GPU instance")
    import fcntl
    from .pipeline import CurriculumRunner
    a.parent_run, a.source_checkpoint, a.output = (p.resolve() for p in (a.parent_run, a.source_checkpoint, a.output))
    if a.output == a.parent_run or a.output.is_relative_to(a.parent_run) or a.parent_run.is_relative_to(a.output):
        p.error("Focus and parent runs require independent directories")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("Another process owns this T1 output directory")
        if not (a.output/"SOURCE.json").exists():
            if any(path.name not in (".lock", "logs", "status.json", "launcher.pid") for path in a.output.iterdir()):
                p.error("Expected an empty focus output directory")
            checkpoints = sorted(p.parent for p in (a.parent_run/"warmup/checkpoints").glob("step_*/COMPLETE.json")
                if re.fullmatch(r"step_\d{5}", p.parent.name))
            if not checkpoints or checkpoints[-1] != a.source_checkpoint:
                p.error("Expected the latest complete parent checkpoint at cutover")
        runner = CurriculumRunner(a)
        (a.output/"launcher.pid").write_text(str(os.getpid())+"\n", encoding="ascii")
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            run(a, runner)
        except KeyboardInterrupt:
            runner.status("interrupted")
            raise
        except Exception as exc:
            runner.status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            runner.close()


if __name__ == "__main__":
    main()
