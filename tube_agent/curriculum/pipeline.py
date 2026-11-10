"""Original -> 18 warm-up batches -> bounded search -> 2-epoch SFT -> stop."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse
from tube_agent.lab.experiment import read_json, write_json
from tube_agent.lab.pipeline import Runner, gpu_check, source_digest
from tube_agent.model_utils import DEFAULT_MODEL, resolve_model
from .config import DEFAULT, ROOT, TASKS, load, digest, file_digest, scene_manifest, schedule, plan


def distributed_command(module, *arguments):
    # Python 3.11 argparse can abbreviate child --run to torchrun --run-path.
    # End launcher options BEFORE the module, not after it.
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=3",
            "--max_restarts=0", "-m", "--", module, *map(str, arguments)]


def validate_launch_arguments(output, model):
    """Exercise the installed torchrun parser before expensive physics/model setup."""
    from torch.distributed.run import get_args_parser
    targets = [("tube_agent.curriculum.train", [stage]) for stage in ("smoke", "warmup", "sft")]
    targets.append(("tube_agent.curriculum.search", []))
    for module, stage in targets:
        arguments = [*stage, "--run", str(output), "--model", str(model)]
        command = distributed_command(module, *arguments)
        try:
            parsed = get_args_parser().parse_args(command[3:])
        except SystemExit as exc:
            raise RuntimeError(f"torchrun could not parse the {module} launch command") from exc
        if not parsed.module or parsed.training_script != module or parsed.training_script_args != arguments:
            raise RuntimeError(f"torchrun changed child arguments for {module}")
    print(json.dumps({"event": "launch_arguments_validated", "commands": len(targets)}), flush=True)


class CurriculumRunner(Runner):
    def start(self, name, command, gpus):
        provenance = self.output/"provenance.json"
        if provenance.exists() and read_json(provenance)["source_sha256"] != source_digest():
            raise RuntimeError("Source code changed during the frozen experiment; refusing mixed-code stages")
        return super().start(name, command, gpus)

    def ensure_sam(self):
        """No experiment or loading deadline; fail on process/protocol errors."""
        import httpx
        url = self.a.sam_url
        parsed = urlparse(url)
        if parsed.hostname not in ("127.0.0.1", "localhost") or parsed.scheme != "http":
            raise ValueError("Managed SAM requires a localhost HTTP URL")
        with httpx.Client(timeout=5, trust_env=False) as client:
            try: existing = client.get(url+"/health")
            except httpx.ConnectError: existing = None
            if existing is not None:
                raise RuntimeError("SAM port is already occupied. Stop the old service, or choose a free --sam-url; external processes were not stopped")
        python = ROOT/".venv-sam/bin/python"
        if not python.exists(): raise FileNotFoundError("Existing .venv-sam is required")
        self.sam = self.start("sam", [str(python), "-m", "tube_agent.lab.sam_server",
            "--checkpoint", str(self.a.sam_checkpoint), "--port", str(parsed.port or 8002)], [self.a.sam_gpu])
        last = 0
        with httpx.Client(timeout=5, trust_env=False) as client:
            while self.sam.poll() is None:
                try:
                    response = client.get(url+"/health")
                    if response.is_success and response.json().get("ready"):
                        health = response.json()
                        if health.get("backend") != "sam3.1_detector": raise ValueError("Wrong SAM backend")
                        write_json(self.output/"sam_provenance.json", health)
                        return
                except httpx.TransportError: pass
                if time.monotonic()-last >= 30:
                    self.status("sam_loading", log=str(self.output/"logs/sam.log")); last = time.monotonic()
                time.sleep(2)
        raise RuntimeError("Managed SAM exited while loading; inspect logs/sam.log")

    def train_stage(self, stage):
        self.step(stage, distributed_command("tube_agent.curriculum.train", stage,
                                            "--run", self.output, "--model", self.a.model))

    def evaluate(self, node):
        marker = self.output/"steps"/("evaluate_"+node+".json")
        if marker.exists(): return
        processes = []
        for shard, gpu in enumerate(self.a.policy_gpus):
            processes.append(self.start(f"evaluate_{node}_{shard}", [sys.executable, "-m",
                "tube_agent.curriculum.evaluate", "--run", str(self.output), "--model", self.a.model,
                "--node", node, "--shard", str(shard), "--shards", "3"], [gpu]))
        self.wait(processes, "evaluate_"+node)
        write_json(marker, {"complete": True, "episodes": 120})


def model_reference(path):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".txt")])
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Expected the existing safetensors Qwen checkpoint")
    return {"resolved_path": str(path), "files": {p.name: {"sha256": file_digest(p), "bytes": p.stat().st_size} for p in files},
            "automatic_base_variant_substitution": False}


def report(root, sft_available):
    root = Path(root); nodes = {}
    for node in ("original", "after_warmup", "after_sft"):
        if node == "after_sft" and not sft_available:
            nodes[node] = {"available": False, "reason": "no complete successful training trajectory"}; continue
        tasks = {}
        for task in TASKS:
            paths = sorted((root/"evaluation"/node).glob(f"{task}_eval_*/result.json"))
            paths = [p for p in paths if ".incomplete-" not in str(p)]
            rows = [read_json(p) for p in paths]
            if len(rows) != 30 or any(r["error"] for r in rows):
                raise ValueError(f"Incomplete/error-contaminated evaluation: {node}/{task}")
            wins = sum(r["success"] for r in rows)
            tasks[task] = {"episodes": 30, "successes": wins, "success_rate": wins/30,
                "failure_types": dict(Counter(r["failure_type"] for r in rows if not r["success"])),
                "results": [{"scene_id": r["scene"]["id"], "success": r["success"],
                             "failure_type": r["failure_type"], "path": str(p)} for p, r in zip(paths, rows)]}
        nodes[node] = {"available": True, "tasks": tasks,
                       "macro_success_rate": sum(t["success_rate"] for t in tasks.values())/4}
    data = read_json(root/"data/manifest.json")
    value = {"status": "completed" if sft_available else "completed_partial", "nodes": nodes,
             "warmup": read_json(root/"warmup/DONE.json"), "search": read_json(root/"search/DONE.json"),
             "sft_data": {k: data[k] for k in ("samples", "successes_by_task", "scene_coverage", "missing_tasks", "includes_tree_search")},
             "comparison_interpretation": {"after_warmup-original": "warm-up including extra T3/T4 groups",
                "after_sft-after_warmup": "success-only SFT; includes search data if indicated, not a search-only ablation"},
             "main_rl_executed": False, "checkpoint_selection_by_evaluation": False}
    write_json(root/"report.json", value)
    lines = ["checkpoint,T1,T2,T3,T4,macro_success_rate"]
    for name, node in nodes.items():
        lines.append(",".join([name]+[str(node["tasks"][t]["success_rate"]) for t in TASKS]+[str(node["macro_success_rate"])])
                     if node["available"] else name+",unavailable,unavailable,unavailable,unavailable,unavailable")
    (root/"comparison.csv").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return value


def execute(runner, config):
    out, a = runner.output, runner.a
    runner.status("preflight", gpus=gpu_check(a))
    runner.step("physics", [sys.executable, "-m", "tube_agent.curriculum.preflight", "--config", str(out/"config.json"),
                           "--manifest", str(out/"scenes.json"), "--output", str(out/"physics")], [a.policy_gpus[0]])
    runner.ensure_sam()
    # Separate output so a cached no-render physics pass cannot hide missing vision checks.
    visual_manifest = read_json(out/"scenes.json")
    visual_manifest["scenes"] = [s for s in visual_manifest["scenes"] if s["index"] == 0 and s["split"] == "train"]
    write_json(out/"vision_scenes.json", visual_manifest)
    runner.step("vision", [sys.executable, "-m", "tube_agent.curriculum.preflight", "--config", str(out/"config.json"),
        "--manifest", str(out/"vision_scenes.json"), "--output", str(out/"vision"), "--render", "--sam-url", a.sam_url], [a.policy_gpus[0]])
    runner.train_stage("smoke")
    runner.evaluate("original")
    runner.train_stage("warmup")
    runner.step("search", distributed_command("tube_agent.curriculum.search", "--run", out, "--model", a.model))
    runner.step("freeze_data", [sys.executable, "-m", "tube_agent.curriculum.data", "--run", str(out)], [a.policy_gpus[0]])
    available = read_json(out/"data/manifest.json")["samples"] > 0
    if available: runner.train_stage("sft")
    else: write_json(out/"checkpoints/after_sft_UNAVAILABLE.json", {"reason": "No complete successful trajectories; SFT cannot run on an empty dataset"})
    runner.evaluate("after_warmup")
    if available: runner.evaluate("after_sft")
    result = report(out, available)
    runner.status(result["status"], report=str(out/"report.json"), main_rl_executed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/tube_curriculum/v1"))
    p.add_argument("--config", type=Path, default=DEFAULT); p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sam-url"); p.add_argument("--plan", action="store_true")
    a = p.parse_args(); c = load(a.config)
    if a.sam_url: c["execution"]["sam_url"] = a.sam_url
    if a.plan:
        print(json.dumps(plan(c), indent=2)); return
    if sys.platform != "linux": p.error("Launch this experiment on the existing Linux GPU instance; --plan works locally")
    import fcntl
    a.output = a.output.resolve(); a.model = str(resolve_model(a.model))
    a.policy_gpus = c["execution"]["policy_gpus"]; a.sam_gpu = c["execution"]["sam_gpu"]
    a.sam_url = c["execution"]["sam_url"]; a.cpu_threads = 4; a.min_free_gpu_gb = 48
    a.sam_checkpoint = ROOT/"assets/sam3.1/sam3.1_multiplex.pt"
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/".lock").open("a")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("This output directory already has an active experiment")
    runner = CurriculumRunner(a)
    def interrupted(signum, frame): raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        validate_launch_arguments(a.output, a.model)
        try: revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError): revision = None
        provenance = {"git_revision": revision, "source_sha256": source_digest(), "config_sha256": digest(c), "model": model_reference(a.model)}
        for name, value in (("config.json", c), ("provenance.json", provenance),
                            ("scenes.json", scene_manifest(c)), ("schedule.json", schedule(c, scene_manifest(c)))):
            if (a.output/name).exists() and read_json(a.output/name) != value:
                raise ValueError(f"Immutable {name} changed; use a new experiment directory")
            write_json(a.output/name, value)
        original = a.output/"checkpoints/original"
        if not (original/"node.json").exists():
            from transformers import AutoProcessor
            original.mkdir(parents=True, exist_ok=True)
            processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
            processor.save_pretrained(original)
            write_json(original/"node.json", {"name": "original", "base_model": provenance["model"],
                "adapter": None, "config_sha256": digest(c), "action_protocol": c["action_protocol"],
                "observation_version": "upright_visual_v1", "inference_config": c["policy"], "evaluation": c["evaluation"]})
        if (a.output/"report.json").exists() and read_json(a.output/"report.json")["status"] in ("completed", "completed_partial"):
            runner.status(read_json(a.output/"report.json")["status"], report=str(a.output/"report.json")); return
        execute(runner, c)
    except KeyboardInterrupt:
        runner.status("interrupted"); raise
    except Exception as exc:
        runner.status("failed", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        runner.close(); lock.close()


if __name__ == "__main__": main()
