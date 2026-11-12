"""Collect fixed-budget original-model T1 rollouts and verify successes by physical replay."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
import random

from tube_agent.lab.experiment import read_json, write_json
from tube_agent.model_utils import DEFAULT_MODEL, resolve_model
from .config import BIMANUAL, digest, file_digest, load, scene_manifest

ORIGIN = "t1_collection"


@contextmanager
def lock(path, *, wait=True):
    # Collection workers run on Linux; flock is released automatically on worker exit.
    import fcntl
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def make_plan(config, scenes, repeats, seed):
    pool = [s for s in scene_manifest(config)["scenes"] if s["task"] == "T1" and s["split"] == "train"]
    if not 1 <= scenes <= len(pool) or repeats <= 0:
        raise ValueError("Expected 1-32 training scenes and at least one sample per scene")
    selected = random.Random(seed).sample(pool, scenes)
    attempts = []
    for repeat in range(repeats):
        for scene in selected:
            index = len(attempts)
            attempts.append({"directory": f"attempts/{index:04d}_{scene['id']}_{repeat}",
                             "policy_seed": seed + index, "scene": scene})
    return attempts


def freeze_plan(root, config, args, model_path, sam_health):
    from tube_agent.curriculum.pipeline import model_reference
    from tube_agent.lab.pipeline import source_digest
    # Compare file identity cheaply on resume; hash the actual base files once, under lock.
    stamps = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in model_path.iterdir()
              if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".txt")}
    plan = {"config_sha256": digest(config), "source_sha256": source_digest(), "policy": "original",
            "model_path": str(model_path), "model_files": stamps, "sam_sha256": sam_health["checkpoint_sha256"],
            "shards": args.shards, "attempts": make_plan(config, args.scenes, args.samples_per_scene, args.seed)}
    with lock(root / ".plan.lock"):
        path = root / "collection.json"
        if path.exists():
            old = read_json(path)
            if (old["plan"] != plan or digest(read_json(root / "config.json")) != plan["config_sha256"]):
                raise ValueError("Output already contains a different collection plan")
        else:
            base = model_reference(model_path)
            write_json(root / "config.json", config)
            write_json(path, {"plan": plan, "model": base})
    return plan


def verify_result(directory, attempt, config):
    from .rollout import read_rows, replay
    result = read_json(directory / "result.json")
    if result["error"]:
        raise RuntimeError(f"Incomplete model attempt {directory.name}: {result['error']}")
    scene, seed = attempt["scene"], attempt["policy_seed"]
    if (result["scene"] != scene or result["policy_seed"] != seed or result["policy"] != "model"
            or result["origin"] != ORIGIN or result["policy_version"] != "original"
            or result["demonstration"] is not None or result["replayed_actions"] != 0
            or result["config_sha256"] != digest(config) or not result["steps"]
            or result["generated_actions"] != result["steps"]):
        raise ValueError("Collection result has incompatible model, scene, actions or configuration")
    record = {"trajectory": attempt["directory"] + "/trajectory.jsonl", "scene_id": scene["id"],
              "policy_seed": seed, "model_success": result["success"], "steps": result["steps"],
              "reward": result["reward"], "failure_type": result["failure_type"],
              "termination": result["termination"], "invalid_actions": result["invalid_actions"],
              "replay_verified": False}
    if result["success"]:
        rows = list(read_rows(directory))
        if (len(rows) != result["steps"] or not rows[-1]["judge_after"]["success"]
                or any(row["source_origin"] != ORIGIN or row["policy_seed"] != seed
                       or not row.get("response_ids") or not row.get("model_input_sha256") for row in rows)):
            raise ValueError("Success candidate lacks complete model-generated action evidence")
        final = replay(scene, config, rows)
        if not (final["success"] and final["held"] and not final["dropped"] and not final["severe_collision"]):
            raise ValueError("Success candidate failed the T1 physical replay")
        record.update(replay_verified=True, trajectory_sha256=file_digest(directory / "trajectory.jsonl"),
                      result_sha256=file_digest(directory / "result.json"),
                      goal_stable_seconds=final["goal_stable_seconds"])
    return record


def summarize(root):
    plan = read_json(root / "collection.json")["plan"]
    records, errors = [], []
    for attempt in plan["attempts"]:
        directory = root / attempt["directory"]
        if (directory / "verified.json").is_file():
            records.append(read_json(directory / "verified.json"))
        elif (directory / "result.json").is_file():
            error = read_json(directory / "result.json")["error"]
            if error:
                errors.append({"directory": attempt["directory"], "error": error})
    successes = [r for r in records if r["replay_verified"]]
    return {"planned": len(plan["attempts"]), "completed": len(records), "verified_successes": len(successes),
            "failures": dict(Counter(r["failure_type"] for r in records if not r["model_success"])),
            "errors": errors, "successes": successes, "records": records, "config_sha256": plan["config_sha256"]}


def update_summary(root):
    with lock(root / ".summary.lock"):
        report = summarize(root)
        write_json(root / "summary.json", report)
    print(f"T1 completed {report['completed']}/{report['planned']} | replay-verified successes "
          f"{report['verified_successes']} | infrastructure errors {len(report['errors'])}", flush=True)


def collect(args):
    from tube_agent.lab.vision import SamClient
    from .learning import Generator, load_model
    from .rollout import episode
    config = load(BIMANUAL)
    config["policy"]["temperature"] = args.temperature
    model_path = resolve_model(args.model)
    if (model_path / "adapter_config.json").exists():
        raise ValueError("T1 collection requires the original Qwen checkpoint")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    segmenter = SamClient(args.sam_url or config["execution"]["sam_url"])
    try:
        plan = freeze_plan(root, config, args, model_path, segmenter.health())
        generator = None
        with lock(root / f".shard_{args.shard}.lock", wait=False):
            for attempt in plan["attempts"][args.shard::args.shards]:
                directory = root / attempt["directory"]
                if (directory / "verified.json").is_file():
                    continue
                result_path = directory / "result.json"
                if directory.exists() and (not result_path.exists() or read_json(result_path)["error"]):
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    directory.rename(directory.with_name(directory.name + ".incomplete-" + stamp))
                if not result_path.exists():
                    if generator is None:
                        model, processor = load_model(model_path, config, trainable=False)
                        generator = Generator(model, processor, config)
                        generator.bind("original")
                    episode(attempt["scene"], config, directory, generator, segmenter, attempt["policy_seed"],
                            "original", origin=ORIGIN, temperature=args.temperature)
                write_json(directory / "verified.json", verify_result(directory, attempt, config))
                update_summary(root)
    finally:
        segmenter.close()
        if (root / "collection.json").is_file():
            update_summary(root)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sam-url")
    p.add_argument("--scenes", type=int, default=6)
    p.add_argument("--samples-per-scene", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args(argv)
    if args.summarize:
        update_summary(args.output)
        return
    if not 0 <= args.shard < args.shards or not math.isfinite(args.temperature) or args.temperature <= 0:
        p.error("Expected a valid shard index and a finite positive sampling temperature")
    collect(args)


if __name__ == "__main__":
    main()
