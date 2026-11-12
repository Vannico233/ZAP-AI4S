"""Branch from replayable grasps in a bounded, verified original-model T1 collection."""
import argparse
from pathlib import Path

from tube_agent import interface
from tube_agent.lab.experiment import read_json, write_json
from tube_agent.model_utils import resolve_model
from . import collect_t1
from .config import digest, file_digest, load, scene_manifest
from .rollout import read_rows, replay, system_prompt

ORIGIN = "t1_model_prefix_continuation"


def prefix_candidate(parent, attempt, config, initial_bottom):
    directory = parent / attempt["directory"]
    verified = directory / "verified.json"
    if not verified.is_file():
        return None
    receipt = read_json(verified)
    result = read_json(directory / "result.json")
    if receipt["model_success"] or result["success"]:
        return None
    if collect_t1.verify_result(directory, attempt, config) != receipt:
        raise ValueError("Parent outcome differs from its verification receipt")
    collect_t1.check_original_policy(result, config)
    scene, seed = attempt["scene"], attempt["policy_seed"]
    rows = list(read_rows(directory))
    if len(rows) != result["steps"] or any(not collect_t1.row_evidence(r, scene, seed, config, i, collect_t1.ORIGIN)
                                          for i, r in enumerate(rows, 1)):
        raise ValueError("Incomplete model provenance in source trajectory")
    # Restrict search to physical states with a succeeding turn in the original episode.
    # Prefer an undepressed tube. Within that group, favor height to the nearest mm, then
    # the earliest grasp at that height; minor solver jitter should not consume turns.
    candidates = []
    for index, row in enumerate(rows[:-1], 1):
        judge = row["judge_after"]
        if (judge["held"] and not judge["success"] and not judge["dropped"] and not judge["severe_collision"]
                and row["feedback"]["accepted"] and row["action"]["action"] in {"move_to", "set_gripper"}):
            rise = judge["geometry"]["bottom"] - initial_bottom
            candidates.append(((rise >= -.002, round(rise, 3), -index), index))
    if not candidates:
        return None
    rank, length = max(candidates)
    return rank, {"directory": attempt["directory"], "policy_seed": seed, "steps": length,
                  "trajectory_sha256": file_digest(directory / "trajectory.jsonl"),
                  "result_sha256": file_digest(directory / "result.json"),
                  "prefix_sha256": digest(rows[:length])}


def load_prefix(parent, attempt):
    source = attempt["parent"]
    directory = parent / source["directory"]
    if (file_digest(directory / "trajectory.jsonl") != source["trajectory_sha256"]
            or file_digest(directory / "result.json") != source["result_sha256"]):
        raise ValueError("Frozen parent trajectory or outcome changed")
    from itertools import islice
    prefix = list(islice(read_rows(directory), source["steps"]))
    if len(prefix) != source["steps"] or digest(prefix) != source["prefix_sha256"]:
        raise ValueError("Frozen model prefix changed")
    return prefix


def select_attempts(parent, parent_plan, config, max_scenes, branches, seed):
    registered = {s["id"]: s for s in scene_manifest(config)["scenes"] if s["task"] == "T1" and s["split"] == "train"}
    rest = {}
    choices = {}
    for attempt in parent_plan["attempts"]:
        scene = attempt["scene"]
        sid = scene["id"]
        if scene != registered.get(sid):
            raise ValueError("Expected a registered T1 training scene")
        if sid not in rest:
            env = interface.make_env(scene, config)
            try:
                rest[sid] = env.result()["geometry"]["bottom"]
            finally:
                env.close()
        candidate = prefix_candidate(parent, attempt, config, rest[sid])
        if candidate and (sid not in choices or candidate[0] > choices[sid][0]):
            choices[sid] = candidate
    ranked = sorted(choices, key=lambda sid: choices[sid][0], reverse=True)[:max_scenes]
    attempts = []
    for sid in ranked:
        source = choices[sid][1]
        representative = {"scene": registered[sid], "parent": source}
        live = replay(registered[sid], config, load_prefix(parent, representative))
        if not live["held"] or live["dropped"] or live["severe_collision"] or live["success"]:
            raise ValueError("Selected prefix did not replay to its recorded unsolved physical grasp")
        for branch in range(branches):
            index = len(attempts)
            attempts.append({**representative, "policy_seed": seed + index,
                             "directory": f"attempts/{index:04d}_{sid}_branch{branch}"})
    if {a["policy_seed"] for a in attempts} & {a["policy_seed"] for a in parent_plan["attempts"]}:
        raise ValueError("Continuation seeds must be distinct from all source seeds")
    return attempts


def freeze_branches(args, parent, root, config, model_path, sam_health):
    from .pipeline import model_reference
    from tube_agent.lab.pipeline import source_digest
    source = read_json(parent / "collection.json")
    old = source["plan"]
    if ("parent_collection_sha256" in old or old["policy"] != "original" or old["config_sha256"] != digest(config)
            or old["sam_sha256"] != sam_health["checkpoint_sha256"]):
        raise ValueError("Incompatible parent policy, scene configuration or SAM weights")
    stamps = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in model_path.iterdir()
              if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".txt")}
    if stamps != old["model_files"]:
        raise ValueError("Parent model files have changed")
    settings = {"parent_collection_sha256": file_digest(parent / "collection.json"),
                "config_sha256": digest(config), "source_sha256": source_digest(), "seed": args.seed,
                "policy": "original", "model_path": str(model_path), "model_files": stamps,
                "sam_sha256": sam_health["checkpoint_sha256"],
                "max_scenes": args.scenes, "branches": args.branches, "shards": args.shards,
                "parent_root": str(parent), "prompt_sha256": digest(system_prompt(config))}
    with collect_t1.lock(root / ".plan.lock"):
        path = root / "collection.json"
        if path.exists():
            plan = read_json(path)["plan"]
            if ({k: v for k, v in plan.items() if k != "attempts"} != settings
                    or digest(read_json(root / "config.json")) != digest(config)):
                raise ValueError("Output has a different continuation plan")
            return plan
        base = model_reference(model_path)
        if base != source["model"]:
            raise ValueError("Original base weights or processor files differ from the source collection")
        attempts = select_attempts(parent, old, config, args.scenes, args.branches, args.seed)
        plan = {**settings, "attempts": attempts}
        write_json(root / "config.json", config)
        write_json(path, {"plan": plan, "model": base})
        return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", help="Expected original base checkpoint from the parent batch")
    p.add_argument("--sam-url")
    p.add_argument("--scenes", type=int, default=3, choices=range(1, 4))
    p.add_argument("--branches", type=int, default=2, choices=(1, 2))
    p.add_argument("--seed", type=int, default=30261113)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args(argv)
    root = args.output.resolve()
    if args.summarize:
        collect_t1.update_summary(root)
        return
    if args.parent is None or not 0 <= args.shard < args.shards:
        p.error("Expected a parent collection and valid shard index")
    parent = args.parent.resolve()
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        p.error("Choose a separate output directory outside the original collection")
    root.mkdir(parents=True, exist_ok=True)
    config = load(parent / "config.json")
    if config["version"] != "tube_curriculum_bimanual_v2":
        raise ValueError("Expected the source bimanual v2 environment")
    from tube_agent.lab.vision import SamClient
    segmenter = SamClient(args.sam_url or config["execution"]["sam_url"])
    try:
        model_path = resolve_model(read_json(parent / "collection.json")["plan"]["model_path"])
        if args.model is not None and resolve_model(args.model) != model_path:
            p.error("Model must resolve to the parent collection's original checkpoint")
        plan = freeze_branches(args, parent, root, config, model_path, segmenter.health())
        collect_t1.collect_shard(args, root, plan, config, model_path, segmenter, origin=ORIGIN,
                                 prefix_reader=lambda attempt: load_prefix(parent, attempt))
    finally:
        segmenter.close()
        if (root / "collection.json").is_file():
            collect_t1.update_summary(root)


if __name__ == "__main__":
    main()
