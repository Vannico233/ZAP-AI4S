"""Audited success-only dataset; frozen train scenes and exact response contexts."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tube_agent.lab.experiment import read_json, write_json
from .config import load, digest, file_digest, TASKS
from .rollout import read_rows, replay


def warmup_episodes(root):
    root = Path(root)
    for number in range(1, 19):
        if not (root/"warmup/checkpoints"/f"step_{number:05d}/COMPLETE.json").exists(): continue
        for result in sorted((root/"warmup/rollouts"/f"round_{number:02d}").glob("*/try_*/result.json")):
            if ".incomplete-" not in str(result): yield result


def source_episodes(root):
    yield from warmup_episodes(root)
    for result in sorted((Path(root)/"search/rollouts").glob("expansion_*/try_*/result.json")):
        if ".incomplete-" not in str(result): yield result


def freeze(root):
    root = Path(root); c = load(root/"config.json")
    target = root/"data"; target.mkdir(exist_ok=True)
    if (target/"manifest.json").exists():
        manifest = read_json(target/"manifest.json")
        if file_digest(target/"samples.jsonl") != manifest["samples_sha256"]:
            raise ValueError("Frozen SFT samples changed")
        return manifest
    train = {s["id"]: s for s in read_json(root/"scenes.json")["scenes"] if s["split"] == "train"}
    accepted = []; samples = []; seen_traces = set(); seen_turns = set(); coverage = {t: set() for t in TASKS}
    counts = Counter(); duplicates = 0
    for path in source_episodes(root):
        result = read_json(path)
        if not result.get("success") or result.get("error"): continue
        if (result.get("policy") != "model" or result.get("demonstration") is not None or
                result["scene"] != train.get(result["scene"]["id"]) or result["config_sha256"] != digest(c)):
            raise ValueError("A success candidate has incompatible scene/policy/config provenance")
        rows = list(read_rows(path.parent))
        if not rows or not rows[-1]["judge_after"]["success"]:
            raise ValueError("Success result disagrees with terminal object-state judge")
        if any("response_ids" not in row or "model_input_sha256" not in row for row in rows):
            raise ValueError("Success lacks exact model inputs/generated token IDs")
        signature = digest([result["scene"]["sha256"],
            [(row["model_input_sha256"], row["response_ids"], row["action"]) for row in rows]])
        if signature in seen_traces:
            duplicates += 1; continue
        verified = replay(result["scene"], c, rows)
        if not verified["success"]: raise ValueError("A successful candidate failed physical replay")
        seen_traces.add(signature)
        trajectory = path.parent/"trajectory.jsonl"; checksum = file_digest(trajectory)
        offsets = []
        with trajectory.open("rb") as stream:
            while True:
                offset = stream.tell(); line = stream.readline()
                if not line: break
                if line.strip(): offsets.append(offset)
        origin = result["origin"]
        accepted.append({"trajectory": str(trajectory.resolve()), "trajectory_sha256": checksum,
            "result_sha256": file_digest(path), "task": result["task"], "scene_id": result["scene"]["id"],
            "origin": origin, "trajectory_signature": signature, "replay_verified": True,
            "invalid_actions": result["invalid_actions"]})
        counts[result["task"]] += 1; coverage[result["task"]].add(result["scene"]["id"])
        for index, row in enumerate(rows):
            key = digest([row["model_input_sha256"], row["response_ids"]])
            if key in seen_turns: continue
            seen_turns.add(key)
            samples.append({"trajectory": str(trajectory.resolve()), "trajectory_sha256": checksum,
                "row": index, "byte_offset": offsets[index], "scene_split": "train", "task": result["task"], "origin": origin,
                "input_response_sha256": key})
    data_path = target/"samples.jsonl"
    temporary = target/"samples.partial.jsonl"
    temporary.write_text("".join(json.dumps(sample)+"\n" for sample in samples), encoding="utf-8")
    temporary.replace(data_path)
    manifest = {"frozen": True, "config_sha256": digest(c), "samples": len(samples),
                "samples_sha256": file_digest(data_path), "trajectories": accepted,
                "successes_by_task": {t: counts[t] for t in TASKS},
                "scene_coverage": {t: sorted(coverage[t]) for t in TASKS},
                "missing_tasks": [t for t in TASKS if not counts[t]],
                "duplicates_removed": duplicates,
                "shared_prefix_policy": "identical input/response assistant turns appear once",
                "includes_tree_search": any(v["origin"] == "search" for v in accepted),
                "evaluation_data_used": False}
    write_json(target/"manifest.json", manifest)
    return manifest


def read_sample(sample, verified_files):
    path = Path(sample["trajectory"])
    identity = (str(path), sample["trajectory_sha256"])
    if identity not in verified_files:
        if file_digest(path) != sample["trajectory_sha256"]:
            raise ValueError("Frozen training trajectory changed")
        verified_files.add(identity)
    with path.open("rb") as stream:
        stream.seek(sample["byte_offset"])
        row = json.loads(stream.readline())
    if digest([row["model_input_sha256"], row["response_ids"]]) != sample["input_response_sha256"]:
        raise ValueError("Frozen training sample/input alignment changed")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run", type=Path, required=True)
    print(json.dumps(freeze(p.parse_args().run), indent=2))
