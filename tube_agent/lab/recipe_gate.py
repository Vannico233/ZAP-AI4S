"""SFT admission for self-generated successes, including GRPO warm-up policies."""
import hashlib
from pathlib import Path
from .experiment import TRAIN_TASK_NAMES, eligible, fingerprint, read_json, rows, write_json
from .success_gate import scan, sha256, replay_check


def provenance_allowed(path, result, output, model):
    """Only this run's base, committed warm-up, or frozen warm-up search data."""
    output, path = Path(output).resolve(), Path(path).resolve()
    if not path.is_relative_to(output):
        return False
    origin = result.get("generator_provenance") or {}
    if origin.get("model") != model:
        return False
    relative = path.relative_to(output)
    if relative.parts[0] == "base_fixed":
        return origin.get("training") is False and origin.get("adapter") is None
    if relative.parts[0] == "warmup":
        update = origin.get("policy_update")
        if type(update) is not int or update < 0:
            return False
        return (origin.get("training") is True and origin.get("on_policy") is True
                and Path(origin.get("training_stage", "")).resolve() == output/"warmup"
                and len(relative.parts) >= 5 and relative.parts[1:3] == ("rollouts", f"update_{update:04d}")
                and (output/"warmup"/f"step_{update+1:05d}"/"COMPLETE.json").is_file())
    if relative.parts[0].startswith("mine_round_"):
        done = output/"warmup/DONE.json"
        if not done.is_file():
            return False
        adapter = read_json(done).get("checkpoint")
        return bool(adapter) and origin.get("training") is False and origin.get("adapter") == adapter
    return False


def inspect_gate(output, sources, seed, model, targets, minimum, verifier=None):
    from .policy import VisualPolicy, parse_visual_action
    output = Path(output)
    verifier = verifier or replay_check
    policy = VisualPolicy(None, action_protocol="zap_pose_v1")
    prompt_hash = hashlib.sha256(policy.system.encode()).hexdigest()
    cache_path = output/"recipe_gate_replays.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    approved = {t: [] for t in TRAIN_TASK_NAMES}
    seen = {t: set() for t in TRAIN_TASK_NAMES}
    rejected = []
    for path, result in scan(sources, seed):
        if (not eligible(result) or result.get("action_protocol") != "zap_pose_v1"
                or result.get("prompt_sha256") != prompt_hash
                or not provenance_allowed(path, result, output, model)):
            continue
        trajectory = path.parent/"trajectory.jsonl"
        record = {"result": str(path.resolve()), "result_sha256": sha256(path),
                  "trajectory": str(trajectory.resolve()), "trajectory_sha256": sha256(trajectory)}
        key = fingerprint(record)
        try:
            trace = rows(trajectory)
            if not trace or len(trace) != result.get("steps"):
                raise ValueError("Incomplete trace")
            for row in trace:
                if row["messages"][0] != {"role": "system", "content": policy.system}:
                    raise ValueError("Prompt mismatch")
                if row["feedback"].get("accepted") is not True:
                    raise ValueError("Rejected turn")
                if parse_visual_action(row["raw_response"])[0] != row["action"]:
                    raise ValueError("Executed action differs from model response")
            identity = fingerprint([r["action"] for r in trace])
            if identity in seen[result["task"]]:
                continue
            if key not in cache:
                if verifier(path.parent) is not True:
                    raise ValueError("Physical replay failed")
                cache[key] = {"passed": True}
                write_json(cache_path, cache)
            if not cache[key]["passed"]:
                raise ValueError(cache[key]["error"])
            seen[result["task"]].add(identity)
            approved[result["task"]].append({**record, "action_sha256": identity})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            cache[key] = {"passed": False, "error": str(exc)}
            write_json(cache_path, cache)
            rejected.append({**record, "error": str(exc)})
    counts = {t: len(v) for t, v in approved.items()}
    gate = {"version": "zap_self_success_v1", "passed": all(n >= minimum for n in counts.values()),
            "seed": seed, "model": model, "minimum_per_task": minimum, "targets": targets,
            "targets_met": all(counts[t] >= targets[t] for t in TRAIN_TASK_NAMES),
            "unique_verified_successes": counts, "approved": approved, "replay_failures": rejected,
            "interpretation": "SFT data admission, not an independent policy success-rate estimate"}
    write_json(output/"recipe_success_gate.json", gate)
    return gate


def require_gate(output):
    path = Path(output)/"recipe_success_gate.json"
    if not path.is_file():
        raise RuntimeError("SFT/post-SFT RL forbidden: recipe success gate missing")
    gate = read_json(path)
    if not gate.get("passed") or gate.get("version") != "zap_self_success_v1" or gate["minimum_per_task"] < 1:
        raise RuntimeError("SFT/post-SFT RL forbidden: recipe data gate did not pass")
    for task in TRAIN_TASK_NAMES:
        records = gate["approved"][task]
        if len({r["action_sha256"] for r in records}) < gate["minimum_per_task"]:
            raise RuntimeError("SFT/post-SFT RL forbidden: incomplete task coverage")
        for record in records:
            for name in ("result", "trajectory"):
                if sha256(record[name]) != record[name+"_sha256"]:
                    raise RuntimeError("SFT/post-SFT RL forbidden: source evidence changed")
            result = read_json(record["result"])
            if not eligible(result) or not provenance_allowed(record["result"], result, output, gate["model"]):
                raise RuntimeError("SFT/post-SFT RL forbidden: inadmissible source")
    return gate
