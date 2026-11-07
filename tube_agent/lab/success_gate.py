"""Auditable, fail-closed admission to post-training. No GPU/model inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .experiment import TRAIN_TASK_NAMES, eligible, fingerprint, read_json, rows, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_check(directory):
    """Replay actual saved actions and compare every physical state, without SAM."""
    import numpy as np
    from .env import LabEnv
    from .zap_env import make_env, ZapLabEnv
    from .policy import VisualPolicy, execute
    from .run import state
    directory = Path(directory)
    result = read_json(directory / "result.json")
    trajectory = rows(directory / "trajectory.jsonl")
    if not trajectory or len(trajectory) != result["steps"]:
        raise ValueError("Empty or incomplete trajectory")
    expected=ZapLabEnv.VERSION if result.get("action_protocol")=="zap_pose_v1" else LabEnv.VERSION
    if result["environment_version"] != expected:
        raise ValueError("Environment version differs")
    env = make_env(result["task"], result["split"], result["seed"],
                 max_steps=result["max_steps"], action_protocol=result["action_protocol"])
    observer = type("ReplayObserver", (), {"queries": []})()
    policy = VisualPolicy(None)
    try:
        for step, row in enumerate(trajectory, 1):
            if row["step"] != step:
                raise ValueError("Non-sequential trajectory")
            feedback = execute(env, observer, policy, row["action"])
            if feedback.get("accepted") is not True or row["feedback"].get("accepted") is not True:
                raise ValueError("Rejected action")
            if not np.allclose(state(env), row["state_after"], atol=1e-5, rtol=0):
                raise ValueError(f"State diverged at step {step}")
        if not env.result()["success"]:
            raise ValueError("Replay did not complete the task")
    finally:
        env.close()
    return True


def scan(sources, seed):
    for source in sources:
        for path in sorted(Path(source).rglob("result.json")):
            if any(".incomplete-" in s for s in path.parts):
                continue
            result = read_json(path)
            if result.get("seed") != seed:
                continue
            yield path, result


def inspect_gate(output, sources, qualification, seed, minimum=10, attempts=10, wins=3,
                 verifier=None):
    verifier = verifier or replay_check
    output = Path(output)
    cache_path = output / "gate_replays.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    approved = {task: [] for task in TRAIN_TASK_NAMES}
    seen = {task: set() for task in TRAIN_TASK_NAMES}
    rejected = []

    def verify(path, result):
        # Old oracle data, demonstrations and trained-policy rollouts cannot be
        # admitted as successes of this fixed base-model acquisition campaign.
        provenance = result.get("generator_provenance") or {}
        if not eligible(result) or provenance.get("adapter") or provenance.get("training") is not False:
            return None
        trajectory = path.parent / "trajectory.jsonl"
        record = {"result": str(path.resolve()), "result_sha256": sha256(path),
                  "trajectory": str(trajectory.resolve()), "trajectory_sha256": sha256(trajectory)}
        key = fingerprint(record)
        if key not in cache:
            try:
                if verifier(path.parent) is not True:
                    raise ValueError("Replay verifier did not pass")
                cache[key] = {"passed": True}
            except Exception as exc:
                cache[key] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            write_json(cache_path, cache)
        if not cache[key]["passed"]:
            rejected.append({**record, **cache[key]})
            return None
        return record

    for path, result in scan(sources, seed):
        if not eligible(result):
            continue
        task = result["task"]
        identity = fingerprint([row["action"] for row in rows(path.parent / "trajectory.jsonl")])
        if identity in seen[task]:
            continue
        record = verify(path, result)
        if record:
            seen[task].add(identity)
            approved[task].append({**record, "action_sha256": identity})

    stats = {task: {"episodes": 0, "successes": 0, "errors": 0, "records": []} for task in TRAIN_TASK_NAMES}
    for path, result in scan([qualification], seed):
        task = result.get("task")
        if task not in stats:
            continue
        item = stats[task]
        item["episodes"] += 1
        item["errors"] += int(bool(result.get("error")))
        if result.get("search_prefix_length", 0):
            item["errors"] += 1
            continue
        record = verify(path, result)
        if record:
            item["successes"] += 1
            item["records"].append(record)
    counts = {task: len(value) for task, value in approved.items()}
    enough = all(n >= minimum for n in counts.values())
    passed = enough and all(s["episodes"] == attempts and s["errors"] == 0 and s["successes"] >= wins
                            for s in stats.values())
    report = {"version": "success_first_v1", "passed": passed, "seed": seed,
              "requirements": {"unique_successes_per_task": minimum, "independent_attempts": attempts,
                               "independent_successes": wins},
              "unique_verified_successes": counts, "acquisition_ready": enough,
              "approved": approved, "qualification": stats, "replay_failures": rejected,
              "interpretation": "Fixed training scene admission only; not generalization evaluation"}
    write_json(output / "success_gate.json", report)
    return report


def require_gate(output):
    path = Path(output) / "success_gate.json"
    if not path.is_file():
        raise RuntimeError("Post-training forbidden: success_gate.json is missing")
    gate = read_json(path)
    req = gate["requirements"]
    if not gate.get("passed"):
        raise RuntimeError("Post-training forbidden: success gate did not pass")
    for task in TRAIN_TASK_NAMES:
        accepted = gate["approved"][task]
        independent = gate["qualification"][task]
        if (len({r["action_sha256"] for r in accepted}) < req["unique_successes_per_task"]
            or independent["episodes"] != req["independent_attempts"] or independent["errors"]
            or independent["successes"] < req["independent_successes"]
            or len(independent["records"]) != independent["successes"]):
            raise RuntimeError("Post-training forbidden: incomplete task evidence")
        for record in accepted + independent["records"]:
            for name in ("result", "trajectory"):
                if sha256(record[name]) != record[name + "_sha256"]:
                    raise RuntimeError("Post-training forbidden: source evidence changed")
    return gate


def diagnostics(output, sources, seed):
    """Bounded text report: failures and best episodes for the next debug round."""
    tasks = {t: {"episodes": 0, "successes": 0, "rejections": Counter(),
                 "unreached_moves": 0, "missing_observations": 0, "best": []} for t in TRAIN_TASK_NAMES}
    for path, result in scan(sources, seed):
        task = result.get("task")
        if task not in tasks:
            continue
        item = tasks[task]; item["episodes"] += 1; item["successes"] += int(bool(result.get("success")))
        item["missing_observations"] += result.get("missing_detection_observations", 0)
        item["best"].append({"episode": str(path.parent), "reward": result.get("reward", 0),
                             "success": result.get("success"), "error": result.get("error")})
        trajectory = path.parent / "trajectory.jsonl"
        for row in rows(trajectory) if trajectory.exists() else []:
            f = row.get("feedback", {})
            if f.get("accepted") is False: item["rejections"][f.get("reason", "unknown")] += 1
            if f.get("reached") is False: item["unreached_moves"] += 1
        item["best"] = sorted(item["best"], key=lambda r: r["reward"], reverse=True)[:3]
    write_json(Path(output) / "acquisition_diagnostics.json", tasks)
    return tasks
