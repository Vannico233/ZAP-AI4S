"""Read-only end-of-GRPO audit for the registered v1 curriculum. Pass the run root."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


def check(condition, where):
    if not condition:
        raise ValueError(where)


def read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def round_metrics(run, spec, scenes, cfg_sha):
    n = spec["round"]
    phase = "base" if n <= 12 else "hard"
    label = f"round {n}"
    m = read(run/f"warmup/metrics/round_{n:02d}.json")
    check((m["round"],m["phase"],spec["phase"]) == (n,phase,phase), label+" phase/index")
    groups = spec["groups"]
    check([g["group"] for g in m["groups"]] == groups, label+" frozen groups")
    tasks = ("T1","T2","T3","T4") if n <= 12 else ("T3","T4")
    check(Counter(g["task"] for g in groups) == {t:2 for t in tasks}, label+" task allocation")
    check(len({g["id"] for g in groups}) == len(groups), label+" duplicate group ID")
    scheduled = {}
    for g in groups:
        scene = scenes[g["scene_id"]]
        check((scene["task"],scene["split"],scene["sha256"]) == (g["task"],"train",g["scene_sha256"]), label+" group scene")
        check(sorted(s["sample"] for s in g["rollouts"]) == list(range(8)), label+" scheduled sample indices")
        for s in g["rollouts"]:
            seed = s["policy_seed"]
            check(seed not in scheduled, f"{label} duplicate seed {seed}")
            scheduled[seed] = (g,s["sample"],scene)
    observed = set()
    tally = Counter()
    version = f"warmup_before_round_{n:02d}"
    for item in m["results"]:
        result = item["result"]
        seed = result["policy_seed"]
        ctx = f"{label}, policy seed {seed}"
        check(seed in scheduled and seed not in observed, ctx+" unexpected or repeated")
        observed.add(seed)
        g, sample, scene = scheduled[seed]
        path = run/"warmup"/"rollouts"/f"round_{n:02d}"/g["id"]/f"try_{sample:02d}"
        check(Path(item["directory"]).resolve() == path, ctx+" source directory")
        check(result["scene"] == scene and result["task"] == g["task"] and result["seed"] == scene["seed"], ctx+" scene/task")
        check((result["origin"],result["policy"],result["policy_version"],result["split"],result["error"],result["demonstration"])
              == ("warmup","model",version,"train",None,None), ctx+" provenance")
        check(result["config_sha256"] == cfg_sha,ctx+" config")
        prov = result["generator_provenance"]
        check(prov["policy_version"] == version and prov["adapter_sha256"] == m["adapter_before"],ctx+" sampled adapter")
        tally[g["task"]] += int(result["success"])
    check(len(observed) == len(scheduled) == m["episodes"],label+" episode count")
    delta = m["optimizer_steps_this_round"]
    check(type(delta) is int and 0 <= delta <= len(scheduled)//32, label+" update count")
    return m, observed, tally


def eval_evidence(run, scenes, tensor_hash, cfg_sha):
    directory = run/"evaluation/after_warmup"
    paths = [p for p in sorted(directory.glob("*/trajectory.jsonl"))
             if p.parent.name in scenes and scenes[p.parent.name]["split"] == "eval"]
    # Prefer a finished result so its recorded live tensor digest can also be checked.
    paths.sort(key=lambda p: not (p.parent/"result.json").is_file())
    for p in paths:
        with p.open("rb") as stream:
            line = stream.readline()
        if not line.endswith(b"\n"):
            continue
        row = json.loads(line)
        scene = scenes[p.parent.name]
        tag = str(p.relative_to(run))
        check(row["scene_id"] == scene["id"] and row["task"] == scene["task"],tag+" scene")
        seed = 30000000+scene["seed"]
        check((row["policy_version"],row["source_origin"],row["policy_seed"],row["step"]) ==
              ("after_warmup","evaluation",seed,1),tag+" first model action identity")
        check(row["response_ids"] and isinstance(row["raw_response"],str) and
              re.fullmatch("[0-9a-f]{64}",row["model_input_sha256"]),tag+" inference evidence")
        view_count = sum(v["type"] == "image_url" for v in row["messages"][-1]["content"])
        check(view_count == 2,tag+" number of images in the current observation")
        ev = {"scene":scene["id"],"step":row["step"],"model_action":row["action"],"response_tokens":len(row["response_ids"]),
              "model_input_sha256":row["model_input_sha256"],"trajectory":tag,"live_tensor_hash_match":None}
        result_path = p.parent/"result.json"
        if result_path.is_file():
            res = read(result_path)
            check(res["scene"] == scene and res["config_sha256"] == cfg_sha,tag+" result scene/config")
            check((res["origin"],res["policy"],res["policy_version"],res["policy_seed"],res["demonstration"],res["error"]) ==
                  ("evaluation","model","after_warmup",seed,None,None),tag+" evaluation result provenance")
            prov = res["generator_provenance"]
            check(prov["policy_version"] == "after_warmup" and prov["adapter_sha256"] == tensor_hash,tag+" live adapter tensor hash")
            ev.update(live_tensor_hash_match=True,live_tensor_sha256=prov["adapter_sha256"],success=res["success"])
        return ev
    return None


def audit(run):
    done_path = run/"warmup/DONE.json"
    if not done_path.is_file():
        return {"status":"pending_warmup"}
    done = read(done_path)
    cfg = read(run/"config.json")
    cfg_sha = digest(cfg)
    manifest = read(run/"scenes.json")
    specs = read(run/"schedule.json")
    check([s["round"] for s in specs] == list(range(1,19)), "frozen schedule rounds")
    check(manifest["config_sha256"] == cfg_sha == read(run/"provenance.json")["config_sha256"],"manifest/run config checksum")
    scenes = {s["id"]:s for s in manifest["scenes"]}
    check(len(scenes) == len(manifest["scenes"]), "duplicate registered scene ID")
    previous = None
    seen = set()
    updates = 0
    rounds = []
    wins = Counter()
    for spec in specs:
        n = spec["round"]
        try:
            m, observed, tally = round_metrics(run,spec,scenes,cfg_sha)
        except (KeyError,TypeError,FileNotFoundError) as exc:
            raise ValueError(f"round {n} metrics: {exc}") from exc
        check(not (seen & observed), f"round {n}: policy seed reused across rounds")
        seen.update(observed)
        if previous is not None:
            check(previous == m["adapter_before"],f"round {n}: adapter chain")
        previous = m["adapter_after"]
        check(isinstance(previous,str) and re.fullmatch("[0-9a-f]{64}",previous), f"round {n}: adapter digest")
        updates += m["optimizer_steps_this_round"]
        check(updates == m["optimizer_steps_total"],f"round {n}: cumulative optimizer count")
        wins.update(tally)
        rounds.append({"round":n,"episodes":len(observed),"updates":m["optimizer_steps_this_round"]})
    check((done["round"],done["rounds"],done["episodes"],done["optimizer_steps"]) == (18,18,len(seen),updates)
          and len(seen) == 960,"warmup completion vs registered episodes and optimizations")
    ck = run/"warmup/checkpoints/step_00018"
    frozen = run/"checkpoints/after_warmup"
    meta = read(ck/"COMPLETE.json")
    check(read(frozen/"COMPLETE.json") == meta,"frozen checkpoint metadata")
    check(meta["config_sha256"] == cfg_sha and (meta["counters"]["round"],meta["counters"]["optimizer_steps"]) ==
          (18,updates),"final checkpoint counters/config")
    node = read(frozen/"node.json")
    check(node["name"] == "after_warmup" and Path(node["source"]).resolve() == ck,"frozen node source")
    check(node["config_sha256"] == cfg_sha and node["scenes_sha256"] == file_hash(run/"scenes.json"),"frozen config/scenes")
    for name,key in (("adapter_model.safetensors","adapter_sha256"),("optimizer_rng.pt","optimizer_sha256")):
        for folder in (ck,frozen):
            check(file_hash(folder/name) == meta[key],str(folder/name)+" checksum")
    ev = eval_evidence(run,scenes,previous,cfg_sha)
    status = "pending_evaluation_action" if ev is None else (
        "verified" if ev["live_tensor_hash_match"] else "evaluation_started_pending_result")
    return {"status":status,"rounds":rounds,"episodes":len(seen),"optimizer_steps":updates,
            "warmup_successes":dict(wins),"final_checkpoint":str(ck),"frozen_node":str(frozen),
            "adapter_file_sha256":meta["adapter_sha256"],"optimizer_rng_sha256":meta["optimizer_sha256"],
            "final_live_tensor_sha256":previous,"evaluation":ev}


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected one argument: run root")
        root = Path(sys.argv[1]).resolve(strict=True)
        result = audit(root)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(json.dumps({"status":"invalid","error":str(exc)},separators=(",",":")))
        sys.exit(2)
    print(json.dumps(result,separators=(",",":")))
