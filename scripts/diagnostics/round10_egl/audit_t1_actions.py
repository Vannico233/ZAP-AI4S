"""Read only registered T1 episode metrics and at most six complete warm-up trajectories."""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trim(value, limit=140):
    text = " ".join(str(value or "").split())
    text = re.sub(r"data:[^ ]*?;base64,[A-Za-z0-9+/=]+", "[image]", text)
    text = re.sub(r"[A-Za-z0-9+/=]{60,}", "[encoded]", text)
    return text[:limit]


def rows_and_signs(metric):
    values = [item for item in metric["results"] if item["result"]["task"] == "T1"]
    means = {g["group"]["id"]: g for g in metric["groups"] if g["group"]["task"] == "T1"}
    pairs = []
    for item in values:
        result = item["result"]
        path = Path(item["directory"])
        group = means[path.parent.name]
        i = int(path.name.removeprefix("try_"))
        if not math.isclose(result["reward"], group["rewards"][i], abs_tol=1e-9):
            raise ValueError(f"Saved individual and group reward disagree: {path}")
        if result["error"] or result["policy"] != "model" or result["demonstration"] is not None:
            raise ValueError(f"Unexpected provenance: {path}")
        pairs.append((result, path, result["reward"] - group["mean"]))
    return pairs


def summarize(pairs):
    wins = Counter()
    for r, _, advantage in pairs:
        wins["episodes"] += 1
        wins["steps"] += r["steps"]
        wins["invalid_actions"] += r["invalid_actions"]
        wins["ever_grasped"] += int(r["ever_grasped"])
        wins["success"] += int(r["success"])
        if not r["success"] and r["termination"] == "done":
            wins["unsuccessful_done_termination"] += 1
            wins["unsuccessful_done_positive_trajectory_advantage"] += int(advantage > 1e-12)
    return dict(wins)


def inspect(pair, cap, collect_example):
    r, path, adv = pair
    counts = Counter()
    reasons, kinds = Counter(), Counter()
    witness = None
    last_action = None
    with (path / "trajectory.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["policy_seed"],row["scene_id"],row["source_origin"]) != (
                    r["policy_seed"],r["scene"]["id"],"warmup"):
                raise ValueError(f"Row provenance mismatch: {path}")
            act = row["action"]["action"]
            last_action = act
            parsed = row["response_format"]
            accepted = row["feedback"]["accepted"]
            ntokens = len(row["response_ids"])
            counts["turns"] += 1
            counts["at_generation_token_cap"] += int(ntokens >= cap)
            kinds[act if re.fullmatch(r"[a-z_]{1,35}",act) else "other"] += 1
            if not accepted:
                counts["rejected"] += 1
                invalid_format = parsed["mode"] == "invalid" or act == "invalid_model_output"
                reason = trim(parsed.get("reason") if invalid_format else row["feedback"].get("reason"), 95)
                category = "parser_rejection" if invalid_format else "move_to_rejection" if act == "move_to" else "other_action_rejection"
                counts[category] += 1
                counts[category+"_at_cap"] += int(ntokens >= cap)
                reasons[f"{category}: {reason}"] += 1
                if collect_example and witness is None and invalid_format:
                    raw = trim(row.get("raw_response"),10000)
                    witness = {"seed":r["policy_seed"],"step":row["step"],"response_tokens":ntokens,
                               "action_json_keys_in_text":raw.count('"action"'),"reason":reason,
                               "response_excerpt":raw if len(raw)<208 else raw[:110]+" ... "+raw[-88:]}
    if counts["turns"] != r["steps"] or counts["rejected"] != r["invalid_actions"]:
        raise ValueError(f"Saved actions disagree with episode metrics: {path}")
    evidence = {"scene":r["scene"]["id"],"seed":r["policy_seed"],"success":r["success"],
                "termination":r["termination"],"last_action":last_action,"group_centered_reward":round(adv,6),
                "counts":dict(counts),"actions":dict(kinds),"top_rejections":reasons.most_common(4)}
    return evidence,witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    cap = read(root/"config.json")["policy"]["max_new_tokens"]
    result = {"source":"saved warmup model rollouts; no demonstration",
              "trajectory_selection":"Within rounds 1, 10, 12, select the lowest invalid fraction (prefer success), then highest invalid fraction among the rest; break ties by policy seed. Purposive sample.",
              "token_cap":cap,"full_t1_round_metrics":{},"sampled_episodes":{},"examples":[]}
    all_pairs=[]
    for n in range(1,13):
        pairs = rows_and_signs(read(root/f"warmup/metrics/round_{n:02}.json"))
        result["full_t1_round_metrics"][n] = summarize(pairs)
        all_pairs.extend(pairs)
        if n not in {1,10,12}:
            continue
        clean = min(pairs,key=lambda p:(not p[0]["success"],p[0]["invalid_actions"]/max(1,p[0]["steps"]),p[0]["policy_seed"]))
        noisy = min((p for p in pairs if p[1]!=clean[1]),key=lambda p:(-p[0]["invalid_actions"]/max(1,p[0]["steps"]),p[0]["policy_seed"]))
        result["sampled_episodes"][n]=[]
        for pair in (clean,noisy):
            data,ex=inspect(pair,cap,collect_example=n in {10,12} and not any(x["round"]==n for x in result["examples"]))
            result["sampled_episodes"][n].append(data)
            if ex:
                result["examples"].append({"round":n,**ex})
    result["all_t1_metrics"]=summarize(all_pairs)
    print(json.dumps(result,ensure_ascii=True,separators=(",",":")))


if __name__=="__main__":
    main()
