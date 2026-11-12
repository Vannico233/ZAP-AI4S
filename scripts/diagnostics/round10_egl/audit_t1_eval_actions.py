"""Count model output, action validation, and token endings in two saved T1 evaluations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root, run, node, eos, cap):
    baseline = read(root / "baselines" / (node + ".json"))
    totals, kinds, accepted, parser_reasons, rejected_actions, reasons = (Counter() for _ in range(6))
    for scene in baseline["results"]:
        path = run / "evaluation" / node / scene["scene_id"] / "result.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == scene["result_sha256"]
        assert (scene["policy_seed"] == 30000000 + read(path)["seed"] and scene["scene_id"].startswith("T1_eval_"))
        steps = rejected = 0
        with path.with_name("trajectory.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                steps += 1
                assert (row["step"] == steps and row["scene_id"] == scene["scene_id"] and
                        row["source_origin"] == "evaluation" and row["policy_version"] == node and
                        row["policy_seed"] == scene["policy_seed"])
                tokens = row["response_ids"]
                at_cap = len(tokens) == cap
                ends_eos = bool(tokens and tokens[-1] in eos)
                action = row["action"].get("action")
                action = action if isinstance(action, str) and len(action) <= 50 else "<missing_or_other>"
                kinds[action] += 1
                passed = row["feedback"]["accepted"]
                parsed = row["response_format"]["mode"] != "invalid"
                totals["turns"] += 1
                totals["at_token_cap"] += at_cap
                totals["ended_in_eos"] += ends_eos
                if passed:
                    accepted[action] += 1
                    continue
                rejected += 1
                if parsed:
                    rejected_actions[action] += 1
                    reasons[str(row["feedback"].get("reason",""))[:135]] += 1
                    totals["parsed_but_rejected"] += 1
                else:
                    parser_reasons[str(row["response_format"].get("reason","")).split(": line ")[0]] += 1
                    totals["parser_rejected"] += 1
                    totals["parser_rejected_at_cap_without_eos"] += at_cap and not ends_eos
                    totals["parser_rejected_with_eos"] += ends_eos
                    totals["parser_rejected_multiple_action_keys_in_text"] += row["raw_response"].count('"action"') >= 2
                totals["first_action_rejected"] += steps == 1
        assert steps == scene["steps"] and rejected == scene["invalid_actions"]
    assert totals["turns"] == baseline["steps"] and sum(accepted.values()) == baseline["accepted_actions"]
    assert totals["parser_rejected"] + totals["parsed_but_rejected"] == baseline["invalid_actions"]
    return {"episodes":baseline["episodes"],"successes":baseline["success"],"counts":dict(totals),
            "accepted_by_action":dict(accepted),"parsed_but_rejected_by_action":dict(rejected_actions),
            "parser_failure_reasons":dict(parser_reasons),"top_execution_rejections":reasons.most_common(8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("focus_run", type=Path)
    args = parser.parse_args()
    root = args.focus_run.resolve()
    parent = Path(read(root/"SOURCE.json")["parent_run"])
    model = Path(read(root/"provenance.json")["model"]["resolved_path"])
    ids = read(model/"generation_config.json")["eos_token_id"]
    eos = {ids} if isinstance(ids, int) else set(ids)
    cap = read(root/"config.json")["policy"]["max_new_tokens"]
    assert read(parent/"config.json")["policy"]["max_new_tokens"] == cap
    print(json.dumps({"generation_cap":cap,"eos_ids":sorted(eos),
        **{node:summarize(root,run,node,eos,cap) for run,node in [(parent,"original"),(root,"source")]}},ensure_ascii=True))


if __name__ == "__main__":
    main()
