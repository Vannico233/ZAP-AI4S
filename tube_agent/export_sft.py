"""Export only clean, successful MODEL rollouts from the TRAIN split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def export(source, output):
    counts = Counter()
    samples = []
    seen = set()
    for result_path in sorted(Path(source).rglob("result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("policy") != "model" or result.get("split") != "train":
            counts["excluded_policy_or_split"] += 1
            continue
        if not result.get("success") or result.get("error") or result.get("invalid_actions"):
            counts["unsuccessful_or_invalid"] += 1
            continue
        messages = json.loads((result_path.parent/"messages.json").read_text(encoding="utf-8"))
        if not messages or not all(isinstance(m.get("content"), str) for m in messages):
            counts["unsupported_multimodal_or_empty"] += 1
            continue
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if digest in seen:
            counts["duplicates"] += 1
            continue
        seen.add(digest)
        samples.append({"messages": messages, "source": {k: result[k] for k in
                       ("task", "split", "seed", "attempt", "policy", "model", "observation_mode")}})
    counts["exported_episodes"] = len(samples)
    if not samples:
        raise ValueError(f"No eligible self-generated successes; no file written. Counts: {dict(counts)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False)+"\n")
    output.with_suffix(".manifest.json").write_text(json.dumps(dict(counts), indent=2), encoding="utf-8")
    return dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
