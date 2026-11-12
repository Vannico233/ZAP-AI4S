"""Audited success-only dataset; frozen train scenes and exact response contexts."""
import argparse
import json
from pathlib import Path
from tube_agent.lab.experiment import read_json
from tube_agent.posttrain import freeze
from .config import digest, file_digest
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
