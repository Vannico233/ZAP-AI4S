"""Fixed-node greedy evaluation: 30 held-out initial states per task."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from tube_agent.lab.experiment import read_json, write_json
from tube_agent.lab.vision import SamClient
from .config import load, file_digest
from .learning import load_model, Generator
from .rollout import episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--model", required=True)
    p.add_argument("--node", choices=("original", "after_warmup", "after_sft"), required=True)
    p.add_argument("--shard", type=int, required=True); p.add_argument("--shards", type=int, default=3)
    a = p.parse_args(); root = a.run.resolve(); c = load(root/"config.json")
    adapter = None if a.node == "original" else root/"checkpoints"/a.node
    if adapter:
        complete = read_json(adapter/"COMPLETE.json")
        if file_digest(adapter/"adapter_model.safetensors") != complete["adapter_sha256"]:
            raise ValueError("Frozen model node was modified")
    model, processor = load_model(a.model, c, adapter, trainable=False)
    generator = Generator(model, processor, c); frozen = generator.bind(a.node)
    segmenter = SamClient(c["execution"]["sam_url"])
    scenes = [s for s in read_json(root/"scenes.json")["scenes"] if s["split"] == "eval"]
    if len(scenes) != 120: raise ValueError("Expected 120 fixed evaluation initial states")
    try:
        segmenter.health()
        for scene in scenes[a.shard::a.shards]:
            directory = root/"evaluation"/a.node/scene["id"]
            if (directory/"result.json").exists() and read_json(directory/"result.json").get("error"):
                directory.rename(directory.with_name(directory.name+".incomplete-"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
            if (directory/"result.json").exists():
                result = read_json(directory/"result.json")
                if result["scene"] != scene or result["policy_version"] != a.node or result["generator_provenance"]["adapter_sha256"] != frozen:
                    raise ValueError("Cached evaluation belongs to another scene/model")
            else:
                if directory.exists(): directory.rename(directory.with_name(directory.name+".incomplete-"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
                result = episode(scene, c, directory, generator, segmenter, 30000000+scene["seed"], a.node,
                                 temperature=0., origin="evaluation", save_frames=False)
            if result["error"]: raise RuntimeError(result["error"])
        write_json(root/"evaluation"/a.node/f"shard_{a.shard}_DONE.json", {"checkpoint": a.node, "shard": a.shard})
    finally: segmenter.close()


if __name__ == "__main__": main()
