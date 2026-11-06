from pathlib import Path
import json

DEFAULT_MODEL = "/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model"


def resolve_model(path):
    """Accept a checkpoint or its parent, but never silently choose among models."""
    root = Path(path).expanduser().resolve()
    if (root/"config.json").is_file():
        return root
    matches = []
    for config in root.glob("*/config.json"):
        info = json.loads(config.read_text(encoding="utf-8"))
        if info.get("model_type") == "qwen3_5":
            matches.append(config.parent)
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Specify the Qwen checkpoint directory containing config.json. Candidates under {root}: {matches}")
