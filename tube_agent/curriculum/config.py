"""Frozen experiment specification, disjoint scene manifests and group schedule."""
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "configs/tube_curriculum_v1.json"
BIMANUAL = ROOT / "configs/tube_curriculum_bimanual.json"
TASKS = ("T1", "T2", "T3", "T4")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_training(settings):
    for name in ("warmup_minibatch", "sft_batch", "sft_epochs"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    for name in ("warmup_lr", "sft_lr", "max_grad_norm", "clip", "weight_decay"):
        value = settings[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} must be finite and nonnegative")
        if name != "weight_decay" and value == 0:
            raise ValueError(f"training.{name} must be positive")
    if settings["clip"] >= 1:
        raise ValueError("training.clip must be less than 1")


def load(path=DEFAULT, *, allow_focus=False):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if "extends" in c:
        if c.pop("extends") != DEFAULT.name:
            raise ValueError("Unknown base curriculum config")
        base = json.loads(DEFAULT.read_text(encoding="utf-8"))
        c["observation"] = {**base["observation"], **c["observation"]}
        c = {**base, **c}
    g, s, t = c["geometry"], c["success"], c["training"]
    focus = allow_focus and c.get("experiment") == "t1_focus_v1"
    bimanual = c["version"] == "tube_curriculum_bimanual_v1"
    protocol = "bimanual_camera_v1" if bimanual else "zap_pose_v1"
    if c["tasks"] != (["T1"] if focus else list(TASKS)) or c["action_protocol"] != protocol:
        raise ValueError("Unexpected task list or action protocol for this experiment")
    if bimanual and (c["interface"]["coordinate_frame"] != "middle_optical"
                     or c["observation"]["version"] != "bimanual_rgbd_v1"
                     or c["observation"]["cameras"] != ["middle", "left_wrist", "right_wrist"]):
        raise ValueError("Bimanual actions and all three camera views require the middle optical frame")
    if not g["tube_radius"] < g["hole_half_width"] < g["rack_outer_half_width"]:
        raise ValueError("Tube must physically fit a nonzero rack hole")
    if abs(s["target_depth_m"] - (g["target_rim_z"] - g["floor_z"])) > 1e-9:
        raise ValueError("Insertion depth must match the actual rim and floor")
    fixed = {"base_rounds": 12, "hard_rounds": 6, "groups_per_task": 2, "group_size": 8,
             "lora_rank": 32, "lora_alpha": 64, "warmup_epochs_per_batch": 1}
    validate_training(t)
    if focus:
        fixed.update(base_rounds=2, hard_rounds=0, groups_per_task=4, warmup_minibatch=32)
    if any(t[k] != v for k, v in fixed.items()):
        raise ValueError("Sampling and training budgets differ from the registered experiment")
    if t["turn_negative_advantage"] or t["failed_done_credit_clamp"] is not focus:
        raise ValueError("Unexpected action credit settings for this experiment")
    if c["seeds"]["train"] != 32 or c["seeds"]["eval"] != 30:
        raise ValueError("Expected 32 training and 30 evaluation scenes per task")
    if c["search"]["max_continuations"] != 128 or c["search"]["continuations_per_expansion"] != 4:
        raise ValueError("Search has one shared 128-continuation budget")
    return c


def scene_manifest(c):
    scenes = []
    g = c["geometry"]
    for task_index, task in enumerate(TASKS):
        for split, offset in (("train", 0), ("eval", 100000)):
            for index in range(c["seeds"][split]):
                seed = offset + task_index * 1000 + index
                rng = random.Random(c["seeds"]["manifest_seed"] + seed)
                jitter = g["scene_xy_jitter"]
                scene = dict(task=task, split=split, seed=seed, index=index,
                    id=f"{task}_{split}_{index:03d}",
                    source_xy=[round(v+rng.uniform(-jitter, jitter), 7) for v in g["source_xy"]],
                    target_xy=[round(v+rng.uniform(-jitter, jitter), 7) for v in g["target_xy"]],
                    insert_offset_xy=[round(rng.uniform(-g["insert_initial_xy_jitter"],
                                                        g["insert_initial_xy_jitter"]), 7) for _ in range(2)])
                scene["sha256"] = digest(scene)
                scenes.append(scene)
    return {"version": c["version"], "config_sha256": digest(c), "scenes": scenes}


def schedule(c, manifest):
    """Fresh independent initial-scene selection each round; no mixed-scene group."""
    rounds = []
    for index in range(18):
        tasks = TASKS if index < 12 else TASKS[2:]
        rng = random.Random(c["seeds"]["manifest_seed"] + 900000 + index)
        groups = []
        for task in tasks:
            pool = [s for s in manifest["scenes"] if s["task"] == task and s["split"] == "train"]
            for j, scene in enumerate(rng.sample(pool, 2)):
                groups.append({"id": f"round_{index+1:02d}_{task}_group_{j}",
                    "task": task, "scene_id": scene["id"], "scene_sha256": scene["sha256"],
                    "rollouts": [{"sample": k, "policy_seed": 10000000 + index*1000 + len(groups)*8+k}
                                 for k in range(8)]})
        rounds.append({"round": index+1, "phase": "base" if index < 12 else "hard", "groups": groups})
    return rounds


def plan(c):
    t = c["training"]
    per_task = t["groups_per_task"] * t["group_size"]
    steps = sum(rounds * math.ceil(per_task * tasks / t["warmup_minibatch"])
                for rounds, tasks in ((t["base_rounds"], len(TASKS)), (t["hard_rounds"], len(TASKS[2:]))))
    return {"environment": c["version"], "action_protocol": c["action_protocol"], "cameras": c["observation"]["cameras"],
            "tasks": list(TASKS), "train_scenes_per_task": 32, "eval_scenes_per_task": 30,
            "warmup_rounds": 18, "warmup_episodes": 960,
            "warmup_by_task": {"T1": 192, "T2": 192, "T3": 288, "T4": 288},
            "planned_warmup_optimizer_steps": steps, "warmup_minibatch_unit": "episodes",
            "search_max_continuations_global": 128, "search_max_expansions_global": 32,
            "sft_epochs": t["sft_epochs"], "sft_batch_unit": "unique assistant responses",
            "evaluation_episodes": 360, "checkpoints": ["original", "after_warmup", "after_sft"],
            "main_rl": False, "wall_clock_budget": None,
            "empty_success_dataset": "after_sft unavailable; never invent successful data or fake an SFT checkpoint"}
