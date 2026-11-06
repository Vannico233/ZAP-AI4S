"""Freeze a hand-authored controller and evaluate specified scene seed ranges."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tube_agent.env import TubeEnv
from tube_agent.manual import run_commands
from tube_agent.run import summarize


def make_spec(split, seed, controller, digest):
    env = TubeEnv(task="transport", split=split, seed=seed, max_steps=controller["max_steps"])
    try:
        observation = env.observe()
    finally:
        env.close()
    tube = next(obj for obj in observation["objects"] if obj["name"] == "tube_red")
    x, y, z = tube["center"]
    tx, ty, tz = observation["tray_center"]
    yaw = math.atan2(tube["axis"][1], tube["axis"][0])

    def move(a, b, c):
        return {"action": "move", "xyz": [a, b, c], "yaw": yaw}

    actions = [move(x, y, controller["approach_z"]),
               move(x, y, z + controller["grasp_center_offset_z"]),
               {"action": "gripper", "opening": 0.0},
               move(x, y, controller["carry_z"]),
               move(tx, ty, controller["carry_z"]),
               move(tx, ty, tz + tube["radius"] + controller["place_surface_radius_offset_z"]),
               {"action": "gripper", "opening": controller["open_width"]},
               move(tx, ty, controller["withdraw_z"]), {"action": "finish"}]
    return {"task": "transport", "split": split, "seed": seed,
            "max_steps": controller["max_steps"], "actions": actions,
            "metadata": {"controller_version": controller["version"], "controller_sha256": digest,
                         "provenance": controller["provenance"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=Path("examples/manual_transport_controller_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20, help="Scenes per split, including every failure")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 99980:
        parser.error("episodes must be in [1, 99980]")
    source = args.controller.read_bytes()
    controller, digest = json.loads(source), hashlib.sha256(source).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {"validation": 0, "test": 20, "ood": 20}
    config = {"policy": "manual", "controller": controller, "controller_sha256": digest,
              "seed_ranges": {split: [start, start+args.episodes-1] for split, start in groups.items()},
              "new_model_inference": False, "rl_training": False}
    (args.output/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {}
    for split, start in groups.items():
        records = []
        for seed in range(start, start+args.episodes):
            spec = make_spec(split, seed, controller, digest)
            result = run_commands(spec, args.output/split/f"seed_{seed:05d}")
            records.append(result)
            print(json.dumps({"split": split, "seed": seed, "success": result["success"],
                              "invalid_actions": result["invalid_actions"], "error": result["error"]}), flush=True)
            summary[split] = {**summarize(records), "policy": "manual", "seed_start": start,
                              "seed_end": seed, "failed_seeds": [r["seed"] for r in records if not r["success"]]}
            (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"split_summary": split, **summary[split]}), flush=True)
    print(f"Saved: {args.output}", flush=True)
    if any(group["errors"] for group in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
