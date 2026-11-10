"""Check every frozen scene's physical feasibility; outputs are NEVER model data."""
import argparse
import json
from pathlib import Path
from tube_agent.lab.experiment import write_json, read_json
from .config import load, scene_manifest, digest
from .env import CurriculumEnv, diagnostic_commands


def check(config, manifest, output, render=False, segmenter=None):
    from tube_agent.lab.vision import Cameras, Observer
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/"config.json").exists() and read_json(output/"config.json") != config:
        raise ValueError("Preflight configuration changed")
    write_json(output/"config.json", config)
    results = []
    for scene in manifest["scenes"]:
        path = output/(scene["id"]+".json")
        if path.exists():
            report = read_json(path)
            if (report["scene"] != scene or not report["result"]["success"] or
                    report.get("config_sha256") != digest(config) or
                    report.get("render_checked") != bool(render and scene["index"] == 0) or
                    report.get("sam_checked") != bool(segmenter and scene["index"] == 0)):
                raise ValueError("Invalid prior physics diagnostic")
            results.append(report); continue
        env = CurriculumEnv(scene, config)
        try:
            for name in ("target", "source"):
                floor = env.model.geom(name+"_floor")
                assert floor.size[2] == config["geometry"]["floor_z"]/2
                assert len([env.model.geom(f"{name}_wall_{axis}_{sign}") for axis in (0, 1) for sign in (-1, 1)]) == 4
            initial = env.result()
            if initial["success"]:
                raise AssertionError("Task already successful at initialization")
            if scene["task"] in ("T2", "T3") and not initial["held"]:
                raise AssertionError("Preheld initial state lacks a true grasp")
            if render and scene["index"] == 0:
                from PIL import Image
                cameras = Cameras(env)
                try:
                    for frame in cameras.capture():
                        Image.fromarray(frame["rgb"]).save(output/(scene["id"]+"_"+frame["calibration"]["name"]+".png"))
                finally: cameras.close()
            if segmenter and scene["index"] == 0:
                observer = Observer(env, segmenter)
                observer.queries = ["red test tube", "blue test tube rack" if scene["task"] == "T1" else "yellow test tube rack"]
                try:
                    obs, _ = observer.observe(output/(scene["id"]+"_vision"))
                    if obs["missing_queries"]:
                        raise RuntimeError(f"Initial scene has missing SAM queries: {scene['id']} {obs['missing_queries']}")
                finally: observer.close()
            actions = []
            for action in diagnostic_commands(env):
                feedback = env.step(action)[1]
                actions.append({"action": action, "feedback": feedback})
                if not feedback["accepted"]:
                    raise RuntimeError(f"Physical diagnostic rejected: {scene['id']} {feedback}")
                if env.ended: break
            result = env.result()
            report = {"scene": scene, "result": result, "actions": actions,
                      "config_sha256": digest(config), "render_checked": bool(render and scene["index"] == 0),
                      "sam_checked": bool(segmenter and scene["index"] == 0),
                      "policy": "privileged_physics_check", "training_eligible": False}
            write_json(path, report)
            results.append(report)
            print(json.dumps({"event": "physics_check", "scene": scene["id"], "success": result["success"],
                              "completed": len(results), "total": len(manifest["scenes"])}), flush=True)
            if not result["success"]:
                raise RuntimeError(f"Physical feasibility failed: {scene['id']} {result}")
        finally: env.close()
    report = {"passed": True, "scenes": len(results), "training_eligible": False,
              "render_checked": render, "sam_checked": bool(segmenter)}
    write_json(output/"report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path); p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--render", action="store_true")
    p.add_argument("--sam-url")
    a = p.parse_args(); c = load(a.config) if a.config else load()
    manifest = read_json(a.manifest) if a.manifest else scene_manifest(c)
    segmenter = None
    try:
        if a.sam_url:
            from tube_agent.lab.vision import SamClient
            segmenter = SamClient(a.sam_url); segmenter.health()
        check(c, manifest, a.output, a.render, segmenter)
    finally:
        if segmenter: segmenter.close()


if __name__ == "__main__": main()
