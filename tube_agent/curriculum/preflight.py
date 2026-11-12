"""Check every frozen scene's physical feasibility; outputs are NEVER model data."""
import argparse
import json
from pathlib import Path
from tube_agent.lab.experiment import write_json, read_json
from .config import BIMANUAL, load, scene_manifest, digest


def check(config, manifest, output, render=False, segmenter=None, all_views=False):
    from tube_agent.interface import make_env, make_observer
    from tube_agent.lab.vision import Cameras
    if config["action_protocol"] == "bimanual_camera_v1":
        from .bimanual import diagnostic_commands
    else:
        from .env import diagnostic_commands
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/"config.json").exists() and read_json(output/"config.json") != config:
        raise ValueError("Preflight configuration changed")
    write_json(output/"config.json", config)
    results = []
    for scene in manifest["scenes"]:
        inspect_views = all_views or scene["index"] == 0
        path = output/(scene["id"]+".json")
        if path.exists():
            report = read_json(path)
            if (report["scene"] != scene or not report["result"]["success"] or
                    report.get("config_sha256") != digest(config) or
                    report.get("render_checked") != bool(render and inspect_views) or
                    report.get("sam_checked") != bool(segmenter and inspect_views)):
                raise ValueError("Invalid prior physics diagnostic")
            results.append(report); continue
        env = make_env(scene, config)
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
            if render and inspect_views:
                from PIL import Image
                cameras = Cameras(env)
                try:
                    for frame in cameras.capture():
                        Image.fromarray(frame["rgb"]).save(output/(scene["id"]+"_"+frame["calibration"]["name"]+".png"))
                finally: cameras.close()
            if segmenter and inspect_views:
                observer = make_observer(env, segmenter)
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
                      "config_sha256": digest(config), "render_checked": bool(render and inspect_views),
                      "sam_checked": bool(segmenter and inspect_views),
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
    p.add_argument("--config", type=Path, default=BIMANUAL); p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--render", action="store_true")
    p.add_argument("--sam-url")
    p.add_argument("--scene", action="append", help="Check a registered scene ID; repeat to select several.")
    a = p.parse_args(); c = load(a.config)
    manifest = read_json(a.manifest) if a.manifest else scene_manifest(c)
    if a.scene:
        selected = set(a.scene)
        scenes = [s for s in manifest["scenes"] if s["id"] in selected]
        if len(scenes) != len(selected):
            p.error("Unregistered scene ID in --scene")
        manifest = {**manifest, "scenes": scenes}
    segmenter = None
    try:
        if a.sam_url:
            from tube_agent.lab.vision import SamClient
            segmenter = SamClient(a.sam_url); segmenter.health()
        check(c, manifest, a.output, a.render, segmenter, all_views=bool(a.scene))
    finally:
        if segmenter: segmenter.close()


if __name__ == "__main__": main()
