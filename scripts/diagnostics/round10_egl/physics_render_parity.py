"""Compare MuJoCo's actual GPU 0 and OSMesa camera pixels on frozen scene resets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def body_masks(cameras, env, name):
    import mujoco
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    r = cameras.renderer
    r.enable_segmentation_rendering()
    try:
        r.update_scene(env.data, camera=name, scene_option=options)
        ids = r.render().copy()
    finally:
        r.disable_segmentation_rendering()
    labels = np.zeros(ids.shape[:2], dtype=np.uint8)
    for i, body in enumerate(("tube_red", "source_rack", "target_rack"), 1):
        geoms = np.flatnonzero(env.model.geom_bodyid == env.model.body(body).id)
        labels[(ids[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(ids[:, :, 0], geoms)] = i
    return labels


def capture(args):
    # Bind before importing any module that initializes an EGL context.
    if args.backend == "egl":
        from egl_devices import active_egl_device, enumerate_devices, rank_map
        device = rank_map(enumerate_devices(1), require_identity=True)[0]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device["egl_index"])
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString
    from tube_agent.curriculum.config import load
    from tube_agent.curriculum.env import CurriculumEnv
    from tube_agent.lab.vision import Cameras
    from zap_egl_runtime import drift, stable_capture
    configs = load(args.run / "config.json")
    manifest = json.loads((args.run / "scenes.json").read_text())
    scenes = {s["id"]: s for s in manifest["scenes"] if s["split"] == "train"}
    folder = args.out / args.backend
    folder.mkdir()
    for scene in args.scenes:
        env = CurriculumEnv(scenes[scene], configs)
        cameras = None
        try:
            cameras = Cameras(env)
            cameras.renderer._gl_context.make_current()
            vendor, renderer = [glGetString(key).decode() for key in (GL_VENDOR, GL_RENDERER)]
            if args.backend == "osmesa" and "llvmpipe" not in renderer.lower():
                raise RuntimeError(f"Expected llvmpipe, found {vendor} / {renderer}")
            if args.backend == "egl" and ("nvidia" not in vendor.lower() or active_egl_device() != 0):
                raise RuntimeError(f"Expected active NVIDIA EGL device for CUDA ordinal 0: {vendor} / {renderer}")
            meta = {"scene": scenes[scene], "state": env.snapshot(), "vendor": vendor,
                    "renderer": renderer, "views": {}, "training_eligible": False}
            if args.backend == "egl":
                meta["device"] = device
            log = {}
            frames, stats = stable_capture(cameras, Cameras.capture, log, folder)
            second, _ = stable_capture(cameras, Cameras.capture, log, folder)
            change = drift(frames, second)
            if any(v > .001 for diffs in change.values() for v in diffs.values()):
                raise RuntimeError(f"{scene}: same-pose reference unstable: {change}")
            meta.update(frame_stats=stats, readbacks=log, repeat_drift=change)
            for fr in frames:
                name = fr["calibration"]["name"]
                meta["views"][name] = fr["calibration"]
                np.savez_compressed(folder / f"{scene}-{name}.npz", rgb=fr["rgb"], depth=fr["depth"],
                                    object_masks=body_masks(cameras, env, name))
            write(folder / f"{scene}.json", meta)
            print(f"{args.backend}: captured {scene} with {vendor} / {renderer}", flush=True)
        finally:
            if cameras is not None:
                cameras.close()
            env.close()


def compare_arrays(a, b):
    delta = np.abs(a["rgb"].astype(float) - b["rgb"])
    color = delta.max(axis=2)
    depth = np.abs(a["depth"] - b["depth"])
    stats = {"rgb_mean_absolute_levels": float(delta.mean()),
             "rgb_p95_max_channel_levels": float(np.quantile(color, .95)),
             "rgb_fraction_over_24_levels": float((color > 24).mean()),
             "depth_p95_m": float(np.quantile(depth, .95)),
             "depth_fraction_over_5mm": float((depth > .005).mean()), "objects": {}}
    passed = (stats["rgb_mean_absolute_levels"] <= 5 and stats["rgb_fraction_over_24_levels"] <= .04
              and stats["depth_fraction_over_5mm"] <= .02)
    # Compare small foreground objects separately; whole-frame statistics can miss a tube.
    for i, name in enumerate(("tube", "source_rack", "target_rack"), 1):
        a_mask, b_mask = a["object_masks"] == i, b["object_masks"] == i
        union, joint = int(np.count_nonzero(a_mask | b_mask)), a_mask & b_mask
        pixels = int(joint.sum())
        m = {"gpu_area_px": int(a_mask.sum()), "software_area_px": int(b_mask.sum()),
             "union_px": union, "iou": pixels / union if union else None}
        if union >= 16:
            m["rgb_mean_absolute_levels"] = float(delta[joint].mean()) if pixels else None
            m["median_depth_delta_m"] = float(np.median(depth[joint])) if pixels else None
            passed = passed and (m["iou"] >= .85 and m["rgb_mean_absolute_levels"] <= 15
                                 and m["median_depth_delta_m"] <= .002)
        stats["objects"][name] = m
    return passed, stats


def compare(args):
    report = {"passed": True, "training_eligible": False, "scenes": {}}
    for scene in args.scenes:
        gpu, cpu = [args.out / name for name in ("egl", "osmesa")]
        a, b = [json.loads((p / f"{scene}.json").read_text()) for p in (gpu, cpu)]
        same = a["scene"] == b["scene"] and a["state"] == b["state"] and a["views"] == b["views"]
        result = {"same_scene_state_and_calibration": same, "views": {}, "passed": same}
        for view in ("top", "side"):
            with np.load(gpu / f"{scene}-{view}.npz") as x, np.load(cpu / f"{scene}-{view}.npz") as y:
                valid, stats = compare_arrays(x, y)
            result["views"][view] = dict(stats, passed=valid)
            result["passed"] &= valid
        for obj in ("tube", "source_rack", "target_rack"):
            result["passed"] &= any(min(v["objects"][obj]["gpu_area_px"],
                                        v["objects"][obj]["software_area_px"]) >= 16
                                    for v in result["views"].values())
        report["scenes"][scene] = result
        report["passed"] &= result["passed"]
    write(args.out / "comparison.json", report)
    print(json.dumps({"passed": report["passed"], "scenes": list(report["scenes"])}), flush=True)
    if not report["passed"]:
        raise SystemExit("Camera parity failed: inspect comparison.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", default=["T1_train_008", "T2_train_019", "T3_train_000", "T4_train_000"])
    p.add_argument("--software-lib-dir", type=Path, action="append")
    p.add_argument("--backend", choices=["egl", "osmesa"], help=argparse.SUPPRESS)
    args = p.parse_args()
    args.root, args.run, args.out = [path.resolve() for path in (args.root, args.run, args.out)]
    if args.out == args.run or args.out.is_relative_to(args.run):
        p.error("Parity output must be outside the training run")
    if args.backend:
        capture(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("egl", "osmesa"):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parent), str(args.root),
                                                            os.environ.get("PYTHONPATH", "")]),
                   OMP_NUM_THREADS="2", LP_NUM_THREADS="2", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   MUJOCO_GL=name, TUBE_MUJOCO_GL=name, PYOPENGL_PLATFORM=name,
                   CUDA_VISIBLE_DEVICES="0" if name == "egl" else "")
        env.pop("ZAP_EGL_AUDIT_DIR", None)
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
        libraries = [args.root / ".runtime/egl-jammy/lib"]
        if name == "osmesa":
            libraries = args.software_lib_dir or [args.root / ".runtime/osmesa-jammy/lib"]
            env.update(GALLIUM_DRIVER="llvmpipe", LIBGL_ALWAYS_SOFTWARE="true")
        else:
            for key in ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE"):
                env.pop(key, None)
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*(str(path.resolve()) for path in libraries),
                                                   env.get("LD_LIBRARY_PATH", "")])
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                   "--run", str(args.run), "--out", str(args.out), "--backend", name, "--scenes", *args.scenes]
        subprocess.run(command, env=env, check=True, cwd=args.root)
    compare(args)


if __name__ == "__main__":
    main()
