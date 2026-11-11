"""Replay the three recorded round-10 prefixes and exercise the original cameras."""
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import warnings

import mujoco
import numpy as np
import torch

from tube_agent.curriculum.env import CurriculumEnv
from tube_agent.curriculum.rollout import read_rows, same_state
from tube_agent.lab.policy import VisualPolicy, execute
from tube_agent.lab.vision import Cameras


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def active_egl_device():
    # EGL_EXT_device_query / EGL_NV_device_cuda are optional driver extensions.
    egl = ctypes.CDLL("libEGL.so.1")
    ptr = ctypes.c_void_p
    egl.eglGetCurrentDisplay.restype = ptr
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ptr
    display = egl.eglGetCurrentDisplay()
    addr = egl.eglGetProcAddress(b"eglQueryDisplayAttribEXT")
    devaddr = egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not (display and addr and devaddr):
        return None
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))
    device, ordinal = ctypes.c_ssize_t(), ctypes.c_ssize_t()
    if query(addr)(display, 0x322C, ctypes.byref(device)) and query(devaddr)(device.value, 0x323A, ctypes.byref(ordinal)):
        return ordinal.value
    return None


def frame_quality(data, channel):
    if channel == "rgb":
        return {"mean_rgb_level_0_255": float(np.mean(data))}
    return {"finite_fraction": float(np.isfinite(data).mean()),
            "nan_fraction": float(np.isnan(data).mean()),
            "inf_fraction": float(np.isinf(data).mean())}


def static_comparison(current, baseline, channel):
    if channel == "rgb":
        difference = np.abs(current.astype(np.int16) - baseline.astype(np.int16))
        # A pixel exceeds the threshold if any one of its three channels does.
        peak = difference.max(axis=2)
        means = difference.mean(axis=(0, 1))
        return {"changed_pixel_fraction": float((peak > 0).mean()),
                "pixel_fraction_over_8_levels": float((peak > 8).mean()),
                "pixel_fraction_over_32_levels": float((peak > 32).mean()),
                **{f"mae_{c}_levels": float(v) for c, v in zip(("r", "g", "b"), means)}}
    finite, baseline_finite = np.isfinite(current), np.isfinite(baseline)
    both = finite & baseline_finite
    metrics = {"finite_mask_disagreement_fraction": float((finite != baseline_finite).mean()),
               "jointly_finite_fraction": float(both.mean())}
    if both.any():
        # Use only co-located, jointly finite pixels for metric differences.
        difference = np.abs(current[both].astype(np.float64) - baseline[both].astype(np.float64))
        metrics.update(mae_m=float(difference.mean()),
                       fraction_over_1mm=float((difference > 0.001).mean()),
                       fraction_over_1cm=float((difference > 0.01).mean()))
    return metrics


def summaries(series):
    return {k: {"n": len(values), "min": min(values), "mean": float(np.mean(values)), "max": max(values)}
            for k, values in series.items()}


def run(args):
    rank, mode, out = args.rank, args.mode, args.out
    seed = (10009012, 10009019, 10009026)[rank]
    run_dir = Path("runs/tube_curriculum/v1_launchfix")
    config = json.loads((run_dir/"config.json").read_text())
    schedule = json.loads((run_dir/"schedule.json").read_text())[9]
    samples = [(g, sample) for g in schedule["groups"] for sample in g["rollouts"]]
    [(idx, group, sample)] = [(i,g,s) for i,(g,s) in enumerate(samples) if s["policy_seed"] == seed]
    assert idx % 3 == rank
    episode = run_dir/"warmup/rollouts/round_10"/group["id"]/f"try_{sample['sample']:02d}"
    init = json.loads((episode/"initialization.json").read_text())
    rows = [{"action":r["action"], "feedback":r["feedback"], "state":r["state_after"]} for r in read_rows(episode)]
    if len(rows) != (3,11,11)[rank] or init["scene"]["id"] != group["scene_id"]:
        raise ValueError("Recorded prefix differs from the failed launch")
    torch.cuda.set_device(rank)
    cuda_token = torch.zeros(1, device=f"cuda:{rank}")
    print("CUDA context:",cuda_token.device,flush=True)
    channels = ["rgb","depth"] if mode == "rgbd" else [mode]
    state = {"rank": rank,"seed":seed,"scene":init["scene"]["id"],"mode":mode,
             "prefix_steps":len(rows),"valid_reads":0,"bad_reads":0,"completed_reads":0}
    counts, bad, messages, mismatches = Counter(), Counter(), Counter(), Counter()
    first_bad = {}
    depth_range = {}
    egl_devices = []
    read_quality = defaultdict(lambda: defaultdict(list))
    static_quality = {}
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    atol = config["search"]["replay_atol"]
    # Keep the last call on disk if the native driver aborts without unwinding Python.
    def event(phase, camera=None, channel=None):
        save(out/f"last_event-{rank}.json",{**state,"phase":phase,"camera":camera,"channel":channel})
    for repetition in range(2):
        state["context"] = repetition
        event("new_environment")
        env = CurriculumEnv(init["scene"],config)
        cams = None
        try:
            if not same_state(env.snapshot(),init["state"],atol):
                raise RuntimeError("Initial physics or judge state diverged")
            observer = SimpleNamespace(queries=[])
            policy = VisualPolicy(None,action_protocol="zap_pose_v1")
            if repetition == 0:
                (args.barrier/f"ready-{rank}").touch()
                deadline = time.monotonic()+125
                while not (args.barrier/"go").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Render barrier timed out")
                    time.sleep(.04)
            event("create_renderer")
            cams = Cameras(env)
            cams.renderer._gl_context.make_current()
            egl_devices.append(active_egl_device())
            static_hashes = {}
            static_frames, comparisons = {}, defaultdict(lambda: defaultdict(list))
            paired_counts = Counter()
            for frame in range(32):
                state["frame"] = frame
                for camera in cams.names:
                    event("update_scene",camera)
                    cams.renderer.disable_depth_rendering()
                    cams.renderer.update_scene(env.data,camera=camera,scene_option=options)
                    for channel in channels:
                        if channel == "depth":
                            cams.renderer.enable_depth_rendering()
                        event("enter_read",camera,channel)
                        with warnings.catch_warnings(record=True) as warns:
                            warnings.simplefilter("always")
                            data = cams.renderer.render().copy()
                        key = f"{camera}/{channel}"
                        counts[key] += 1
                        state["completed_reads"] += 1
                        messages.update(f"{key}: {w.message}" for w in warns)
                        for metric, v in frame_quality(data, channel).items():
                            read_quality[key][metric].append(v)
                        if channel == "rgb":
                            valid = data.shape == (480,640,3) and data.dtype == np.uint8 and data.std() >= 1
                        else:
                            valid = data.shape == (480,640) and np.issubdtype(data.dtype,np.floating)
                            valid = valid and np.isfinite(data).all() and data.min() > 0
                            finite = data[np.isfinite(data)]
                            if finite.size:
                                lo,hi = float(finite.min()),float(finite.max())
                                old = depth_range.get(key,[lo,hi])
                                depth_range[key] = [min(old[0],lo),max(old[1],hi)]
                                valid = valid and hi <= env.model.stat.extent*env.model.vis.map.zfar*1.01
                        if not valid:
                            bad[key] += 1
                            first_bad.setdefault(key,[repetition,frame])
                        state["valid_reads" if valid else "bad_reads"] += 1
                        if frame >= len(rows):
                            value = hashlib.blake2b(data,digest_size=16).hexdigest()
                            if key in static_hashes and static_hashes[key] != value:
                                mismatches[key] += 1
                            static_hashes.setdefault(key,value)
                            if key not in static_frames:
                                static_frames[key] = data.copy()
                            elif data.shape == static_frames[key].shape and (channel != "rgb" or data.ndim == 3 and data.shape[-1] == 3):
                                paired_counts[key] += 1
                                for metric, v in static_comparison(data, static_frames[key], channel).items():
                                    comparisons[key][metric].append(v)
                    cams.renderer.disable_depth_rendering()
                if frame < len(rows):
                    row = rows[frame]
                    feedback = execute(env,observer,policy,row["action"])
                    if feedback != row["feedback"] or not same_state(env.snapshot(),row["state"],atol):
                        raise RuntimeError(f"Recorded replay diverged at step {frame+1}")
            static_quality[str(repetition)] = {
                k: {"baseline_quality": frame_quality(b, k.split("/")[-1]), "comparisons": paired_counts[k],
                    "metrics": summaries(comparisons[k])}
                for k, b in static_frames.items()}
            event("close_renderer")
        finally:
            if cams:
                cams.close()
            env.close()
    result = {**state,"read_counts":counts,"invalid_counts":bad,"first_invalid":first_bad,
              "warnings":messages,"static_hash_mismatches":mismatches,"depth_range_m":depth_range,
              "egl_cuda_device_attribute":egl_devices,"visible_cuda_devices":os.getenv("CUDA_VISIBLE_DEVICES"),
              "mujoco_egl_device_id":os.getenv("MUJOCO_EGL_DEVICE_ID"),"mujoco":mujoco.__version__,
              "all_read_quality": {k: summaries(series) for k, series in read_quality.items()},
              "static_quality_by_context": static_quality,
              "verified_replays":2}
    save(out/f"result-{rank}.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank",required=True,type=int,choices=range(3))
    p.add_argument("--mode",required=True,choices=("rgb","depth","rgbd"))
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--barrier",required=True,type=Path)
    run(p.parse_args())
