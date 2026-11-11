"""Bind rendering to this policy worker's GPU and check its RGB-D observations."""
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import warnings


def worker_target(argv, env):
    args = iter(argv[1:])
    for arg in args:
        if arg == "-m":
            entry = next(args, None)
            break
        if arg == "-c":
            return None
        if not arg.startswith("-"):
            entry = Path(arg).name
            break
    else:
        return None

    if entry in {"tube_agent.curriculum.evaluate", "tube_agent.curriculum.preflight"}:
        if env.get("CUDA_VISIBLE_DEVICES") not in {"0", "2", "3"}:
            raise RuntimeError("Expected a single visible policy GPU for this curriculum worker")
        return entry, 0, 1
    if entry not in {"tube_agent.curriculum.train", "tube_agent.curriculum.search", "real_sampling_smoke.py"}:
        return None
    rank = env.get("LOCAL_RANK")
    if rank is None and entry == "real_sampling_smoke.py":
        arguments = list(args)
        if "--rank" not in arguments:
            return None
        rank = arguments[arguments.index("--rank")+1]
    if rank not in {"0", "1", "2"} or env.get("CUDA_VISIBLE_DEVICES") != "0,2,3":
        raise RuntimeError("Expected LOCAL_RANK 0/1/2 and CUDA_VISIBLE_DEVICES=0,2,3")
    return entry, int(rank), 3


def frame_stats(cameras, frames):
    import numpy as np
    if len(frames) != len(cameras.names):
        raise ValueError("Wrong number of camera views")
    shape = cameras.height, cameras.width
    far = float(cameras.env.model.stat.extent * cameras.env.model.vis.map.zfar)
    stats = {}
    for name, frame in zip(cameras.names, frames):
        if frame["calibration"]["name"] != name:
            raise ValueError(f"Wrong camera identity: {name}")
        rgb, depth = frame["rgb"], frame["depth"]
        if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"{name}: unexpected RGB shape or dtype")
        variation = float(rgb.std())
        if variation < 1.0:
            raise ValueError(f"{name}: blank RGB frame (std={variation})")
        if depth.shape != shape or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"{name}: unexpected depth shape or dtype")
        if not np.isfinite(depth).all():
            raise ValueError(f"{name}: nonfinite depth")
        lo, hi = float(depth.min()), float(depth.max())
        if lo <= 0 or hi > far * 1.01:
            raise ValueError(f"{name}: depth [{lo}, {hi}] outside valid range")
        stats[name] = {"rgb_std": variation, "depth_min_m": lo, "depth_max_m": hi}
    return stats


def configure_worker(argv, env):
    target = worker_target(argv, env)
    if not target:
        return
    entry, rank, count = target
    from egl_devices import active_egl_device, enumerate_devices, rank_map
    import sys
    if "mujoco" in sys.modules or "OpenGL.EGL" in sys.modules:
        raise RuntimeError("MuJoCo/EGL imported before renderer device selection")
    if env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    info = enumerate_devices(expected_cuda_devices=count)
    selected = rank_map(info, require_identity=True)[rank]
    if selected["cuda_ordinal"] != rank:
        raise RuntimeError("EGL device identity and CUDA ordinal disagree")
    env["MUJOCO_EGL_DEVICE_ID"] = str(selected["egl_index"])
    record = {"entry": entry, "rank": rank, "pid": os.getpid(), "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
              "selected": selected, "verified_contexts": 0, "valid_captures": 0, "invalid_captures": 0}
    path = Path(env["ZAP_EGL_AUDIT_DIR"])/f"egl-binding-{os.getpid()}.json"

    def publish():
        record["time"] = datetime.now(timezone.utc).isoformat()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf8")
        temporary.replace(path)

    publish()
    print(json.dumps({"event":"egl_binding_selected","entry":entry,"rank":rank,"egl_index":selected["egl_index"],
                      "pci":selected["pci"],"uuid":selected["uuid"]}), flush=True)
    from tube_agent.lab.vision import Cameras
    original_init, original_capture = Cameras.__init__, Cameras.capture

    @wraps(original_init)
    def checked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            self.renderer._gl_context.make_current()
            actual = active_egl_device()
            record["observed_cuda_ordinal"] = actual
            if actual != rank:
                raise RuntimeError(f"Wrong active EGL CUDA ordinal: {actual}, expected {rank}")
            record["verified_contexts"] += 1
        except Exception as exc:
            record["error"] = str(exc)
            self.close()
            raise
        finally:
            publish()

    @wraps(original_capture)
    def checked_capture(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning, module=r"mujoco\.renderer")
                frames = original_capture(self)
            stats = frame_stats(self, frames)
        except Exception as exc:
            record["invalid_captures"] += 1
            record["error"] = str(exc)
            publish()
            raise
        record["valid_captures"] += 1
        record.setdefault("first_frames", stats)
        record["latest_frames"] = stats
        publish()
        return frames

    Cameras.__init__ = checked_init
    Cameras.capture = checked_capture
