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

    if entry in {"tube_agent.curriculum.evaluate", "tube_agent.curriculum.preflight",
                 "tube_agent.curriculum.collect_t1"}:
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
        if (rgb.sum(axis=2, dtype=np.uint16) < 9).mean() > .70:
            raise ValueError(f"{name}: more than 70 percent near-black pixels")
        if depth.shape != shape or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"{name}: unexpected depth shape or dtype")
        if not np.isfinite(depth).all():
            raise ValueError(f"{name}: nonfinite depth")
        lo, hi = float(depth.min()), float(depth.max())
        if lo <= 0 or hi > far * 1.01:
            raise ValueError(f"{name}: depth [{lo}, {hi}] outside valid range")
        stats[name] = {"rgb_std": variation, "depth_min_m": lo, "depth_max_m": hi}
    return stats


def drift(first, second):
    import numpy as np
    result = {}
    for a, b in zip(first, second):
        rgb_delta = np.max(np.abs(a["rgb"].astype(np.int16)-b["rgb"].astype(np.int16)), axis=2)
        depth_delta = np.abs(a["depth"]-b["depth"])
        result[a["calibration"]["name"]] = {
            "rgb_fraction_over_8":float((rgb_delta>8).mean()),
            "depth_fraction_over_1mm":float((depth_delta>.001).mean())}
    return result


def stable_capture(cameras, capture, record, evidence_dir):
    """Only deliver a fresh observation after two consecutive physical readbacks agree."""
    previous = None
    reads = []
    comparisons = []
    def evidence():
        if record.get("readback_evidence"):
            return
        from PIL import Image
        folder = evidence_dir / f"readback-{os.getpid()}"
        folder.mkdir(exist_ok=True)
        for i, frames in enumerate(reads):
            for frame in frames:
                cam = frame["calibration"]["name"]
                Image.fromarray(frame["rgb"]).save(folder/f"read-{i+1}-{cam}.png")
        (folder/"comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")
        record["readback_evidence"] = str(folder)

    for index in range(3):
        frames = capture(cameras)
        reads.append(frames)
        try:
            stats = frame_stats(cameras, frames)
        except ValueError as exc:
            comparisons.append({"read":index+1,"rejected":str(exc)})
            record["rejected_raw_reads"] = record.get("rejected_raw_reads",0)+1
            previous = None
            continue
        if previous is not None:
            changes = drift(previous, frames)
            comparisons.append({"ending_read":index+1,"drift":changes})
            if all(v <= .001 for camera in changes.values() for v in camera.values()):
                if index == 2:
                    record["three_read_captures"] = record.get("three_read_captures",0)+1
                    evidence()
                return frames, stats
        previous = frames
    evidence()
    raise RuntimeError(f"No stable valid pair of raw RGB-D observations in three reads: {comparisons}")


def select_renderer(env, rank, count):
    """Select a backend in this worker only; CUDA visibility stays untouched."""
    from egl_devices import enumerate_devices, rank_map
    physical = env["CUDA_VISIBLE_DEVICES"].split(",")[rank]
    software_mask = env.get("ZAP_SOFTWARE_RENDER_GPUS", "")
    if software_mask:
        requested = software_mask.split(",")
        if len(requested) != len(set(requested)) or not set(requested) <= {"0", "2", "3"}:
            raise ValueError("ZAP_SOFTWARE_RENDER_GPUS must list distinct policy GPUs from 0,2,3")
        if physical in requested:
            backend = env.get("ZAP_SOFTWARE_RENDERER", "osmesa")
            if backend != "osmesa":
                raise ValueError(f"Unsupported software backend: {backend}")
            env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LIBGL_ALWAYS_SOFTWARE="true",
                       GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS=env.get("LP_NUM_THREADS", "4"))
            env.pop("MUJOCO_EGL_DEVICE_ID", None)
            return {"backend": backend, "policy_gpu": physical, "egl_index": None}

    info = enumerate_devices(expected_cuda_devices=count)
    selected = rank_map(info, require_identity=True)[rank]
    if selected["cuda_ordinal"] != rank:
        raise RuntimeError("EGL device identity and CUDA ordinal disagree")
    env["MUJOCO_EGL_DEVICE_ID"] = str(selected["egl_index"])
    return dict(selected, backend="nvidia-egl", policy_gpu=physical)


def active_renderer(selected, rank):
    if selected["backend"] == "nvidia-egl":
        from egl_devices import active_egl_device
        actual = active_egl_device()
        if actual != rank:
            raise RuntimeError(f"Wrong active EGL CUDA ordinal: {actual}, expected {rank}")
        return {"observed_cuda_ordinal": actual}
    from OpenGL import GL
    renderer = (GL.glGetString(GL.GL_RENDERER) or b"").decode("ascii", errors="replace")
    if "llvmpipe" not in renderer.lower():
        raise RuntimeError(f"Expected a software llvmpipe context, found {renderer!r}")
    return {"gl_renderer": renderer, "observed_cuda_ordinal": None}


def configure_worker(argv, env):
    target = worker_target(argv, env)
    if not target:
        return
    entry, rank, count = target
    import sys
    if "mujoco" in sys.modules or "OpenGL.platform" in sys.modules:
        raise RuntimeError("MuJoCo/EGL imported before renderer device selection")
    if env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    selected = select_renderer(env, rank, count)
    record = {"entry": entry, "rank": rank, "pid": os.getpid(), "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
              "selected": selected, "verified_contexts": 0, "valid_captures": 0, "invalid_captures": 0}
    path = Path(env["ZAP_EGL_AUDIT_DIR"])/f"egl-binding-{os.getpid()}.json"

    def publish():
        record["time"] = datetime.now(timezone.utc).isoformat()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf8")
        temporary.replace(path)

    publish()
    print(json.dumps({"event":"egl_binding_selected","entry":entry,"rank":rank,**selected}), flush=True)
    from tube_agent.lab.vision import Cameras
    original_init, original_capture = Cameras.__init__, Cameras.capture

    @wraps(original_init)
    def checked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            self.renderer._gl_context.make_current()
            identity = active_renderer(selected, rank)
            self.zap_renderer_identity = {**selected, **identity}
            record.update(identity)
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
                frames, stats = stable_capture(self, original_capture, record, path.parent)
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
