"""Coordinate short RGB/depth probes. Run on Linux with the project's Python."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_worker_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return {"read_error": str(exc)}


def stop_own_workers(workers):
    live = [p for p in workers.values() if p.poll() is None]
    for p in live:
        p.terminate()
    deadline = time.monotonic() + 4
    for p in live:
        try:
            p.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def enumerate_devices():
    """Obtain GPU IDs with the worker's exact CUDA mask, in a fresh process."""
    import ctypes as C
    ptr, integer, boolean = C.c_void_p, C.c_int, C.c_uint
    pi = C.POINTER(integer)
    egl = C.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [C.c_char_p]
    egl.eglGetProcAddress.restype = ptr
    def proc(name, output, *args):
        address = egl.eglGetProcAddress(name.encode())
        return C.CFUNCTYPE(output, *args)(address) if address else None
    devices = proc("eglQueryDevicesEXT", boolean, integer, C.POINTER(ptr), pi)
    attrib = proc("eglQueryDeviceAttribEXT", boolean, ptr, integer, C.POINTER(C.c_ssize_t))
    string = proc("eglQueryDeviceStringEXT", C.c_char_p, ptr, integer)
    binary = proc("eglQueryDeviceBinaryEXT", boolean, ptr, integer, integer, ptr, pi)
    count = integer()
    if not devices or not attrib or not string or not devices(0, None, C.byref(count)) or count.value < 3:
        raise RuntimeError("EGL device enumeration/attribute queries unavailable")
    handles = (ptr * count.value)()
    if not devices(len(handles), handles, C.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    egl_rows = []
    for index, device in enumerate(handles[:count.value]):
        extensions = (string(device, 0x3055) or b"").split()
        ordinal = C.c_ssize_t(-1)
        gpu_uuid = (C.c_ubyte * 16)()
        size = integer()
        row = {"egl_index": index, "cuda_ordinal": None, "uuid": None, "pci": None}
        if b"EGL_NV_device_cuda" in extensions and attrib(device, 0x323A, C.byref(ordinal)):
            row["cuda_ordinal"] = ordinal.value
        if binary and b"EGL_EXT_device_persistent_id" in extensions:
            if binary(device, 0x335C, 16, gpu_uuid, C.byref(size)) and size.value == 16:
                row["uuid"] = bytes(gpu_uuid).hex()
        for ext, key in [(b"EGL_EXT_device_drm_render_node", 0x3377),(b"EGL_EXT_device_drm", 0x3233)]:
            node = string(device,key) if ext in extensions else None
            if node:
                device_link = Path("/sys/class/drm") / Path(os.fsdecode(node)).name / "device"
                if device_link.exists():
                    row["pci"] = device_link.resolve().name.lower()
                    break
        egl_rows.append(row)
    cuda = C.CDLL("libcuda.so.1")
    def check(ret):
        if ret != 0:
            raise RuntimeError(f"CUDA device identification failed ({ret})")
    check(cuda.cuInit(0))
    check(cuda.cuDeviceGetCount(C.byref(count)))
    if count.value != 3:
        raise RuntimeError(f"Expected 3 visible CUDA devices, found {count.value}")
    cuda_rows = []
    for rank in range(count.value):
        handle, bus, gpu_uuid = integer(), C.create_string_buffer(40), (C.c_ubyte * 16)()
        check(cuda.cuDeviceGet(C.byref(handle), rank))
        check(cuda.cuDeviceGetPCIBusId(bus, len(bus), handle))
        check(cuda.cuDeviceGetUuid(C.byref(gpu_uuid), handle))
        cuda_rows.append({"rank":rank, "uuid":bytes(gpu_uuid).hex(), "pci":bus.value.decode().lower()})
    return {"egl": egl_rows, "cuda": cuda_rows}


def rank_map(info):
    selected = {}
    for gpu in info["cuda"]:
        matches = []
        for row in info["egl"]:
            if row["cuda_ordinal"] is None or row["cuda_ordinal"] < 0:
                continue
            identity = [key for key in ("uuid","pci") if row[key] is not None]
            if identity and all(row[key] == gpu[key] for key in identity):
                matches.append({**row, "identity_verified_by":identity})
        if not matches:
            # The extension returns the CUDA ordinal in this process. Only use
            # this fallback when the complete EGL ordinal set fits this mask.
            reported = [r["cuda_ordinal"] for r in info["egl"] if r["cuda_ordinal"] is not None and r["cuda_ordinal"] >= 0]
            if sorted(reported) == list(range(len(info["cuda"]))):
                matches = [{**r, "identity_verified_by":["same-process CUDA ordinal only"]}
                           for r in info["egl"] if r["cuda_ordinal"] == gpu["rank"] and not r["uuid"] and not r["pci"]]
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous/unavailable EGL identity for rank {gpu['rank']}: {len(matches)} matches")
        selected[gpu["rank"]] = matches[0]
    if len({r["egl_index"] for r in selected.values()}) != len(info["cuda"]):
        raise RuntimeError("EGL devices do not have a unique mapping to CUDA workers")
    return selected


def run_case(root, worker, out, layout, mode, binding):
    case_dir = out / f"{layout}-{mode}"
    case_dir.mkdir()
    barrier = case_dir / "barrier"
    barrier.mkdir()
    ranks = [2] if layout == "single" else [0, 1, 2]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,2,3"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    entry = {"layout": layout, "mode": mode, "ranks": ranks, "egl_binding": binding, "started": timestamp(),
             "status": "starting", "barrier_released": False}
    save(case_dir / "summary.json", entry)
    workers = {}
    try:
        for rank in ranks:
            worker_env = env.copy()
            if binding:
                worker_env["MUJOCO_EGL_DEVICE_ID"] = str(binding[rank]["egl_index"])
            args = [sys.executable, "-u", str(worker), "--rank", str(rank), "--mode", mode,
                    "--out", str(case_dir), "--barrier", str(barrier)]
            with (case_dir/f"worker-{rank}.stdout").open("wb") as stdout, \
                    (case_dir/f"worker-{rank}.stderr").open("wb") as stderr:
                workers[rank] = subprocess.Popen(args, cwd=root, env=worker_env, stdin=subprocess.DEVNULL,
                                                 stdout=stdout, stderr=stderr)

        deadline = time.monotonic() + 120
        while True:
            ready = [rank for rank in ranks if (barrier / f"ready-{rank}").is_file()]
            if len(ready) == len(ranks) and all(p.poll() is None for p in workers.values()):
                (barrier / "go").touch(exist_ok=False)
                entry.update(status="rendering", barrier_released=True, released_at=timestamp())
                save(case_dir / "summary.json", entry)
                break
            if all(p.poll() is not None for p in workers.values()):
                entry["status"] = "exited_before_barrier"
                break
            if time.monotonic() >= deadline:
                entry["status"] = "readiness_timeout"
                break
            time.sleep(.15)
        entry["ready_ranks"] = ready

        if entry["barrier_released"]:
            deadline = time.monotonic() + 180
            while any(p.poll() is None for p in workers.values()):
                if time.monotonic() >= deadline:
                    entry["status"] = "render_timeout"
                    break
                time.sleep(.15)
            if entry["status"] == "rendering":
                entry["status"] = "finished" if all(p.returncode == 0 for p in workers.values()) else "worker_failed"
    finally:
        entry["stopped_by_driver"] = [rank for rank, p in workers.items() if p.poll() is None]
        stop_own_workers(workers)
        entry["workers"] = {
            rank: {"pid": p.pid, "returncode": p.returncode,
                   "result": read_worker_json(case_dir / f"result-{rank}.json"),
                   "last_event": read_worker_json(case_dir / f"last_event-{rank}.json")}
            for rank, p in workers.items()
        }
        if binding:
            for rank, data in entry["workers"].items():
                observed = (data["result"] or {}).get("egl_cuda_device_attribute", [])
                data["active_egl_verified"] = bool(observed) and all(
                    ordinal == binding[rank]["cuda_ordinal"] for ordinal in observed)
                if entry["status"] == "finished" and not data["active_egl_verified"]:
                    entry["status"] = "binding_unverified"
        entry["ended"] = timestamp()
        save(case_dir / "summary.json", entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", choices=("rgb", "depth", "rgbd"), default=["rgb", "depth", "rgbd"])
    parser.add_argument("--layouts", nargs="+", choices=("single", "parallel"), default=["single", "parallel"])
    parser.add_argument("--egl-layout", choices=("inherited", "by-rank"), default="inherited")
    args = parser.parse_args()
    worker = Path(__file__).resolve().with_name("zap_probe_worker.py")
    root, out = args.root.resolve(), args.out.resolve()
    if not root.is_dir() or not worker.is_file():
        parser.error("Experiment checkout or probe worker is missing")
    out.mkdir(parents=True, exist_ok=True)
    cases = [(layout, mode) for layout in args.layouts for mode in args.modes]
    if (out / "summary.json").exists() or any((out / f"{a}-{b}").exists() for a,b in cases):
        parser.error("Choose a fresh output directory")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    result = {"host": socket.gethostname(), "started": timestamp(), "cases": [], "status": "running",
              "egl_layout": args.egl_layout}
    save(out/"summary.json", result)
    binding = None
    if args.egl_layout == "by-rank":
        try:
            invocation = [sys.executable, str(Path(__file__).resolve()), "--enumerate-devices"]
            proc = subprocess.run(invocation, cwd=root, env=dict(os.environ,CUDA_VISIBLE_DEVICES="0,2,3",
                                  CUDA_DEVICE_ORDER="PCI_BUS_ID"), text=True, capture_output=True, timeout=45)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-1400:] or f"Enumeration exited {proc.returncode}")
            info = json.loads(proc.stdout)
            result["device_inventory"] = info
            binding = rank_map(info)
            result["selected_egl"] = binding
        except Exception as exc:
            result.update(status="mapping_failed", error=str(exc), ended=timestamp())
            save(out/"summary.json", result)
            raise
        save(out/"summary.json", result)
    for layout, mode in cases:
        print(f"Starting {layout} {mode}", flush=True)
        case = run_case(root, worker, out, layout, mode, binding)
        result["cases"].append(case)
        save(out/"summary.json", result)
        print(f"Finished {layout} {mode}: {case['status']}", flush=True)
    result.update(status="finished", ended=timestamp())
    save(out/"summary.json", result)


if __name__ == "__main__":
    if sys.argv[1:] == ["--enumerate-devices"]:
        print(json.dumps(enumerate_devices()))
    else:
        main()
