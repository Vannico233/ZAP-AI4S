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


from egl_devices import enumerate_devices, rank_map


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
