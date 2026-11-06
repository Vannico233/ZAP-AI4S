"""Verify optional private GL libraries or render both cameras without model services."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime(root=ROOT, check_platform=True):
    folder = Path(root) / ".runtime/egl-jammy"
    manifest = json.loads((folder / "EGL_RUNTIME.json").read_text(encoding="utf-8"))
    if check_platform:
        libc, version = platform.libc_ver()
        if (sys.platform != "linux" or platform.machine() != "x86_64" or libc != "glibc"
                or tuple(map(int, version.split("."))) < (2, 34)):
            raise RuntimeError("Private EGL runtime requires Linux x86_64 with glibc >= 2.34")
    if set(manifest["files"]) != {"lib/libEGL.so.1", "lib/libOpenGL.so.0", "lib/libGLdispatch.so.0"}:
        raise ValueError("Unexpected EGL runtime library inventory")
    for name, record in manifest["files"].items():
        path = folder / name
        if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Damaged EGL runtime file: {name}; re-extract the EGL patch")
    print("Verified private EGL/OpenGL runtime (Ubuntu 22.04, libglvnd 1.4.0-1).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_runtime:
        verify_runtime(); return
    if args.output is None:
        parser.error("--output is required for rendering")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"success": False, "stage": "load_libraries"}
    env = cameras = None
    try:
        if sys.platform == "linux":
            import os
            if os.environ.get("MUJOCO_GL", "egl") == "egl":
                for name in ("libEGL.so.1", "libOpenGL.so.0", "libGLdispatch.so.0"):
                    ctypes.CDLL(name)
        report["stage"] = "render_cameras"
        sys.path.insert(0, str(ROOT))
        from tube_agent.lab.env import LabEnv
        from tube_agent.lab.vision import Cameras
        from PIL import Image
        from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
        env = LabEnv("pick", "train", 0)
        cameras = Cameras(env)
        report["cameras"] = []
        for frame in cameras.capture():
            rgb = frame["rgb"]
            if rgb.std() < 1:
                raise RuntimeError("Rendered frame is blank")
            name = frame["calibration"]["name"]
            Image.fromarray(rgb).save(args.output / (name + ".png"))
            report["cameras"].append({"name": name, "shape": list(rgb.shape)})
        for name, key in (("vendor", GL_VENDOR), ("renderer", GL_RENDERER), ("version", GL_VERSION)):
            value = glGetString(key)
            report[name] = value.decode(errors="replace") if value else None
        report.update(success=True, stage="complete")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if cameras:
            cameras.close()
        if env:
            env.close()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
