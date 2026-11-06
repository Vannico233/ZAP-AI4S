"""Verify and install the uploaded Linux/Python 3.11 bundle without network access."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty offline bundle manifest")
    for relative, expected in manifest["files"].items():
        relative = PurePosixPath(relative)
        path = (root / str(relative)).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError(f"Unsafe bundle path: {relative}")
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Missing or damaged bundle file: {relative}; upload the archive again")
    expected_wheels = {name for name in manifest["files"] if name.startswith("wheels/")}
    actual_wheels = {path.relative_to(root).as_posix() for path in (root / "wheels").glob("*.whl")}
    if not expected_wheels or actual_wheels != expected_wheels:
        raise ValueError("Wheel directory differs from the bundle manifest")
    print(f"Bundle verified: {manifest['source_commit'][:12]}, {len(expected_wheels)} wheels.", flush=True)
    return manifest


def check_target():
    if sys.platform != "linux" or platform.machine().lower() != "x86_64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("This bundle requires Linux x86_64 and Python 3.11; use DiG/.runtime/py311/bin/python")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(int(part) for part in version.split(".")[:2]) < (2, 28):
        raise RuntimeError(f"This bundle requires glibc >= 2.28; found {libc} {version}")


def install(root):
    root = Path(root).resolve()
    verify(root)
    check_target()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the project virtual environment before installing")
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("PIP_"):
            del environment[key]
    environment.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    # The complete non-Torch dependency closure is pinned and hashed in this file.
    # Torch's dependency closure has already been copied and checked by reuse_torch.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                    "--require-hashes", "--only-binary=:all:", "--disable-pip-version-check",
                    "--find-links", str(root / "wheels"), "-r", str(root / "requirements-offline.txt")],
                   env=environment, check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "install"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-target", action="store_true")
    args = parser.parse_args()
    if args.check_target:
        check_target()
    if args.action == "install":
        install(args.root)
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
