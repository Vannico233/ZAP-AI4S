"""Package committed code and verified Linux cp311 wheels for upload to the server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

if __package__:
    from .offline_bundle import sha256
else:
    from offline_bundle import sha256


TARGET = dict(implementation_name="cpython", implementation_version="3.11.18", os_name="posix",
              platform_machine="x86_64", platform_python_implementation="CPython",
              platform_release="", platform_system="Linux", platform_version="",
              python_full_version="3.11.18", python_version="3.11", sys_platform="linux", extra="")
PROVIDED = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128"}


def read_requirements(path):
    return [Requirement(line.split("#", 1)[0].strip()) for line in Path(path).read_text().splitlines()
            if line.split("#", 1)[0].strip()]


def inspect_wheels(directory, lock, roots):
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = set(cpython_tags((3, 11), ["cp311", "abi3", "none"], platforms))
    tags.update(compatible_tags((3, 11), "cp311", platforms))
    wheels = {}
    for path in sorted(Path(directory).glob("*.whl")):
        name, version, _, wheel_tags = parse_wheel_filename(path.name)
        if name in wheels or not wheel_tags.intersection(tags):
            raise ValueError(f"Duplicate or incompatible wheel: {path.name}")
        with zipfile.ZipFile(path) as archive:
            names = [entry for entry in archive.namelist() if entry.endswith(".dist-info/METADATA")]
            if len(names) != 1 or archive.testzip() is not None:
                raise ValueError(f"Invalid wheel archive: {path.name}")
            info = Parser().parsestr(archive.read(names[0]).decode("utf-8"))
        if info.get("Requires-Python") and not SpecifierSet(info["Requires-Python"]).contains("3.11.18"):
            raise ValueError(f"Wrong Python version for {path.name}")
        if canonicalize_name(info["Name"]) != name or Version(info["Version"]) != version:
            raise ValueError(f"Wheel metadata mismatch: {path.name}")
        wheels[name] = dict(path=path, version=str(version), requires=info.get_all("Requires-Dist", []),
                            sha256=sha256(path), size=path.stat().st_size)
    expected = {canonicalize_name(req.name): req for req in read_requirements(lock)}
    if set(wheels) != set(expected):
        raise ValueError(f"Wheel set differs from lock: {set(wheels) ^ set(expected)}")
    for name, req in expected.items():
        if not req.specifier.contains(wheels[name]["version"]):
            raise ValueError(f"Wheel doesn't satisfy {req}")
    extras = {name: set() for name in wheels}

    def check(requirement):
        name = canonicalize_name(requirement.name)
        version = wheels[name]["version"] if name in wheels else PROVIDED.get(name)
        if requirement.url or version is None or not requirement.specifier.contains(version):
            raise ValueError(f"Offline dependency is missing or incompatible: {requirement}")
        if name in extras:
            previous = len(extras[name])
            extras[name].update(requirement.extras)
            return previous != len(extras[name])
        return False

    for req in roots:
        if not req.marker or req.marker.evaluate(TARGET):
            check(req)
    changed = True
    while changed:
        changed = False
        for name, wheel in wheels.items():
            for text in wheel["requires"]:
                req = Requirement(text)
                if not req.marker or any(req.marker.evaluate(dict(TARGET, extra=extra))
                                         for extra in {"", *extras[name]}):
                    changed |= check(req)
    return wheels


def verify_pypi(item):
    name, wheel = item
    url = f"https://pypi.org/pypi/{name}/{wheel['version']}/json"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                info = json.load(response)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    release = next((entry for entry in info["urls"] if entry["filename"] == wheel["path"].name), None)
    if release is None or release["digests"]["sha256"] != wheel["sha256"] or release["size"] != wheel["size"]:
        raise ValueError(f"Official PyPI hash/size mismatch: {wheel['path'].name}")
    return name, dict(version=wheel["version"], filename=wheel["path"].name,
                     sha256=wheel["sha256"], size=wheel["size"], source_url=release["url"])


def build(root, wheels, provenance, output):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("Commit source changes before packaging the offline bundle")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=root)
    payload = {}
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        for member in archive:
            if member.isfile():
                payload[member.name] = archive.extractfile(member).read()
            elif not member.isdir():
                raise ValueError(f"Unsupported source archive member: {member.name}")
    payload["requirements-offline.txt"] = "".join(
        f"{name}=={info['version']} --hash=sha256:{info['sha256']}\n"
        for name, info in sorted(wheels.items())).encode()
    manifest = dict(format=1, source_commit=commit, target=TARGET, provided_by_runtime=PROVIDED,
                    wheels=provenance, files={})
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "ZAP-AI4S-offline-20261106"
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        for relative, content in sorted(payload.items()):
            member = tarfile.TarInfo(f"{prefix}/{relative}")
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
            manifest["files"][relative] = dict(size=len(content), sha256=hashlib.sha256(content).hexdigest())
        for info in wheels.values():
            relative = "wheels/" + info["path"].name
            archive.add(info["path"], arcname=f"{prefix}/{relative}", recursive=False)
            manifest["files"][relative] = dict(size=info["size"], sha256=info["sha256"])
        content = json.dumps(manifest, indent=2).encode()
        member = tarfile.TarInfo(f"{prefix}/MANIFEST.json")
        member.size = len(content)
        member.mode = 0o644
        archive.addfile(member, io.BytesIO(content))
    digest = sha256(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"Created {output}\nSize: {output.stat().st_size / 2**20:.1f} MiB\nSHA256: {digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    roots = read_requirements(root / "requirements.txt") + read_requirements(root / "requirements-model.txt")
    roots.append(Requirement("pytest>=8"))
    wheels = inspect_wheels(args.wheels, args.lock, roots)
    with ThreadPoolExecutor(max_workers=6) as pool:
        provenance = dict(pool.map(verify_pypi, wheels.items()))
    print(f"Verified {len(wheels)} Linux cp311 wheels: target tags, dependency closure, ZIP integrity, official PyPI hashes.")
    if not args.check_only:
        if not args.output:
            parser.error("--output is required unless --check-only is used")
        build(root, wheels, provenance, args.output)


if __name__ == "__main__":
    main()
