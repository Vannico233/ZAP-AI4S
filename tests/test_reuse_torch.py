import csv
import hashlib
import importlib.metadata as metadata
from pathlib import Path

import pytest

from scripts.reuse_torch import copy_packages, dependency_plan, safe_path


def make_distribution(root, name, requires=()):
    package = name.replace("-", "_")
    folder = root / f"{package}-1.0.dist-info"
    folder.mkdir(parents=True)
    module = root / package / "__init__.py"
    module.parent.mkdir()
    module.write_text("VALUE = 1\n")
    (folder / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n" +
        "".join(f"Requires-Dist: {requirement}\n" for requirement in requires))
    files = [module.relative_to(root).as_posix(), f"{folder.name}/METADATA", f"{folder.name}/RECORD"]
    with (folder / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows((path, "", "") for path in files + ["../../../bin/source-only-command"])
    return metadata.Distribution.at(folder)


def test_dependency_copy_is_independent_and_excludes_unrelated_packages(tmp_path, monkeypatch):
    source = tmp_path / "source"
    distributions = {
        "zap-root": make_distribution(source, "zap-root", ["zap-leaf>=1", 'not-needed; extra == "test"']),
        "zap-leaf": make_distribution(source, "zap-leaf"),
        "unrelated": make_distribution(source, "unrelated"),
    }
    monkeypatch.setattr("scripts.reuse_torch.metadata.distribution", distributions.__getitem__)
    monkeypatch.setattr("scripts.reuse_torch.sys.prefix", str(source))
    plan = dependency_plan(["zap-root"])
    assert set(plan) == {"zap-root", "zap-leaf"}
    target = tmp_path / "target"
    before = hashlib.sha256((source / "zap_root/__init__.py").read_bytes()).hexdigest()
    copy_packages(plan, target)
    assert {d.metadata["Name"] for d in metadata.distributions(path=[str(target)])} == {"zap-root", "zap-leaf"}
    (target / "zap_root/__init__.py").write_text("VALUE = 2\n")
    assert hashlib.sha256((source / "zap_root/__init__.py").read_bytes()).hexdigest() == before
    assert not (target / "unrelated").exists()
    assert "source-only-command" not in (target / "zap_root-1.0.dist-info/RECORD").read_text()


@pytest.mark.parametrize("relative", ["../outside", "/outside"])
def test_copy_rejects_paths_outside_environment(tmp_path, relative):
    with pytest.raises(ValueError):
        safe_path(tmp_path, relative)


def test_source_dependency_mismatch_is_rejected(tmp_path, monkeypatch):
    dist = make_distribution(tmp_path, "zap-leaf")
    monkeypatch.setattr("scripts.reuse_torch.metadata.distribution", lambda name: dist)
    monkeypatch.setattr("scripts.reuse_torch.sys.prefix", str(tmp_path))
    with pytest.raises(RuntimeError, match="mismatch"):
        dependency_plan(["zap-leaf>=2"])
