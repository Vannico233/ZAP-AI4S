import json

import pytest

from scripts.offline_bundle import check_target, sha256, verify


def make_bundle(root):
    (root / "wheels").mkdir(parents=True)
    (root / "code.py").write_bytes(b"value=1\n")
    (root / "wheels/probe.whl").write_bytes(b"wheel bytes")
    files = {name: {"size": (root / name).stat().st_size, "sha256": sha256(root / name)}
             for name in ("code.py", "wheels/probe.whl")}
    manifest = {"format": 1, "source_commit": "a" * 40, "files": files}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    return manifest


def test_bundle_verification_detects_same_size_corruption(tmp_path):
    make_bundle(tmp_path)
    assert verify(tmp_path)["source_commit"] == "a" * 40
    (tmp_path / "code.py").write_bytes(b"value=2\n")
    with pytest.raises(ValueError, match="damaged"):
        verify(tmp_path)


def test_missing_uploaded_wheel_is_rejected(tmp_path):
    make_bundle(tmp_path)
    (tmp_path / "wheels/probe.whl").unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify(tmp_path)


def test_extra_wheel_is_rejected(tmp_path):
    make_bundle(tmp_path)
    (tmp_path / "wheels/unexpected.whl").write_bytes(b"other")
    with pytest.raises(ValueError, match="differs"):
        verify(tmp_path)


def test_manifest_cannot_access_parent_directory(tmp_path):
    root = tmp_path / "bundle"
    manifest = make_bundle(root)
    manifest["files"]["../outside"] = {"size": 1, "sha256": "0" * 64}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        verify(root)


@pytest.mark.parametrize("version,libc", [((3, 13), ("glibc", "2.35")), ((3, 11), ("glibc", "2.27"))])
def test_offline_install_rejects_incompatible_runtime(monkeypatch, version, libc):
    monkeypatch.setattr("scripts.offline_bundle.sys.platform", "linux")
    monkeypatch.setattr("scripts.offline_bundle.sys.version_info", version)
    monkeypatch.setattr("scripts.offline_bundle.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("scripts.offline_bundle.platform.libc_ver", lambda: libc)
    with pytest.raises(RuntimeError):
        check_target()
