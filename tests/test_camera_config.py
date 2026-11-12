import copy

import pytest

from tube_agent.curriculum.config import BIMANUAL, digest, file_digest, load, scene_manifest, validate_adapter_checkpoint
from tube_agent.curriculum.preflight import check
from tube_agent.lab.experiment import write_json


def test_new_mount_changes_frozen_provenance_but_not_registered_scenes(tmp_path):
    current = load(BIMANUAL)
    previous = copy.deepcopy(current)
    previous["version"] = "tube_curriculum_bimanual_v1"
    previous["observation"]["version"] = "bimanual_rgbd_v1"
    del previous["observation"]["middle_camera"]
    path = tmp_path / "frozen.json"
    write_json(path, previous)
    assert load(path) == previous
    manifest = scene_manifest(previous)
    assert manifest["scenes"] == scene_manifest(current)["scenes"]
    with pytest.raises(ValueError, match="Scene manifest"):
        check(current, manifest, tmp_path / "preflight")

    # Tuning the v2 camera must also invalidate reuse within v2.
    calibrated = copy.deepcopy(current)
    calibrated["observation"]["middle_camera"]["position_world_m"][2] += .01
    write_json(path, calibrated)
    assert load(path) == calibrated
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"test adapter")
    write_json(tmp_path / "COMPLETE.json", {"adapter_sha256": file_digest(adapter), "config_sha256": digest(current)})
    validate_adapter_checkpoint(tmp_path, digest(current))
    for other in (previous, calibrated):
        with pytest.raises(ValueError, match="checkpoint/config"):
            validate_adapter_checkpoint(tmp_path, digest(other))


@pytest.mark.parametrize(("field", "value"), [
    ("fovy_deg", 180),
    ("look_at_world_m", [0, 0, 0.9]),                  # Same as camera position.
    ("image_right_world", [.45, -.01, -.85]),          # Parallel to viewing direction.
    ("position_world_m", [0, 0, float("nan")]),
])
def test_invalid_middle_camera_calibration_rejected(tmp_path, field, value):
    config = load(BIMANUAL)
    config["observation"]["middle_camera"][field] = value
    path = tmp_path / "invalid.json"
    write_json(path, config)
    with pytest.raises(ValueError, match="middle_camera"):
        load(path)
