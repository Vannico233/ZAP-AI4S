"""Independent camera axes and moving-camera regression checks for the bimanual API."""
from types import SimpleNamespace
import json

import mujoco
import numpy as np
import pytest

from tube_agent import interface
from tube_agent.lab.vision import unproject


@pytest.fixture
def rig():
    # Rz(90 degrees) makes the middle optical axes world +y, +x, -z.
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <camera name="middle" pos="1 2 3" xyaxes="0 1 0 -1 0 0"/>
      <body pos="0.5 1 2">
        <joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.01"/>
        <camera name="left_wrist" pos="0 0.1 0.05"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)


def test_pose_position_and_orientation_are_in_middle_optical(rig):
    xyz = np.array([1.2, 2.3, 2.6])
    axes = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    pose = interface.world_pose_to_middle(rig, xyz, axes)
    np.testing.assert_allclose(pose["position"], [.3, .2, .4])
    np.testing.assert_allclose(pose["quaternion_wxyz"], [1., 0., 0., 0.], atol=1e-12)
    # Rotating 90 degrees about middle optical Z maps TCP X to middle Y.
    xyz2, basis = interface.middle_pose_to_world(rig, [.3, .2, .4], [2**-.5, 0., 0., 2**-.5])
    np.testing.assert_allclose(xyz2, xyz)
    np.testing.assert_allclose(basis[:,0], [1., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(basis[:,1], [0., -1., 0.], atol=1e-12)


def test_wrist_pixel_uses_live_extrinsics(rig):
    mask = np.zeros((3,3), dtype=bool)
    mask[2,2] = True
    intrinsic = {"fx":2., "fy":2., "cx":1., "cy":1.}
    # At unit axial depth, the bottom-right pixel is +0.5x, -0.5y, -1z
    # in world axes for this identity-rotation MuJoCo wrist camera.
    for slide in (0., .2):
        rig.data.qpos[0] = slide
        mujoco.mj_forward(rig.model, rig.data)
        cal = {**intrinsic, "T_middle_from_view":interface.view_extrinsics(rig, "left_wrist")}
        points = unproject(np.ones((3,3)), mask, cal)
        np.testing.assert_allclose(points, [[-1.4, slide, 1.95]], atol=1e-12)
        np.testing.assert_allclose(interface.view_extrinsics(rig, "middle"), np.eye(4), atol=1e-12)


def test_actions_require_an_arm_and_a_finite_unit_camera_pose(rig):
    action = {"action":"move_to", "arm":"right", "position":[0.,0.,.5], "quaternion_wxyz":[1.,0.,0.,0.]}
    rig.move_pose_world = lambda arm, xyz, rot: (arm, xyz, rot)
    name, xyz, _ = interface.execute_motor(rig, action)
    assert name == "right"
    np.testing.assert_allclose(xyz, [1.,2.,2.5])
    for changes in ({"arm":"unknown"}, {"position":[1.,2.,None]}, {"position":[True,2.,0.]},
                    {"quaternion_wxyz":[0.,0.,0.,0.]}, {"position":[np.inf,0.,1.]},
                    {"xyz":[0.,0.,.5]}):
        with pytest.raises(ValueError):
            interface.execute_motor(rig, {**action, **changes})
    with pytest.raises(ValueError):
        interface.execute_motor(rig, {"action":"set_gripper","opening":0.5})


@pytest.mark.requires_render
def test_bimanual_rollout_and_replay_use_the_public_camera_frame(tmp_path):
    from tube_agent.curriculum.config import BIMANUAL, load, scene_manifest
    from tube_agent.curriculum.rollout import episode, read_rows, replay
    config = load(BIMANUAL)
    scene = scene_manifest(config)["scenes"][0]

    def generate(messages, seed, temperature):
        content = messages[-1]["content"]
        sensed = json.loads(content[0]["text"])["observation"]
        assert "middle camera optical frame" in messages[0]["content"]
        assert sensed["coordinate_frame"] == "middle_optical"
        assert list(sensed["arms"]) == ["left", "right"]
        assert sensed["view_order"] == ["middle","left_wrist","right_wrist"]
        assert [part["type"] for part in content[1:]] == ["image_url"] * 3
        assert not {"tube_position","geometry","reward","success","tcp"} & sensed.keys()
        return '{"action":"done"}'

    directory = tmp_path/"camera_interface_only"
    result = episode(scene, config, directory, generate, SimpleNamespace(segment=lambda rgb, queries: []),
                     3, "interface_test", origin="interface_test")
    assert result["error"] is None, result["error"]
    assert result["steps"] == 1 and not result["success"]
    rows = list(read_rows(directory))
    for camera in rows[0]["camera_calibration"]:
        assert camera["coordinate_frame"] == "middle_optical" and "T_middle_from_view" in camera
        assert "rotation_camera_to_world" not in camera
    assert replay(scene, config, rows)["reward"] == result["reward"]
