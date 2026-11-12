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


def test_base_mounted_middle_camera_projects_forward_up_and_measures_vertical_lifts():
    env = interface.open_scene("T1_train_000")
    try:
        camera = env.model.camera("middle")
        origin = env.data.cam_xpos[camera.id]
        midpoint = (env.data.body("link0").xpos + env.data.body("right_link0").xpos) / 2
        np.testing.assert_allclose(origin, midpoint + [0., 0., .90], atol=1e-7)
        assert env.model.cam_bodyid[camera.id] == 0  # Fixed to the world, not a wrist.

        # Sightline is fixed to the nominal rack midpoint, independent of scene jitter.
        target = np.array([.45, -.01, .05])
        sightline = (target - origin) / np.linalg.norm(target - origin)
        right = np.cross(sightline, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(sightline, right)
        world_from_optical = np.column_stack((right, down, sightline))

        def optical_position(xyz):
            return np.asarray(interface.world_pose_to_middle(env, xyz, np.eye(3))["position"])

        center = optical_position(target)
        np.testing.assert_allclose(center[:2], [0., 0.], atol=1e-7)
        assert center[2] > 0
        assert optical_position(target + [0., -.05, 0.])[0] > .049
        assert optical_position(target + [.05, 0., 0.])[1] < -.04
        lift = optical_position(target + [0., 0., .05]) - center
        np.testing.assert_allclose(lift, .05 * world_from_optical[2], atol=1e-7)
        assert lift[1] < 0 and lift[2] < 0
        sensed = interface.proprioception(env)
        np.testing.assert_allclose(sensed["table_up_middle"], lift / .05, atol=1e-7)

        for name, arm in env.arms.items():
            pose = sensed["arms"][name]["ee_pose"]
            local_rotation = np.empty(9)
            mujoco.mju_quat2Mat(local_rotation, np.array(pose["quaternion_wxyz"]))
            np.testing.assert_allclose(world_from_optical @ pose["position"] + origin,
                                       env.data.site_xpos[arm.site], atol=1e-7)
            np.testing.assert_allclose(world_from_optical @ local_rotation.reshape(3, 3),
                                       env.data.site_xmat[arm.site].reshape(3, 3), atol=1e-7)
    finally:
        env.close()


def test_frozen_v1_scene_keeps_the_original_across_table_view():
    from tube_agent.curriculum.config import BIMANUAL, load, scene_manifest
    config = load(BIMANUAL)
    config["version"] = "tube_curriculum_bimanual_v1"
    config["observation"]["version"] = "bimanual_rgbd_v1"
    config["observation"].pop("middle_camera")
    env = interface.make_env(scene_manifest(config)["scenes"][0], config)
    try:
        camera = env.model.camera("middle")
        origin = np.array([1.20, 0., .92])
        target = np.array([.32, 0., .32])
        direction = (target - origin) / np.linalg.norm(target - origin)
        np.testing.assert_allclose(env.data.cam_xpos[camera.id], origin)
        axes = env.data.cam_xmat[camera.id].reshape(3, 3)
        np.testing.assert_allclose(axes[:, 0], [0., 1., 0.], atol=1e-7)
        np.testing.assert_allclose(-axes[:, 2], direction, atol=1e-7)
        assert env.model.cam_fovy[camera.id] == 66
        assert env.result()["environment_version"] == "bimanual_panda_middle_camera_v1"
    finally:
        env.close()


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
    # A done-only replay produces identical physics even from the wrong view.
    # The mount check must also catch missing markers in older bimanual rows.
    old_row = rows[0].copy()
    del old_row["middle_camera_sha256"]
    with pytest.raises(ValueError, match="different middle camera mount"):
        replay(scene, config, [old_row])
    wrong_row = {**rows[0], "middle_camera_sha256": "other-mount"}
    mismatch = episode(scene, config, tmp_path/"wrong_mount", generate,
                       SimpleNamespace(segment=lambda rgb, queries: []),
                       4, "interface_test", prefix=[wrong_row], origin="interface_test")
    assert "different middle camera mount" in mismatch["error"]
