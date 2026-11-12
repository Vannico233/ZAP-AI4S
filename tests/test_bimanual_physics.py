"""Real MuJoCo control and contact checks for each Panda in the shared T1 scene."""
import numpy as np
import pytest

from tube_agent import interface
from tube_agent.curriculum.config import BIMANUAL, load, scene_manifest
from tube_agent.lab.zap_pose import rotation


@pytest.mark.parametrize("active,idle", [("left","right"), ("right","left")])
def test_either_arm_can_lift_with_stationary_other_wrist(active, idle):
    config = load(BIMANUAL)
    env = interface.make_env(scene_manifest(config)["scenes"][0], config)
    try:
        idle_start = interface.view_extrinsics(env, idle + "_wrist")
        active_start = interface.view_extrinsics(env, active + "_wrist")
        def to_height(z):
            return interface.move_action_from_world(env, active, [*env.source_rack[:2], z], rotation(0, 0))

        # Privileged feasibility check; these poses never enter a policy dataset.
        for command in (to_height(.24), to_height(.10),
                        {"action": "set_gripper", "arm": active, "opening": 0.}, to_height(.23),
                        {"action": "done"}):
            _, feedback, _ = env.step(command)
            assert feedback["accepted"] and feedback.get("reached", True), feedback
        assert env.result()["success"] and env.grasp_arm == active
        assert env.gripper_opening_m(idle) > .075
        np.testing.assert_allclose(interface.view_extrinsics(env, idle + "_wrist"), idle_start, atol=2e-4)
        assert np.linalg.norm(interface.view_extrinsics(env, active + "_wrist")[:3, 3] - active_start[:3, 3]) > .10
    finally:
        env.close()


def test_extreme_pose_rejected_before_ik_and_contacts_do_not_mix_grippers():
    config = load(BIMANUAL)
    env = interface.make_env(scene_manifest(config)["scenes"][0], config)
    try:
        t0 = env.data.time
        _, feedback, _ = env.step({"action": "move_to", "arm": "right",
            "position": [1e90, 1e90, -1e90], "quaternion_wxyz": [1.,0.,0.,0.]})
        assert not feedback["accepted"] and env.data.time == t0
        env.touching = {next(iter(env.arms[arm].pads)) for arm in ("left", "right")}
        assert not env._held_now()
    finally:
        env.close()
