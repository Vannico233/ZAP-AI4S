import math

import numpy as np
import pytest

from tube_agent.env import TubeEnv
from tube_agent.policy import ScriptedPolicy


def execute(env, limit=None):
    policy = ScriptedPolicy(env.task)
    observation, feedback = env.observe(), None
    for _ in range(limit or env.max_steps):
        action, _ = policy.act(env.instruction, observation, feedback)
        observation, feedback, ended = env.step(action)
        assert feedback["accepted"]
        if ended:
            break
    return env.result()


@pytest.mark.parametrize("task", ["pick", "transport"])
@pytest.mark.parametrize("split,seed", [("test", 0), ("test", 7), ("ood", 2)])
def test_contact_control_completes_tasks(task, split, seed):
    env = TubeEnv(task, split, seed)
    try:
        assert execute(env)["success"]
        assert env.lifted_once
    finally:
        env.close()


def test_finish_does_not_assert_success():
    env = TubeEnv()
    try:
        _, feedback, ended = env.step({"action": "finish"})
        assert ended and feedback == {"accepted": True, "submitted": True}
        assert not env.result()["success"]
        assert "success" not in env.observe() and "reward" not in env.observe()
        with pytest.raises(RuntimeError):
            env.step({"action": "finish"})
    finally:
        env.close()


@pytest.mark.parametrize("action", [
    {"action": "move", "xyz": [99, 0, .2], "yaw": 0},
    {"action": "move", "xyz": [.4, 0], "yaw": 0},
    {"action": "move", "xyz": [.4, 0, .2], "yaw": float("nan")},
    {"action": "gripper", "opening": -.1},
    {"action": "gripper", "opening": float("inf")},
    {"action": "move", "xyz": [.4, 0, .2], "yaw": 0, "attach": True},
    {"action": "grasp_tube"}, None,
])
def test_invalid_actions_cannot_mutate_physics(action):
    env = TubeEnv()
    try:
        before = env.data.qpos.copy()
        _, feedback, ended = env.step(action)
        assert not feedback["accepted"] and not ended
        assert np.array_equal(before, env.data.qpos)
        assert env.invalid_actions == 1
    finally:
        env.close()


def test_holding_over_tray_is_not_transport_success():
    env = TubeEnv(task="transport")
    try:
        result = execute(env, limit=5)
        assert result["held"] and result["lifted_once"]
        assert not result["success"]
    finally:
        env.close()


def test_dropped_tube_is_not_pick_success():
    env = TubeEnv(task="pick")
    try:
        assert execute(env, limit=4)["success"]
        env.step({"action": "gripper", "opening": .08})
        env.step({"action": "finish"})
        assert not env.result()["success"]
    finally:
        env.close()


def test_pushing_or_teleporting_into_tray_cannot_pass_without_lift():
    env = TubeEnv()
    try:
        # Test-only intervention: even perfect placement must not bypass pick-up.
        adr = env.model.jnt_qposadr[env.tube_joint]
        env.data.qpos[adr:adr+3] = [*env.tray[:2], env.tray[2]+env.radius]
        env._advance(1.)
        assert not env.result()["success"] and not env.lifted_once
    finally:
        env.close()


def test_seed_replay_and_split_separation():
    a, b, c = TubeEnv(seed=5), TubeEnv(seed=5), TubeEnv(split="test", seed=5)
    try:
        assert a.observe() == b.observe()
        assert not np.allclose(a.tube_pos, c.tube_pos)
        assert a.model.neq == 1  # Only finger synchronisation; no tube weld.
    finally:
        for env in (a, b, c):
            env.close()


def test_invalid_attempts_consume_budget():
    env = TubeEnv(max_steps=2)
    try:
        env.step({})
        _, _, ended = env.step({})
        assert ended and env.steps == 2 and not env.result()["success"]
    finally:
        env.close()
