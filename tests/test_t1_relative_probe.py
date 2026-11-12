import numpy as np
import pytest

from tube_agent.curriculum.t1_relative_probe import resolve_relative


@pytest.fixture
def observation():
    return {"coordinate_frame": "middle_optical", "table_up_middle": [0., -.6, -.8],
            "arms": {"left": {"ee_pose": {"position": [1., 2., 3.], "quaternion_wxyz": [1., 0., 0., 0.]}},
                     "right": {"ee_pose": {"position": [.1, .2, .3], "quaternion_wxyz": [.5, -.5, .5, -.5]}}}}


@pytest.mark.parametrize("axis,direction", [
    ("table_normal", [0., -.6, -.8]), ("camera_x", [1., 0., 0.]),
    ("camera_y", [0., 1., 0.]), ("camera_z", [0., 0., 1.])])
@pytest.mark.parametrize("distance", [-.08, .08, 0.])
def test_signed_translation_uses_selected_measured_pose(observation, axis, direction, distance):
    action = {"action": "translate", "arm": "right", "axis": axis, "distance_m": distance}
    result = resolve_relative(action, observation)
    assert result["action"] == "move_to" and result["arm"] == "right"
    assert result["quaternion_wxyz"] == [.5, -.5, .5, -.5]
    np.testing.assert_allclose(result["position"], [.1, .2, .3] + distance * np.array(direction))
    assert observation["arms"]["right"]["ee_pose"]["position"] == [.1, .2, .3]


@pytest.mark.parametrize("change", [
    {"distance_m": True}, {"distance_m": float("nan")}, {"distance_m": float("inf")},
    {"distance_m": "0.08"}, {"axis": "up"}, {"axis": []}, {"arm": "both"}, {"extra": 1}])
def test_rejects_invalid_relative_action(observation, change):
    action = {"action": "translate", "arm": "left", "axis": "table_normal", "distance_m": .02}
    with pytest.raises(ValueError):
        resolve_relative({**action, **change}, observation)


def test_relative_move_requires_measured_axis_and_pose(observation):
    action = {"action": "translate", "arm": "left", "axis": "table_normal", "distance_m": .02}
    for field, invalid in (("coordinate_frame", "world"), ("table_up_middle", [0., 0., 0.]), ("arms", {})):
        with pytest.raises(ValueError):
            resolve_relative(action, {**observation, field: invalid})
