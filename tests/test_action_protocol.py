import math
import numpy as np
import pytest
from tube_agent.lab.env import LabEnv


def test_logged_first_pick_action_accepts_rounded_pi_and_matches_exact_pi():
    rounded=LabEnv("pick");exact=LabEnv("pick")
    action={"action":"move","xyz":[.4647,-.1814,.0163],"rpy":[3.1416,0,0]}
    try:
        feedback=rounded.step(action)[1]
        reference=exact.step({**action,"rpy":[math.pi,0,0]})[1]
        assert feedback["accepted"] and reference["accepted"]
        assert feedback["reached"] and reference["reached"]
        assert feedback["rpy_normalization"]["executed"]==[math.pi,0,0]
        assert action["rpy"]==[3.1416,0,0]
        assert np.allclose(rounded.data.qpos,exact.data.qpos,atol=1e-12,rtol=0)
        assert np.allclose(rounded.data.qvel,exact.data.qvel,atol=1e-12,rtol=0)
    finally:rounded.close();exact.close()


@pytest.mark.parametrize("angle",[3.1416,-3.1416,3.14159,-3.14159])
def test_pi_rounding_is_symmetric_and_explicit(angle):
    env=LabEnv()
    try:
        received=[]
        env.move_pose=lambda xyz,rpy: received.append(rpy.copy()) or {"accepted":True}
        feedback=env.step({"action":"move","xyz":[.4,0,.2],"rpy":[angle,0,0]})[1]
        assert feedback["accepted"] and received[0][0]==math.copysign(math.pi,angle)
    finally:env.close()


@pytest.mark.parametrize("xyz,rpy,reason",[
    ([.4,0,-.01],[3.1416,0,0],"z=-0.01"),
    ([.4,0,.2],[3.142,0,0],"roll=3.142"),
    ([.4,0,.2],[180,0,0],"roll=180"),
    ([.4,0,.2],[math.nan,0,0],"finite"),
])
def test_rounding_fix_does_not_accept_real_out_of_range_poses(xyz,rpy,reason):
    env=LabEnv()
    try:
        before=env.data.qpos.copy()
        feedback=env.step({"action":"move","xyz":xyz,"rpy":rpy})[1]
        assert not feedback["accepted"] and reason in feedback["reason"]
        assert np.array_equal(before,env.data.qpos)
    finally:env.close()


def test_old_protocol_preserves_rejection_for_historical_replay():
    env=LabEnv(action_protocol="strict_rpy_v1")
    try:
        feedback=env.step({"action":"move","xyz":[.4647,-.1814,.0163],"rpy":[3.1416,0,0]})[1]
        assert not feedback["accepted"]
        assert feedback["reason"]=="TCP outside x[.2,.65],y[-.4,.4],z[.009,.5] or rpy outside [-pi,pi]"
    finally:env.close()


def test_action_errors_explain_missing_orientation_and_response_format():
    env=LabEnv()
    try:
        assert "rpy is missing" in env.step({"action":"move","xyz":[.4,0,.2]})[1]["reason"]
        assert "one JSON" in env.step({"action":"invalid_model_output"})[1]["reason"]
    finally:env.close()
