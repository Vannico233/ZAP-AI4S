import copy
import json

import pytest

from tube_agent.env import TubeEnv
from tube_agent.export_sft import export
from tube_agent.manual import run_commands
from tube_agent.policy import ScriptedPolicy


def transport_spec():
    # Test-only fixture: no generated model actions are claimed here.
    env = TubeEnv(task="transport", split="train", seed=0)
    try:
        policy = ScriptedPolicy("transport")
        policy.act(env.instruction, env.observe())
        return {"task": "transport", "split": "train", "seed": 0,
                "actions": policy.actions}
    finally:
        env.close()


def test_manual_transport_logs_all_states_and_cannot_enter_self_sft(tmp_path):
    spec = transport_spec()
    original = copy.deepcopy(spec)
    output = tmp_path / "manual_transport"
    result = run_commands(spec, output)
    assert spec == original
    assert result["success"] and not result["incomplete"] and result["error"] is None
    assert result["commands_executed"] == result["commands_planned"] == 9
    assert result["policy"] == "manual" and result["model"] is None
    rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    assert [r["action"] for r in rows] == spec["actions"]
    assert rows[-1]["judge_after"]["success"]
    assert all(a["observation_after"] == b["observation"] for a, b in zip(rows, rows[1:]))
    replay = json.loads((output / "replay.jsonl").read_text())
    assert replay["source"]["policy"] == "manual"
    assert replay["source"]["provenance"] == "hand_authored"
    assert len(replay["messages"]) == 19
    with pytest.raises(ValueError, match="No eligible self-generated successes"):
        export(output, tmp_path / "self_success.jsonl")
    assert not (tmp_path / "self_success.jsonl").exists()
    with pytest.raises(FileExistsError):
        run_commands(spec, output)


def test_invalid_command_is_preserved_and_never_corrected(tmp_path):
    invalid = {"action": "move", "xyz": [.45, .22, .2]}  # Deliberately missing yaw.
    spec = {"task": "transport", "split": "train", "seed": 0,
            "actions": [invalid, {"action": "finish"}]}
    output = tmp_path / "invalid"
    result = run_commands(spec, output)
    assert not result["success"] and result["invalid_actions"] == 1
    assert result["commands_executed"] == 2 and not result["incomplete"]
    rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    assert rows[0]["action"] == invalid
    assert json.loads(rows[0]["raw_response"]) == invalid
    assert not rows[0]["feedback"]["accepted"]
    assert rows[0]["observation"]["tcp"] == rows[0]["observation_after"]["tcp"]
    assert rows[0]["observation"]["objects"] == rows[0]["observation_after"]["objects"]


@pytest.mark.parametrize("limit", [4, 8])
def test_partial_commands_are_not_silently_finished(tmp_path, limit):
    spec = transport_spec()
    spec["actions"] = spec["actions"][:limit]
    result = run_commands(spec, tmp_path / "partial")
    assert result["incomplete"] and result["commands_executed"] == limit
    assert result["lifted_once"]
    assert not result["success"] and result["reward"] == 0
    assert result["final_judge"]["success"] == (limit == 8)


def test_commands_after_finish_are_reported_not_executed(tmp_path):
    spec = transport_spec()
    spec["actions"].append({"action": "gripper", "opening": 0})
    result = run_commands(spec, tmp_path / "trailing")
    assert result["incomplete"] and result["commands_executed"] == 9
    assert "trailing commands" in result["error"]
    assert not result["success"] and result["final_judge"]["success"]
