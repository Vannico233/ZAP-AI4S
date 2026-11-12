import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tube_agent.curriculum import collect_t1, rollout
from tube_agent.curriculum.config import BIMANUAL, digest, load
from tube_agent.lab.experiment import write_json


def test_kinematic_note_is_a_separate_reset_collection_condition():
    config = load(BIMANUAL)
    original_prompt = rollout.system_prompt(config)
    original_digest = digest(config)
    config["policy"]["kinematic_note"] = True
    assert "p + h*u" in rollout.system_prompt(config) and "p + h*u" not in original_prompt
    assert digest(config) != original_digest


def test_shards_cover_distinct_model_seeds_on_training_scenes_only():
    config = load(BIMANUAL)
    attempts = collect_t1.make_plan(config, 6, 2, 20261113)
    assert len(attempts) == len({a["policy_seed"] for a in attempts}) == 12
    assert len({a["scene"]["id"] for a in attempts}) == 6
    assert all(a["scene"]["task"] == "T1" and a["scene"]["split"] == "train" for a in attempts)
    assert attempts == collect_t1.make_plan(config, 6, 2, 20261113)
    assert [len(attempts[i::3]) for i in range(3)] == [4, 4, 4]
    assert len({a["directory"] for shard in range(3) for a in attempts[shard::3]}) == 12


@pytest.mark.requires_render
def test_real_rollout_and_infrastructure_error_are_separate(tmp_path):
    # Test-only text generator emits done on the first step. No real model data is produced.
    config = load(BIMANUAL)
    plan = collect_t1.make_plan(config, 2, 1, 104)
    root = tmp_path
    write_json(root / "collection.json", {"plan": {"attempts": plan, "config_sha256": digest(config)}})
    attempt, failed = plan
    path = root / attempt["directory"]
    segmenter = SimpleNamespace(segment=lambda rgb, queries: [])
    rollout.episode(attempt["scene"], config, path, lambda *_: '{"action":"done"}', segmenter,
                    attempt["policy_seed"], "original", origin=collect_t1.ORIGIN)
    record = collect_t1.verify_result(path, attempt, config)
    assert record["termination"] == "done" and not record["model_success"]
    write_json(path / "verified.json", record)
    write_json(root / failed["directory"] / "result.json", {"error": "Camera service unavailable"})
    report = collect_t1.summarize(root)
    assert (report["completed"], report["verified_successes"], len(report["errors"])) == (1, 0, 1)


def test_success_at_step_limit_requires_replay_and_token_evidence(tmp_path, monkeypatch):
    config = load(BIMANUAL)
    attempt, = collect_t1.make_plan(config, 1, 1, 20261113)
    # Tiny synthetic records exercise the gate. Stubbed replay is local to this unit test.
    result = {"policy": "model", "policy_version": "original", "demonstration": None, "error": None,
              "scene": attempt["scene"], "policy_seed": attempt["policy_seed"], "origin": collect_t1.ORIGIN,
              "replayed_actions": 0, "generated_actions": 1, "steps": 1, "success": True,
              "reward": 1, "invalid_actions": 0, "failure_type": "none", "termination": "max_steps",
              "config_sha256": digest(config)}
    write_json(tmp_path / "result.json", result)
    row = {"source_origin": collect_t1.ORIGIN, "policy_seed": attempt["policy_seed"],
           "response_ids": [123], "model_input_sha256": "fixture", "judge_after": {"success": True}}
    path = tmp_path / "trajectory.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    physical = {"success": True, "held": True, "dropped": False, "severe_collision": False, "goal_stable_seconds": 1.}
    monkeypatch.setattr(rollout, "replay", lambda *args: physical)
    assert collect_t1.verify_result(tmp_path, attempt, config)["replay_verified"]
    physical["held"] = False
    with pytest.raises(ValueError, match="physical replay"):
        collect_t1.verify_result(tmp_path, attempt, config)
    row.pop("response_ids")
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        collect_t1.verify_result(tmp_path, attempt, config)
    write_json(tmp_path / "result.json", {**result, "demonstration": "scripted"})
    with pytest.raises(ValueError, match="incompatible"):
        collect_t1.verify_result(tmp_path, attempt, config)


def test_collect_worker_uses_single_device_mapped_renderer(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/diagnostics/round10_egl"
    monkeypatch.syspath_prepend(str(directory))
    from zap_egl_runtime import worker_target
    name = "tube_agent.curriculum.collect_t1"
    for gpu in ("0", "2", "3"):
        assert worker_target(["python", "-m", name], {"CUDA_VISIBLE_DEVICES": gpu}) == (name, 0, 1)
