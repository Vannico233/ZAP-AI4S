import copy

import pytest

from tube_agent.curriculum.config import file_digest, load, scene_manifest
from tube_agent.curriculum.t1_focus import focus_plan, source_record, evaluation_summary
from tube_agent.curriculum.terminal_credit import failed_final_done, response_advantage
from tube_agent.lab.experiment import write_json


@pytest.mark.parametrize("advantage", [-0.25, 0, 0.25])
def test_credit_only_on_final_accepted_failed_done(advantage):
    result = {"error": None, "success": False, "steps": 40, "termination": "max_steps"}
    row = {"step": 40, "action": {"action": "done"}, "feedback": {"accepted": True},
           "judge_after": {"success": False}}
    assert failed_final_done(result, row)
    assert response_advantage(advantage, 40, 40) == min(advantage, 0)
    assert response_advantage(advantage, 39, 40) == advantage
    for change in ({"step": 39}, {"feedback": {"accepted": False}}, {"action": {"action": "move_to"}},
                   {"action": {"action": "invalid_model_output"}}, {"judge_after": {"success": True}}):
        assert not failed_final_done(result, {**row, **change})
        assert response_advantage(advantage, 40, None) == advantage
    assert not failed_final_done({**result, "success": True}, row)
    assert not failed_final_done({**result, "error": "infrastructure"}, row)


def test_schedule_and_opt_in(tmp_path):
    parent = load()
    manifest = scene_manifest(parent)
    c, scenes, rounds = focus_plan(parent, manifest)
    assert (c, scenes, rounds) == focus_plan(parent, manifest)
    groups = [g for r in rounds for g in r["groups"]]
    assert [len(r["groups"]) for r in rounds] == [4, 4]
    assert len({g["scene_id"] for g in groups}) == 8
    assert all(g["scene_id"].startswith("T1_train_") and len(g["rollouts"]) == 8 for g in groups)
    assert len({a["policy_seed"] for g in groups for a in g["rollouts"]}) == 64
    assert len([s for s in scenes["scenes"] if s["split"] == "eval"]) == 30
    assert not parent["training"]["failed_done_credit_clamp"] and c["training"]["failed_done_credit_clamp"]
    path = tmp_path/"config.json"
    write_json(path, c)
    assert load(path, allow_focus=True) == c
    with pytest.raises(ValueError):
        load(path)
    modified = copy.deepcopy(manifest)
    modified["scenes"][0]["seed"] += 1
    with pytest.raises(ValueError):
        focus_plan(parent, modified)


def test_checkpoint_requires_intact_complete_and_matches_parent(tmp_path):
    from tube_agent.curriculum.config import digest
    parent = tmp_path/"parent"
    ckpt = parent/"warmup/checkpoints/step_00015"
    ckpt.mkdir(parents=True)
    adapter, optimizer = ckpt/"adapter_model.safetensors", ckpt/"optimizer_rng.pt"
    adapter.write_bytes(b"adapter")
    optimizer.write_bytes(b"optimizer")
    config = load()
    write_json(ckpt/"COMPLETE.json", {"config_sha256": digest(config), "adapter_sha256": file_digest(adapter),
        "optimizer_sha256": file_digest(optimizer), "counters": {"round": 15, "optimizer_steps": 27}})
    write_json(parent/"warmup/metrics/round_15.json",
               {"round": 15, "optimizer_steps_total": 27, "adapter_after": "tensor-sha"})
    source = source_record(parent, ckpt, config)
    assert source["inherited_optimizer_steps"] == source["inherited_rounds"] == 0
    assert source["tensor_sha256"] == "tensor-sha"
    optimizer.write_bytes(b"partial")
    with pytest.raises(ValueError, match="modified"):
        source_record(parent, ckpt, config)


def test_matched_evaluation_rejects_manual_or_replayed_actions(tmp_path):
    from tube_agent.curriculum.config import digest
    c = load()
    scenes = [s for s in scene_manifest(c)["scenes"] if s["task"] == "T1" and s["split"] == "eval"]
    for scene in scenes:
        path = tmp_path/"evaluation/original"/scene["id"]/"result.json"
        result = {"scene": scene, "seed": scene["seed"], "split": "eval", "task": "T1", "success": True,
            "generated_actions": 2, "steps": 2, "invalid_actions": 1, "ever_grasped": True, "extracted": True,
            "termination": "done", "policy_seed": 30000000+scene["seed"], "failure_type": "none", "origin": "evaluation",
            "policy_version": "original", "policy": "model", "error": None, "demonstration": None, "replayed_actions": 0,
            "search_prefix_length": 0, "config_sha256": digest(c), "observation_mode": "rgbd_sam3.1_detector",
            "generator_provenance": {"policy_version": "original", "adapter_sha256": "base", "temperature": 0,
                                    "deterministic": True, "sampler": "same_live_model_object"}}
        write_json(path, result)
    summary = evaluation_summary(tmp_path, "original", scenes, c, "base")
    assert summary["success"] == summary["accepted_actions"] == summary["invalid_actions"] == 30
    for change in ({"policy": "manual"}, {"replayed_actions": 1}, {"demonstration": "operator"}, {"policy_seed": -1}):
        write_json(path, {**result, **change})
        with pytest.raises(ValueError, match="provenance"):
            evaluation_summary(tmp_path, "original", scenes, c, "base")
