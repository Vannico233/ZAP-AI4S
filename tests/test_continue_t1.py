"""Physical integration test with a synthetic motor policy confined to pytest's temp directory."""
import json
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from tube_agent import interface
from tube_agent.curriculum import collect_t1, continue_t1, rollout
from tube_agent.curriculum.config import BIMANUAL, digest, load
from tube_agent.lab.experiment import write_json
from tube_agent.lab.zap_pose import rotation


class FixturePolicy:
    # Test double for schema and physics only. No test artifact is production data.
    provenance = {"sampler": "same_live_model_object", "model_class": "Qwen3_5ForConditionalGeneration",
                  "policy_version": "original", "adapter_sha256": collect_t1.EMPTY_ADAPTER}
    last_response_ids = [123]
    last_input_sha256 = "a" * 64

    def __init__(self, actions):
        self.actions = iter(actions)

    def __call__(self, messages, seed, temperature):
        observed = json.loads(messages[-1]["content"][0]["text"])["observation"]
        assert {"judge_after", "goal_stable_seconds", "scene", "reward", "held"}.isdisjoint(observed)
        return json.dumps(next(self.actions))


@pytest.mark.requires_render
@pytest.mark.parametrize("kinematic_note", (False, True))
def test_original_model_only_prefix_gate_and_stitched_physical_replay(tmp_path, kinematic_note):
    c = load(BIMANUAL)
    if kinematic_note:
        c["policy"]["kinematic_note"] = True
    source, = collect_t1.make_plan(c, 1, 1, 77)
    scene = source["scene"]
    env = interface.make_env(scene, c)
    try:
        def move(z):
            return interface.move_action_from_world(env, "left", [*env.source_rack[:2], z], rotation(0, 0))
        before = [move(.24), move(.10), {"action": "set_gripper", "arm": "left", "opening": 0.}, {"action": "done"}]
        after = [move(.23), {"action": "done"}]
    finally:
        env.close()
    segmenter = SimpleNamespace(segment=lambda rgb, queries: [])
    parent = tmp_path / "original"
    directory = parent / source["directory"]
    failed = rollout.episode(scene, c, directory, FixturePolicy(before), segmenter, source["policy_seed"],
                             "original", origin=collect_t1.ORIGIN)
    assert failed["error"] is None and failed["held"] and not failed["success"]
    assert ("p + h*u" in rollout.system_prompt(c)) is kinematic_note
    assert failed["prompt_sha256"] == digest(rollout.system_prompt(c))
    write_json(directory / "verified.json", collect_t1.verify_result(directory, source, c))
    planned = continue_t1.select_attempts(parent, {"attempts": [source]}, c, 3, 2, 700)
    assert len(planned) == 2 and {a["policy_seed"] for a in planned} == {700, 701}
    assert planned[0]["parent"] == planned[1]["parent"]
    attempt = planned[0]
    prefix = continue_t1.load_prefix(parent, attempt)
    assert len(prefix) == 3
    branch_dir = tmp_path / "continuations" / attempt["directory"]
    result = rollout.episode(scene, c, branch_dir, FixturePolicy(after), segmenter, attempt["policy_seed"],
                             "original", prefix=prefix, origin=continue_t1.ORIGIN)
    assert result["error"] is None and result["success"]
    assert result["prompt_sha256"] == failed["prompt_sha256"]
    record = collect_t1.verify_result(branch_dir, attempt, c, origin=continue_t1.ORIGIN, prefix=prefix)
    assert record["replay_verified"] and (record["replayed_actions"], record["generated_actions"]) == (3, 2)
    root = tmp_path / "continuations"
    write_json(root / "collection.json", {"plan": {"attempts": planned, "config_sha256": digest(c),
                                                     "parent_collection_sha256": "fixture"}})
    # Two successful completions following identical actions constitute one unique trajectory.
    for entry in planned:
        write_json(root / entry["directory"] / "verified.json",
                   {**record, "trajectory": entry["directory"] + "/trajectory.jsonl", "policy_seed": entry["policy_seed"]})
    summary = collect_t1.summarize(root)
    assert summary["verified_successes"] == summary["verified_successful_continuations"] == 2
    assert summary["distinct_success_trajectories"] == len(summary["unique_successes"]) == 1
    assert summary["distinct_successful_scenes"] == 1
    assert (summary["replayed_actions"], summary["generated_actions"]) == (6, 4)
    with pytest.raises(ValueError, match="incompatible"):
        collect_t1.verify_result(branch_dir, attempt, c)
    # Any source edit after selection, including a changed response, invalidates the branch.
    raw = directory / "trajectory.jsonl"
    with raw.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="trajectory"):
        continue_t1.load_prefix(parent, attempt)


def test_freeze_requires_identical_base_weights_and_package_schema(tmp_path, monkeypatch):
    from tube_agent.curriculum.pipeline import model_reference
    c = load(BIMANUAL)
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"fixture")
    parent = tmp_path / "parent"
    root = tmp_path / "branches"
    stamps = {weights.name: [weights.stat().st_size, weights.stat().st_mtime_ns]}
    source_plan = {"model_files": stamps, "policy": "original", "config_sha256": digest(c), "sam_sha256": "sam",
                   "attempts": []}
    args = SimpleNamespace(seed=5, scenes=3, branches=2, shards=3)
    health = {"checkpoint_sha256": "sam"}
    monkeypatch.setattr(collect_t1, "lock", lambda *_: nullcontext())
    baseline = model_reference(model)
    write_json(parent / "collection.json", {"plan": source_plan, "model": {**baseline, "files": {}}})
    with pytest.raises(ValueError, match="base weights"):
        continue_t1.freeze_branches(args, parent, root, c, model, health)
    write_json(parent / "collection.json", {"plan": source_plan, "model": baseline})
    plan = continue_t1.freeze_branches(args, parent, root, c, model, health)
    assert plan["policy"] == "original" and plan["model_path"] == str(model)
    assert plan["sam_sha256"] == "sam" and plan["prompt_sha256"] == digest(rollout.system_prompt(c))
    assert plan["attempts"] == [] and plan["parent_collection_sha256"]
