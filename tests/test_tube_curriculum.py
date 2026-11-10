import copy
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tube_agent.curriculum.config import load, scene_manifest, schedule, digest, plan, TASKS
from tube_agent.curriculum.env import CurriculumEnv, diagnostic_commands, move
from tube_agent.curriculum.learning import group_advantages, tensor_digest, training_inputs
from tube_agent.curriculum.rollout import replay, same_state, context, restore_context
from tube_agent.curriculum.search import Forest
from tube_agent.curriculum.train import flatten_round
from tube_agent.lab.experiment import write_json, read_json


def scene(task="T1", split="train", index=0):
    return next(s for s in scene_manifest(load())["scenes"] if s["task"] == task and s["split"] == split and s["index"] == index)


def test_frozen_scene_pools_and_exact_schedule():
    c = load(); manifest = scene_manifest(c); rounds = schedule(c, manifest)
    assert len(manifest["scenes"]) == 248
    assert len({s["sha256"] for s in manifest["scenes"]}) == 248
    assert Counter((s["task"], s["split"]) for s in manifest["scenes"]) == {
        (t, split): n for t in TASKS for split, n in (("train", 32), ("eval", 30))}
    assert len(rounds) == 18
    counts = Counter(); policy_seeds = []
    for number, row in enumerate(rounds):
        batch = flatten_round(row)
        assert len(batch) == (64 if number < 12 else 32)
        for group in row["groups"]:
            assert "_train_" in group["scene_id"] and len(group["rollouts"]) == 8
            counts[group["task"]] += 8
            policy_seeds += [r["policy_seed"] for r in group["rollouts"]]
        # The three policy ranks cover every rollout once without changing group size.
        assert sorted(i for rank in range(3) for i in range(rank, len(batch), 3)) == list(range(len(batch)))
    assert counts == {"T1": 192, "T2": 192, "T3": 288, "T4": 288}
    assert len(set(policy_seeds)) == 960
    assert plan(c)["planned_warmup_optimizer_steps"] == 30


def group(rewards):
    return [{"scene": scene(), "policy_version": "batch_1", "reward": r, "reward_terms": {"p": r},
             "action_signature": str(i), "error": None} for i, r in enumerate(rewards)]


def test_advantages_are_group_local_and_no_noise():
    one, _ = group_advantages(group([0, 1, 2, 3, 4, 5, 6, 7]))
    two, _ = group_advantages(group([100, 101, 102, 103, 104, 105, 106, 107]))
    assert one == two == [-3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5]
    values, report = group_advantages(group([.2]*8))
    assert values == [0.]*8 and report["zero_advantage"]
    assert report["zero_group_diagnosis"] == "different_actions_same_reward_inspect_terms"
    mixed = group([0]*8); mixed[0]["scene"] = scene("T2")
    with pytest.raises(ValueError, match="identical"): group_advantages(mixed)
    mixed = group([0]*8); mixed[0]["policy_version"] = "stale"
    with pytest.raises(ValueError, match="policy"): group_advantages(mixed)


@pytest.mark.parametrize("task", TASKS)
def test_new_tasks_are_not_successful_at_reset_or_from_done(task):
    env = CurriculumEnv(scene(task))
    try:
        assert not env.result()["success"]
        assert env.result()["held"] == (task in ("T2", "T3"))
        env.step({"action": "done"})
        assert not env.result()["success"]
    finally: env.close()


@pytest.mark.parametrize("task", TASKS)
def test_four_tasks_physically_succeed_without_attachment(task):
    c = load(); env = CurriculumEnv(scene(task), c)
    rows = []
    try:
        for action in diagnostic_commands(env):
            feedback = env.step(action)[1]
            rows.append({"action": action, "feedback": feedback, "state_after": env.snapshot()})
            assert feedback["accepted"]
        result = env.result()
        assert result["success"] and result["invalid_actions"] == 0
        assert result["reward"] == pytest.approx(5-.01*len(rows))
        assert result["goal_stable_seconds"] >= (.5 if task in ("T1", "T2") else 1.)
        if task in ("T3", "T4"):
            assert result["geometry"]["seated"] and result["geometry"]["released"]
        assert not any("weld" in str(env.model.equality(i).name) for i in range(env.model.neq))
    finally: env.close()
    assert replay(scene(task), c, rows)["success"]


def test_empty_close_and_empty_hand_at_goal_cannot_earn_carry_reward():
    env = CurriculumEnv(scene("T2"))
    try:
        env.step({"action": "set_gripper", "opening": 1.})
        result = env.result()
        assert not result["held"] and not result["success"]
        assert result["reward_terms"]["terminal_progress"] == 0.
        assert result["dropped"] and result["reward_terms"]["failure_cost"] == 1.
    finally: env.close()
    env = CurriculumEnv(scene("T1"))
    try:
        env.step({"action": "set_gripper", "opening": 0.})
        assert not env.result()["held"]
        assert env.result()["reward_terms"]["grasp_lift"] == 0.
    finally: env.close()


def test_seated_but_held_is_not_release_success_and_reward_is_terminal():
    env = CurriculumEnv(scene("T3"))
    try:
        for action in diagnostic_commands(env):
            if action == {"action": "set_gripper", "opening": 1.}: break
            env.step(action)
        result = env.result()
        assert result["held"] and result["geometry"]["seated"]
        assert not result["success"] and result["reward_terms"]["release"] == 0.
        before = result["reward"]
        env.steps += 5
        assert env.result()["reward"] == pytest.approx(before-.05)
        env.invalid_actions += 2
        assert env.result()["reward"] == pytest.approx(before-.05-.2)
        env.severe_collision = True
        assert env.result()["reward"] == pytest.approx(before-.05-.2-1.)
    finally: env.close()


def test_contact_without_relative_motion_stability_is_not_grasp(monkeypatch):
    env = CurriculumEnv(scene("T1"))
    try:
        monkeypatch.setattr(env, "_contacts", lambda: {1, 2})
        env.relative_speed = 1.; env.relative_angular_speed = 0.
        assert not env._held_now()
        env.relative_speed = 0.; env.relative_angular_speed = 1.
        assert not env._held_now()
    finally: env.close()


def test_collision_geometry_really_has_open_hole():
    env = CurriculumEnv(scene())
    try:
        for name in ("source", "target"):
            center = env.data.body(name+"_rack").xpos.copy(); center[2] = .02
            for axis in (0, 1):
                for sign in (-1, 1):
                    geom = env.model.geom(f"{name}_wall_{axis}_{sign}")
                    position = env.data.geom_xpos[geom.id]
                    assert not np.all(np.abs(center-position) < geom.size)
    finally: env.close()


def test_context_restore_preserves_all_agent_memory():
    from tube_agent.lab.policy import VisualPolicy
    class Observer: queries = ["red tube"]
    observer = Observer(); policy = VisualPolicy(None, action_protocol="zap_pose_v1")
    policy.history = [{"role": "assistant", "content": "x"}]
    policy.action_memory = [{"feedback": {"accepted": True}}]
    policy.plan = "insert"; policy.initial_visual_reference = [{"id": "top:0"}]
    snapshot = context(policy, observer)
    policy.plan = "wrong"; observer.queries = []
    restore_context(policy, observer, snapshot)
    assert context(policy, observer) == snapshot
    policy.action_memory.append({})
    assert len(snapshot["action_memory"]) == 1


def test_training_reconstructs_exact_prompt_and_only_appends_response(monkeypatch):
    import tube_agent.curriculum.learning as learning
    source = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]]),
              "pixel_values": torch.arange(8.).reshape(2, 4)}
    row = {"response_ids": [4, 5], "model_input_sha256": tensor_digest(source)}
    def inputs(*args):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.ones(1, 5, dtype=torch.int64),
                "pixel_values": source["pixel_values"]}, 3
    monkeypatch.setattr(learning, "existing_inputs", inputs)
    rebuilt, length = training_inputs(None, row, load())
    assert length == 3 and rebuilt["input_ids"][0, length:].tolist() == row["response_ids"]
    source["pixel_values"][0, 0] = -1
    with pytest.raises(ValueError, match="context"): training_inputs(None, row, load())


def test_token_loss_backpropagates_only_response_predictions():
    from tube_agent.lab.train import token_logps
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.logits = torch.nn.Parameter(torch.zeros(1, 5, 8))
        @property
        def device(self): return self.logits.device
        def forward(self, input_ids, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.logits[:, -kwargs["logits_to_keep"]:])
    model = Toy(); logps = token_logps(model, {"input_ids": torch.tensor([[0, 1, 2, 3, 4]])}, 3)
    assert logps.shape == (2,)
    (-logps.mean()).backward()
    assert model.logits.grad[0, :2].abs().sum() == 0
    assert model.logits.grad[0, 2:4].abs().sum() > 0
    assert model.logits.grad[0, 4].abs().sum() == 0


def test_pipeline_has_fixed_nodes_no_main_rl_and_empty_data_is_explicit(tmp_path, monkeypatch):
    import tube_agent.curriculum.pipeline as pipeline
    from types import SimpleNamespace
    c = load(); write_json(tmp_path/"scenes.json", scene_manifest(c))
    write_json(tmp_path/"data/manifest.json", {"samples": 0})
    monkeypatch.setattr(pipeline, "gpu_check", lambda a: {})
    monkeypatch.setattr(pipeline, "report", lambda root, available: {"status": "completed" if available else "completed_partial"})
    calls = []
    class Runner:
        output = tmp_path
        a = SimpleNamespace(policy_gpus=[0, 2, 3], model="unchanged-model", sam_url="localhost")
        def status(self, *args, **kwargs): pass
        def step(self, name, *args): calls.append(name)
        def train_stage(self, stage): calls.append(stage)
        def evaluate(self, node): calls.append("eval:"+node)
        def ensure_sam(self): pass
    pipeline.execute(Runner(), c)
    assert calls == ["physics", "vision", "smoke", "eval:original", "warmup", "search", "freeze_data", "eval:after_warmup"]
    assert (tmp_path/"checkpoints/after_sft_UNAVAILABLE.json").exists()
    calls.clear(); write_json(tmp_path/"data/manifest.json", {"samples": 10})
    pipeline.execute(Runner(), c)
    assert calls == ["physics", "vision", "smoke", "eval:original", "warmup", "search", "freeze_data", "sft", "eval:after_warmup", "eval:after_sft"]


def test_search_prunes_same_failure_and_shared_budget_contract():
    c = load(); forest = Forest(c)
    node = {"key": "one", "parent": None, "task": "T1", "visits": 0, "wins": 0,
            "failures": {}, "progress": .3}
    child = {**node, "key": "two", "parent": "one"}
    forest.nodes = {"one": node, "two": child}
    forest.add = lambda directory: None
    failures = [("path", {"success": False, "failure_type": "not_grasped", "replayed_actions": 2, "generated_actions": 3})]*4
    forest.update(dict(node), failures)
    assert forest.blocked == {"one", "two"} and forest.select() is None
    assert sum(e["attempts"] for e in forest.expansions) == 4
    assert c["search"]["max_continuations"] == 128


@pytest.mark.requires_render
def test_real_visual_episode_and_prefix_restore(tmp_path):
    from tube_agent.curriculum.rollout import episode, read_rows
    class Masks:
        def segment(self, *args): return []
    actions = iter([move([.45, -.17, .24]), {"action": "update_todo", "plan": "grasp later"}, {"action": "done"}])
    def generator(messages, *args):
        text = json.loads(messages[-1]["content"][0]["text"])
        assert "reward" not in text["observation"] and "held" not in text["observation"]
        assert len(messages[-1]["content"]) == 3
        return json.dumps(next(actions))
    c = load(); first = tmp_path/"first"
    result = episode(scene(), c, first, generator, Masks(), 12, "test", origin="test")
    assert result["error"] is None
    prefix = list(read_rows(first))[:2]
    def finish(messages, *args):
        text = json.loads(messages[-1]["content"][0]["text"])
        assert text["remaining_plan"] == "grasp later" and len(text["action_history"]) == 2
        return '{"action":"done"}'
    result = episode(scene(), c, tmp_path/"second", finish, Masks(), 13, "test", prefix=prefix, origin="test")
    assert result["error"] is None and result["replayed_actions"] == 2 and result["generated_actions"] == 1
