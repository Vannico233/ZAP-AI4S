import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest

from tube_agent.lab.experiment import TRAIN_TASK_NAMES, read_json, write_json
from tube_agent.lab.zap_recipe import warmup_schedule


def put_success(root, task="pick", index=0, stage="warmup", **patch):
    from tube_agent.lab.zap_policy import SYSTEM
    if stage == "warmup":
        folder = root/f"warmup/rollouts/update_{index:04d}/rank_00/try_000"
        write_json(root/f"warmup/step_{index+1:05d}/COMPLETE.json", {"step": index+1})
        origin = {"model": "model", "training": True, "on_policy": True,
                  "training_stage": str(root/"warmup"), "policy_update": index}
    else:
        folder = root/f"{stage}/{task}/try_{index:03d}"
        origin = {"model": "model", "training": False, "adapter": None}
    result = {"policy": "model", "split": "train", "task": task, "seed": 0,
              "steps": 1, "success": True, "reward": 5., "error": None, "invalid_actions": 0,
              "observation_mode": "rgbd_sam3.1_detector", "demonstration": None,
              "action_protocol": "zap_pose_v1", "generator_provenance": origin,
              "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), **patch}
    action = {"action": "set_gripper", "opening": index*.001}
    trace = {"step": 1, "action": action, "raw_response": json.dumps(action),
             "feedback": {"accepted": True}, "messages": [{"role": "system", "content": SYSTEM}],
             "judge_after": {"reward": 5.}}
    write_json(folder/"result.json", result)
    (folder/"trajectory.jsonl").write_text(json.dumps(trace)+"\n")
    return folder


def test_recipe_plan_allocates_all_tasks_and_no_legacy_training_gate():
    output = subprocess.check_output([sys.executable, "-m", "tube_agent.lab.pipeline", "--plan",
                "--zap-recipe", "--action-protocol", "zap_pose_v1"], text=True)
    plan = json.loads(output)
    assert plan["warmup_rollouts_by_task"] == {"pick": 144, "transport": 120, "insert": 72, "unscrew": 72}
    assert not plan["no_training_before_gate"]
    assert plan["max_total_episodes"] == 1407
    schedule = warmup_schedule([24, 20, 12, 12])
    from collections import Counter
    assert Counter(schedule) == dict(zip(TRAIN_TASK_NAMES, [24, 20, 12, 12]))
    assert set(schedule[:4]) == set(TRAIN_TASK_NAMES)
    assert "unscrew" in schedule[-8:]  # Hard tasks are not all left at stale early weights.


def test_dr_grpo_has_no_std_or_sampled_length_normalization():
    import numpy as np
    import torch
    from tube_agent.lab.train import advantages, dr_grpo_loss
    assert np.allclose(advantages([0., .2, .4], False), [-.2, 0., .2])
    assert np.allclose(advantages([0., 2., 4.], False), [-2., 0., 2.])
    assert np.allclose(advantages([.2, .2], False), 0.)
    x = torch.tensor([-.5, -.5], requires_grad=True)
    loss = dr_grpo_loss(x, x.detach(), 2., token_budget=10)
    loss.backward()
    assert torch.allclose(x.grad, torch.tensor([-.2, -.2]))
    y = torch.tensor([-.5], requires_grad=True)
    dr_grpo_loss(y, y.detach(), -1., token_budget=10).backward()
    assert torch.allclose(y.grad, torch.tensor([.1]))  # Invalid-turn likelihood decreases.


def test_schedule_rejects_heldout_or_wrong_length(tmp_path):
    from tube_agent.lab.distributed_train import task_schedule
    file = tmp_path/"schedule.json"
    write_json(file, ["pick", "rack_transfer"])
    with pytest.raises(ValueError): task_schedule(file, TRAIN_TASK_NAMES, 2)
    write_json(file, ["pick"])
    with pytest.raises(ValueError): task_schedule(file, TRAIN_TASK_NAMES, 2)


@pytest.mark.parametrize("bad", ["none", "manual", "test", "wrong_model", "wrong_prompt", "rewritten", "incomplete", "failed_replay", "uncommitted"])
def test_warmup_success_admission_is_provenance_and_replay_checked(tmp_path, bad):
    from tube_agent.lab.recipe_gate import inspect_gate
    folder = put_success(tmp_path)
    result = read_json(folder/"result.json")
    if bad == "manual": result["demonstration"] = {"manual": True}
    if bad == "test": result["split"] = "test"
    if bad == "wrong_model": result["generator_provenance"]["model"] = "teacher"
    if bad == "wrong_prompt": result["prompt_sha256"] = "other"
    write_json(folder/"result.json", result)
    if bad == "rewritten":
        row = json.loads((folder/"trajectory.jsonl").read_text())
        row["raw_response"] = '{"action":"done"}'
        (folder/"trajectory.jsonl").write_text(json.dumps(row)+"\n")
    if bad == "incomplete": folder.rename(folder.with_name("try_000.incomplete-old"))
    if bad == "uncommitted": (tmp_path/"warmup/step_00001/COMPLETE.json").unlink()
    gate = inspect_gate(tmp_path, [tmp_path/"warmup/rollouts"], 0, "model",
                        dict.fromkeys(TRAIN_TASK_NAMES, 3), 1, verifier=lambda p: bad != "failed_replay")
    assert gate["unique_verified_successes"]["pick"] == int(bad == "none")
    assert not gate["passed"]  # Other three tasks are missing.


def test_gate_separates_preferred_targets_from_minimum_and_rejects_changed_data(tmp_path):
    from tube_agent.lab.recipe_gate import inspect_gate, require_gate
    for i, task in enumerate(TRAIN_TASK_NAMES): put_success(tmp_path, task, i)
    gate = inspect_gate(tmp_path, [tmp_path/"warmup/rollouts"], 0, "model",
                        dict.fromkeys(TRAIN_TASK_NAMES, 3), 1, verifier=lambda p: True)
    assert gate["passed"] and not gate["targets_met"]
    require_gate(tmp_path)
    Path(gate["approved"]["pick"][0]["trajectory"]).write_text("tampered")
    with pytest.raises(RuntimeError, match="evidence changed"): require_gate(tmp_path)


def test_recipe_warmup_is_allowed_without_data_but_sft_is_not(tmp_path):
    from tube_agent.lab.pipeline import Runner, parser
    a = parser().parse_args(["--zap-recipe", "--action-protocol", "zap_pose_v1"])
    a.output = tmp_path
    runner = Runner(a)
    commands = []
    def intercept(name, command, gpus):
        commands.append(command)
        raise LookupError("launch intercepted")
    runner.start = intercept
    for name, stage in (("training_smoke", "smoke"), ("warmup", "grpo")):
        with pytest.raises(LookupError): runner.train(name, stage)
    assert all("dr_grpo" in cmd for cmd in commands)
    for name, stage in (("sft", "sft"), ("rl_fixed", "grpo")):
        with pytest.raises(RuntimeError, match="gate missing"): runner.train(name, stage)


@pytest.mark.parametrize("has_success", [False, True])
def test_complete_recipe_and_zero_success_budget_exhaustion(tmp_path, monkeypatch, has_success):
    import tube_agent.lab.pipeline as pipeline
    import tube_agent.lab.recipe_gate as gate
    from tube_agent.lab.zap_recipe import execute
    a = pipeline.parser().parse_args(["--zap-recipe", "--action-protocol", "zap_pose_v1",
              "--warmup-updates", "1", "1", "1", "1", "--success-targets", "1", "1", "1", "1",
              "--sft-min-successes", "1", "--search-rounds", "2", "--search-attempts", "1"])
    a.output, a.model = tmp_path, "model"
    monkeypatch.setattr(gate, "replay_check", lambda p: True)
    monkeypatch.setattr(pipeline, "make_videos", lambda r: {})
    class Runner:
        output = tmp_path
        def __init__(self): self.a, self.training, self.samples = a, [], []
        def status(self, *args, **kwargs): pass
        def train(self, name, stage, **kwargs):
            if name not in ("training_smoke", "warmup"): gate.require_gate(tmp_path)
            self.training.append((name, kwargs))
            if name == "warmup" and has_success:
                for i, task in enumerate(TRAIN_TASK_NAMES): put_success(tmp_path, task, i)
            adapter = tmp_path/name/"step_00004"
            write_json(tmp_path/name/"DONE.json", {"checkpoint": str(adapter)})
            return adapter if stage != "smoke" else None
        def sample(self, name, adapter, job_list):
            self.samples.append((name, adapter, job_list))
            for job in job_list:
                write_json(tmp_path/name/job["id"]/"summary.json", {"job": job, "episodes": job["attempts"],
                            "successes": 0, "errors": 0, "mean_reward": 0., "invalid_actions": 0})
    runner = Runner()
    execute(runner)
    names = [name for name, _ in runner.training]
    assert names[:2] == ["training_smoke", "warmup"]
    if has_success:
        assert names == ["training_smoke", "warmup", "sft", "rl_fixed", "rl_diverse"]
        assert "adapter" not in dict(runner.training)["sft"]  # Self-distill into base.
        assert read_json(tmp_path/"report.json")["status"] == "completed"
    else:
        assert names == ["training_smoke", "warmup"]
        assert len([name for name, _, _ in runner.samples if name.startswith("mine_round_")]) == 2
        assert read_json(tmp_path/"report.json")["status"] == "completed_partial"
    for name, adapter, jobs in runner.samples:
        if name.startswith("mine_round_"):
            assert adapter == tmp_path/"warmup/step_00004"
        if name not in ("base_test", "final_test"):
            assert all(j["task"] != "rack_transfer" for j in jobs)
    assert {name for name, _, _ in runner.samples} >= {"base_test", "final_test", "final_fixed"}


def test_search_prefixes_must_use_identical_frozen_adapter(tmp_path):
    from tube_agent.lab.worker import seed_tree
    from tube_agent.lab.run import PrefixTree
    folder = put_success(tmp_path, stage="mine_round_000")
    result = read_json(folder/"result.json")
    result["generator_provenance"]["adapter"] = "checkpoint-A"
    write_json(folder/"result.json", result)
    job = {"task": "pick", "seed": 0}
    for adapter, expected in ((None, 0), ("checkpoint-B", 0), ("checkpoint-A", 1)):
        assert seed_tree(PrefixTree(), job, [tmp_path], "zap_pose_v1", "model", adapter) == expected


def test_training_progress_uses_committed_updates_only(tmp_path):
    from tube_agent.lab.zap_recipe import training_summary
    for step in (1, 2):
        write_json(tmp_path/f"metrics/step_{step:05d}.json", {"task": "pick", "step": step,
            "group_size": 6, "successes": 1, "optimizer_step_taken": True, "invalid_turns": 2, "rewards": [0., 1.]})
    write_json(tmp_path/"step_00001/COMPLETE.json", {"step": 1})
    report = training_summary(tmp_path)
    assert report["updates"] == 1 and report["tasks"]["pick"]["episodes"] == 6


def test_review_archive_keeps_recipe_admission_and_committed_training_metadata(tmp_path):
    import tarfile
    root=tmp_path/"run"
    write_json(root/"config.json",{"zap_recipe":True})
    write_json(root/"recipe_success_gate.json",{"approved":{}})
    write_json(root/"warmup/metrics/step_00001.json",{"step":1})
    write_json(root/"warmup/step_00001/COMPLETE.json",{"step":1})
    script=Path(__file__).resolve().parents[1]/"scripts/collect_success_first.py"
    subprocess.run([sys.executable,str(script),"--run",str(root)],cwd=tmp_path,check=True,capture_output=True)
    with tarfile.open(next(tmp_path.glob("success_first_review_*.tar.gz"))) as archive:
        names=set(archive.getnames())
    assert {"recipe_success_gate.json","warmup/metrics/step_00001.json","warmup/step_00001/COMPLETE.json"} <= names
