"""Exercise the real batching/checkpoint loop using a tiny CPU policy, not Qwen."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import pytest
from tube_agent.curriculum.config import BIMANUAL, DEFAULT, load, scene_manifest, schedule, file_digest, digest
from tube_agent.lab.experiment import write_json, read_json


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lora_weight = torch.nn.Parameter(torch.tensor([.2]))
    def save_pretrained(self, path, **kwargs):
        torch.save(self.state_dict(), Path(path)/"adapter_model.safetensors")


class Processor:
    def save_pretrained(self, path):
        write_json(Path(path)/"processor_config.json", {"test_only": True})


def setup_runtime(monkeypatch, tmp_path, config_path=DEFAULT, training=None):
    import tube_agent.curriculum.train as trainer
    c = load(config_path)
    c["training"].update(training or {})
    manifest = scene_manifest(c)
    for name, value in (("config.json", c), ("scenes.json", manifest), ("schedule.json", schedule(c, manifest)),
                        ("provenance.json", {"model": {"resolved_path": "original-checkpoint"}})):
        write_json(tmp_path/name, value)
    monkeypatch.setenv("WORLD_SIZE", "1"); monkeypatch.setenv("RANK", "0"); monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *args: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *args: None)
    monkeypatch.setattr(trainer, "runtime_device", lambda local: "cpu")
    loads = []
    def load_model(path, config, adapter=None, *args):
        model = TinyPolicy()
        if adapter:
            model.load_state_dict(torch.load(Path(adapter)/"adapter_model.safetensors", weights_only=True))
        loads.append((adapter, model.lora_weight.detach().clone()))
        return model, Processor()
    monkeypatch.setattr(trainer, "load_model", load_model)
    monkeypatch.setattr(trainer, "audit_parameters", lambda *args: [{"name": "test_lora"}])
    class Generator:
        def __init__(self, model, *args): self.model = model; self.provenance = {}
        def bind(self, version, expected_digest=None):
            current = trainer.adapter_digest(self.model)
            if expected_digest: assert expected_digest == current
            self.provenance = {"adapter_sha256": current}
            return current
    monkeypatch.setattr(trainer, "Generator", Generator)
    monkeypatch.setattr(trainer, "training_inputs", lambda processor, row, config: ({"label": row["response_ids"][0]}, 0))
    monkeypatch.setattr(trainer, "token_logps", lambda model, inputs, length:
                        torch.nn.functional.logsigmoid(model.lora_weight*(1 if inputs["label"] else -1)))
    class SAM:
        def __init__(self, *args): pass
        def health(self): pass
        def close(self): pass
    monkeypatch.setattr(trainer, "SamClient", SAM)
    calls = []
    def episode(scene, config, directory, generator, segmenter, policy_seed, version, **kwargs):
        calls.append((scene["id"], version, generator.provenance["adapter_sha256"]))
        directory.mkdir(parents=True)
        label = int(policy_seed % 8 >= 4)
        row = {"response_ids": [label], "action": {"action": "done"}, "model_input_sha256": "toy"}
        (directory/"trajectory.jsonl").write_text(json.dumps(row)+"\n")
        result = {"scene": scene, "policy_version": version, "generator_provenance": generator.provenance.copy(),
                  "policy_seed": policy_seed, "config_sha256": digest(config),
                  "reward": float(label), "reward_terms": {"progress": float(label)}, "success": False, "error": None}
        write_json(directory/"result.json", result)
        return result
    monkeypatch.setattr(trainer, "episode", episode)
    return trainer, loads, calls


@pytest.mark.parametrize(("config_path", "training", "warmup_steps", "epochs", "sft_steps"), [
    (DEFAULT, {}, 30, 2, 4),
    (BIMANUAL, {"warmup_minibatch": 24, "sft_batch": 8, "sft_epochs": 3}, 48, 3, 9),
])
def test_real_loop_uses_frozen_settings(monkeypatch, tmp_path, config_path, training, warmup_steps, epochs, sft_steps):
    trainer, loads, calls = setup_runtime(monkeypatch, tmp_path, config_path, training)
    monkeypatch.setattr(sys, "argv", ["train", "warmup", "--run", str(tmp_path), "--model", "original-checkpoint"])
    trainer.main()
    assert len(calls) == 960
    done = read_json(tmp_path/"warmup/DONE.json")
    assert done["round"] == 18 and done["optimizer_steps"] == warmup_steps
    assert len({h for _, _, h in calls}) == 18  # Next batch really uses updated weights.
    assert len(list((tmp_path/"warmup/checkpoints").glob("*/COMPLETE.json"))) == 18
    # A completed resume must not draw more rollouts or perform more updates.
    trainer.main(); assert len(calls) == 960
    warmup = tmp_path/"checkpoints/after_warmup"
    assert read_json(warmup/"node.json")["observation_version"] == load(config_path)["observation"]["version"]
    before = torch.load(warmup/"adapter_model.safetensors", weights_only=True)["lora_weight"]
    path = tmp_path/"toy_success"; path.mkdir()
    rows = [{"response_ids": [1], "model_input_sha256": "toy"} for _ in range(17)]
    trajectory = path/"trajectory.jsonl"
    trajectory.write_text("".join(json.dumps(row)+"\n" for row in rows))
    samples = []; offset = 0
    with trajectory.open("rb") as stream:
        for index, line in enumerate(stream):
            samples.append({"trajectory": str(trajectory), "trajectory_sha256": file_digest(trajectory),
                "row": index, "byte_offset": offset, "scene_split": "train",
                "input_response_sha256": digest(["toy", [1]])})
            offset += len(line)
    data = tmp_path/"data"; data.mkdir()
    (data/"samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in samples))
    write_json(data/"manifest.json", {"samples": 17, "samples_sha256": file_digest(data/"samples.jsonl")})
    monkeypatch.setattr(sys, "argv", ["train", "sft", "--run", str(tmp_path), "--model", "original-checkpoint"])
    trainer.main()
    assert loads[-1][0] == warmup and torch.equal(loads[-1][1], before)
    sft = read_json(tmp_path/"sft/DONE.json")
    assert sft["epochs"] == epochs and sft["optimizer_steps"] == sft_steps and sft["sft_batches"] == sft_steps
    final = tmp_path/"checkpoints/after_sft"
    assert read_json(final/"COMPLETE.json")["epoch_completed"] == epochs
    after = torch.load(final/"adapter_model.safetensors", weights_only=True)["lora_weight"]
    assert not torch.equal(after, before)
    assert torch.equal(torch.load(warmup/"adapter_model.safetensors", weights_only=True)["lora_weight"], before)


def test_interrupted_batch_resumes_old_policy_without_duplicate_successes(monkeypatch, tmp_path):
    trainer, loads, calls = setup_runtime(monkeypatch, tmp_path)
    original = trainer.episode
    def interrupted(*args, **kwargs):
        if len(calls) == 75: raise RuntimeError("injected interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, "episode", interrupted)
    monkeypatch.setattr(sys, "argv", ["train", "warmup", "--run", str(tmp_path), "--model", "original-checkpoint"])
    with pytest.raises(RuntimeError, match="injected"): trainer.main()
    assert len(list((tmp_path/"warmup/checkpoints").glob("*/COMPLETE.json"))) == 1
    monkeypatch.setattr(trainer, "episode", original)
    cached = next((tmp_path/"warmup/rollouts/round_02").glob("*/try_00/result.json"))
    saved = read_json(cached)
    write_json(cached, {**saved, "config_sha256": "different-environment"})
    with pytest.raises(ValueError, match="different policy, environment, scene or seed"):
        trainer.main()
    write_json(cached, saved)
    trainer.main()
    assert len(calls) == 960
    assert read_json(tmp_path/"warmup/DONE.json")["optimizer_steps"] == 30


def test_t1_warm_start_clamps_only_failed_done_and_resumes(monkeypatch, tmp_path):
    from tube_agent.curriculum.t1_focus import focus_plan
    root = tmp_path/"focus"
    trainer, loads, calls = setup_runtime(monkeypatch, root)
    parent = load()
    config, scenes, rounds = focus_plan(parent, scene_manifest(parent))
    source = tmp_path/"parent/warmup/checkpoints/step_00015"
    source.mkdir(parents=True)
    model = TinyPolicy()
    with torch.no_grad():
        model.lora_weight.fill_(0.6)
    model.save_pretrained(source)
    write_json(source/"COMPLETE.json", {"config_sha256": digest(parent),
               "adapter_sha256": file_digest(source/"adapter_model.safetensors")})
    for name, value in (("config.json", config), ("scenes.json", scenes), ("schedule.json", rounds),
                        ("SOURCE.json", {"source_checkpoint": str(source), "parent_config_sha256": digest(parent),
                                         "tensor_sha256": trainer.adapter_digest(model)})):
        write_json(root/name, value)
    original = trainer.episode

    def focus_episode(scene, config, directory, generator, segmenter, policy_seed, version, **kwargs):
        result = original(scene, config, directory, generator, segmenter, policy_seed, version, **kwargs)
        path = directory/"trajectory.jsonl"
        row = json.loads(path.read_text())
        row.update(step=1, feedback={"accepted": True}, judge_after={"success": False}, action={"action": "move_to"})
        final_row = {**row, "action": {"action": "done"}, "step": 2}
        path.write_text(json.dumps(row)+"\n"+json.dumps(final_row)+"\n")
        result.update(task="T1", split="train", policy="model", policy_seed=policy_seed, origin="warmup",
            demonstration=None, search_prefix_length=0, replayed_actions=0, config_sha256=digest(config),
            generated_actions=2, steps=2)
        write_json(directory/"result.json", result)
        return result

    def interrupt(*args, **kwargs):
        if len(calls) == 35:
            raise RuntimeError("test interruption")
        return focus_episode(*args, **kwargs)

    monkeypatch.setattr(trainer, "episode", interrupt)
    monkeypatch.setattr(sys, "argv", ["train", "warmup", "--run", str(root), "--model", "original-checkpoint"])
    with pytest.raises(RuntimeError, match="test interruption"):
        trainer.main()
    assert loads[0][0] == source and torch.equal(loads[0][1], model.lora_weight)
    assert read_json(root/"warmup/training_config.json")["optimizer_restored_from"] is None
    monkeypatch.setattr(trainer, "episode", focus_episode)
    trainer.main()
    assert loads[-1][0] == root/"warmup/checkpoints/step_00001"
    assert len(calls) == 64
    done = read_json(root/"warmup/DONE.json")
    assert (done["rounds"], done["episodes"], done["optimizer_steps"]) == (2, 64, 2)
    for index in (1, 2):
        metric = read_json(root/"warmup/metrics"/f"round_{index:02d}.json")
        assert metric["failed_done_credit"] == {"failed_accepted_done": 32, "positive_before": 16, "positive_after": 0}
        assert all(g["failed_done_credit"]["positive_before"] == 4 for g in metric["groups"])
