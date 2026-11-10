"""Exercise the real batching/checkpoint loop using a tiny CPU policy, not Qwen."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import pytest
from tube_agent.curriculum.config import load, scene_manifest, schedule, file_digest, digest
from tube_agent.lab.experiment import write_json, read_json


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lora_weight = torch.nn.Parameter(torch.tensor([.2]))
    def save_pretrained(self, path, **kwargs):
        torch.save(self.state_dict(), Path(path)/"adapter_model.safetensors")


class Processor:
    def save_pretrained(self, path):
        write_json(Path(path)/"processor_config.json", {"test_only": True})


def setup_runtime(monkeypatch, tmp_path):
    import tube_agent.curriculum.train as trainer
    c = load(); manifest = scene_manifest(c)
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
                  "reward": float(label), "reward_terms": {"progress": float(label)}, "success": False, "error": None}
        write_json(directory/"result.json", result)
        return result
    monkeypatch.setattr(trainer, "episode", episode)
    return trainer, loads, calls


def test_real_loop_has_960_rollouts_18_batches_30_optimizer_steps_then_two_sft_epochs(monkeypatch, tmp_path):
    trainer, loads, calls = setup_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "warmup", "--run", str(tmp_path), "--model", "original-checkpoint"])
    trainer.main()
    assert len(calls) == 960
    done = read_json(tmp_path/"warmup/DONE.json")
    assert done["round"] == 18 and done["optimizer_steps"] == 30
    assert len({h for _, _, h in calls}) == 18  # Next batch really uses updated weights.
    assert len(list((tmp_path/"warmup/checkpoints").glob("*/COMPLETE.json"))) == 18
    # A completed resume must not draw more rollouts or perform more updates.
    trainer.main(); assert len(calls) == 960
    warmup = tmp_path/"checkpoints/after_warmup"
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
    assert sft["epochs"] == 2 and sft["optimizer_steps"] == 4 and sft["sft_batches"] == 4
    final = tmp_path/"checkpoints/after_sft"
    assert read_json(final/"COMPLETE.json")["epoch_completed"] == 2
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
    trainer.main()
    assert len(calls) == 960
    assert read_json(tmp_path/"warmup/DONE.json")["optimizer_steps"] == 30
