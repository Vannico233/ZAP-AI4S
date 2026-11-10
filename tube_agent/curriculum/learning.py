"""Shared multimodal model boundary; same inputs in rollout and teacher forcing."""
import hashlib
import json
from pathlib import Path
import torch
from tube_agent.serve import build_inputs
from tube_agent.lab.train import load as load_existing, sample_inputs as existing_inputs


def tensor_digest(values):
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def adapter_digest(model):
    return tensor_digest({name: value for name, value in model.named_parameters() if "lora_" in name})


def audit_parameters(model, config):
    entries = []
    for name, value in model.named_parameters():
        if value.requires_grad:
            if "lora_" not in name or ".language_model." not in name or any(
                    term in name for term in (".visual.", ".vision.", ".merger.", ".connector.")):
                raise ValueError(f"Unexpected trainable parameter: {name}")
            entries.append({"name": name, "shape": list(value.shape), "numel": value.numel(), "dtype": str(value.dtype)})
    if not entries: raise ValueError("No language LoRA parameters found")
    adapters = getattr(model, "peft_config", {})
    if len(adapters) != 1:
        raise ValueError("Exactly one adapter must be active; refusing stacked adapters")
    for adapter in adapters.values():
        if adapter.r != config["training"]["lora_rank"] or adapter.lora_alpha != config["training"]["lora_alpha"]:
            raise ValueError("Adapter rank/alpha do not match the frozen experiment")
    return entries


def load_model(path, config, adapter=None, device="cuda:0", trainable=True):
    model, processor = load_existing(path, adapter, config["training"]["lora_rank"], device, trainable)
    if trainable: audit_parameters(model, config)
    return model, processor


def training_inputs(processor, row, config):
    if "response_ids" not in row or "model_input_sha256" not in row:
        raise ValueError("Training requires the exact sampled token IDs and input fingerprint")
    inputs, length = existing_inputs(processor, row, config["policy"]["max_context"])
    prompt = {}
    for name, value in inputs.items():
        prompt[name] = value[:, :length] if name in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids") else value
    if tensor_digest(prompt) != row["model_input_sha256"]:
        raise ValueError("Training input differs from the model's actual rollout context")
    return inputs, length


class Generator:
    def __init__(self, model, processor, config):
        self.model, self.processor, self.config = model, processor, config
        self.provenance = {}

    def bind(self, version, expected_digest=None):
        current = adapter_digest(self.model)
        if expected_digest is not None and current != expected_digest:
            raise RuntimeError("Sampler is not using the expected updated adapter")
        self.provenance = {"policy_version": version, "adapter_sha256": current,
            "temperature": self.config["policy"]["temperature"], "top_p": self.config["policy"]["top_p"],
            "enable_thinking": False, "sampler": "same_live_model_object", "model_class": type(self.model).__name__}
        return current

    def __call__(self, messages, seed, temperature):
        inputs = build_inputs(self.processor, messages)
        self.last_input_sha256 = tensor_digest(inputs)
        length = inputs["input_ids"].shape[1]
        p = self.config["policy"]
        if length+p["max_new_tokens"] > p["max_context"]:
            raise ValueError("Exact visual context exceeds the frozen context budget")
        torch.manual_seed(seed)
        self.model.eval()
        options = {"max_new_tokens": p["max_new_tokens"], "do_sample": temperature > 0, "use_cache": True}
        if temperature > 0:
            options.update(temperature=temperature, top_p=p["top_p"], top_k=p["top_k"])
        with torch.inference_mode():
            generated = self.model.generate(**inputs.to(self.model.device), **options)
        self.last_response_ids = generated[0, length:].tolist()
        return self.processor.decode(generated[0, length:], skip_special_tokens=True)


def group_advantages(results):
    """Only center within a task+scene group; no noise or per-turn overrides."""
    import numpy as np
    if len(results) != 8 or len({(r["scene"]["task"], r["scene"]["sha256"]) for r in results}) != 1:
        raise ValueError("A GRPO group must contain 8 trajectories from the identical task/scene")
    if len({r["policy_version"] for r in results}) != 1 or any(r.get("error") for r in results):
        raise ValueError("Mixed policy versions or infrastructure errors in GRPO group")
    rewards = np.asarray([r["reward"] for r in results], dtype=float)
    if not np.isfinite(rewards).all(): raise ValueError("Nonfinite rewards")
    values = rewards-rewards.mean()
    same = bool(np.all(rewards == rewards[0]))
    return values.tolist(), {"rewards": rewards.tolist(), "mean": float(rewards.mean()),
        "std": float(rewards.std()), "all_rewards_equal": same,
        "zero_advantage": bool(np.all(np.abs(values) < 1e-12)),
        "behavior_signatures": [r.get("action_signature") for r in results],
        "reward_terms": [r["reward_terms"] for r in results],
        "zero_group_diagnosis": ("identical_action_sequences" if len({r.get("action_signature") for r in results}) == 1
                                  else "different_actions_same_reward_inspect_terms") if same else "nonzero"}
