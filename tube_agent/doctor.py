"""Check the CPU-prepared environment or the offline GPU instance."""
import argparse
import ctypes.util
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from .env import TubeEnv
from .model_utils import DEFAULT_MODEL, resolve_model
from .policy import SYSTEM, observation_message
from .serve import build_inputs
from .train_sft import make_examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--sim-only", action="store_true")
    args = p.parse_args()
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "osmesa_library": ctypes.util.find_library("OSMesa")}
    for name in ("mujoco", "numpy", "torch", "torchvision", "transformers", "accelerate", "peft"):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report[name] = "not installed"
    env = TubeEnv()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": observation_message(env.instruction, env.observe())}]
    report["simulator"] = "OK"
    env.close()
    if not args.sim_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig
        path = resolve_model(args.model)
        processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        inputs = build_inputs(processor, messages)
        messages.append({"role": "assistant", "content": '{"action":"finish"}'})
        examples, _ = make_examples([{"source": {"split": "train", "policy": "model"},
                                     "messages": messages}], processor.tokenizer, 8192)
        prefix = inputs["input_ids"][0].tolist()
        if examples[0]["input_ids"][:len(prefix)] != prefix:
            raise RuntimeError("Inference and training prompt tokens differ")
        report["sft_prompt_matches_inference"] = True
        index = path/"model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            missing = [name for name in shards if not (path/name).is_file()]
            if missing:
                raise RuntimeError(f"Missing model shards: {missing}")
        elif not list(path.glob("*.safetensors")):
            raise RuntimeError("No safetensors model weights found")
        report.update(model_path=str(path), cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), visible_gpus=torch.cuda.device_count())
        if args.gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: activate the prepared cu128 environment on the GPU instance")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
