"""GPU runtime and checkpoint handling for the curriculum training workers."""

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch
import torch.distributed as dist

from tube_agent import posttrain
from tube_agent.posttrain import flatten_round as flatten_round
from tube_agent.lab.experiment import read_json, write_json
from tube_agent.lab.train import token_logps
from tube_agent.lab.vision import SamClient

from .config import digest, file_digest, load
from .learning import Generator, adapter_digest, audit_parameters, load_model, training_inputs
from .rollout import episode, read_rows


def runtime_device(local):
    return f"cuda:{local}"


class TrainingSession:
    def __init__(self, root, stage, model_path, rank, world, local):
        self.root, self.output = root, root / stage
        self.rank, self.world, self.local = rank, world, local
        self.device = runtime_device(local)
        self.config = load(root / "config.json", allow_focus=True)
        focus = self.config.get("experiment") == "t1_focus_v1"
        if focus and stage != "warmup":
            raise ValueError("T1 focus runs only support warmup")
        self.source = read_json(root / "SOURCE.json") if focus else None
        self.seed = self.config["seeds"]["manifest_seed"]
        self.model_path = model_path
        self.segmenter = None
        self.output.mkdir(parents=True, exist_ok=True)
        complete = sorted(p.parent for p in self.output.glob("checkpoints/step_*/COMPLETE.json")
                          if re.fullmatch(r"step_[0-9]{5}", p.parent.name))
        self.resume = complete[-1] if complete else None
        adapter = self.resume or (root / "checkpoints/after_warmup" if stage == "sft" else None)
        if focus and self.resume is None:
            adapter = Path(self.source["source_checkpoint"])
        torch.manual_seed(self.seed)
        random.seed(self.seed + rank)
        np.random.seed(self.seed + rank)
        if adapter:
            metadata = read_json(adapter / "COMPLETE.json")
            expected_config = self.source["parent_config_sha256"] if focus and not self.resume else digest(self.config)
            if metadata["config_sha256"] != expected_config or (
                file_digest(adapter / "adapter_model.safetensors") != metadata["adapter_sha256"]
            ):
                raise ValueError("Adapter checkpoint/config checksum mismatch")
        self.model, self.processor = load_model(model_path, self.config, adapter, self.device)
        self.parameters = [value for value in self.model.parameters() if value.requires_grad]
        if world > 1:
            for value in self.parameters:
                dist.broadcast(value.data, src=0)
        if rank == 0:
            write_json(self.output / "trainable_parameters.json", audit_parameters(self.model, self.config))
            write_json(self.output / "training_config.json", {
                "stage": stage, "model": model_path, "initial_adapter": str(adapter) if adapter else None,
                "config_sha256": digest(self.config), "world_size": world, "microbatch": 1,
                "warmup_minibatch_unit": "episodes", "sft_minibatch_unit": "unique assistant responses",
                "per_turn_negative_advantage": False, "optimizer_reset_for_sft": stage == "sft" and self.resume is None,
                "focus_source_adapter": self.source["source_checkpoint"] if focus else None,
                "optimizer_restored_from": str(self.resume) if self.resume else None,
                "failed_done_credit_clamp": self.config["training"]["failed_done_credit_clamp"],
            })
        settings = self.config["training"]
        self.generator = Generator(self.model, self.processor, self.config)
        self.optimizer = posttrain.create_optimizer(self.parameters, settings, stage)
        self.counters = {"round": 0, "optimizer_steps": 0, "sft_batches": 0}
        if self.resume:
            self.restore_optimizer()

    def restore_optimizer(self):
        path = self.resume / "optimizer_rng.pt"
        if file_digest(path) != read_json(self.resume / "COMPLETE.json")["optimizer_sha256"]:
            raise ValueError("Optimizer checkpoint checksum mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.counters.update(saved["counters"])
        state = saved["rng"][self.rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], self.local)

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def gather(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def sample(self, scene, directory, seed, version):
        return episode(scene, self.config, directory, self.generator, self.segmenter, seed, version, origin="warmup")

    def logps(self, row):
        return token_logps(self.model, *training_inputs(self.processor, row, self.config))

    def adapter_digest(self):
        return adapter_digest(self.model)

    def commit(self, index, extra=None):
        states = self.gather({"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.local)})
        final = self.output / "checkpoints" / f"step_{index:05d}"
        if self.rank == 0:
            temporary = final.with_name(final.name + f".partial-{os.getpid()}")
            if temporary.exists():
                temporary = temporary.with_name(temporary.name + datetime.now(timezone.utc).strftime("%H%M%S%f"))
            temporary.mkdir(parents=True)
            self.model.save_pretrained(temporary, safe_serialization=True)
            self.processor.save_pretrained(temporary)
            torch.save({"optimizer": self.optimizer.state_dict(), "counters": self.counters.copy(), "rng": states},
                       temporary / "optimizer_rng.pt")
            write_json(temporary / "COMPLETE.json", {
                "counters": self.counters.copy(), "config_sha256": digest(self.config),
                "adapter_sha256": file_digest(temporary / "adapter_model.safetensors"),
                "optimizer_sha256": file_digest(temporary / "optimizer_rng.pt"), **(extra or {}),
            })
            temporary.rename(final)
        self.barrier()
        return final

    def freeze_node(self, name, source):
        final = self.root / "checkpoints" / name
        if self.rank == 0 and not final.exists():
            temporary = final.with_name(name + ".partial")
            if temporary.exists():
                temporary.rename(temporary.with_name(temporary.name + datetime.now(timezone.utc).strftime("%H%M%S%f")))
            shutil.copytree(source, temporary)
            write_json(temporary / "node.json", {
                "name": name, "base_model": self.model_path, "source": str(source),
                "config_sha256": digest(self.config), "scenes_sha256": file_digest(self.root / "scenes.json"),
                "base_model_provenance": read_json(self.root / "provenance.json")["model"],
                "runtime_config": self.config, "observation_version": self.config["observation"]["version"],
                "data_manifest_sha256": file_digest(self.root / "data/manifest.json") if name == "after_sft" else None,
                "selection": "fixed_final_node_no_evaluation_selection",
            })
            temporary.rename(final)
        self.barrier()

    def smoke(self, scenes):
        before = self.generator.bind("smoke_before")
        config = {**self.config, "policy": {**self.config["policy"], "max_steps": 1}}
        directory = self.output / f"rank_{self.rank}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        seed = self.seed + self.rank
        result = episode(scenes["T1_train_000"], config, directory, self.generator, self.segmenter, seed,
                         "smoke_before", origin="training_smoke_not_data")
        if result["error"]:
            raise RuntimeError(result["error"])
        row = next(read_rows(directory))
        posttrain.start_minibatch(self)
        loss = -self.logps(row).mean() / self.world
        loss.backward()
        if not posttrain.step_optimizer(self):
            raise RuntimeError("No gradients in multimodal update smoke test")
        after = self.adapter_digest()
        if before == after:
            raise RuntimeError("Optimizer update did not change the adapter")
        self.generator.bind("smoke_after", after)
        self.generator(row["messages"], seed, 0.)
        if self.generator.model is not self.model or adapter_digest(self.generator.model) != after:
            raise RuntimeError("Post-update sampling used stale parameters")
        reports = self.gather({"rank": self.rank, "before": before, "after": after,
                               "new_parameters_used_for_generation": True})
        if self.rank == 0:
            write_json(self.output / "DONE.json", {"passed": True, "weights_discarded": True, "ranks": reports})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "warmup", "sft"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    try:
        session = TrainingSession(root, args.stage, args.model, rank, world, local)
        if args.stage == "sft":
            posttrain.run_sft(session)
            return
        scenes = {s["id"]: s for s in read_json(root / "scenes.json")["scenes"]}
        session.segmenter = SamClient(session.config["execution"]["sam_url"])
        try:
            session.segmenter.health()
            if args.stage == "smoke":
                session.smoke(scenes)
            else:
                posttrain.run_grpo(session, scenes, read_json(root / "schedule.json"))
        finally:
            session.segmenter.close()
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
