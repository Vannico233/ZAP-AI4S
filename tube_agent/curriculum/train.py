"""Synchronous multimodal warm-up: complete batch -> one epoch of minibatches.

Reuses existing Qwen/PEFT loading, response-token likelihoods and explicit gradient
summation. Groups of 8 are independent of the three policy GPU replica count.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import numpy as np
import torch
import torch.distributed as dist
from tube_agent.lab.distributed_train import synchronize_gradients
from tube_agent.lab.train import token_logps, dr_grpo_loss
from tube_agent.lab.experiment import read_json, write_json
from tube_agent.lab.vision import SamClient
from .config import load, digest, file_digest
from .learning import load_model, Generator, audit_parameters, adapter_digest, training_inputs, group_advantages
from .rollout import episode, read_rows
from .terminal_credit import failed_final_done, response_advantage


def runtime_device(local):
    return f"cuda:{local}"


def flatten_round(round_spec):
    return [{"group": group["id"], "scene_id": group["scene_id"], **sample}
            for group in round_spec["groups"] for sample in group["rollouts"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "warmup", "sft"))
    p.add_argument("--run", type=Path, required=True); p.add_argument("--model", required=True)
    a = p.parse_args(); root = a.run.resolve(); c = load(root/"config.json", allow_focus=True)
    focus = c.get("experiment") == "t1_focus_v1"
    if focus and a.stage != "warmup":
        p.error("T1 focus runs only support warmup")
    source = read_json(root/"SOURCE.json") if focus else None
    manifest = read_json(root/"scenes.json"); scenes = {s["id"]: s for s in manifest["scenes"]}
    rounds = read_json(root/"schedule.json")
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    torch.cuda.set_device(local); device = runtime_device(local)
    distributed = dist if world > 1 else None
    if distributed: dist.init_process_group("nccl", timeout=timedelta(hours=24))
    def barrier():
        if distributed: dist.barrier()
    def gather(value):
        if not distributed: return [value]
        values = [None]*world; dist.all_gather_object(values, value); return values
    output = root/a.stage; output.mkdir(parents=True, exist_ok=True)
    complete = sorted(p for p in output.glob("checkpoints/step_*/COMPLETE.json")
                      if re.fullmatch(r"step_[0-9]{5}", p.parent.name))
    resume = complete[-1].parent if complete else None
    adapter = resume or (root/"checkpoints/after_warmup" if a.stage == "sft" else None)
    if focus and resume is None:
        adapter = Path(source["source_checkpoint"])
    seed = c["seeds"]["manifest_seed"]
    torch.manual_seed(seed); random.seed(seed+rank); np.random.seed(seed+rank)
    if adapter:
        metadata = read_json(Path(adapter)/"COMPLETE.json")
        expected_config = source["parent_config_sha256"] if focus and not resume else digest(c)
        if metadata["config_sha256"] != expected_config or file_digest(Path(adapter)/"adapter_model.safetensors") != metadata["adapter_sha256"]:
            raise ValueError("Adapter checkpoint/config checksum mismatch")
    model, processor = load_model(a.model, c, adapter, device)
    parameters = [v for v in model.parameters() if v.requires_grad]
    if distributed:
        for value in parameters: dist.broadcast(value.data, src=0)
    if rank == 0:
        write_json(output/"trainable_parameters.json", audit_parameters(model, c))
        write_json(output/"training_config.json", {"stage": a.stage, "model": a.model, "initial_adapter": str(adapter) if adapter else None,
            "config_sha256": digest(c), "world_size": world, "microbatch": 1,
            "warmup_minibatch_unit": "episodes", "sft_minibatch_unit": "unique assistant responses",
            "per_turn_negative_advantage": False, "optimizer_reset_for_sft": a.stage == "sft" and resume is None,
            "focus_source_adapter": source["source_checkpoint"] if focus else None,
            "optimizer_restored_from": str(resume) if resume else None,
            "failed_done_credit_clamp": c["training"]["failed_done_credit_clamp"]})
    generator = Generator(model, processor, c)
    optimizer = torch.optim.AdamW(parameters, lr=c["training"]["sft_lr" if a.stage == "sft" else "warmup_lr"],
                                  weight_decay=c["training"]["weight_decay"])
    counters = {"round": 0, "optimizer_steps": 0, "sft_batches": 0}
    if resume:
        # Only locally created, checksummed checkpoint files are accepted.
        metadata = read_json(resume/"COMPLETE.json")
        if file_digest(resume/"optimizer_rng.pt") != metadata["optimizer_sha256"]:
            raise ValueError("Optimizer checkpoint checksum mismatch")
        saved = torch.load(resume/"optimizer_rng.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"]); counters.update(saved["counters"])
        state = saved["rng"][rank]
        random.setstate(state["python"]); np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state(state["cuda"], local)
    def optimize():
        updated = synchronize_gradients(parameters, distributed)
        if updated:
            torch.nn.utils.clip_grad_norm_(parameters, c["training"]["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step(); counters["optimizer_steps"] += 1
        optimizer.zero_grad(set_to_none=True)
        return updated
    def train_mode():
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout): module.eval()
    def commit(index, extra=None):
        states = gather({"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(local)})
        final = output/"checkpoints"/f"step_{index:05d}"
        if rank == 0:
            temporary = final.with_name(final.name+f".partial-{os.getpid()}")
            if temporary.exists(): temporary = temporary.with_name(temporary.name+datetime.now(timezone.utc).strftime("%H%M%S%f"))
            temporary.mkdir(parents=True)
            model.save_pretrained(temporary, safe_serialization=True); processor.save_pretrained(temporary)
            torch.save({"optimizer": optimizer.state_dict(), "counters": counters.copy(), "rng": states}, temporary/"optimizer_rng.pt")
            write_json(temporary/"COMPLETE.json", {"counters": counters.copy(), "config_sha256": digest(c),
                "adapter_sha256": file_digest(temporary/"adapter_model.safetensors"),
                "optimizer_sha256": file_digest(temporary/"optimizer_rng.pt"), **(extra or {})})
            temporary.rename(final)
        barrier()
        return final
    def freeze_node(name, source):
        if rank == 0:
            final = root/"checkpoints"/name
            if not final.exists():
                temporary = final.with_name(name+".partial")
                if temporary.exists(): temporary.rename(temporary.with_name(temporary.name+datetime.now(timezone.utc).strftime("%H%M%S%f")))
                shutil.copytree(source, temporary)
                write_json(temporary/"node.json", {"name": name, "base_model": a.model, "source": str(source),
                    "config_sha256": digest(c), "scenes_sha256": file_digest(root/"scenes.json"),
                    "base_model_provenance": read_json(root/"provenance.json")["model"],
                    "runtime_config": c, "observation_version": "upright_visual_v1",
                    "data_manifest_sha256": file_digest(root/"data/manifest.json") if name == "after_sft" else None,
                    "selection": "fixed_final_node_no_evaluation_selection"})
                temporary.rename(final)
        barrier()
    segmenter = None
    try:
        if a.stage in ("smoke", "warmup"):
            segmenter = SamClient(c["execution"]["sam_url"]); segmenter.health()
        if a.stage == "smoke":
            before = generator.bind("smoke_before")
            smoke_c = json.loads(json.dumps(c)); smoke_c["policy"]["max_steps"] = 1
            directory = output/f"rank_{rank}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
            result = episode(scenes["T1_train_000"], smoke_c, directory, generator, segmenter, seed+rank,
                             "smoke_before", origin="training_smoke_not_data")
            if result["error"]: raise RuntimeError(result["error"])
            row = next(read_rows(directory)); train_mode()
            loss = -token_logps(model, *training_inputs(processor, row, c)).mean()/world
            loss.backward()
            if not optimize(): raise RuntimeError("No gradients in multimodal update smoke test")
            after = adapter_digest(model)
            if before == after: raise RuntimeError("Optimizer update did not change the adapter")
            generator.bind("smoke_after", after)
            generator(row["messages"], seed+rank, 0.)
            if generator.model is not model or adapter_digest(generator.model) != after:
                raise RuntimeError("Post-update sampling used stale parameters")
            reports = gather({"rank": rank, "before": before, "after": after,
                              "new_parameters_used_for_generation": True})
            if rank == 0: write_json(output/"DONE.json", {"passed": True, "weights_discarded": True, "ranks": reports})
        elif a.stage == "warmup":
            last = resume
            for specification in rounds[counters["round"]:]:
                number = specification["round"]; version = f"warmup_before_round_{number:02d}"
                expected = None
                if focus:
                    expected = source["tensor_sha256"] if number == 1 else read_json(
                        output/"metrics"/f"round_{number-1:02d}.json")["adapter_after"]
                before = generator.bind(version, expected)
                if len(set(gather(before))) != 1: raise RuntimeError("Policy replicas differ before sampling")
                batch = flatten_round(specification); batch_dir = output/"rollouts"/f"round_{number:02d}"
                local_results = []
                for index in range(rank, len(batch), world):
                    item = batch[index]; directory = batch_dir/item["group"]/f"try_{item['sample']:02d}"
                    if (directory/"result.json").exists() and read_json(directory/"result.json").get("error"):
                        directory.rename(directory.with_name(directory.name+".incomplete-"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
                    if (directory/"result.json").exists():
                        result = read_json(directory/"result.json")
                        if result["policy_version"] != version or result["generator_provenance"]["adapter_sha256"] != before or result["scene"] != scenes[item["scene_id"]]:
                            raise ValueError("Prior rollout does not belong to this policy/scene batch")
                    else:
                        if directory.exists(): directory.rename(directory.with_name(directory.name+".incomplete-"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
                        result = episode(scenes[item["scene_id"]], c, directory, generator, segmenter,
                            item["policy_seed"], version, origin="warmup")
                    if result["error"]: raise RuntimeError(result["error"])
                    actions = []
                    final_row = None
                    for row in read_rows(directory):
                        actions.append(row["action"])
                        final_row = row
                    result["action_signature"] = digest(actions)
                    if focus:
                        if (result["task"] != "T1" or result["split"] != "train" or result["policy"] != "model"
                                or result["policy_seed"] != item["policy_seed"] or result["origin"] != "warmup"
                                or result["demonstration"] is not None or result["search_prefix_length"] != 0
                                or result["replayed_actions"] != 0 or result["config_sha256"] != digest(c)
                                or result["generated_actions"] != len(actions) or len(actions) != result["steps"]
                                or final_row is None):
                            raise ValueError("Invalid T1 warmup trajectory provenance or step count")
                        result["failed_final_done_step"] = final_row["step"] if failed_final_done(result, final_row) else None
                    local_results.append({"index": index, "group": item["group"], "directory": str(directory), "result": result})
                results = sorted([r for group in gather(local_results) for r in group], key=lambda r: r["index"])
                if len(results) != len(batch): raise RuntimeError("Incomplete sampling batch; no update allowed")
                advantages = {}; metrics = []
                for group in specification["groups"]:
                    members = [r for r in results if r["group"] == group["id"]]
                    values, report = group_advantages([r["result"] for r in members])
                    advantages.update({r["index"]: v for r, v in zip(members, values)})
                    if focus:
                        audit = []
                        for member, value in zip(members, values):
                            result = member["result"]
                            step = result["failed_final_done_step"]
                            if step is not None:
                                audit.append({"policy_seed": result["policy_seed"], "step": step,
                                    "group_advantage": value, "response_advantage": response_advantage(value, step, step)})
                        report["failed_done_credit"] = {"failed_accepted_done": len(audit),
                            "positive_before": sum(v["group_advantage"] > 0 for v in audit),
                            "positive_after": sum(v["response_advantage"] > 0 for v in audit), "responses": audit}
                    metrics.append({"group": group, **report})
                # All old likelihoods are computed before the first minibatch update.
                model.eval()
                for item in results[rank::world]:
                    with torch.no_grad():
                        old = [token_logps(model, *training_inputs(processor, row, c)).cpu()
                               for row in read_rows(item["directory"])]
                    torch.save(old, Path(item["directory"])/"old_logps.pt")
                barrier()
                order = list(range(len(results))); random.Random(seed+number).shuffle(order)
                updates_before = counters["optimizer_steps"]
                for start in range(0, len(order), c["training"]["warmup_minibatch"]):
                    indices = order[start:start+c["training"]["warmup_minibatch"]]
                    train_mode(); optimizer.zero_grad(set_to_none=True)
                    for index in indices[rank::world]:
                        advantage = advantages[index]
                        if abs(advantage) < 1e-12: continue
                        item = results[index]
                        old = torch.load(Path(item["directory"])/"old_logps.pt", weights_only=True)
                        for turn, row in enumerate(read_rows(item["directory"])):
                            weight = response_advantage(advantage, row["step"], item["result"].get("failed_final_done_step")) if focus else advantage
                            if weight == 0: continue
                            current = token_logps(model, *training_inputs(processor, row, c))
                            loss = dr_grpo_loss(current, old[turn].to(device), weight,
                                c["policy"]["max_steps"]*c["policy"]["max_new_tokens"], c["training"]["clip"])/len(indices)
                            loss.backward()
                    optimize()
                after = generator.bind(f"warmup_after_round_{number:02d}", adapter_digest(model))
                if len(set(gather(after))) != 1: raise RuntimeError("Replicas differ after update")
                counters["round"] = number
                if rank == 0:
                    report = {"round": number, "phase": specification["phase"], "episodes": len(batch),
                        "groups": metrics, "optimizer_steps_this_round": counters["optimizer_steps"]-updates_before,
                        "optimizer_steps_total": counters["optimizer_steps"],
                        "zero_advantage_groups": sum(m["zero_advantage"] for m in metrics),
                        "zero_advantage_group_fraction": sum(m["zero_advantage"] for m in metrics)/len(metrics),
                        "adapter_before": before, "adapter_after": after,
                        "results": [{"directory": r["directory"], "result": r["result"]} for r in results]}
                    if focus:
                        report["failed_done_credit"] = {k: sum(m["failed_done_credit"][k] for m in metrics)
                            for k in ("failed_accepted_done", "positive_before", "positive_after")}
                    write_json(output/"metrics"/f"round_{number:02d}.json", report)
                    print(json.dumps({"event": "warmup_round_complete", "round": number,
                        "optimizer_steps": counters["optimizer_steps"], "episodes": len(batch),
                        "zero_advantage_groups": report["zero_advantage_groups"]}), flush=True)
                last = commit(number)
            freeze_node("after_warmup", last)
            if rank == 0:
                write_json(output/"DONE.json", {"rounds": len(rounds),
                    "episodes": sum(len(flatten_round(r)) for r in rounds), **counters})
        else:
            from .data import read_sample
            verified_files = set()
            data_manifest = read_json(root/"data/manifest.json")
            if not data_manifest["samples"]: raise ValueError("No successful training data; SFT unavailable")
            if file_digest(root/"data/samples.jsonl") != data_manifest["samples_sha256"]:
                raise ValueError("Frozen SFT sample manifest was modified")
            samples = [json.loads(line) for line in (root/"data/samples.jsonl").read_text().splitlines()]
            if any(s["scene_split"] != "train" for s in samples): raise ValueError("Evaluation data in SFT")
            size = c["training"]["sft_batch"]; batches = math.ceil(len(samples)/size)
            last = resume
            for epoch in range(2):
                order = list(range(len(samples))); random.Random(seed+epoch).shuffle(order)
                for batch in range(batches):
                    step = epoch*batches+batch+1
                    if step <= counters["sft_batches"]: continue
                    indices = order[batch*size:(batch+1)*size]
                    train_mode(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.
                    for index in indices[rank::world]:
                        sample = samples[index]
                        row = read_sample(sample, verified_files)
                        loss = -token_logps(model, *training_inputs(processor, row, c)).mean()/len(indices)
                        loss.backward(); loss_sum += float(loss.detach())
                    optimize(); counters["sft_batches"] = step
                    losses = gather(loss_sum)
                    # Epoch boundaries and every fourth minibatch are resumable.
                    if step % 4 == 0 or batch == batches-1:
                        last = commit(step, {"epoch_completed": epoch+1 if batch == batches-1 else epoch,
                                             "data_manifest_sha256": file_digest(root/"data/manifest.json")})
                    if rank == 0:
                        write_json(output/"metrics"/f"batch_{step:05d}.json", {"epoch": epoch+1,
                            "batch": batch+1, "optimizer_steps": counters["optimizer_steps"], "loss": sum(losses),
                            "samples": len(indices), "sample_indices": indices})
                        print(json.dumps({"event": "sft_batch_complete", "epoch": epoch+1, "batch": batch+1,
                                          "total_batches_per_epoch": batches}), flush=True)
            freeze_node("after_sft", last)
            if rank == 0: write_json(output/"DONE.json", {"epochs": 2, "samples": len(samples), **counters})
    finally:
        if segmenter: segmenter.close()
        if distributed: dist.destroy_process_group()


if __name__ == "__main__": main()
