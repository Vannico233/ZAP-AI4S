"""Edit the curriculum, GRPO loss, search, SFT data selection and SFT loop here.

Use `python -m tube_agent.posttrain plan` to preview a run and
`python -m tube_agent.posttrain inspect --metrics FILE` to debug a saved GRPO group.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import random
import sys

from .curriculum.config import TASKS, digest, file_digest, load, validate_adapter_checkpoint
from .lab.credit import failed_done
from .lab.experiment import read_json, write_json


# Edit a value here to override the input JSON for a new experiment.
# None inherits the value from --config. Explicit CLI values take precedence.
TRAINING_OVERRIDES = {
    "warmup_lr": None,  # Default 1e-5.
    "sft_lr": None,  # Default 1e-4.
    "clip": None,  # Default 0.2.
    "max_grad_norm": None,  # Default 1.0.
    "weight_decay": None,  # Default 0.01.
    "warmup_minibatch": None,  # Default 32 episodes.
    "sft_batch": None,  # Default 16 assistant responses.
    "sft_epochs": None,  # Default 2 complete passes.
}


def add_training_arguments(parser):
    for name in TRAINING_OVERRIDES:
        kind = int if name in ("warmup_minibatch", "sft_batch", "sft_epochs") else float
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=kind,
            help=f"Override training.{name} before freezing a new run.",
        )


def configure_training(config, arguments):
    from .curriculum.config import validate_training
    settings = config["training"]
    for name, default in TRAINING_OVERRIDES.items():
        value = getattr(arguments, name)
        if value is None:
            value = default
        if value is not None:
            settings[name] = value
    validate_training(settings)


def run_posttraining(runner):
    """Execute the registered stages after the physics and vision preflights."""
    from .curriculum.pipeline import distributed_command, report

    out, a = runner.output, runner.a
    runner.train_stage("smoke")
    runner.evaluate("original")
    runner.train_stage("warmup")
    runner.step(
        "search", distributed_command("tube_agent.curriculum.search", "--run", out, "--model", a.model)
    )
    runner.step(
        "freeze_data",
        [sys.executable, "-m", "tube_agent.curriculum.data", "--run", str(out)],
        [a.policy_gpus[0]],
    )
    available = read_json(out / "data/manifest.json")["samples"] > 0
    if available:
        runner.train_stage("sft")
    else:
        write_json(
            out / "checkpoints/after_sft_UNAVAILABLE.json",
            {"reason": "No complete successful trajectories; SFT cannot run on an empty dataset"},
        )
    runner.evaluate("after_warmup")
    if available:
        runner.evaluate("after_sft")
    result = report(out, available)
    runner.status(result["status"], report=str(out / "report.json"), main_rl_executed=False)


def flatten_round(round_spec):
    return [
        {"group": group["id"], "scene_id": group["scene_id"], **sample}
        for group in round_spec["groups"]
        for sample in group["rollouts"]
    ]


def log_event(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def group_advantages(results):
    """Only center within a task+scene group; no noise or per-turn overrides."""
    import numpy as np

    if len(results) != 8 or len({(r["scene"]["task"], r["scene"]["sha256"]) for r in results}) != 1:
        raise ValueError("A GRPO group must contain 8 trajectories from the identical task/scene")
    if len({r["policy_version"] for r in results}) != 1 or any(r.get("error") for r in results):
        raise ValueError("Mixed policy versions or infrastructure errors in GRPO group")
    rewards = np.asarray([r["reward"] for r in results], dtype=float)
    if not np.isfinite(rewards).all():
        raise ValueError("Nonfinite rewards")
    values = rewards - rewards.mean()
    same = bool(np.all(rewards == rewards[0]))
    return values.tolist(), {
        "rewards": rewards.tolist(),
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "all_rewards_equal": same,
        "zero_advantage": bool(np.all(np.abs(values) < 1e-12)),
        "behavior_signatures": [r.get("action_signature") for r in results],
        "reward_terms": [r["reward_terms"] for r in results],
        "zero_group_diagnosis": (
            "identical_action_sequences"
            if len({r.get("action_signature") for r in results}) == 1
            else "different_actions_same_reward_inspect_terms"
        )
        if same
        else "nonzero",
    }


def dr_grpo_loss(new_logp, old_logp, advantage, token_budget, clip=0.2):
    """Sum the clipped token objective using the fixed per-episode token budget."""
    import torch

    if token_budget < 1:
        raise ValueError("Positive token budget required")
    ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = ratio.clamp(1 - clip, 1 + clip)
    return -torch.minimum(ratio * advantage, clipped_ratio * advantage).sum() / token_budget


def failed_final_done(result, final_row):
    # The 40th step may be labeled "max_steps" even when its action was done.
    return (result["error"] is None and result["success"] is False
            and final_row["step"] == result["steps"] and failed_done(final_row))


def response_advantage(group_advantage, step, adjusted_step):
    return min(group_advantage, 0.0) if step == adjusted_step else group_advantage


def score_rollout(directory, result, config, expected_seed, focus):
    """Record the action signature and identify an eligible terminal response."""
    from .curriculum.rollout import read_rows
    actions = []
    last = None
    for row in read_rows(directory):
        actions.append(row["action"])
        last = row
    result["action_signature"] = digest(actions)
    if not focus:
        return
    if (result["task"] != "T1" or result["split"] != "train" or result["policy"] != "model"
            or result["policy_seed"] != expected_seed or result["origin"] != "warmup"
            or result["demonstration"] is not None or result["search_prefix_length"] != 0
            or result["replayed_actions"] != 0 or result["config_sha256"] != digest(config)
            or result["generated_actions"] != len(actions) or len(actions) != result["steps"] or last is None):
        raise ValueError("Invalid T1 warmup trajectory provenance or step count")
    result["failed_final_done_step"] = last["step"] if failed_final_done(result, last) else None


def terminal_credit_report(members, advantages):
    responses = []
    for member, advantage in zip(members, advantages):
        result = member["result"]
        step = result["failed_final_done_step"]
        if step is not None:
            responses.append({"policy_seed": result["policy_seed"], "step": step,
                              "group_advantage": advantage, "response_advantage": response_advantage(advantage, step, step)})
    return {"failed_accepted_done": len(responses),
            "positive_before": sum(r["group_advantage"] > 0 for r in responses),
            "positive_after": sum(r["response_advantage"] > 0 for r in responses), "responses": responses}


def create_optimizer(parameters, settings, stage):
    import torch
    return torch.optim.AdamW(parameters, lr=settings["sft_lr" if stage == "sft" else "warmup_lr"],
                             weight_decay=settings["weight_decay"])


def start_minibatch(session):
    import torch
    session.model.train()
    for module in session.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    session.optimizer.zero_grad(set_to_none=True)


def step_optimizer(session):
    import torch
    import torch.distributed as dist
    from .lab.distributed_train import synchronize_gradients
    updated = synchronize_gradients(session.parameters, dist if session.world > 1 else None)
    if updated:
        torch.nn.utils.clip_grad_norm_(session.parameters, session.config["training"]["max_grad_norm"],
                                       error_if_nonfinite=True)
        session.optimizer.step()
        session.counters["optimizer_steps"] += 1
    session.optimizer.zero_grad(set_to_none=True)
    return updated


def read_completed_rollout(directory):
    """Keep valid results on resume and archive interrupted attempts before retry."""
    if not directory.exists():
        return None
    path = directory / "result.json"
    if path.exists():
        result = read_json(path)
        if not result.get("error"):
            return result
    stamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
    directory.rename(directory.with_name(f"{directory.name}.incomplete-{stamp}"))
    return None


def run_grpo(session, scenes, rounds):
    """Collect the whole round, cache old token likelihoods, then update minibatches."""
    import torch
    from .curriculum.rollout import read_rows

    config = session.config
    config_sha = digest(config)
    rank, world = session.rank, session.world
    last = session.resume
    for specification in rounds[session.counters["round"] :]:
        number = specification["round"]
        version = f"warmup_before_round_{number:02d}"
        expected = None
        if session.source:
            expected = session.source["tensor_sha256"] if number == 1 else read_json(
                session.output / "metrics" / f"round_{number-1:02d}.json")["adapter_after"]
        before = session.generator.bind(version, expected)
        if len(set(session.gather(before))) != 1:
            raise RuntimeError("Policy replicas differ before sampling")
        batch = flatten_round(specification)
        batch_dir = session.output / "rollouts" / f"round_{number:02d}"
        local_results = []
        for index in range(rank, len(batch), world):
            item = batch[index]
            directory = batch_dir / item["group"] / f"try_{item['sample']:02d}"
            result = read_completed_rollout(directory)
            if result is not None:
                if (
                    result["policy_version"] != version
                    or result["generator_provenance"]["adapter_sha256"] != before
                    or result["scene"] != scenes[item["scene_id"]]
                    or result["config_sha256"] != config_sha
                    or result["policy_seed"] != item["policy_seed"]
                ):
                    raise ValueError("Prior rollout has a different policy, environment, scene or seed")
            else:
                result = session.sample(scenes[item["scene_id"]], directory, item["policy_seed"], version)
            if result["error"]:
                raise RuntimeError(result["error"])
            score_rollout(directory, result, config, item["policy_seed"], session.source is not None)
            local_results.append(
                {"index": index, "group": item["group"], "directory": str(directory), "result": result}
            )
        results = sorted(
            [r for group in session.gather(local_results) for r in group], key=lambda r: r["index"]
        )
        if len(results) != len(batch):
            raise RuntimeError("Incomplete sampling batch; no update allowed")
        advantages = {}
        metrics = []
        for group in specification["groups"]:
            members = [r for r in results if r["group"] == group["id"]]
            values, report = group_advantages([r["result"] for r in members])
            advantages.update({r["index"]: v for r, v in zip(members, values)})
            if session.source:
                report["failed_done_credit"] = terminal_credit_report(members, values)
            metrics.append({"group": group, **report})
        # All old likelihoods are computed before the first minibatch update.
        session.model.eval()
        for item in results[rank::world]:
            with torch.no_grad():
                old = [session.logps(row).cpu() for row in read_rows(item["directory"])]
            torch.save(old, Path(item["directory"]) / "old_logps.pt")
        session.barrier()
        order = list(range(len(results)))
        random.Random(session.seed + number).shuffle(order)
        updates_before = session.counters["optimizer_steps"]
        for start in range(0, len(order), config["training"]["warmup_minibatch"]):
            indices = order[start : start + config["training"]["warmup_minibatch"]]
            start_minibatch(session)
            for index in indices[rank::world]:
                advantage = advantages[index]
                if abs(advantage) < 1e-12:
                    continue
                item = results[index]
                old = torch.load(Path(item["directory"]) / "old_logps.pt", weights_only=True)
                for turn, row in enumerate(read_rows(item["directory"])):
                    weight = advantage
                    if session.source:
                        weight = response_advantage(advantage, row["step"], item["result"]["failed_final_done_step"])
                        if weight == 0:
                            continue
                    current = session.logps(row)
                    loss = dr_grpo_loss(
                        current,
                        old[turn].to(session.device),
                        weight,
                        config["policy"]["max_steps"] * config["policy"]["max_new_tokens"],
                        config["training"]["clip"],
                    ) / len(indices)
                    loss.backward()
            step_optimizer(session)
        after = session.generator.bind(f"warmup_after_round_{number:02d}", session.adapter_digest())
        if len(set(session.gather(after))) != 1:
            raise RuntimeError("Replicas differ after update")
        session.counters["round"] = number
        if rank == 0:
            report = {
                "round": number,
                "phase": specification["phase"],
                "episodes": len(batch),
                "groups": metrics,
                "optimizer_steps_this_round": session.counters["optimizer_steps"] - updates_before,
                "optimizer_steps_total": session.counters["optimizer_steps"],
                "zero_advantage_groups": sum(m["zero_advantage"] for m in metrics),
                "zero_advantage_group_fraction": sum(m["zero_advantage"] for m in metrics) / len(metrics),
                "adapter_before": before,
                "adapter_after": after,
                "results": [{"directory": r["directory"], "result": r["result"]} for r in results],
            }
            if session.source:
                report["failed_done_credit"] = {k: sum(m["failed_done_credit"][k] for m in metrics)
                    for k in ("failed_accepted_done", "positive_before", "positive_after")}
            write_json(session.output / "metrics" / f"round_{number:02d}.json", report)
            log_event("warmup_round_complete", round=number, optimizer_steps=session.counters["optimizer_steps"],
                      episodes=len(batch), zero_advantage_groups=report["zero_advantage_groups"])
        last = session.commit(number)
    session.freeze_node("after_warmup", last)
    if rank == 0:
        write_json(session.output / "DONE.json", {"rounds": len(rounds),
                   "episodes": sum(len(flatten_round(r)) for r in rounds), **session.counters})


def run_sft(session):
    """Train on unique replay-verified assistant responses from the frozen manifest."""
    from .curriculum.data import read_sample

    rank, world = session.rank, session.world

    verified_files = set()
    data_manifest = read_json(session.root / "data/manifest.json")
    if not data_manifest["samples"]:
        raise ValueError("No successful training data; SFT unavailable")
    if file_digest(session.root / "data/samples.jsonl") != data_manifest["samples_sha256"]:
        raise ValueError("Frozen SFT sample manifest was modified")
    samples = [json.loads(line) for line in (session.root / "data/samples.jsonl").read_text().splitlines()]
    if any(s["scene_split"] != "train" for s in samples):
        raise ValueError("Evaluation data in SFT")
    epochs = session.config["training"]["sft_epochs"]
    size = session.config["training"]["sft_batch"]
    batches = math.ceil(len(samples) / size)
    last = session.resume
    for epoch in range(epochs):
        order = list(range(len(samples)))
        random.Random(session.seed + epoch).shuffle(order)
        for batch in range(batches):
            step = epoch * batches + batch + 1
            if step <= session.counters["sft_batches"]:
                continue
            indices = order[batch * size : (batch + 1) * size]
            start_minibatch(session)
            loss_sum = 0.0
            for index in indices[rank::world]:
                sample = samples[index]
                row = read_sample(sample, verified_files)
                loss = -session.logps(row).mean() / len(indices)
                loss.backward()
                loss_sum += float(loss.detach())
            step_optimizer(session)
            session.counters["sft_batches"] = step
            losses = session.gather(loss_sum)
            # Epoch boundaries and every fourth minibatch are resumable.
            if step % 4 == 0 or batch == batches - 1:
                last = session.commit(
                    step,
                    {
                        "epoch_completed": epoch + 1 if batch == batches - 1 else epoch,
                        "data_manifest_sha256": file_digest(session.root / "data/manifest.json"),
                    },
                )
            if rank == 0:
                write_json(
                    session.output / "metrics" / f"batch_{step:05d}.json",
                    {
                        "epoch": epoch + 1,
                        "batch": batch + 1,
                        "optimizer_steps": session.counters["optimizer_steps"],
                        "loss": sum(losses),
                        "samples": len(indices),
                        "sample_indices": indices,
                    },
                )
                log_event("sft_batch_complete", epoch=epoch + 1, batch=batch + 1, total_batches_per_epoch=batches)
    session.freeze_node("after_sft", last)
    if rank == 0:
        write_json(
            session.output / "DONE.json", {"epochs": epochs, "samples": len(samples), **session.counters}
        )


def node_key(scene, rows):
    return digest([scene["sha256"], [r["action"] for r in rows]])


class Forest:
    def __init__(self, config):
        self.config = config
        self.nodes = {}
        self.task_expansions = Counter()
        self.solved = set()
        self.expansions = []
        self.blocked = set()

    def add(self, directory, limit=None):
        from .curriculum.rollout import read_rows

        result = read_json(Path(directory) / "result.json")
        if result.get("error") or result["scene"]["split"] != "train" or result.get("policy") != "model":
            return
        rows = list(read_rows(directory))
        prefix = []
        parent = None
        for row in rows[:limit]:
            prefix.append(row)
            if not row["feedback"].get("accepted"):
                break
            if not row.get("node_eligible", False):
                continue
            if row["action"]["action"] not in ("move_to", "set_gripper"):
                continue
            if row["action"]["action"] == "move_to" and not row["feedback"].get("reached"):
                continue
            judge = row["judge_after"]
            if judge["success"] or judge["severe_collision"] or judge["dropped"]:
                break
            if len(prefix) >= self.config["policy"]["max_steps"]:
                break
            if judge["reward_terms"]["terminal_progress"] <= 0.001:
                continue
            key = node_key(result["scene"], prefix)
            if key in self.blocked:
                break
            self.nodes.setdefault(
                key,
                {
                    "key": key,
                    "parent": parent,
                    "task": result["task"],
                    "scene": result["scene"],
                    "directory": str(Path(directory).resolve()),
                    "length": len(prefix),
                    "progress": judge["reward_terms"]["terminal_progress"],
                    "visits": 0,
                    "wins": 0,
                    "failures": {},
                },
            )
            parent = key
        for task in TASKS:
            active = [n for n in self.nodes.values() if n["task"] == task]
            cap = self.config["search"]["max_nodes_per_task"]
            for node in sorted(active, key=lambda n: (n["progress"], -n["length"]))[
                : max(0, len(active) - cap)
            ]:
                self.nodes.pop(node["key"], None)

    def select(self):
        active = [
            n for n in self.nodes.values() if n["task"] not in self.solved and n["key"] not in self.blocked
        ]
        if not active:
            return None
        tasks = {n["task"] for n in active}
        task = min(tasks, key=lambda t: (self.task_expansions[t], t))

        def score(n):
            return (
                n["wins"] / (1 + n["visits"])
                + n["progress"]
                + self.config["search"]["exploration"]
                * math.sqrt(math.log(2 + sum(self.task_expansions.values())) / (1 + n["visits"]))
            )

        return dict(max((n for n in active if n["task"] == task), key=score))

    def update(self, node, results):
        self.task_expansions[node["task"]] += 1
        stored = self.nodes[node["key"]]
        for directory, result in results:
            stored["visits"] += 1
            stored["wins"] += int(result["success"])
            if result["success"]:
                self.solved.add(node["task"])
            else:
                reason = result["failure_type"]
                stored["failures"][reason] = stored["failures"].get(reason, 0) + 1
            self.add(directory)
        prune = any(n >= self.config["search"]["same_failure_prune"] for n in stored["failures"].values())
        if prune:
            self.blocked.add(node["key"])
            changed = True
            while changed:
                changed = False
                for key, child in self.nodes.items():
                    if child["parent"] in self.blocked and key not in self.blocked:
                        self.blocked.add(key)
                        changed = True
        self.expansions.append(
            {
                "selected_node": node,
                "attempts": len(results),
                "pruned": prune,
                "results": [
                    {
                        "directory": str(d),
                        "success": r["success"],
                        "failure_type": r["failure_type"],
                        "replayed_actions": r["replayed_actions"],
                        "generated_actions": r["generated_actions"],
                    }
                    for d, r in results
                ],
            }
        )

    def state(self):
        return {
            "nodes": self.nodes,
            "task_expansions": dict(self.task_expansions),
            "solved": sorted(self.solved),
            "blocked": sorted(self.blocked),
            "expansions": self.expansions,
        }

    def restore(self, state):
        self.nodes = state["nodes"]
        self.task_expansions = Counter(state["task_expansions"])
        self.solved = set(state["solved"])
        self.blocked = set(state["blocked"])
        self.expansions = state["expansions"]


def run_search(root, model_path):
    """Expand eligible prefixes with the frozen warm-up policy on all ranks."""
    import torch
    import torch.distributed as dist
    from .curriculum.data import warmup_episodes
    from .curriculum.learning import Generator, load_model
    from .curriculum.rollout import episode, read_rows
    from .lab.vision import SamClient

    c = load(root / "config.json")
    config_sha = digest(c)
    output = root / "search"
    output.mkdir(exist_ok=True)
    rank, world, local = (
        int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24))

    def broadcast(value):
        values = [value]
        if world > 1:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    forest = Forest(c)
    candidates = []
    if rank == 0:
        successes = Counter()
        paths = list(warmup_episodes(root))
        for path in paths:
            result = read_json(path)
            successes[result["task"]] += int(result["success"])
        for path in paths:
            if not successes[read_json(path)["task"]]:
                forest.add(path.parent)
        candidates = sorted({n["task"] for n in forest.nodes.values()})
        if (output / "state.json").exists():
            forest.restore(read_json(output / "state.json"))
        write_json(
            output / "plan.json",
            {
                "candidate_tasks": candidates,
                "warmup_successes": dict(successes),
                "budget_global": 128,
                "continuations_per_node": 4,
                "model": "after_warmup",
                "selection": "round-robin least expanded task; wins/(visits+1) + terminal_progress + 0.45*sqrt(log(2+total_expansions)/(visits+1))",
                "pruning": "3 continuations with the same failure type prune node and descendants",
                "root_replay_actions_count_separately": True,
                "weight_updates": 0,
            },
        )
    candidates = broadcast(candidates)
    if not candidates:
        if rank == 0:
            write_json(
                output / "DONE.json", {"continuations": 0, "reason": "no_zero_success_task_with_valid_prefix"}
            )
        if world > 1:
            dist.destroy_process_group()
        return
    adapter = root / "checkpoints/after_warmup"
    validate_adapter_checkpoint(adapter, config_sha)
    model, processor = load_model(model_path, c, adapter, f"cuda:{local}", False)
    generator = Generator(model, processor, c)
    frozen = generator.bind("after_warmup")
    segmenter = SamClient(c["execution"]["sam_url"])
    try:
        segmenter.health()
        while True:
            selection = None
            if rank == 0 and len(forest.expansions) < 32:
                node = forest.select()
                if node:
                    selection = {"node": node, "expansion": len(forest.expansions)}
            selection = broadcast(selection)
            if selection is None:
                break
            node, number = selection["node"], selection["expansion"]
            prefix = list(read_rows(node["directory"]))[: node["length"]]
            results = []
            for sample in range(rank, 4, world):
                directory = output / "rollouts" / f"expansion_{number:03d}" / f"try_{sample:02d}"
                seed = 20000000 + number * 4 + sample
                result = read_completed_rollout(directory)
                if result is not None:
                    if (
                        result["scene"] != node["scene"]
                        or result["generator_provenance"]["adapter_sha256"] != frozen
                        or result["search_prefix_sha256"] != digest([r["action"] for r in prefix])
                        or result["config_sha256"] != config_sha or result["policy_seed"] != seed
                    ):
                        raise ValueError("Search continuation belongs to another node or policy")
                else:
                    generator.bind("after_warmup", frozen)
                    result = episode(
                        node["scene"],
                        c,
                        directory,
                        generator,
                        segmenter,
                        seed,
                        "after_warmup",
                        prefix=prefix,
                        origin="search",
                    )
                if result["error"]:
                    raise RuntimeError(result["error"])
                results.append((str(directory), result))
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, results)
            else:
                gathered = [results]
            if rank == 0:
                complete = sorted([item for batch in gathered for item in batch])
                if len(complete) != 4:
                    raise RuntimeError("Incomplete four-continuation expansion")
                forest.update(node, complete)
                write_json(output / "state.json", forest.state())
                log_event("search_expansion_complete", expansions=number + 1, continuations=(number + 1) * 4,
                          task=node["task"], solved=sorted(forest.solved))
        if rank == 0:
            write_json(
                output / "DONE.json",
                {
                    "continuations": len(forest.expansions) * 4,
                    "solved_tasks": sorted(forest.solved),
                    "candidate_tasks": candidates,
                    "replayed_actions": sum(
                        r["replayed_actions"] for e in forest.expansions for r in e["results"]
                    ),
                    "new_generated_actions": sum(
                        r["generated_actions"] for e in forest.expansions for r in e["results"]
                    ),
                    "weight_updates": 0,
                },
            )
    finally:
        segmenter.close()
        if world > 1:
            dist.destroy_process_group()


def freeze(root):
    """Replay successful training trajectories and freeze a deduplicated SFT index."""
    from .curriculum.data import read_rows, replay, source_episodes

    root = Path(root)
    c = load(root / "config.json")
    target = root / "data"
    target.mkdir(exist_ok=True)
    if (target / "manifest.json").exists():
        manifest = read_json(target / "manifest.json")
        if file_digest(target / "samples.jsonl") != manifest["samples_sha256"]:
            raise ValueError("Frozen SFT samples changed")
        return manifest
    train = {s["id"]: s for s in read_json(root / "scenes.json")["scenes"] if s["split"] == "train"}
    accepted = []
    samples = []
    seen_traces = set()
    seen_turns = set()
    coverage = {t: set() for t in TASKS}
    counts = Counter()
    duplicates = 0
    for path in source_episodes(root):
        result = read_json(path)
        if not result.get("success") or result.get("error"):
            continue
        if (
            result.get("policy") != "model"
            or result.get("demonstration") is not None
            or result["scene"] != train.get(result["scene"]["id"])
            or result["config_sha256"] != digest(c)
        ):
            raise ValueError("A success candidate has incompatible scene/policy/config provenance")
        rows = list(read_rows(path.parent))
        if not rows or not rows[-1]["judge_after"]["success"]:
            raise ValueError("Success result disagrees with terminal object-state judge")
        if any("response_ids" not in row or "model_input_sha256" not in row for row in rows):
            raise ValueError("Success lacks exact model inputs/generated token IDs")
        signature = digest(
            [
                result["scene"]["sha256"],
                [(row["model_input_sha256"], row["response_ids"], row["action"]) for row in rows],
            ]
        )
        if signature in seen_traces:
            duplicates += 1
            continue
        verified = replay(result["scene"], c, rows)
        if not verified["success"]:
            raise ValueError("A successful candidate failed physical replay")
        seen_traces.add(signature)
        trajectory = path.parent / "trajectory.jsonl"
        checksum = file_digest(trajectory)
        offsets = []
        with trajectory.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        origin = result["origin"]
        accepted.append(
            {
                "trajectory": str(trajectory.resolve()),
                "trajectory_sha256": checksum,
                "result_sha256": file_digest(path),
                "task": result["task"],
                "scene_id": result["scene"]["id"],
                "origin": origin,
                "trajectory_signature": signature,
                "replay_verified": True,
                "invalid_actions": result["invalid_actions"],
            }
        )
        counts[result["task"]] += 1
        coverage[result["task"]].add(result["scene"]["id"])
        for index, row in enumerate(rows):
            key = digest([row["model_input_sha256"], row["response_ids"]])
            if key in seen_turns:
                continue
            seen_turns.add(key)
            samples.append(
                {
                    "trajectory": str(trajectory.resolve()),
                    "trajectory_sha256": checksum,
                    "row": index,
                    "byte_offset": offsets[index],
                    "scene_split": "train",
                    "task": result["task"],
                    "origin": origin,
                    "input_response_sha256": key,
                }
            )
    data_path = target / "samples.jsonl"
    temporary = target / "samples.partial.jsonl"
    temporary.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    temporary.replace(data_path)
    manifest = {
        "frozen": True,
        "config_sha256": digest(c),
        "samples": len(samples),
        "samples_sha256": file_digest(data_path),
        "trajectories": accepted,
        "successes_by_task": {t: counts[t] for t in TASKS},
        "scene_coverage": {t: sorted(coverage[t]) for t in TASKS},
        "missing_tasks": [t for t in TASKS if not counts[t]],
        "duplicates_removed": duplicates,
        "shared_prefix_policy": "identical input/response assistant turns appear once",
        "includes_tree_search": any(v["origin"] == "search" for v in accepted),
        "evaluation_data_used": False,
    }
    write_json(target / "manifest.json", manifest)
    return manifest


def inspect_group(path, index=0):
    """Recompute one saved rollout group's advantage using the training implementation."""
    metrics = read_json(path)
    groups = metrics["groups"]
    if not 0 <= index < len(groups):
        raise ValueError(f"Expected a group index between 0 and {len(groups)-1}")
    group = groups[index]["group"]
    by_seed = {item["result"]["policy_seed"]: item["result"] for item in metrics["results"]}
    results = [by_seed[sample["policy_seed"]] for sample in group["rollouts"]]
    advantages, report = group_advantages(results)
    print(f"Group {group['id']} | Scene {group['scene_id']} | Mean reward {report['mean']:.6f}")
    print("Policy seed      Reward   Group advantage   Steps   Invalid   Success")
    for result, advantage in zip(results, advantages):
        print(f"{result['policy_seed']:>11} {result['reward']:>11.6f} {advantage:>17.6f} "
              f"{result['steps']:>7} {result['invalid_actions']:>9}   {result['success']}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    inspect = commands.add_parser("inspect", help="Inspect the advantages from a saved GRPO round on CPU")
    inspect.add_argument("--metrics", required=True, type=Path)
    inspect.add_argument("--group", type=int, default=0, help="Zero-based group within the round")
    commands.add_parser("plan", help="Preview a configured run")
    commands.add_parser("run", help="Start or resume the production pipeline")
    if not argv:
        parser.print_help()
        return
    if argv[0] in ("plan", "--plan", "run"):
        from .curriculum.pipeline import main as launch_experiment
        launch_experiment((["--plan"] if argv[0] in ("plan", "--plan") else []) + argv[1:])
        return
    args = parser.parse_args(argv)
    if args.command == "inspect":
        inspect_group(args.metrics, args.group)


if __name__ == "__main__":
    main()
