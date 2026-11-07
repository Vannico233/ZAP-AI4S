"""Bounded warm-up -> self-success SFT -> GRPO recipe for the four lab tasks.

This is the 4B/Franka/MuJoCo adaptation of ZAP, not its published benchmark.
Warm-up needs no prior successes. Only the later SFT stage has a data gate.
"""
from pathlib import Path

from .experiment import TRAIN_TASK_NAMES, TASK_NAMES, read_json, write_json

VERSION = "zap_warmup_recipe_v1"


def add_arguments(parser):
    parser.add_argument("--zap-recipe", action="store_true", help="Warm-up RL before self-success SFT")
    parser.add_argument("--warmup-updates", nargs=4, type=int, default=[24, 20, 12, 12],
                        metavar=("PICK", "TRANSPORT", "INSERT", "UNSCREW"))
    parser.add_argument("--success-targets", nargs=4, type=int, default=[20, 20, 10, 10],
                        metavar=("PICK", "TRANSPORT", "INSERT", "UNSCREW"))
    parser.add_argument("--sft-min-successes", type=int, default=3,
                        help="Minimum replay-verified distinct successes in EACH task; targets are preferred")
    parser.add_argument("--recipe-fixed-updates", type=int, default=48)
    parser.add_argument("--recipe-diverse-updates", type=int, default=48)


def validate(a, parser):
    if not a.zap_recipe:
        return
    if a.pick_probe or a.pose_probe or a.success_source:
        parser.error("--zap-recipe cannot be combined with probe/import modes; use a new output")
    if a.action_protocol != "zap_pose_v1":
        parser.error("--zap-recipe requires --action-protocol zap_pose_v1")
    if min(*a.warmup_updates, *a.success_targets, a.sft_min_successes,
           a.recipe_fixed_updates, a.recipe_diverse_updates) < 1:
        parser.error("Recipe budgets must be positive")
    if min(a.success_targets) < a.sft_min_successes:
        parser.error("Every success target must be >= sft-min-successes")


def warmup_schedule(counts):
    """Spread each task's quota throughout one shared policy's warm-up."""
    if len(counts) != 4 or min(counts) < 1:
        raise ValueError("Four positive warm-up quotas are required")
    completed = [0] * 4
    result = []
    while len(result) < sum(counts):
        index = min((i for i in range(4) if completed[i] < counts[i]),
                    key=lambda i: (completed[i] / counts[i], i))
        result.append(TRAIN_TASK_NAMES[index])
        completed[index] += 1
    return result


def plan(a):
    budgets = {
        "training_smoke": len(a.policy_gpus),
        "base_and_final_fixed": 8 * a.fixed_attempts,
        "warmup": sum(a.warmup_updates) * a.group_size,
        "search_max": 4 * a.search_rounds * a.search_attempts,
        "post_sft_rl": (a.recipe_fixed_updates + a.recipe_diverse_updates) * a.group_size,
        "validation_max": 5 * 4 * 2 * a.validation_attempts,
        "base_and_final_test": 2 * 5 * len(a.test_seeds) * a.test_attempts,
    }
    return {"recipe": VERSION, "episode_budgets": budgets,
            "max_total_episodes": sum(budgets.values()), "no_training_before_gate": False,
            "warmup_updates_by_task": dict(zip(TRAIN_TASK_NAMES, a.warmup_updates)),
            "warmup_rollouts_by_task": dict(zip(TRAIN_TASK_NAMES, [n*a.group_size for n in a.warmup_updates])),
            "success_targets": dict(zip(TRAIN_TASK_NAMES, a.success_targets)),
            "sft_minimum_per_task": a.sft_min_successes,
            "gate": {"applies_to": "SFT and post-SFT RL only", "replay_verified_minimum_per_task": a.sft_min_successes,
                     "pre_warmup_successes_required": 0, "independent_base_qualification_required": False},
            "loss_normalization": "dr_grpo", "lora_rank": 16,
            "budget_exhaustion": "Evaluate warm-up policy and export available videos; mark completed_partial if SFT gate fails",
            "allocation_note": "More warm-up for easy primitives is this run's allocation, not the paper's hard-task oversampling"}


def training_summary(root):
    """Read committed metrics only; old partial rollouts do not count as updates."""
    root = Path(root)
    tasks = {t: {"groups": 0, "episodes": 0, "successes": 0, "optimizer_steps": 0,
                 "invalid_turns": 0, "last_mean_reward": None} for t in TRAIN_TASK_NAMES}
    for path in sorted((root / "metrics").glob("step_*.json")):
        row = read_json(path)
        if row.get("task") not in tasks:
            continue
        # A metrics file can be written just before a failed checkpoint commit.
        if not (root / f"step_{row['step']:05d}" / "COMPLETE.json").is_file():
            continue
        item = tasks[row["task"]]
        item["groups"] += 1
        item["episodes"] += row["group_size"]
        item["successes"] += row["successes"]
        item["optimizer_steps"] += int(row["optimizer_step_taken"])
        item["invalid_turns"] += row["invalid_turns"]
        item["last_mean_reward"] = sum(row["rewards"]) / len(row["rewards"])
    return {"tasks": tasks, "updates": sum(v["groups"] for v in tasks.values()),
            "interpretation": "Changing-policy training yield, not independent evaluation success rate"}


def execute(runner):
    from .pipeline import jobs, export_sources, select_candidate, make_videos, report
    from .recipe_gate import inspect_gate, require_gate
    from .success_gate import diagnostics

    a, out = runner.a, runner.output
    write_json(out / "recipe_plan.json", plan(a))
    # Run real multimodal forward/backward and gradient synchronization before
    # spending hours mining. The smoke weights and traces are discarded.
    runner.train("training_smoke", "smoke")
    runner.sample("base_fixed", None, jobs(TRAIN_TASK_NAMES, [a.seed], "train", a.fixed_attempts))
    schedule_path = out / "plans/warmup_schedule.json"
    write_json(schedule_path, warmup_schedule(a.warmup_updates))
    warmup = runner.train("warmup", "grpo", updates=sum(a.warmup_updates), schedule=schedule_path)
    sources = [out / "base_fixed", out / "warmup/rollouts"]
    targets = dict(zip(TRAIN_TASK_NAMES, a.success_targets))

    def inspect():
        return inspect_gate(out, sources, a.seed, a.model, targets, a.sft_min_successes)

    gate = inspect()
    search_sources = []  # Frozen final warm-up policy only; never reuse earlier-policy prefixes.
    for index in range(a.search_rounds):
        counts = gate["unique_verified_successes"]
        missing = [t for t in TRAIN_TASK_NAMES if counts[t] < targets[t]]
        if not missing:
            break
        name = f"mine_round_{index:03d}"
        search_jobs = jobs(missing, [a.seed], "train", a.search_attempts, "search",
                           attempt_start=100000 + index*a.search_attempts)
        for job in search_jobs:
            job["min_successes"] = targets[job["task"]] - counts[job["task"]]
            job["search_sources"] = [str(p) for p in search_sources]
        runner.sample(name, warmup, search_jobs)
        sources.append(out/name)
        search_sources.append(out/name)
        gate = inspect()
        runner.status("self_success_progress", round=index+1, rounds=a.search_rounds,
                      verified_successes=gate["unique_verified_successes"], targets=targets,
                      sft_ready=gate["passed"])
        # Zero-success rounds do not cancel other tasks or the bounded search.
    diagnostics(out, sources, a.seed)
    candidates = [("base", None, out/"validation_base"),
                  ("warmup", warmup, out/"validation_warmup")]
    if gate["passed"]:
        require_gate(out)
        admitted = [Path(r["result"]).parent for records in gate["approved"].values() for r in records]
        data = out/"data/self_success.jsonl"
        info = export_sources(admitted, data)
        write_json(out/"data/export.json", info)
        if not info["actions"] or set(info["episodes_by_task"]) != set(TRAIN_TASK_NAMES):
            raise RuntimeError("SFT export lost task coverage; inspect data/export.json")
        # Self-distill successes into the pretrained base, then continue RL.
        adapter = runner.train("sft", "sft", data=data)
        candidates.append(("sft", adapter, out/"validation_sft"))
        adapter = runner.train("rl_fixed", "grpo", adapter=adapter, updates=a.recipe_fixed_updates)
        candidates.append(("rl_fixed", adapter, out/"validation_rl_fixed"))
        adapter = runner.train("rl_diverse", "grpo", adapter=adapter, updates=a.recipe_diverse_updates,
                               seeds=[a.seed+i for i in range(4)])
        candidates.append(("rl_diverse", adapter, out/"validation_rl_diverse"))
    else:
        runner.status("sft_skipped_insufficient_data", training_started=True,
                      verified_successes=gate["unique_verified_successes"],
                      message="Warm-up and search budgets exhausted. Evaluate available policies; no fabricated SFT data.")
    for name, adapter, directory in candidates:
        runner.sample(directory.name, adapter, jobs(TRAIN_TASK_NAMES, [0, 1], "validation", a.validation_attempts))
    winner, scores = select_candidate(candidates)
    write_json(out/"selection.json", {"selected": winner, "candidates": scores,
               "criterion": "Four-task validation macro success, then reward; no test/rack_transfer selection"})
    final = Path(winner["adapter"]) if winner["adapter"] else None
    runner.sample("base_test", None, jobs(TASK_NAMES, a.test_seeds, "test", a.test_attempts))
    runner.sample("final_test", final, jobs(TASK_NAMES, a.test_seeds, "test", a.test_attempts))
    runner.sample("final_fixed", final, jobs(TRAIN_TASK_NAMES, [a.seed], "train", a.fixed_attempts))
    make_videos(runner)
    state = "completed" if gate["passed"] else "completed_partial"
    summary = report(out, state)
    runner.status(state, training_started=True, sft_completed=gate["passed"],
                  report=str(out/"report.json"), videos=str(out/"videos/index.json"),
                  all_five_have_success_videos=summary["all_five_have_success_videos"])
