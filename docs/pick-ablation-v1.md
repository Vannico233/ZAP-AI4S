# Failed-done credit ablation (2026-11-10)

## Evidence and scope

The supplied v5 archive reports 68 GRPO updates, 408 warm-up episodes and one
eligible pick success. All 256 prefix-search episodes failed; SFT never started;
the final five-task test was 0/30. The successful rollout was sampled after update
38 and used for update 39. It does not establish checkpoint 39 or 68 performance.

The archived update 49 unscrew episode `update_0048/rank_01/try_002` ended with an
accepted `done` while unsuccessful. Its reward was 0.17, the group mean was
0.08372333333333333, and its advantage was +0.08627666666666668. Previously that
positive advantage trained every accepted turn, including the failed `done`.
This is one demonstrated failure mechanism, not proof of the sole cause.

## Intervention

Both trainers expose `--failed-done-credit legacy|nonpositive`. The default stays
`legacy` for explicit historical comparison. The ablation invokes each mode
explicitly. `nonpositive` maps the advantage of an accepted unsuccessful `done`
to `min(advantage, 0)`. Negative credit remains negative. Successful `done` and
earlier movement/gripper turns retain their trajectory credit; rejected actions
retain -1. Success is read from the existing post-action judge, not model claims.
The judge remains training-only. No action is supplied, prohibited or rewritten.
The reward ladder, length penalty, success thresholds and policy inputs are unchanged.

This removes the direct positive likelihood gradient on failed termination. It
does not guarantee that shared parameters, optimizer momentum or AdamW decay
cannot indirectly change its probability. A zero clamp does not actively penalize
above-average failed termination; stronger penalties and length-cost changes are
separate future experiments, not combined into this first comparison.

## Predeclared experiment

1. Compare `step_00038`, `step_00039`, `step_00068` using 12 independent pick
   rollouts each: train split, physical seed 0, temperature 0.8, attempts 80000–80011.
2. Start BOTH arms from `step_00038`, without selecting by evaluation reward.
   Reset AdamW in both arms; this is a matched branch, not exact optimizer resume.
   Run eight pick-only updates per arm: group size 6, physical seed 0,
   LoRA rank 16, learning rate 1e-5, 40 turns, 16k context, Dr.GRPO normalization,
   no reference KL. The only arm difference is failed-done credit.
3. Evaluate each final arm with 12 new fixed-scene rollouts (attempts 81000–81011)
   and 12 validation-split rollouts (seed 0, attempts 82000–82011).
4. Stop. Total: 180 episodes, 16 optimizer updates. No SFT, search, test-set
   selection, rack_transfer sampling, or automatic four-task continuation.

Sampling seeds are matched across arms/checkpoints. Different policies need not
produce matching trajectories. Training is stochastic; one training seed and
12-rollout cells provide a diagnostic pilot, not a precise success-rate estimate.
Report counts and denominators. `report.json` includes failed-done episode counts,
ever-held and stable-lift judge counts, true task successes, and total steps,
reward and invalid actions. Training metrics record positive failed-done credit
before and after the intervention. Grasp/lift diagnostics are not task successes.

Proceed to further work only after reviewing these results. A fall in failed-done
frequency alone is insufficient; longer unsuccessful trajectories are not an
improvement. Inspect successful traces and replay them before admitting any to
future SFT. If neither arm succeeds independently, diagnose perception and grasp
execution rather than automatically scaling the run. The validation scene is
untrained but is a development set, not a final generalization claim.

## Existing server environment

CPU instance, from the existing Git checkout:

```bash
git pull --ff-only
git log -1 --oneline
```

GPU instance, from that same shared checkout:

```bash
bash scripts/start_pick_ablation.sh runs/lab_pick_ablation/v1
```

The launcher reuses `.venv-linux`, `.venv-sam`, the model, SAM weights and private
EGL runtime. It manages SAM on GPU 1 and policies on GPUs 0/2/3. No download or
installation is performed. An occupied SAM port is rejected without stopping
external processes. Ctrl+C exits the watcher; the background run continues.

Reattach:

```bash
.venv-linux/bin/python scripts/watch_lab_run.py runs/lab_pick_ablation/v1
```

Print budgets without models:

```bash
.venv-linux/bin/python -m tube_agent.lab.pick_ablation --plan
```

Repeat the identical launch to resume interrupted work. Code/settings or source
adapter weight changes require a new output directory. Original v5 outputs are
read-only inputs. A lock prevents concurrent writers to the same ablation run.
Per-action English logs are in `logs/`; summaries are in `report.json` and
`train_*/metrics/`. For review, retain full result/trajectory files as well as
these summaries; images are needed to diagnose perception failures.

## Differences from ZAP

Source: https://arxiv.org/html/2610.35817v3 (sections 3.3 and Appendix C).

The paper uses Qwen 3.6 27B, simulated bimanual YAM in Isaac Sim and 40 training
tasks. This project uses Qwen3.5-4B, MuJoCo/Panda and four tube tasks; this pilot
trains only pick. The paper reports rank 64, groups of 8, 50 turns, 127k context,
asynchronous training with STAPO/DPPO and bounded staleness. This implementation
uses rank 16, groups of 6, 40 turns, 16k context and synchronous replicas without
those mechanisms. It retains self-generated data, semantic poses, vision input
and the warm-up/search/SFT/RL structure, but has not reached SFT in v5.

The failed-done clamp is an evidence-motivated engineering adaptation. It is not
presented as a paper-prescribed correction or a replication of published scores.
No improved success rate has yet been measured for this change on the H100 server.

## Local verification

260 tests passed across the full suite and permission-corrected reruns; one
existing Windows distributed-communication test was skipped. The first full run
had seven local HTTP failures caused by Windows socket permissions; all seven
passed when rerun with normal permissions. Bash syntax and `git diff --check`
passed. Tests cover turn-credit gradient direction, preserving earlier actions
and successful termination, matched training arms, evaluation RNG separation,
bounded orchestration and rejection of manual evaluation evidence. These checks
do not substitute for running Qwen3.5-4B on the H100 server.
