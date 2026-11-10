# Fixed-node upright-tube experiment

This experiment is **original -> warm-up GRPO -> necessary prefix search -> SFT**.
It never launches main RL, checkpoint selection or hyperparameter search. It uses
the existing Qwen3.5-4B checkpoint, Panda/MuJoCo controller, semantic ZAP poses,
native SAM3.1 service and shared Python environments. Old experiments are retained.

## Launch on the existing server

On the networked CPU instance, in the existing shared checkout:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git
git pull --ff-only
```

On the four-H100 GPU instance, in that same checkout:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git
bash scripts/start_tube_curriculum.sh runs/tube_curriculum/v1
```

No package installation or model download is performed. The launcher uses the
existing `.venv-linux`, `.venv-sam`, `assets/panda`, SAM weights, and private EGL
runtime. The original model is resolved from the existing path:

```text
/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model
```

For a GPU job entrypoint accepting an `sh` script, use the checked-in wrapper:

```sh
sh /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git/scripts/submit_tube_curriculum.sh
```

Its contents can also be pasted directly into the job's startup-script field.
It activates the existing project environment, prints the Python environment,
and launches this same experiment with the explicit model path. Live progress is
shown in the job output and appended to the project's `start_self.log`. The
wrapper waits for the experiment monitor to finish and preserves its exit code;
it does not exit immediately after starting the background pipeline. No Conda
environment from another project is required. Allocate all four H100 GPUs to the
job and synchronize the code on the CPU instance before submitting it.

If an earlier job stopped with `ambiguous option: --run could match --run-path,
--run_path`, update the checkout and use a new output directory:

```sh
TUBE_RUN_DIR=runs/tube_curriculum/v1_launchfix sh /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git/scripts/submit_tube_curriculum.sh
```

The launcher now separates torchrun options from child-module arguments with
`--` before the module name. This avoids Python 3.11 option-abbreviation conflicts
in smoke, warm-up, SFT and search. All four commands are also checked with the
installed PyTorch parser before physics or model loading. Reusing a directory
from an older code revision is intentionally rejected by the frozen provenance
check; preserve the old directory for diagnosis. The task configuration and
sampling budgets are unchanged.

It records the exact resolved directory and hashes model, tokenizer and processor
files; it does not substitute a checkpoint with a `-Base` suffix. SAM occupies
GPU 1 and policy replicas occupy GPUs 0/2/3. Stop prior experiment/services before
launching. An occupied SAM port is rejected without stopping unrelated processes.
Alternatively use a free localhost port, consistently on resume:

```bash
bash scripts/start_tube_curriculum.sh runs/tube_curriculum/v1 --sam-url http://127.0.0.1:8003
```

Progress streams until the experiment completes/fails. Ctrl+C only detaches the
watcher. To watch again:

```bash
.venv-linux/bin/python scripts/watch_lab_run.py runs/tube_curriculum/v1
```

Repeat the identical launch to resume committed work after interruption. Code,
configuration, model or scene changes require a new output directory. Updating
the checkout while the experiment runs is rejected at subsequent stage launch.
There is no wall-clock experiment budget or deadline. HTTP/communication failures
remain operational errors, not a reason to count missing trials as failures.

Print the registered plan without loading models:

```bash
.venv-linux/bin/python -m tube_agent.curriculum.pipeline --plan
```

## Task geometry and success

All dimensions, success tolerances, reward coefficients and training settings are
in `configs/tube_curriculum_v1.json`, copied and frozen into the run directory.

- Tube: capsule collision body, radius 12 mm, end-to-end length 114 mm.
- Each support/rack has **four separate collision walls and a floor**, not a
  solid collision box. The square hole is 28 mm wide; outer width is 70 mm.
- Floors: world z=8 mm. Source support rim: 26 mm. Target rack rim: 50 mm.
- Insertion depth: 42 mm from target rim to tube bottom; tolerance 3 mm.
- Insertion cross-section must fit the real hole, allowing 0.3 mm contact tolerance.
- Release: measured opening >=60 mm and neither finger contacting the tube.
- Withdrawal: TCP-to-tube-center distance >=90 mm.
- Placement stability: linear speed <=30 mm/s and angular speed <=0.3 rad/s.
- True hold: bilateral finger contacts plus low tube-to-gripper relative linear
  and angular motion continuously for 0.1 s; a close command alone is insufficient.

| Task | Initial state | Goal |
|---|---|---|
| T1 | Upright tube in shallow blue support, hand open | Tube bottom raised >=50 mm; stable grasp and stationary object for >=0.5 s |
| T2 | Airborne upright tube physically held near source | Bottom horizontally <=5 mm from yellow hole center, bottom z=85 +/-5 mm, tilt <=8 degrees, held and stable >=0.5 s |
| T3 | Physically held above target, XY offset up to +/-5 mm per axis | Correct insertion depth and hole, tilt <=8 degrees, release and withdraw, stable >=1 s |
| T4 | Same type of start as T1 | Grasp, clear source rim by 5 mm, transport, then T3 final conditions |

T2/T3 setup is produced by deterministic physical motions, closing the real
fingers, lifting, and verifying the grasp. No object-to-hand attachment or weld is
used. Setup commands are logged as initialization, never model actions or SFT data.

The controller is unchanged: absolute world TCP in metres, slot_yaw/tilt/
tilt_direction in degrees, normalized gripper commands [0,1]. Maximum episode
length is 40 agent turns, including query/todo tools. `done` advances 0.5 s then
ends the episode; it does not declare success. Collision/drop failures and the
turn limit also end episodes. The same rules apply to every checkpoint.

## Observation and perception

Two 640x480 RGB-D views, `top` and `side`, use the existing SAM3.1 detector and
mask/depth world-coordinate measurements. Initial queries are task-configured;
the agent can change them. No object ground truth, reward or success flag is
inserted into model inputs. The privileged judge is used for rewards, search and
evaluation only. Camera calibration and simulator state are retained for audit.

Each turn stores the exact multimodal messages, generated response token IDs and
a fingerprint of the processor's input tensors. Training verifies this fingerprint
after reconstructing that same turn. It never rebuilds an expanded conversation
from the full trajectory. Old image history remains cropped exactly as in rollout.

## Sampling and training

The frozen manifest contains 32 training and 30 held-out evaluation scenes per
task: 248 physical scenes total. Source/target positions vary by +/-12 mm; T3 also
has insertion offsets. Evaluation scenes are excluded from search and SFT.

Every group has one task, one scene, one instruction and one policy version. Only
policy sampling seeds differ among its eight members. Each round selects two
distinct training scenes per participating task afresh. The entire schedule is
recorded before training; it never adapts to evaluation results.

| Phase | Rounds | Groups/round | Episodes/round | Total |
|---|---:|---:|---:|---:|
| All four tasks | 12 | 8 | 64 | 768 |
| T3/T4 supplement | 6 | 4 | 32 | 192 |
| Total | 18 | 120 total groups | | 960 |

Per-task totals: T1=192, T2=192, T3=288, T4=288. Three ranks partition rollout
indices; a group of eight is not changed to fit the number of GPUs. Advantages
are centered separately in each group. No standard-deviation normalization, noise
injection, per-turn invalid advantage or failed-done credit clamp is applied.
Zero-advantage groups record action signatures, rewards and reward components.

Each sampling batch completes before updates start. Old response likelihoods for
the whole batch are computed before any minibatch update. One epoch uses episode
minibatches of 32, accumulated with microbatch one. Consequently the 18 sampling
rounds have 30 planned optimizer steps (24+6); genuinely zero-gradient minibatches
are skipped and reported separately. The loss uses clipped token likelihood
ratios and a fixed maximum episode-token denominator, reusing the existing
multimodal Dr.GRPO-style helpers. Reference KL=0, clip=0.2, AdamW weight decay=0.01,
gradient norm limit=1. No STAPO/DPPO or asynchronous stale-policy training is claimed.

Both stages use BF16 base weights and language-only LoRA rank32/alpha64. Vision
encoder and connector parameters remain frozen. Actual trainable parameter names,
shapes and dtypes are checked and saved. Warm-up LR=1e-5, temperature=0.7,
top-p=0.95, top-k disabled; `enable_thinking=False`, max response 256 tokens,
max context 16384. Sampling calls the same live model object after each update.
Before formal training, a discarded multimodal optimizer step checks nonzero
gradients, changed adapter hashes and a new generation with those changed weights.

## Reward

`R = 5*S + (1-S)*progress - .01*turns - .1*invalid_actions - catastrophic`

Progress is terminal-state partial credit in [0,.9], not an accumulated sum of
visits to subgoals. Full success uses the physical judge, not TCP reach or model
claims. Catastrophic is a binary, latched cost of 1: robot-rack/table contact force
>=80 N sustained for >=0.03 s, or a previously held tube lost outside a valid
seated goal with bottom below 3 mm or tilt above 70 degrees. It also prevents
success. Ordinary finger/tube gripping forces are excluded from the collision cost.

Normalized distance signals use 0.30 m approach, 0.35 m transport, 0.03 m final
alignment, 0.10 m height and 45-degree orientation scales; all clip to [0,1].

| Task | Progress formula/gates |
|---|---|
| T1 | max(.20*approach, .25+.65*normalized_lift if truly held else 0) |
| T2 | While truly held: .20 + .35*horizontal + .20*height*horizontal + .15*upright*height*horizontal; otherwise 0 |
| T3 | .20*alignment + .15*alignment*upright while held or validly seated; .35*depth only if actual cross-section fits; .10 release only seated and released; .10 withdrawal only after release |
| T4 | max(.10*approach, .15 grasp + .20 extraction + .15 transport + .15 depth + .075 release + .075 withdrawal), with predecessor gates |

For T4, transport/depth require recorded extraction and current physical holding
or valid seating. Empty-hand motion cannot earn transport credit. Insertion
depth is zero unless the actual tube cross-section fits the correct hole and
orientation is valid. Release credit is zero while still held. Historical flags
record actual physical events; repetitive opens/closes do not accumulate credit.
This version deliberately omits ZAP's per-turn negative-advantage mechanism as
requested; invalid actions are penalized at episode level only.

## Prefix search and SFT

Only tasks with zero complete warm-up successes and a valid physical prefix are
searched. The fixed `after_warmup` adapter is used without updates. Nodes follow
accepted move/gripper actions after the object and joints have settled. Each node
references a real training scene, complete root prefix, saved contexts and states.

Eligible tasks are scheduled by least number of expansions; within a task select
`wins/(visits+1) + progress + .45*sqrt(log(2+total_expansions)/(visits+1))`.
Three same-type continuation failures prune the node and descendants. There are
at most 128 nodes per task, retaining higher-progress prefixes when full.

One node selection generates four continuations; the **global** limit is 32
expansions/128 continuations across all candidate tasks. Stop searching a task
after its first complete success. Replayed and newly generated action counts are
separate. Every continuation resets and replays from the deterministic initial
state, verifies physical/judge state within 1e-5, restores queries, todo, initial
visual reference and action history, then samples only a new suffix. Stored
successes are single real root-to-leaf paths, never concatenated sibling branches.

The SFT manifest merges complete warm-up/search successes, verifies physical
replay and train-scene membership, removes exact duplicate trajectories, and uses
each identical input/assistant-response pair once so common prefixes cannot
dominate. Missing tasks are explicitly reported. Successful trajectories with
invalid turns are not silently relabeled as failures; their counts are retained.
All assistant responses in admitted successes are supervised as requested.

SFT loads **after_warmup and its one existing adapter**, resets AdamW, uses LR1e-4,
global batch16 assistant responses with gradient accumulation, and runs exactly
two complete epochs. Only generated response tokens contribute to the loss.
The second epoch's endpoint is `after_sft`; no score-based selection occurs.
If there are no complete successes anywhere, SFT is unavailable rather than
inventing data or copying after_warmup under a misleading after_sft name. The run
reports `completed_partial` after evaluating the two available fixed nodes.

## Checkpoints, evaluation and recovery

`checkpoints/original` contains the exact base reference and tokenizer/processor.
`checkpoints/after_warmup` is the end of round18, and `checkpoints/after_sft` is the
end of epoch2. Adapters, tokenizer/processor, source/config/model/scene hashes and
data references are associated with each node. Optimizer and Python/NumPy/Torch/
CUDA RNG state are saved per rank in committed checkpoints. Warm-up commits each
sampling round; SFT commits every fourth minibatch and every epoch boundary.
Incomplete checkpoint directories are ignored. Resuming an interrupted round
reuses only same-policy complete rollouts and repeats any uncommitted updates
from the last committed optimizer state; failed infrastructure attempts are archived.

Each node receives exactly the same 120 evaluation scenes, greedy decoding,
interface and termination rules. Total planned model evaluations: 360. There is
no `after_search` model and no evaluation-based checkpoint replacement.

Important artifacts:

```text
config.json / scenes.json / schedule.json / provenance.json
physics/ / vision/ / smoke/
checkpoints/original/ / after_warmup/ / after_sft/
warmup/rollouts/round_*/.../trajectory.jsonl
warmup/metrics/round_*.json
search/state.json / search/rollouts/
data/manifest.json / data/samples.jsonl
sft/metrics/ / sft/checkpoints/
evaluation/{original,after_warmup,after_sft}/T*_eval_*/result.json
comparison.csv / report.json / status.json / logs/
```

The original-vs-warm-up change includes the T3/T4 supplement. The warm-up-vs-SFT
change includes any search-supplied SFT data; this design does not isolate the
contribution of search. After comparison, the process stops. Physical diagnostic
successes and discarded smoke episodes are never reported as Qwen successes.

## Verification before publication

Local verification on Windows/Python 3.13 covered all 248 frozen initial scenes:
248/248 passed the scripted physical feasibility check. These are controller
diagnostics, **not learned-policy success rates**. The committed
`validation/curriculum_v1/physics_summary.json` records the per-scene outcomes,
and eight initial camera images show both views of all four tasks. The archived
sweep configuration differs from the final configuration only in interface/
observation metadata and the tree-search node settling threshold; physics,
rewards, scene generation and success thresholds are identical.

The test suite passed 287 tests, with one Windows-only distributed-transport
test skipped, across the corrected test invocations. Eight rendering tests used
GLFW; local HTTP tests needed normal Windows socket permissions. The new tests
cover actual contact/release/withdrawal conditions, reward gates, disjoint scenes,
exact context fingerprints, prefix replay, success-only data selection and the
18-round/960-rollout/30-planned-step schedule. A small CPU model exercises the
actual training loop, SFT continuation and checkpoint recovery; it does not stand
in for a Qwen3.5-4B training result. Shell syntax and Python compilation passed.

The full Qwen checkpoint, native SAM weights and H100/NCCL execution are available
on the server, not this Windows workspace. Accordingly, the launcher still runs
native perception and a real multimodal parameter-update check there before the
formal experiment. Local checks establish executable plumbing and physical
feasibility; they cannot guarantee learned-policy success.
