# Four-task visual laboratory reproduction

This is an independently implemented, small-scale ZAP-method reproduction for
Qwen3.5-4B, not a reproduction of the paper's reported scores. All commands and
program output are English/ASCII. The original `tube_agent.run` baseline remains
available. The older acquisition campaign has a closeout helper; the user has
subsequently started the bounded v4 pose probe described below.

## Stream progress until the experiment finishes

For the already running pose probe, fetch just the read-only monitor on the CPU
instance, without changing the experiment's running source files:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git &&
git fetch origin main &&
git show origin/main:scripts/watch_lab_run.py > watch_lab_run.py
```

On the GPU instance, attach to the existing experiment:

```bash
.venv-linux/bin/python watch_lab_run.py runs/lab_pose_probe/seed0_v4
```

The monitor first shows a bounded tail of existing logs, then streams new action
lines and episode outcomes, stage changes, completed-trial counts and final
video paths. It returns automatically after the identified launcher exits and
the final logs are read. It reports crashes without final status instead of
waiting forever. Exit codes: 0 for a completed workflow (task failures may still
exist), 1 for errors, 2 for an unmet success gate, and 130 for interruption.
`[WAIT]` every 30 quiet seconds means the launcher is alive, not proof of action
progress. No simulation, training, network request or process signal is started
by this standard-library-only monitor. Run it on the actual GPU host.

For future launches, after updating the checkout when no experiment is running,
add `--follow` to the existing launcher command:

```bash
bash scripts/start_lab_all.sh runs/lab_pose_probe/seed0_v4 \
  --action-protocol zap_pose_v1 --pose-probe \
  --sam-url http://127.0.0.1:8002 --follow
```

Do not repeat the launch command just to view an active run: use the attach
command above. `--follow` is consumed by the launcher and does not change model
or experiment settings. Ctrl+C closes the monitor; the detached experiment
continues. Different GPU workers' output is interleaved, not a global action
execution order. Exact task/attempt/step identifiers are printed on each action.

## Stop and archive the current experiment

On the networked **CPU instance**, fetch only the standalone closeout helper.
This leaves the running experiment's tracked source files unchanged:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git &&
git fetch origin main &&
git show origin/main:scripts/close_lab_run.py > close_lab_run.py
```

On the **GPU instance running the experiment**, execute:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git &&
.venv-linux/bin/python close_lab_run.py \
  --stop \
  --run runs/lab_success_first/seed0_v3_1 \
  --include runs/lab_pick_debug/seed0_v3
```

Use the actual active output directory for `--run`. If the later pose probe was
started, its directory is `runs/lab_pose_probe/seed0_v4`. `--include` is optional:
it adds an already stopped run for evidence collection and never stops that run.
The old pick probe is included to retain its existing successful replay video.

The standard-library-only helper matches the pipeline by its exact output
directory and verifies process identity before SIGTERM; a stale launcher PID
is never trusted. It prefers pidfd when available. If Python lacks either pidfd
API, or the container/kernel rejects that API as unavailable, it rechecks the
same-user process's start time and command line immediately before ordinary
SIGTERM. This fallback has the normal check-to-signal race of PID-based kill;
it is not an atomic pidfd identity guarantee. `STOP_SIGNAL` prints the backend
and PID used. No Python upgrade or package installation is required.
It waits for the pipeline and identified children to exit,
then holds the run lock while copying evidence. A timeout or a still-locked run
is reported without deleting data. Unrelated services, including a separately
started SAM server, are not stopped. Run this on the GPU host, not the CPU host.

Outputs are printed as `CLOSEOUT`, `ARCHIVE` and `SHA256`:

- `closeouts/lab_closeout_<UTC timestamp>/SUMMARY.md`: per-stage results and video links.
- `summary.json`: original paths, shutdown scope, errors and existing training evidence.
- `evidence/`: configurations, job summaries, episode results, selected trajectories,
  log tails and existing successful model videos with matching replay reports.
- `MANIFEST.json`: source paths, byte counts and SHA256 hashes for copied evidence.
- A sibling `.tar.gz` archive and LF-only `.tar.gz.sha256` checksum file.

Original runs, environments and checkpoints remain in place. No sampling,
training, model loading or video rendering starts. This is a bounded evidence
archive, not a full checkpoint backup: at most 64 MiB per file (configurable),
1 GiB total, ten successful trajectories per task per run, first/best failures,
and the last 2 MiB of each log. Skipped files are listed in `summary.json`.
Imported successes, search yield and independent trials are counted separately.
An eligible-result filter is not a claim that the training admission gate passed.

## ZAP interface correction and bounded diagnosis (v4)

Source audit: [paper v3, section 3.3 and Appendix A](https://arxiv.org/html/2610.35817v3),
[official orientation widget](https://zero-real.github.io/assets/js/orientation-widget.js),
and [Open-RUA pose math](https://github.com/Open-RUA/uni-robot-agent/blob/f3e0ec006295f74321b6a349c9b47ec82071eb28/uni_robot_agent/tools/gripper/pose.py).
ZAP uses absolute world TCP positions in metres and **slot yaw, tilt, tilt
direction in degrees**, not RPY. Its level slot is yawed about world Z, then
tilted about `[sin(phi), -cos(phi), 0]`. Open-RUA currently uses fixed-axis RPY
at its boundary; it is an execution-design reference, not the source of the
paper's orientation convention. Our previous RPY interface was a substantive
deviation. It remains available only to preserve old experiments/replays.

The new single-arm JSON serialization is:

```json
{"action":"move_to","xyz":[0.46,-0.18,0.20],"slot_yaw":0,"tilt":0,"tilt_direction":0}
```

`set_gripper` takes an opening fraction (0 closed, 1 fully open); `done` ends
the episode. The other tools remain `change_query` and `update_todo`. No grasp,
approach, transport or insertion macro is exposed to the model. Policy output
remains one JSON object rather than the paper's native multi-arm tool schema.

The controller prechecks all IK waypoints and joint continuity before moving.
It refuses invalid paths without clipping a target or executing a prefix.
Reached/short feedback contains measured position and rotation errors. This
small MuJoCo controller is not a copy of the original collision planner.

Reward is a **final-state ladder**, not a sum of step bonuses: approach <=0.2,
grasp 0.2, additional lift <=0.55 over 8 cm, transport total <=0.9, task success
5.0. Every step logs its current `reward_terms`, but GRPO consumes the final
episode reward. Local insert/unscrew extensions, a 0.01/turn penalty and a
0.05/invalid-action penalty are explicit adaptations. Completion must remain
stable for 0.5 s. The lab pick instruction still requires 6 cm clearance;
8 cm is the reward saturation height, not a changed instruction.

The paper mines successes with short GRPO warm-up plus prefix search, then SFT
and longer GRPO. **At the user's request we retain no training before verified
successes**, so that initial warm-up is intentionally absent. Our 10-success
and 3/10-qualification thresholds are engineering choices, not paper settings.

After the previous experiment has stopped, update on the networked CPU instance:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git
git pull --ff-only
```

On the GPU instance, run this bounded diagnostic first:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git
bash scripts/start_lab_all.sh runs/lab_pose_probe/seed0_v4 \
  --action-protocol zap_pose_v1 --pose-probe \
  --sam-url http://127.0.0.1:8002
```

It runs physical checks, then 12 independent model trials: each of pick,
transport, insert and unscrew at temperatures 0, 0.3 and 0.7, all scene seed 0.
It **never trains**, even if every trial succeeds. SAM uses GPU 1; policies
use GPUs 0, 2 and 3. Read `status.json`, `logs/pose_probe_gpu*.log`, per-job
summaries and available verified videos. Collect evidence with:

```bash
.venv-linux/bin/python scripts/collect_success_first.py --run runs/lab_pose_probe/seed0_v4
```

Once this interface is validated on the actual model, continue acquisition with:

```bash
bash scripts/start_lab_all.sh runs/lab_success_first/seed0_v4 \
  --action-protocol zap_pose_v1 --sam-url http://127.0.0.1:8002 \
  --success-source runs/lab_pose_probe/seed0_v4
```

Only compatible new-interface successes can be imported. Prefix search now
starts from same-task, same-seed, same-policy model traces in previous batches,
instead of discarding their tree between rounds. Logs show `search_initialized`
and `search_expand` with source/node counts and prefix length. A semantic-pose
round adding no verified successes stops for diagnosis. The existing gate
still blocks SFT, optimizer smoke and RL until all four tasks qualify.
The supplied v3.1 trace showed an empty closure at z=0.04 m followed by empty
lifting, with all four steps reused as a prefix. Search now stops adding nodes
at a judge-confirmed empty closure and resamples before that action. A non-root
prefix failing three expansions is pruned with its descendants. The search
judge can select branches; it never supplies target coordinates, actions or
hidden object state to the policy. These pruning thresholds are local choices.

For explicit step-by-step command debugging at the same scene seed:

```bash
bash scripts/lab.sh command-probe \
  --task transport --seed 0 --output runs/manual_pose/transport_trial_01
```

This applies the private EGL runtime automatically. Enter one JSON action per line, ending with
`{"action":"done"}`; alternatively pass `--commands your_actions.json` with a
JSON array. Each step writes top/side images and measured pose/feedback/reward.
Repeat with a new output directory to change commands from the same initial
state. These traces are labeled `manual_command_debug`, never eligible for
self-success SFT or model success-rate claims. `zap_check` similarly uses
privileged geometry only to validate physics, never as model input.

Do not reinterpret the prior 4/9 legacy pick probe or scripted physics success
as evidence that the corrected model interface already succeeds. H100 model
results are pending the new probe.

## Scope and honest boundaries

| Component | Implementation |
|---|---|
| Training tasks | `pick`, `transport`, `insert`, `unscrew` |
| Held-out composition | `rack_transfer`: blue source rack to yellow destination rack; forbidden in the train split |
| Robot | Single Panda; v4 semantic full pose and normalized opening; legacy RPY retained for replay |
| Vision | Two calibrated RGB-D cameras; native **SAM 3.1 multiplex checkpoint**, official predictor, text queries; 3D visible-surface statistics from masks and depth |
| Policy input | Annotated RGB, estimated surface geometry, robot proprioception and execution feedback. No simulator object coordinates or success/reward fields |
| Data mining | Independent fixed-scene stochastic trials; optional deterministic prefix replay and UCB tree search over model-generated actions |
| Training | Multimodal self-success LoRA SFT; online group-relative clipped policy optimization, terminal ladder rewards, invalid-turn negative advantages |
| Differences from paper | MuJoCo rather than Isaac Sim; 4B rather than 27B; four tasks rather than forty; synchronous SAM snapshots at action boundaries rather than asynchronous temporal perception; LoRA and a small synchronous GRPO implementation, beta=0 |
| Not claimed | Real-robot transfer, original benchmark scores, original training scale, or verified 4B/SAM GPU results before the server smoke run |

The `unscrew` task explicitly clamps the tube body in a fixture. Its cap is a
separate dynamic body: a helical constraint couples rotation to 6 mm rise per
turn, releases after one actual positive turn and sufficient rise while held,
and then permits removal. This is a **joint/constraint approximation of threads**,
not resolved thread-mesh contact or a fluid simulation. No task welds an object
to the gripper. `insert` starts with a horizontal tube and requires cap-up
reorientation, physical entry into a collision-bounded square hole, release and
withdrawal. The nominal hole is 28 mm wide, floor 8 mm, rim 50 mm above the table.
The new environment uses a 1 ms timestep and a 2000 N/m finger servo spring;
these changes do not apply to the old baseline.

## GitHub delivery and reuse of the successful pick probe (v3.1)

The server's v3 pick probe finished 9 independent attempts with 4 successes:
1/1 at temperature 0, 0/4 at 0.3, and 3/4 at 0.7. This is a mixed-temperature,
single-scene diagnostic, not a held-out success rate. No training ran. The
5-step greedy trajectory was independently replayed locally: all five states
matched and the whole-tube lift was 0.0679 m, with zero invalid/prose actions.
SAM masks were unavailable in three later observations; RGB, proprioception,
and the initial visual reference remained available. This is not evidence of
continuous tracking or generalization. The other three successful trajectories
were present in server logs but absent from that uploaded archive.

Use a separate Git checkout on the networked CPU instance:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong
git clone https://github.com/Vannico233/ZAP-AI4S.git ZAP-AI4S-git
cd ZAP-AI4S-git
../ZAP-AI4S-offline-20261106/.venv-linux/bin/python scripts/link_shared_runtime.py \
  --existing ../ZAP-AI4S-offline-20261106
```

The helper validates all destinations before linking `.venv-linux`, `.venv-sam`,
`.runtime`, `runs`, and `assets/sam3.1`. It installs/downloads nothing and never
replaces an existing different destination. KEEP the old offline directory:
its environments, assets and runs remain the shared storage. The checkout has
its own tracked source files and Panda assets. `offline_bundle.py verify` is for
the archived offline distribution, not for this Git checkout.

On later updates, when no experiment is running, use `git pull --ff-only` in
`ZAP-AI4S-git` on the CPU instance. On the GPU instance, use that same checkout:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git
bash scripts/start_lab_all.sh runs/lab_success_first/seed0_v3_1 \
  --sam-url http://127.0.0.1:8002 \
  --success-source runs/lab_pick_debug/seed0_v3
```

The visual policy/prompt is unchanged from v3. Compatible completed probe
successes are copied into the new run. Model, seed, prompt, environment/action
versions, raw-response/action consistency and provenance are checked. Only
subsequent successful physical replay and deduplication admit them to the gate;
`IMPORT.json` candidate count alone is not an admission count. Probe trials
cannot substitute for the 10 fresh qualification trials. All four tasks still
need 10 unique verified successes each and at least 3/10 fresh successes per
task before any optimizer step. No manual diagnosis enters the training set.

The collector now includes up to 10 successful episodes per task, an independent
failure if available, and verified videos/replay reports rather than only an
index. To collect this new run:

```bash
.venv-linux/bin/python scripts/collect_success_first.py --run runs/lab_success_first/seed0_v3_1
```

## Focused pick debugging (v3, 2026-11-07)

The v2 review archive was captured mid-search: 74 completed episodes, zero
full successes. The first pick trace moved upward without closing the gripper
until step 36. Its raw prose claimed a grasp that had not occurred. Three
provided invalid responses contained a single complete JSON action after prose.

Changes in v3:

- Accept a SINGLE trailing JSON action after explanatory prose, keeping the raw
  response and parse audit. Reject arrays, multiple alternatives, duplicate keys,
  trailing prose, truncated objects and non-finite JSON constants. Never invent
  coordinates, gripper commands or a missing field. The old strict baseline
  parser is unchanged. This changes the visual output parsing protocol.
- Feed canonical actions back into history, not unsupported assistant claims.
  Include executed action/feedback history and a frozen initial VISUAL reference.
  Robot opening is explicitly interpreted as an uncertain grasp cue; neither
  simulator contacts nor ground-truth held/success flags enter the policy.
- Retain camera calibration in trajectory audits while omitting full matrices
  from the model decision text; object measurements already use world coordinates.
- Collector selects failed episodes and their images even before stage-level
  diagnostics exist, and includes the launcher log.

Start with a small pick-only probe AFTER the old run has ended:

```bash
bash scripts/start_lab_all.sh runs/lab_pick_debug/seed0_v3 \
  --pick-probe --sam-url http://127.0.0.1:8002
```

Nine independent base-model episodes, fixed train seed 0: one greedy trial on
GPU 0, four trials at temperature 0.3 on GPU 2, four at 0.7 on GPU 3; managed SAM
on GPU 1. No prefix search, optimizer smoke, SFT or RL. Even if all nine succeed,
the probe cannot train. It ends at `probe_complete`, with summaries, candidate
trajectories and verified success videos if available. Probe successes are not
automatically imported into the full four-task gate.

```bash
cat runs/lab_pick_debug/seed0_v3/status.json
.venv-linux/bin/python scripts/collect_success_first.py --run runs/lab_pick_debug/seed0_v3
```

Local evidence: replaying the server's first pick actions reproduced all final
reported physical metrics (failure). A separately labelled manual five-action
diagnostic using ONLY the supplied initial SAM position/axis estimates succeeded.
This demonstrates geometric/actuation feasibility for that initial scene only;
it is NOT a model success or training example. Actual v3 Qwen/SAM success rates
must be measured on the GPU server.

## Success-first four-GPU experiment (v3, 2026-11-07)

The old v1 flow incorrectly allowed warm-up and fixed-scene RL with ZERO
successes. Do not resume that flow. Use a NEW output directory with this patch;
old evidence and checkpoints remain untouched. No new packages or weights.

Run on the GPU instance, from the project directory:

```bash
bash scripts/start_lab_all.sh runs/lab_success_first/fixed_seed0_v3
```

GPU 1 hosts managed SAM; GPUs 0, 2, 3 host base-model policy replicas for
sampling. Stop old Qwen/training processes on those policy GPUs first. The
pipeline now requires its own SAM process, stores `logs/sam.log`, checks its
process and health, and stops if it fails. If port 8001 is already occupied,
stop your previous SAM terminal before launching, or explicitly select a free
local port with `--sam-url http://127.0.0.1:8002`. It never kills external jobs.
SAM inference remains serialized; threaded HTTP handling keeps health requests
responsive. Requests log start/end/error and CUDA allocation. A 120-second queue
limit and 660-second client deadline report failure rather than invent masks.
This improves diagnosis, not proof that the previous service failure is fixed.

Stages and mandatory admission:

1. Scripted physics/camera checks establish environment feasibility only.
2. Base-model independent trials: 3 attempts per task, train scene seed 0.
3. Search the same scene in batches of 16 attempts, at most 4 batches per task.
   Only unfinished tasks continue. New batches use distinct sampling seeds;
   physics scene seed stays fixed. Replay prior model-generated prefixes within
   a batch, without scripted/oracle actions or policy updates. If the first
   batch yields zero verified successes across ALL tasks, stop for debugging.
4. Require at least 10 different successful action sequences PER task. Each
   must have zero invalid actions, no demonstration, visual SAM/RGB-D inputs,
   no adapter, and a full physical replay matching every stored state.
5. Independently run the unchanged base policy from initial states 10 times per
   task, with new sampling seeds and NO prefix replay. Require at least 3 valid,
   replay-verified complete successes per task and no infrastructure errors.
   This is a pragmatic small-run admission rule, not evidence of generalization.
6. Only when BOTH checks pass: optimizer smoke, self-success SFT, 12 fixed-scene
   GRPO updates, then 16 updates across train seeds 0-3. The smoke weights are
   discarded. Every training entry checks the gate and hashes of its evidence.
7. Select base/SFT/fixed-RL/diverse-RL using four-task validation; evaluate all
   five tasks on held-out scenes, and replay successful episodes as videos.
   `rack_transfer` never enters acquisition, qualification, SFT or RL.

No warm-up fallback exists. A failed gate ends at `awaiting_success` and creates
`success_gate.json`, `acquisition_diagnostics.json`, `report.json` and any available
verified success videos. It does not start SFT, RL or the optimizer smoke.
These partial results are sufficient to debug the next batch. Repeating the
same command resumes interrupted work but does not extend an exhausted budget.
New code/settings require a new output directory. After a gate passes, training
continues automatically; no additional approval is needed.

The prompt now explains grasp orientation, surface-coordinate uncertainty,
object/TCP offsets, and revising unreachable poses. This is procedural prompt
assistance, recorded by `prompt_sha256`; it is not a demonstrated trajectory or
a hidden controller. Its effect on Qwen success rates has NOT been measured
locally. The base policy for this release differs from the previous prompt.

```bash
cat runs/lab_success_first/fixed_seed0_v3/status.json
cat runs/lab_success_first/fixed_seed0_v3/success_gate.json
tail -n 30 runs/lab_success_first/fixed_seed0_v3/launcher.log
.venv-linux/bin/python scripts/collect_success_first.py
```

The collector packages counts, service logs, top candidate trajectories and
initial input images into `success_first_review_*.tar.gz` in the current directory.
It does not perform inference or modify experiments. Large omitted files are
listed. Upload this archive for diagnosis, especially at `awaiting_success`.

Use `bash scripts/lab.sh all --plan` for budgets. The upper bound is 615 model
episodes, excluding interrupted retries; acquisition can stop much earlier.
The following manual CLI sections document individual components. Direct
`train`, `grpo` and `sft` commands bypass the coordinator and must NOT be used
for this success-first campaign. The supported command is `start_lab_all.sh`.

## 1. Upload and prepare (networked CPU instance)

Upload `ZAP-AI4S-lab-v1-patch.tar.gz` and its SHA256 sidecar into:

```text
/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong
```

The cumulative patch overlays the existing offline project, preserves its venv,
models and runs, and adds the pinned SAM wheel and extra Linux/Python 3.11
dependency wheels. It requires the existing base offline v2 bundle's 42 wheels.
If using a fresh machine, extract the base offline v2 bundle first.

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong &&
sha256sum -c ZAP-AI4S-lab-v1-patch.tar.gz.sha256 &&
tar -xzf ZAP-AI4S-lab-v1-patch.tar.gz &&
cd ZAP-AI4S-offline-20261106 &&
.venv-linux/bin/python scripts/offline_bundle.py verify &&
bash scripts/lab.sh prepare &&
bash scripts/lab.sh physics
```

SAM gets a separate `.venv-sam`: Torch/CUDA is copied locally from the shared DiG
Python 3.11 installation and the other packages install from verified local
wheels. This needs approximately another 7-8 GiB of shared storage. It does not
change `.venv-linux`, DiG, or system Python 3.13.5. The upstream README recommends
Python 3.12+, while this pinned package declares Python >=3.8; this bundle uses
the user's shared Python 3.11 and includes a mandatory native import check.
The official SAM source revision is
`f1647e8d0c1d280991f4d1b7d974c2704537fe66`.

CPU preparation and `physics` explicitly disable OpenGL. Renderer-dependent
tests are reserved for the GPU instance; `check` there performs the mandatory
camera/SAM preflight. A CPU instance does not need an EGL display for physics.

## 2. Obtain SAM 3.1 weights once (no login needed)

If you upload the separately prepared `ZAP-AI4S-sam3.1-weights.tar` and its SHA256
sidecar, use the following commands from the shared parent directory and skip
the download below:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong &&
sha256sum -c ZAP-AI4S-sam3.1-weights.tar.sha256 &&
tar -xf ZAP-AI4S-sam3.1-weights.tar
```

This creates `ZAP-AI4S-offline-20261106/assets/sam3.1` with the original checkpoint,
model license and provenance manifest. The archive is separate from code updates.

The default source is the public `facebook/sam3.1` repository on ModelScope:
https://modelscope.cn/models/facebook/sam3.1 . No ModelScope SDK, login or token
is needed. The original 3,502,755,717-byte `sam3.1_multiplex.pt` is used, not a
third-party FP16 conversion. The downloader pins the repository revision and
SHA256, supports HTTP range resume, and publishes the final filename only after
verification. Its LFS pointer matches Meta's official Hugging Face listing.

On the networked CPU instance:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-offline-20261106 &&
source .venv-linux/bin/activate &&
python scripts/download_sam.py --output assets/sam3.1
```

If the server download is unreliable, run the same Python download script on a
networked local machine using any Python 3.10+ installation, then upload the **whole**
`sam3.1` directory to the project's `assets/sam3.1`. The download is resumable;
it records the official model revision and a SHA256 manifest. The GPU service
checks the checkpoint hash before loading and requires complete weight loading.
Do not rename or substitute SAM 3.0 weights. The image-only SAM 3 builder is not
interchangeable with the SAM 3.1 multiplex architecture.

The default download bypasses stale proxy settings. Add `--use-proxy` only if
your network requires a working proxy. A Hugging Face fallback is available as
`--source huggingface`; that source does require model access and `hf auth login`.
Read the included SAM LICENSE. Never paste account tokens into chat.

## 3. Start two services on the SAME GPU instance

Terminal A (SAM, physical GPU 1):

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-offline-20261106 &&
bash scripts/lab.sh sam
```

Terminal B (Qwen, physical GPU 0): stop any old Qwen server on port 8000 with
Ctrl+C first, then:

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-offline-20261106 &&
bash scripts/lab.sh model
```

Wait for both `Ready` messages; keep both terminals running. The shared model
directory defaults to the previously supplied DiG `assets/model` path. Use
`bash scripts/lab.sh model --model /absolute/model/path` if necessary. These are
localhost services; CPU and GPU instances do not share localhost.

Terminal C (preflight, GPU EGL rendering):

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-offline-20261106 &&
bash scripts/lab.sh check
```

This checks all physical tasks, actual camera visibility, RGB-D rendering, SAM
detections of each task's target/goal queries and a real image-bearing Qwen request. It saves RGB, depth, annotated
masks and a report in `runs/lab_v1/preflight_*`. Inspect the annotated images
before collecting a long run. Missing EGL support, service/API errors, bad weight
loads or malformed model responses stop the check. Empty SAM detections are
reported separately as perception warnings, as described below. `MUJOCO_GL=egl` is selected
by the wrapper; `TUBE_MUJOCO_GL` can explicitly select an installed alternative.
Do not install OSMesa or change drivers preemptively.

If `libEGL.so.1` is missing but `libEGL_nvidia.so.0` loads and the NVIDIA vendor
JSON exists, use the separate `ZAP-AI4S-egl-jammy-patch.tar.gz` for this Ubuntu
22.04 x86_64 server. It adds verified Ubuntu libglvnd 1.4.0-1 EGL/OpenGL/dispatch
libraries under `.runtime/egl-jammy`, including licenses and source hashes.
`lab.sh` validates these files and prepends their directory to `LD_LIBRARY_PATH`
before starting Python. It does not install system packages or NVIDIA drivers.
It requires glibc >= 2.34 and an already-working NVIDIA EGL vendor library.

After extracting the patch over the existing offline project, run:

```bash
bash scripts/lab.sh render-check &&
bash scripts/lab.sh check &&
bash scripts/lab.sh sample
```

`render-check` does not contact SAM or Qwen. It saves both camera PNGs and the GL
vendor/renderer/version in `runs/lab_v1/render_*`. Keep services A/B running.

For the pinned SAM source, the application applies a narrow runtime adapter for
the unsupported `offload_state_to_cpu=False` session argument. When upgrading
with `ZAP-AI4S-sam-session-fix.tar.gz`, restart Terminal A after extraction and
wait for `Ready`; its health JSON should include `session_init_compat: true`.
Terminal B does not need restarting. No checkpoint or dependency download is
required. Future SAM errors include their actual message in Terminal C and the
full traceback in Terminal A.

### Missing masks and bounded appearance-query retries

`sam_query_retry_v1` first uses the original task queries in both cameras. Only
if a concept has no usable mask in either view does it try up to two predefined
appearance descriptions. For example, the simulated flat teal target is queried
as `teal tray`, then `cyan rectangle`, then `teal rectangular plate`. Yellow and
blue racks can be retried as colored blocks/containers. Arbitrary model-specified
queries are passed through unchanged. Retries use the same RGB and depth, no
oracle coordinates, no color-threshold masks and no fabricated detections.

Each detection retains both its canonical `query` and actual
`segmentation_query`. `perception.json` records every attempted phrase, camera,
valid-mask count and remaining missing queries. Model observations explicitly
contain `missing_queries`; RGB and the `change_query` tool remain available.
Mask availability is not proof of correct identity: inspect annotated images.

The check reports **all** task results. If services work but masks are missing,
it reports `READY_WITH_PERCEPTION_WARNINGS`, sets `perception_complete=false`,
and permits policy sampling with the missing observations. It does not report
perception as passed, fill in poses or alter the physical success criterion.
Service/inference errors still stop execution. To require complete initial mask
coverage, pass `--strict-perception`; this exits nonzero after reporting all tasks.

After physical checks have already passed, use the shorter visual-only check:

```bash
bash scripts/lab.sh vision-check &&
bash scripts/lab.sh sample --attempts 1
```

This probes all five initial scenes, then samples one model trial per training
task (four trials). It does not rerun the scripted physical trajectories.
Reports are in `runs/lab_v1/vision_check_*`, with `physical_checks=not_run`.
Per-episode and summary files record the perception version and the number of
observations with missing masks. This changes the perception/prompt protocol;
retain the recorded version and prompt hash when comparing experiments.

## 4. Fixed-scene baseline: 4 tasks x 8 attempts = 32 trials

Action protocol `rpy_rounding_v2` corrects the old strict +/-pi boundary bug:
RPY values within 0.0001 radians of +/-pi are executed as exact +/-pi and the
normalization is recorded in feedback. XYZ limits are unchanged; larger angular
violations, missing RPY and non-JSON responses still fail with specific feedback.
Robot orientation observations now use six decimal places. Results record the
action protocol. Historical results lacking that field replay with
`strict_rpy_v1`, preserving their original action rejection behavior.

The initial four-task model trial was affected by the old validator rejecting
its own four-decimal pi representation (`3.1416`). Retain those logs as a bug
diagnostic; rerun a fixed-seed pilot before drawing policy-performance conclusions.

```bash
bash scripts/lab.sh sample
```

Each task uses train scene seed 0; model sampling RNG differs between attempts.
There is **no human demonstration** in the prompt. Default temperature is 0.8,
budget 40 actions, and each observation contains both RGB views. A service error
stops the run after saving its diagnostic result instead of generating many
misleading zero-score episodes. Results are saved after every episode.

```bash
find runs/lab_v1/sample -name summary.json -print -exec cat {} \;
```

If helpful, repeat another fixed scene with `bash scripts/lab.sh sample --seed 1`.
Do not call the seed-0 result a generalization score.

## 5. Mine model successes using prefix search

```bash
bash scripts/lab.sh search
bash scripts/lab.sh export
```

Search defaults to 32 continuation trials per task on the same train scene. It
restores both physical state by action replay and the original model context;
it refuses a continuation if replay differs. Search uses judge progress for
node selection, but never sends that privileged score to Qwen. Only model
prefixes are used. Search yield is recorded separately from independent policy
success rate. The export command prints its concrete output file path.

Export includes only visual, model-generated, successful, error-free train
trajectories with zero invalid actions. It rejects assisted/manual/held-out
data and deduplicates repeated state-response examples. Images are embedded in
the exported JSONL, so moving it does not break image paths. A no-success export
fails without creating an empty training file.

## 6. SFT, then explicit RL

Substitute the **actual printed data path** in the first command below. Training
uses physical GPU 2 by default and a separate in-process Qwen copy; SAM stays on
GPU 1. It does not use the HTTP Qwen server for policy updates. The vision encoder
is frozen; image-conditioned language-model LoRA weights are trained.

```bash
bash scripts/lab.sh sft --data data/YOUR_EXPORTED_FILE.jsonl --epochs 1
```

After SFT, use the actual saved `epoch_001` directory:

```bash
bash scripts/lab.sh grpo --adapter checkpoints/YOUR_SFT_RUN/epoch_001 --updates 4 --group-size 4
```

The four-update pilot trains once on each task and saves an adapter after every
update. Increase to `--updates 20 --group-size 8` only after this pilot works.
Do not run base-model GRPO before the success gate. Use the coordinator above.

## 7. Compare trained policy and held-out composition

Before training, run `bash scripts/lab.sh eval --seed 20` to store the base-model
test results. After training, stop Terminal B and restart it with the actual
adapter directory:

```bash
bash scripts/lab.sh model --adapter checkpoints/YOUR_GRPO_RUN/update_0004
```

Then rerun the identical evaluation command in Terminal C:

```bash
bash scripts/lab.sh eval --seed 20
```

This evaluates all four tasks plus `rack_transfer`, four independent attempts
per task. Run seeds 21-24 without changing the policy/prompt for a larger held-out
check. The split offsets ensure train seed 0 and test seed 0 are different scenes.
Each run records the service's adapter path; keep both before/after runs. Four
attempts are a smoke comparison, not a precise generalization estimate.

## 8. Replay one successful visual episode as video

No SAM or Qwen inference is needed for replay:

```bash
bash scripts/lab.sh replay --episode runs/lab_v1/sample/YOUR_RUN/try_000 --output runs/lab_v1/video_success_01
```

The output includes a two-view MP4 and a state/result comparison report. Replay
requires a working EGL renderer on the GPU instance. Alternatively, send the
episode's `trajectory.jsonl` and `result.json` back for local rendering.

## Validation and limitations

Local validation is recorded separately in `LAB_VALIDATION.md`. A successful
scripted physics diagnostic is **not a Qwen result**. Random tiny-model gradient
tests verify code paths, not actual 4B learning. The target H100 machine is
not accessible from this workspace, so the
live SAM + 4B + EGL preflight and GPU training pilot remain server acceptance
steps. Never report them as passed before their logs exist.
