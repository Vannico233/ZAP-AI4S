# Four-task visual laboratory reproduction

This is an independently implemented, small-scale ZAP-method reproduction for
Qwen3.5-4B, not a reproduction of the paper's reported scores. All commands and
program output are English/ASCII. The original `tube_agent.run` baseline remains
available. New experiments use `tube_agent.lab` and `runs/lab_v1`.

## Scope and honest boundaries

| Component | Implementation |
|---|---|
| Training tasks | `pick`, `transport`, `insert`, `unscrew` |
| Held-out composition | `rack_transfer`: blue source rack to yellow destination rack; forbidden in the train split |
| Robot | Single Panda; XYZ + full RPY and gripper opening |
| Vision | Two calibrated RGB-D cameras; native **SAM 3.1 multiplex checkpoint**, official predictor, text queries; 3D visible-surface statistics from masks and depth |
| Policy input | Annotated RGB, estimated surface geometry, robot proprioception and execution feedback. No simulator object coordinates or success/reward fields |
| Data mining | Independent fixed-scene stochastic trials; optional deterministic prefix replay and UCB tree search over model-generated actions |
| Training | Multimodal self-success LoRA SFT; online group-relative clipped policy optimization, terminal ladder rewards, invalid-turn negative advantages |
| Differences from paper | MuJoCo rather than Isaac Sim; 4B rather than 27B; four tasks rather than forty; synchronous SAM snapshots at action boundaries rather than asynchronous temporal perception; LoRA and a small single-GPU GRPO implementation, beta=0 |
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
For a short GRPO warm-up directly from the base model (the paper's success-mining
stage), omit `--adapter`; it is an explicit action, never started by sampling.
`--scene-seeds 0 1 2` expands the training scenes. `rack_transfer` is not accepted
by the training CLI. To continue from an adapter, pass `--adapter` to a new run;
this warm-starts weights with a new optimizer, **not** an exact optimizer resume.

Reward ladder (one terminal score, not cumulative collection of repeated bonuses):

| Progress | Score before costs |
|---|---:|
| Approach target | 0 to 0.15 |
| Contact grasp | at least 0.20 |
| Grasp and lift | up to 0.55 |
| Transport toward goal | up to 0.90 |
| Align and insert | up to 0.90 |
| Grasp, unscrew, then separate cap | up to 0.90 |
| Full stable task success | 5.00 |

Subtract 0.01 per action and 0.05 per rejected action. GRPO computes relative
advantages within each same-scene group; an invalid turn gets advantage -1
regardless of final success. Equal-reward groups yield zero group advantage.
The loss uses clipped token probability ratios (epsilon .2), equal rollout
weighting, and beta=0 (no reference KL). Stored generation token IDs ensure
training scores the actual sampled responses, including EOS/truncation.

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
