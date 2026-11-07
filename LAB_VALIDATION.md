# Lab validation - 2026-11-07

## Completed v4 probe: evidence review and controlled diagnostics

The five uploaded parts reconstructed the server archive with SHA256
`9c02f11e83449527f087b39b2d1ea4a91729511bdd51d2f7dbc26d141c7441ee`.
The archive contains the completed fixed-seed pose probe, 12 model episodes,
446 actions, zero task successes, zero infrastructure errors and no training.
The held-out rack-transfer task was not sampled. The server's scripted pose
calibration passed all five tasks with zero invalid actions; that is not a model
success result.

| Model task | Episodes | Successes | Actions | Rejected actions | IK rejections |
|---|---:|---:|---:|---:|---:|
| Pick | 3 | 0 | 86 | 80 | 80 |
| Transport | 3 | 0 | 120 | 58 | 53 |
| Insert | 3 | 0 | 120 | 65 | 63 |
| Unscrew | 3 | 0 | 120 | 7 | 2 |

198 of 210 rejected actions reported no continuous IK path. Four others were
below the workspace floor; eight were unsupported/malformed actions (including
five parse failures). Pick at temperatures 0 and 0.3 repeated a rejected pose
40 times each. The 0.7 pick rollout reached valid poses but never closed its
gripper and submitted an empty-hand lift. Transport at temperatures 0 and 0.3
established two-finger contact, then tried low, horizontal destination poses
without first lifting. Insert had one contact-containing trace but no successful
lift or seating. Unscrew's greedy rollout repeated an already reached pose 40
times without closing; the 0.3 rollout only closed at step 40. No cap was released.

Initial rendered views and SAM measurements locate the tube in both cameras.
After approach/grasp, some views lose masks under the hand; no rollout uses
change_query. These observations support action-selection/recovery failures,
with additional occlusion, rather than a service outage. They do not prove all
poses requested by the model are physically impossible or the controller is
free of numerical/planning limitations.

Local physical counterfactuals kept seed 0, geometry, rewards and the executor
unchanged. The local source digest matches the server record exactly:
`6a9fe865581b3ef7583b211b80aa7d06ede6611a2586966451464dc21e3ae73d`.
The original first pick action was rejected at waypoint 60/60, as on
the server. Changing only its tilt to top-down made that same XYZ reachable
(position error 0.29 mm). A five-action engineer-authored pick sequence using
the saved SAM coordinates succeeded, with 12.27 cm lift and zero rejections.
The first three model transport actions followed by an engineer-authored lift,
carry, release and withdraw sequence also succeeded, with 1.79 seconds stable
placement and zero rejections. Both local action-replay videos passed every
saved-state comparison and the success check; their frames explicitly label
them SCRIPTED PHYSICS DIAGNOSTIC (NOT QWEN).

The fixed JSON command files are under `examples/lab_diagnostics/`. They are
manual diagnostics, not demonstrations to silently add to the model prompt,
not autonomous Qwen successes, and not eligible self-success training data.
No Qwen sampling, SFT or RL was performed during this local evidence review.

## Foreground progress monitor

The user started the bounded `seed0_v4` pose probe and requested continuous
progress until completion. Added a standalone read-only monitor and optional
launcher `--follow`. **18 local tests passed** for complete/partial/truncated log
lines, bounded historical tails, per-stage result counts, live action output,
completion/error/interruption exit codes, waiting for launcher cleanup, stale
status after a restart, missing final status, and matching the actual output
directory rather than a reused PID, and quiet-worker notifications despite
launcher heartbeats. Python 3.11 syntax and CLI help passed. Git Bash syntax and
a dummy-worker/monitor launch check passed, including paths with spaces, removal
of `--follow` from worker arguments and propagation of the monitor's exit code.
Linux /proc and liveness operations are simulated in the Windows test suite;
this is monitoring validation, not a new Qwen experiment result.

## User-requested experiment closeout

Server correction: the original helper stopped at its pidfd capability check
before sending any signal. The server's Python does not expose at least one of
`os.pidfd_open` and `signal.pidfd_send_signal`; the previous assumption that the
reported Ubuntu/Python versions guaranteed those APIs was incorrect. Python's
[os documentation](https://docs.python.org/3.11/library/os.html#os.pidfd_open)
and [signal documentation](https://docs.python.org/3.11/library/signal.html#signal.pidfd_send_signal)
describe platform-dependent availability. No environment reinstall is needed.

The helper now uses an immediately identity-checked ordinary SIGTERM fallback
for missing APIs or supported capability errors; it logs the chosen backend.
**32 closeout tests passed** after this correction, including absence of either
or both APIs, unavailable/blocked pidfd calls, rechecking after a failed pidfd
call, stale/changed/exited processes, cleanup and exact-run shutdown. Unrelated
resource errors remain visible. Process syscalls are still simulated locally
on Windows; native execution on the user's GPU host has not been observed.
The fallback does not claim pidfd's atomic protection against PID reuse.

The current campaign is being stopped; no additional experiment is authorized
by the closeout helper. Local tests cover output-directory scoping, stale PID
rejection, parent/child shutdown, unrelated service exclusion, timeout behavior,
PID identity checks with pidfd, exit-before-signal races, Linux proc stat parsing,
stage-separated evidence counts, original file preservation, video provenance,
archive hashes/LF checksums, bounded file/log collection and offline-snapshot
labeling. Linux process operations are mocked on Windows; no remote GPU process
has been stopped from this workspace. The helper requires execution on the
actual GPU instance and reports its observed shutdown scope.

Closeout validation: **12 tests passed**, Python 3.11 syntax and CLI help passed.
An end-to-end collection of the user's previously downloaded pick evidence
completed without loading a model or rendering: one available complete episode
and its existing replay-verified success video were archived, with no files
skipped. This partial local evidence package is explicitly labeled an offline
snapshot; its one episode does not replace the full server probe's 9-trial count.

## Source-audited ZAP pose / reward / prefix correction (v4)

The uploaded v3.1 summary snapshot contained 72 completed new episodes, one task
success and zero eligible new successes. It does not include the imported
legacy successes. Follow-up action evidence showed: pick closed at z=0.04 m,
measured zero jaw opening, then lifted an empty hand; the next attempt reused
that four-action failure prefix. Transport similarly closed above the object
and later descended with closed jaws. Insert eventually lifted but did not seat
the tube. Unscrew closed at z=0.35 m above a cap near z=0.13 m, never released
the thread and included format errors. These are motion/recovery failures;
changing the angle representation alone is not evidence of a model improvement.

The corrected optional `zap_pose_v1` uses the original project's semantic
orientation convention, verified against its public JavaScript, SHA256
`207094aae53d473e56ff3c87a7d78fb40b8c17ecaee48386277269bbba289668`.
Physical slot/tool axes match eight reference cases, including both sides of
horizontal tilt and inverted orientation. Panda's body axes are explicitly
adapted. Full path IK failure is checked before dynamics are advanced.

Five fixed-seed physical diagnostics passed through the new API: pick,
transport, insert, unscrew and rack_transfer, with zero rejected actions.
These are scripted, privileged calibration checks, NOT Qwen successes or
training data. Completion is held for 0.5 s; the task geometry stays explicit.

Reward checks cover the terminal-state approach/grasp/lift ladder, no repeated
step accumulation, and loss of lift credit after release. Prefix tests cover
same-policy reuse, rejection of manual/other-task/old-protocol sources, rollback
before a judge-confirmed empty closure, and pruning after repeated failures.
Visual-loop tests use a no-mask test double, not a SAM accuracy claim. Both
protocols retain gate tests for zero/duplicate/missing-task/bad-replay and low
independent success cases. The 12-episode pose probe cannot enter training.

Final full local regression: **181 passed, 1 skipped**, 131.03 s. A subsequently
added physical new-probe import/replay admission test passed separately (1/1).
Python 3.11 syntax passed for 58 source/script/test files; `git diff --check`
passed. The skipped test is Windows distributed transport; native H100/NCCL
and actual Qwen3.5-4B + SAM success under this protocol remain server checks.
No new model success rate or RL improvement is claimed. The server should run
the bounded four-task pose probe before starting another acquisition campaign.

## Server pick success and Git migration (v3.1, 2026-11-07)

The uploaded completed probe reports 4 successes / 9 episodes, zero errors and
six invalid actions. Temperature strata: 0 -> 1/1; 0.3 -> 0/4; 0.7 -> 3/4.
Weights were untrained and no demonstration or prefix was used. This diagnostic
mixes decoding settings and uses one training scene, so it is not an independent
held-out success-rate or RL-improvement claim.

The archive contains one full successful trajectory (attempt 200000). Its five
saved actions replayed locally with all state_matches=true, success=true,
whole-tube lift 0.0679 m and held=true. All five raw responses were direct JSON;
there was no prose extraction. Three later observations lacked SAM masks;
RGB, proprioception and the initial measured reference were still inputs.
The remaining three successes are supported by server logs, pending independent
trajectory replay. The improved collector now includes multiple successes,
a failure and verified videos/replay reports.

v3.1 preserves the v3 policy code and prompt. Compatible completed probe traces
can be copied into a new acquisition, but only replay-verified, deduplicated
successes enter its existing gate. Imported episodes never replace fresh
qualification trials. Incompatible prompts, adapters, models, scene seeds,
action protocols and changed raw-response/action pairs are rejected/skipped.

Local verification: **150 passed, 1 skipped**; Python 3.11 syntax passed.
The import path was additionally exercised on the real supplied trajectory:
one verified pick success, training gate remains false. Shared-runtime linking
was tested for its complete plan and no-overwrite behavior; actual Linux
symlinks remain a CPU-instance deployment step. No package reinstall is needed.


## Visual action evidence and focused pick probe (v3, 2026-11-07)

Evidence: the uploaded 08:55:55 UTC archive was captured during acquisition,
contained 74 completed episode log records and no full successes. The collector
had omitted trajectories because its stage-level diagnostic index was not yet
written. This collector dependency is removed.

The user supplied the first input and three raw invalid responses from pick
attempt 0. All three contained one complete action after explanatory prose.
The recorded first episode did not close the gripper until step 36, but its
prose asserted a grasp. Replaying all saved actions locally reproduced every
reported final physical metric. A separate MANUAL five-action sequence using
only supplied initial SAM center/axis measurements achieved a 0.1644 m lift;
it is an execution diagnostic, not Qwen exploration or SFT training evidence.

Parser v3 accepts exactly one complete trailing JSON object, records the original
text, rejects ambiguous/multiple/truncated/duplicate-key outputs, and does not
repair coordinates or supply missing fields. History retains canonical actions
and executed feedback, plus a fixed initial visual reference. Current opening
is an explicitly uncertain cue, never an oracle grasp flag. Per-episode
`leading_prose_responses` and `parse_failures` expose extraction frequency.

Local regression: **139 passed, 1 skipped** (Windows distributed transport),
including parser boundaries, history/reference integrity, collection during a
failed running experiment, no-training probe mode, and prior gate/physics/LoRA
checks. A separate episode smoke verified raw text preservation and the new
parse counters. Python 3.11 syntax validation passed. Native SAM/4B/H100 gains
are not claimed; the next run is nine independent pick episodes, no training.


## Success-first correction (2026-11-07, supersedes v1 scheduling)

The prior automatic pipeline violated the requested acquisition-before-training
order: eight warm-up updates and one fixed-scene update occurred with zero
complete visual successes. This release removes that fallback, changes the
prompt, requires a new output path and does not reuse the failed run's adapters.

Default admission: 10 distinct fully successful visual trajectories per task,
zero invalid actions, no demonstrations, base policy only; every trajectory
physically replayed with state comparisons. Then 10 new independent trials per
task must yield at least 3 eligible replay-verified successes and zero errors.
All four tasks must pass before ANY optimizer step, including smoke. Evidence
hashes are rechecked at each training entry. Replay or infrastructure failure
never becomes a training success. No-success first-batch acquisition stops
for debugging; exhausted budgets also stop without training.

Local verification: `pytest -q --basetemp=.bootstrap-success-first-full`
returned **125 passed, 1 skipped**. The skip is Linux distributed transport
validation unavailable on the Windows host. Tests cover zero successes,
duplicate sequences, a missing task, divergent replay, low independent yield,
positive stage ordering, altered source evidence, a direct training bypass
attempt, distinct batch sampling seeds, physical replay and the existing
physics/vision/LoRA regressions. Loopback HTTP tests ran outside the sandbox.
Python 3.11 syntax checks passed. Local tiny-model gradient tests are NOT
full Qwen3.5-4B or H100 training acceptance.

SAM now has managed process logs and request start/end/error records. HTTP
health can respond during serialized detector inference. This addresses missing
diagnostics; it does not establish or prove a repair for the previous unexplained
SAM outage. The new prompt and full native SAM/4B behavior need GPU-server runs.

Everything below is historical validation of earlier releases, not current
success-first scheduling or evidence of visual-policy task success.


## Four-GPU automatic experiment release

Full local regression: **118 passed, 1 skipped** in 87.78 seconds. The skipped
test is a real three-process Gloo reduction: this Windows Torch build reports
`unsupported gloo device`, including with explicit UV transport. No local NCCL
or real four-H100 training result is claimed. The production pipeline instead
requires a real three-rank NCCL + SAM + Qwen visual generation/backward/optimizer
smoke check on the user's GPU instance before spending the mining budget.

The targeted follow-up checkpoint/phase tests also passed (3 tests). On a random
tiny Qwen3.5 multimodal model, checkpointed SFT and GRPO backward work; saving
LoRA and AdamW state, restoring both, and taking the next update matches the
uninterrupted model's response log probabilities within 1e-6. This uses tiny
random weights and synthetic pixels, not evidence of a 4B task success.

Additional coverage includes: gradient sums with unequal per-rank work and one
empty-gradient rank; all-zero-gradient groups; model-only data provenance;
excluding held-out/test/assisted/invalid/interrupted traces; resumable search
without regenerating complete episodes; distinct-success counting; complete
stage sequencing with and without SFT-eligible samples; validation-only model
selection; and never inventing a success video when no success exists. Existing
physical/camera/SAM contract, HTTP and legacy tests remain passing. Shell syntax
and Python compilation checks passed. The PEFT warning about no original model
config occurs in the random tiny-model tests, whose base has no model directory.

Review fixes include atomic episode result writes, bounded shared-prefix storage,
explicit action protocol provenance, preserving historical replay, and keeping
EGL device selection independent of CUDA local ranks. Default limits are 559
model episodes before interrupted retries, with early stops for successful
searches. Scripted feasibility tests and smoke-test actions are excluded from
the SFT data source directories. The existing dependency/weight bundle is reused.

Remaining server-only evidence: real Qwen3.5-4B/NCCL training, convergence on all
four primitives, held-out composition success and success videos from those
actual model runs. The launcher records failures and can resume; it cannot
guarantee convergence or turn a failed model rollout into a success.

## Audit of the first four real visual-policy trajectories

The user supplied 160 recorded actions: 146 were rejected. Of those, 110 had
valid XYZ but a four-decimal +/-pi representation rejected by the 1e-6 angular
tolerance; one combined that issue with a genuinely invalid z=-0.01; 30 were
noncompliant JSON responses and five omitted RPY. The original robot observation
itself supplied `tcp_rpy=[3.1416,-0.0,-0.0]`, exposing an inconsistent interface.
No successful trial can be inferred from these logs or from fixing this bug.

`rpy_rounding_v2` accepts only a 1e-4 neighborhood of +/-pi and records execution
at exact +/-pi, without clamping XYZ or inferring absent orientations. It uses
six decimals for robot orientation observations and reports specific rejection
reasons. All 43 action/lab/perception tests passed in 22.64 seconds. The exact
first pick action from the user's log now reaches its commanded pose and yields
the same physical state as an exact-pi command; true height/range/NaN errors are
still rejected. This is an action execution check, not a new model success.

Historical videos select the old protocol for results without `action_protocol`.
Local replay matched every saved state and feedback for the old pick, transport
and unscrew trajectories. The insert trajectory diverged across the server/local
platforms and must not be presented as an exact local replay; retain the existing
divergence check and render on the originating server when needed.

## Recoverable missing-mask handling and appearance retries

The latest server report progressed through native SAM inference and the
pick-stage image-bearing Qwen check, then stopped because `teal tray` had no
usable detection. The fixed-scene render depicts this target as a thin teal
rectangle, so a successful SAM service request is not a guarantee that the
original noun phrase will produce a mask. No replacement mask or pose is used.

The new `sam_query_retry_v1` retries at most two documented appearance phrases
only after both original views miss a known query. It preserves every mask's
actual text prompt and records unresolved misses. Preflight distinguishes
service failures from recoverable missing observations and reports all tasks;
`--strict-perception` retains an optional complete-coverage gate. A visual-only
entry point avoids rerunning physical trajectories that have already passed.

All 31 lab/perception tests passed in 33.34 seconds. After adding an end-to-end
sampling check, all seven perception tests passed in 2.69 seconds (32 distinct
lab/perception tests). The CLI check confirmed an all-missing observation stays
visible to the model and an immediate finish is recorded as a physical failure,
not a service error or success. Tests cover alias recovery,
bounded all-empty retries, original-query success without extra calls, unchanged
SAM masks, propagation of service exceptions, all-task warning reports and
strict-mode exit status. Their segmenters are test doubles: **actual SAM
recognition accuracy for the new phrases is not locally validated**. The
server's saved annotated images remain the evidence for that assessment.

## SAM 3.1 session API compatibility repair

The server passed H100 EGL rendering and all five scripted physical/visibility
checks. Its first SAM request then exposed a mismatch in the pinned upstream
API: `Sam3BasePredictor.start_session` always passes `offload_state_to_cpu`,
while `Sam3MultiplexTrackingWithInteractivity.init_state` does not accept it.
The application adapter consumes only the unsupported false/default flag;
true CPU-offload requests and unrelated unknown arguments still fail explicitly.
SAM health exposes `session_init_compat`. Upstream wheel files are unchanged.

The unmodified upstream base predictor reproduced the TypeError locally, then
passed start/prompt/close dispatch with the adapter and stub inference. All 25
lab tests passed in 19.66 seconds. This verifies the API repair, not native GPU
segmentation. The client now preserves the service's error text, the SAM server
prints the traceback, and raw RGB-D is saved before a segmentation request.

## EGL loader repair for the target Ubuntu 22.04 instance

The user's server diagnostics found that `libEGL_nvidia.so.0` and
`libGLdispatch.so.0` load, NVIDIA's EGL vendor JSON exists, and glibc is 2.35,
but `libEGL.so.1` is absent. The optional EGL patch supplies Ubuntu Jammy amd64
libglvnd 1.4.0-1 client libraries in a private project directory. Three upstream
DEB hashes were verified against packages.ubuntu.com; extracted ELF binaries
were checked as x86_64 and have a maximum GLIBC requirement of 2.34.

The new `render-check` CLI rendered both 640x480 cameras locally on Windows
(NVIDIA RTX 4060 Laptop GPU), and shell syntax / private-file checks passed.
This does **not** validate those Linux binaries on the remote H100. Run
`bash scripts/lab.sh render-check` there before the normal service preflight.

These are local software/physics checks, not Qwen3.5-4B task scores. Structured
results are retained in `validation/lab_v1_local.json`. Run the mandatory server
preflight in `LAB_GUIDE.md` before collecting model episodes.

## Passed locally

- Full regression suite: **88 passed**, 70.70 seconds, one existing PEFT warning
  about a missing config file in the randomly initialized tiny-model test.
  After the final search visit-count fix, all 23 lab tests passed again in
  22.90 seconds, including the new ancestor-count regression (89 distinct tests).
- Graphics-disabled CPU acceptance: **86 passed, 3 renderer tests deselected**,
  57.30 seconds. The renderer tests passed in the full/targeted runs above.
- Runtime: Windows, Python 3.13.5, MuJoCo 3.3.7, Torch 2.10.0 CPU,
  Transformers 5.3.0, PEFT 0.18.1. New Python files also parse with Python 3.11
  grammar; Linux shell entry points pass `bash -n`.
- All five seed-0 scripted physical diagnostics passed. This only establishes
  that the environment is solvable with the low-level motion/gripper interface.
  These privileged diagnostics are excluded from model training/export.

| Task | Split | Actions | Final success | Initial target pixels, view 1 / view 2 |
|---|---|---:|---|---:|
| pick | train | 5 | true | 763 / 499 |
| transport | train | 9 | true | 763 / 499 |
| insert | train | 10 | true | 763 / 499 |
| unscrew | train | 25 | true | 531 / 289 (cap) |
| rack_transfer | test | 9 | true | 507 / 349 |

- Initial images and per-action camera images were rendered. The previous
  overhead elbow occlusion was corrected with two oblique viewpoints. Tube,
  cap, tray and rack openings are visible in the initial images. Simulator ID
  masks are used **only in the visibility diagnostic**, never in policy input.
- Four two-view diagnostic MP4s were rendered through the same replay module
  used for saved model episodes. All saved physical states matched replay
  within `atol=1e-5, rtol=0`; all four success judgments matched. Videos label
  their scripted provenance explicitly. No model inference was used.
- Tests cover premature finish failure, physical cap removal requiring a
  turn, insertion/release stability, invalid pose rejection, bounded reward,
  train/held-out separation, mask encoding and metric depth unprojection,
  absence of oracle coordinates in observations, deterministic search-prefix
  replay/context restoration, and success-export provenance filtering.
- A randomly initialized tiny **Qwen3.5 vision-language model** with actual
  `pixel_values`, `image_grid_thw`, multimodal token types and language-only LoRA
  passed SFT/GRPO backward checks; positive advantage increased response
  likelihood. These tests do not measure 4B learning or H100 memory use.
- The actual public Qwen3.5-4B processor/chat template was checked using two
  640x480 images: 631 prompt tokens, 600 image tokens, 637 tokens including the
  response target. Answer-only scoring and generation-prefix formatting worked.
- Local mock HTTP integration verified image-bearing requests and mask wire
  format. Native SAM request-schema tests verified per-query session cleanup,
  including inference failure. These are interface tests, not SAM inference.
- Downloader tests verify partial-file resume, already-verified file reuse,
  rejection of incorrect byte ranges and refusal to publish a bad checksum.
- ModelScope `facebook/sam3.1` allowed an anonymous HTTP 206 download. The
  pinned 3,502,755,717-byte checkpoint's size/SHA reconstruct the official
  Hugging Face LFS pointer blob `88589b6b0bee708b57010d3ed6f413476e4be255`.
  The full checkpoint subsequently downloaded and passed SHA256 verification;
  a safe `torch.load(weights_only=True, mmap=True)` CPU read found 1,623 tensor
  entries under `detector` and `tracker`, totaling 874,365,676 elements.
  This is a checkpoint structure check, not a native SAM forward-pass result.
- SAM offline dependency metadata resolves for Linux/Python 3.11 using 13 new
  wheels, the existing 42-wheel base bundle, and the user's local Torch 2.10.0
  + CUDA 12.8 / torchvision 0.25.0 closure. No new Torch download is needed.

## Required target-server acceptance

The target server is not remotely accessible from this workspace. These have
**not** passed here and must not be reported as completed GPU experiments:

1. Native SAM 3.1 import and complete checkpoint load in Linux `.venv-sam`.
2. Actual text-query detections of tube/cap/racks on rendered task images.
3. Full Qwen3.5-4B inference from the user's local model directory.
4. Headless EGL rendering with the user's H100/driver installation.
5. Full-size multimodal SFT and online GRPO memory/time/learning behavior.
6. Fixed-seed model success rates and held-out generalization results.

`bash scripts/lab.sh check` checks physics, cameras, every task's initial SAM
queries and one image-bearing Qwen request, saving partial diagnostics even when
a service check fails. `sample` then runs 32 model trials. Do not begin long
training until the preflight, success mining and a short GPU training pilot pass.

## Method limitations

The implementation changes the paper's model size, simulator, task count,
perception scheduling and optimization scale. SAM uses action-boundary snapshots
rather than asynchronous tracking. GRPO uses LoRA, one synchronous policy update
per rollout group and no reference KL. Cap threads are a helical-constraint
approximation with a clamped tube, not mesh-resolved screw contact. Depth is an
ideal simulated sensor. These adaptations are recorded in `LAB_GUIDE.md`.
