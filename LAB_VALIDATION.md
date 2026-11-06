# Lab v1 validation - 2026-11-06

These are local software/physics checks, not Qwen3.5-4B task scores. Structured
results are retained in `validation/lab_v1_local.json`. Run the mandatory server
preflight in `LAB_GUIDE.md` before collecting model episodes.

## Passed locally

- Full regression suite: **88 passed**, 70.70 seconds, one existing PEFT warning
  about a missing config file in the randomly initialized tiny-model test.
  After the final search visit-count fix, all 23 lab tests passed again in
  22.90 seconds, including the new ancestor-count regression (89 distinct tests).
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
