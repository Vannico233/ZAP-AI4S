# Sources and licenses

## MuJoCo Menagerie / Franka Emika Panda

- Source: https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda
- Revision: `98a70c569c4c1b6da467a20012f766ee1f8dc521`
- License: Apache-2.0; full license in `assets/panda/LICENSE`.
- Bundled: robot XML and 12 collision meshes. Visual-only meshes were removed;
  collision meshes are used for visualization. The runtime adds a TCP site,
  fingertip friction parameters, gravity/bias compensation, a tabletop, a tray,
  and procedurally generated rigid test tubes. No downloaded task demonstrations.

## Design references

- ZAP: https://arxiv.org/html/2610.35817v3
- Orientation math checked against the project's public demonstration:
  https://zero-real.github.io/assets/js/orientation-widget.js (`orient`).
  Retrieved 2026-11-07; SHA256
  `207094aae53d473e56ff3c87a7d78fb40b8c17ecaee48386277269bbba289668`.
  `zap_pose.py` independently expresses its rotation convention and adapts
  physical slot/tool axes to Panda. Eight reference vectors were evaluated
  directly from the official JavaScript with Node for regression checks.
  No project-site JavaScript is bundled or executed by the lab runtime.
- Open-RUA uni-robot-agent: https://github.com/Open-RUA/uni-robot-agent
  (MIT). Consulted `docs/architecture.md`, `docs/sdk.md` and
  `uni_robot_agent/tools/gripper/motion.py` for the separation of observations,
  pose commands, measured execution feedback and official task outcomes.
  This implementation is independently written; no Open-RUA Python source is vendored.
  The v4 audit additionally inspected `tools/gripper/pose.py`, `contracts.py`,
  and integration gripper geometry at revision
  `f3e0ec006295f74321b6a349c9b47ec82071eb28`. Open-RUA's RPY interface is not
  ZAP's semantic orientation representation; these references are distinct.

The original two-task baseline is a simulator/task and self-success SFT prototype.
The new `tube_agent.lab` independently implements a scaled visual ZAP recipe:
queryable SAM masks and depth, model-prefix search, self-success SFT and online
GRPO with staged rewards. It changes the simulator, model size, task suite,
perception scheduling and optimizer scale; it does not reproduce published scores.

## SAM 3.1

- Native source: https://github.com/facebookresearch/sam3
- Pinned revision: `f1647e8d0c1d280991f4d1b7d974c2704537fe66`.
- License: SAM License, retained in the built wheel's license metadata.
- The offline extension includes an unmodified wheel built from that revision,
  including the upstream tokenizer BPE resource. No SAM Python source is copied
  into this repository's application modules.
- Checkpoint: `facebook/sam3.1/sam3.1_multiplex.pt`, downloaded from the public
  ModelScope mirror at revision `616acbee0b9ed4177f1f389e3c13594a0a1f6398`.
- Source: https://modelscope.cn/models/facebook/sam3.1 ; official model listing:
  https://huggingface.co/facebook/sam3.1 . The SHA256/size reconstruct the exact
  LFS pointer Git blob `88589b6b0bee708b57010d3ed6f413476e4be255` in that listing.
- The download retains the model LICENSE, README and revision/hash manifest.
  Weights are not tracked in Git.
- Uses fresh single-frame native predictor sessions. This is synchronous visual
  detection at action boundaries, not a reproduction of asynchronous tracking.

## Optional Ubuntu EGL loader runtime

- Ubuntu 22.04 amd64 packages `libegl1`, `libopengl0`, `libglvnd0`, version 1.4.0-1.
- Source: https://packages.ubuntu.com/jammy/libegl1 and
  https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/ .
- The optional offline patch extracts three unmodified shared-library binaries
  into `.runtime/egl-jammy/lib`; upstream copyright/license files and package
  download SHA256 records are retained alongside them.
- The patch contains no NVIDIA driver, replaces no system files and requires
  the host's existing NVIDIA EGL vendor implementation.
