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
- Open-RUA uni-robot-agent: https://github.com/Open-RUA/uni-robot-agent
  (MIT). Consulted `docs/architecture.md`, `docs/sdk.md` and
  `uni_robot_agent/tools/gripper/motion.py` for the separation of observations,
  pose commands, measured execution feedback and official task outcomes.
  This implementation is independently written; no Open-RUA Python source is vendored.

The project is an initial simulator/task and self-success SFT implementation,
not a reproduction of the published ZAP results or the complete ZAP RL recipe.
