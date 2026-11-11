# EGL binding and real-sampler check

This directory can be exported from a Git revision on the CPU instance and run from shared storage on the offline GPU instance. Keep the original experiment checkout at the revision recorded in its provenance. The separate Git revision identifies the launcher, GPU placement checks and frame validation.

Run from the original project directory, with the existing Linux and SAM environments and original model path:

```bash
.venv-linux/bin/python "$TOOLS/run_real_smoke.py" \
  --root "$PWD" --run "$RUN" --model "$MODEL" --out "$DIAGNOSTICS/smoke"
```

`TOOLS` is the absolute path to this exported directory. `RUN` is the original `v1_launchfix` output and `DIAGNOSTICS` is a separate output directory. The check loads warm-up checkpoint 9, starts the native SAM service on GPU 1 and generates actions with three Qwen workers. The temporary SAM service exits after the check. Inspect the three worker reports, the camera images, and the overall `smoke/summary.json`. A passing sampler check describes the image and action-generation path. Task success is measured by the experiment's fixed evaluation.

To use the same placement for the existing experiment, after reviewing the smoke:

```bash
bash "$TOOLS/with_mapped_egl.sh" "$DIAGNOSTICS/training-bindings" \
  bash scripts/start_tube_curriculum.sh "$RUN" --model "$MODEL"
```

The experiment owns its run lock, checkpoint validation and resume counters. Renderer binding records belong to the new audit directory. Record the launcher revision, the checkpoint and the disposition of any partial-round rollouts with the run when changing the runtime placement.
