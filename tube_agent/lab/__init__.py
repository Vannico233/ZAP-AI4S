"""Visual laboratory manipulation experiments; separate from the legacy baseline."""
import os
import sys
if sys.platform=="linux":
    os.environ.setdefault("MUJOCO_GL","egl")
