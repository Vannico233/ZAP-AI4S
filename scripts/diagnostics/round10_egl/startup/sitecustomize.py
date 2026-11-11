"""Opt-in Python startup hook for the curriculum's rendering workers."""
import os
import sys


if "ZAP_EGL_AUDIT_DIR" in os.environ:
    try:
        from zap_egl_runtime import configure_worker
        configure_worker(sys.orig_argv, os.environ)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        # Python site initialization otherwise logs ordinary exceptions and continues.
        os._exit(78)
