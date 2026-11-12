"""Exercise the GPU job shell entry with a harmless stub launcher."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_submit_selects_the_requested_checkout(tmp_path):
    if os.name == "nt":
        git = shutil.which("git")
        shell = Path(git).resolve().parents[1]/"bin/bash.exe" if git else None
        if not shell or not shell.is_file():
            pytest.skip("Git Bash is required for this shell test on Windows")
    else:
        shell = "/bin/sh"
    wrapper = Path(__file__).resolve().parents[1]/"scripts/submit_tube_curriculum.sh"
    checkout = tmp_path/"new checkout"
    for folder in ("scripts", "configs", ".venv-linux/bin"):
        (checkout/folder).mkdir(parents=True)
    start = checkout/"scripts/start_tube_curriculum.sh"
    start.write_bytes(b'printf "Stub started %s in %s\\n" "$1" "${PWD##*/}"\n')
    (checkout/".venv-linux/bin/activate").write_bytes(b"python() { :; }\n")
    (checkout/"configs/tube_curriculum_bimanual.json").write_text("{}", encoding="utf-8")
    script = checkout/"scripts"/wrapper.name
    script.write_bytes(wrapper.read_bytes().replace(b"\r\n", b"\n"))
    base_env = dict(os.environ)
    base_env.pop("ZAP_PROJECT_ROOT", None)
    base_env["TUBE_RUN_DIR"] = "test run"
    for command, cwd, extra in [
        ([shell, "new checkout/scripts/submit_tube_curriculum.sh"], tmp_path, {}),
        ([shell, "-s"], checkout, {}),
        ([shell, "-s"], tmp_path, {"ZAP_PROJECT_ROOT": "new checkout"}),
    ]:
        result = subprocess.run(command, cwd=cwd, env={**base_env, **extra}, input=script.read_bytes(),
                                capture_output=True, check=True)
        assert b"Stub started test run in new checkout" in result.stdout
    rejected = subprocess.run([shell, "-s"], cwd=tmp_path, env=base_env,
                              input=script.read_bytes(), capture_output=True)
    assert rejected.returncode and b"Expected the bimanual checkout" in rejected.stderr
