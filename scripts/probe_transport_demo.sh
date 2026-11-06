#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -x .venv-linux/bin/python ]]; then
  printf '%s\n' 'Missing .venv-linux/bin/python. Run this script in the prepared Linux project on the GPU instance.' >&2
  exit 1
fi
source .venv-linux/bin/activate

python - <<'PY'
import json
import sys
import urllib.request

try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8000/health", timeout=10) as response:
        health = json.load(response)
    if health.get("ready") is not True:
        raise RuntimeError("The model server did not report ready=true")
    if "adapter" not in health or health["adapter"] is not None:
        raise RuntimeError("This probe requires the original model with adapter=null")
except Exception as exc:
    print("Health check failed:", json.dumps(str(exc), ensure_ascii=True), file=sys.stderr)
    print("Keep the model server running in another terminal on the SAME GPU instance.", file=sys.stderr)
    print("From this project directory, start the original model with:", file=sys.stderr)
    print("CUDA_VISIBLE_DEVICES=0 bash scripts/start_model.sh", file=sys.stderr)
    print("Wait for Ready, then run this probe again. This script does not start or stop any server.", file=sys.stderr)
    raise SystemExit(1)
print("HEALTH:", json.dumps(health, ensure_ascii=True), flush=True)
PY

probe_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
probe_root="runs/transport_demo_probe/${probe_stamp}_$$"
mkdir -p runs/transport_demo_probe
mkdir "$probe_root"

set +e
python -m tube_agent.run \
  --policy model --prompt-variant manual_demo_v1 \
  --task transport --split train --seed-start 0 \
  --episodes 1 --attempts 8 --temperature 0.8 \
  --output "$probe_root"
probe_status=$?
set -e

if ! python - "$probe_root" <<'PY'
import json
from pathlib import Path
import sys

def show(label, value):
    print(label, json.dumps(value, ensure_ascii=True), flush=True)

root = Path(sys.argv[1])
summaries = sorted(root.glob("*_model_transport_train/summary.json"))
if len(summaries) != 1:
    show("DIAGNOSTIC_ERROR:", {"root": str(root), "summary_files": len(summaries),
                               "message": "Expected exactly one run summary; inspect the run output above."})
    raise SystemExit(1)
run = summaries[0].parent
summary = json.loads(summaries[0].read_text(encoding="utf-8"))
items = [(path, json.loads(path.read_text(encoding="utf-8")))
         for path in sorted(run.glob("seed_*/result.json"))]
eligible = [result for _, result in items
            if result.get("success") and not result.get("error")
            and result.get("invalid_actions") == 0
            and result.get("policy") == "model" and result.get("split") == "train"]
show("SAVED_RUN:", str(run))
show("SUMMARY:", summary)
show("ELIGIBLE_DATA:", {"eligible_successes": len(eligible),
                        "scene_count": len({r["seed"] for r in eligible}),
                        "episodes_with_result": len(items)})

def show_episode(label, item, raw_printed):
    path, result = item
    show(label + "_RESULT:", result)
    keys = ("task", "split", "seed", "attempt", "policy", "model", "observation_mode",
            "prompt_variant", "system_prompt_sha256", "feedback_version", "demonstration")
    show(label + "_SOURCE:", {key: result.get(key) for key in keys})
    trajectory = path.parent / "trajectory.jsonl"
    if not trajectory.is_file():
        show(label + "_TRAJECTORY:", "No trajectory file was produced")
        return raw_printed
    for line in trajectory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        show(label + "_STEP:", {key: row.get(key) for key in ("step", "action", "feedback")})
        if not raw_printed and row.get("feedback", {}).get("accepted") is False:
            show("FIRST_REJECTED_RAW_RESPONSE:", row.get("raw_response"))
            raw_printed = True
    return raw_printed

success = next((item for item in items if item[1].get("success") and not item[1].get("error")), None)
failure = next((item for item in items if not item[1].get("success") or item[1].get("error")), None)
raw_printed = False
for label, item in (("FIRST_SUCCESS", success), ("FIRST_FAILURE", failure)):
    if item is None:
        show(label + ":", "None")
    else:
        raw_printed = show_episode(label, item, raw_printed)
show("NOTE:", "Read-only diagnostics; no training data was exported and no training was started.")
PY
then
  printf '%s\n' 'Probe diagnostics could not be completed; the original run exit code is preserved.' >&2
fi

exit "$probe_status"
