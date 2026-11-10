"""Stream existing lab progress until the identified launcher exits (stdlib only)."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

TERMINAL = {"completed": 0, "already_completed": 0, "pose_probe_complete": 0,
            "probe_complete": 0, "completed_partial": 2, "awaiting_success": 2, "failed": 1, "interrupted": 130}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def launcher_alive(pid, root, proc_root=Path("/proc")):
    """Read-only identity check; never trust a recycled launcher PID by itself."""
    folder = proc_root / str(pid)
    try:
        if folder.stat().st_uid != os.geteuid():
            return False
        state = (folder / "stat").read_text().rsplit(")", 1)[1].split()[0]
        if state in ("Z", "X"):
            return False
        args = (folder / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
        is_pipeline = any(args[i] == "-m" and args[i+1] in
                          ("tube_agent.lab.pipeline", "tube_agent.lab.pick_ablation", "tube_agent.curriculum.pipeline")
                          for i in range(len(args)-1))
        is_wrapper = any(arg.endswith("scripts/lab.sh") for arg in args) and any(
            command in args for command in ("all", "pick-ablation", "tube-curriculum"))
        if not (is_pipeline or is_wrapper):
            return False
        for index, arg in enumerate(args):
            target = args[index+1] if arg == "--output" and index+1 < len(args) else None
            if arg.startswith("--output="):
                target = arg.split("=", 1)[1]
            if target:
                path = Path(target)
                if not path.is_absolute():
                    path = (folder / "cwd").resolve() / path
                return path.resolve() == root
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (PermissionError, IndexError) as error:
        raise RuntimeError("Cannot verify launcher process; run this monitor on the same GPU instance and user") from error
    return False


class LogTail:
    def __init__(self):
        self.files = {}

    def poll(self, path):
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            previous = self.files.get(path)
            with path.open("rb") as stream:
                if previous is None or previous[0] != identity:
                    offset = max(0, stat.st_size - 16384)
                    stream.seek(offset)
                    if offset:
                        stream.readline()  # Skip the first partial historical line.
                    pending = b""
                else:
                    _, offset, pending = previous
                    if stat.st_size < offset:
                        offset, pending = 0, b""
                    stream.seek(offset)
                data = pending + stream.read(1024 * 1024)
                lines = data.split(b"\n")
                self.files[path] = (identity, stream.tell(), lines.pop()[-65536:])
            return [line.decode("utf-8", errors="replace").rstrip("\r") for line in lines]
        except FileNotFoundError:
            return []


def display_line(path, line, emit):
    if not line.strip():
        return False
    label = path.stem
    try:
        event = json.loads(line)
    except ValueError:
        event = None
    if isinstance(event, dict):
        if path.name == "launcher.log" and "stage" in event:
            return False  # status.json supplies the current stage without old heartbeats.
        kind = event.get("event")
        if kind == "action":
            action = json.dumps(event.get("action", {}), separators=(",", ":"))
            feedback = event.get("feedback", {})
            emit(f"[{label}] ACTION task={event.get('task')} try={event.get('attempt')} "
                 f"step={event.get('step')} reward={event.get('reward')} "
                 f"accepted={feedback.get('accepted')} action={action}")
            if feedback.get("accepted") is False:
                emit(f"[{label}] REJECTED: {feedback.get('reason', '')}")
            return True
        if kind == "training_group_start":
            emit(f"[{label}] UPDATE {event.get('update')}/{event.get('total_updates')} "
                 f"task={event.get('task')} scene_seed={event.get('scene_seed')} "
                 f"rollouts={event.get('group_size')} loss={event.get('loss_normalization')}")
            return True
        if event.get("stage") == "grpo" and "optimizer_step_taken" in event:
            rewards = event.get("rewards", [])
            mean = sum(rewards)/len(rewards) if rewards else 0.
            emit(f"[{label}] UPDATE_DONE {event.get('step')}/{event.get('total_updates', '?')} "
                 f"task={event.get('task')} successes={event.get('successes')}/{event.get('group_size')} "
                 f"mean_reward={mean:.4f} invalid_turns={event.get('invalid_turns')} "
                 f"optimizer_step={event.get('optimizer_step_taken')}")
            return True
        if kind in ("episode", "training_episode"):
            outcome = "ERROR" if event.get("error") else "SUCCESS" if event.get("success") else "FAIL"
            emit(f"[{label}] EPISODE {outcome} task={event.get('task')} try={event.get('attempt')} "
                 f"steps={event.get('steps')} reward={event.get('reward')} error={event.get('error')}")
            return True
    emit(f"[{label}] {line[:4000]}")
    return True


def progress(root):
    stages = defaultdict(lambda: defaultdict(lambda: {"episodes": 0, "successes": 0}))
    for path in sorted(root.glob("*/*/summary.json")):
        if any(".incomplete-" in part for part in path.relative_to(root).parts):
            continue
        value = read_json(path)
        task = value.get("job", {}).get("task")
        if not task:
            continue
        item = stages[path.relative_to(root).parts[0]][task]
        for name in item:
            item[name] += value.get(name, 0)
    config = read_json(root / "config.json")
    if config.get("version") == "tube_curriculum_v1":
        committed = []
        for path in sorted((root/"warmup/metrics").glob("round_*.json")):
            value = read_json(path)
            if (root/"warmup/checkpoints"/f"step_{value.get('round', 0):05d}/COMPLETE.json").exists():
                committed.append(value)
        done = sum(v["episodes"] for v in committed)
        steps = committed[-1]["optimizer_steps_total"] if committed else 0
        lines = [f"[WARMUP] committed_rounds={len(committed)}/18 episodes={done}/960 optimizer_steps={steps}"]
        search = read_json(root/"search/state.json")
        if search:
            lines.append(f"[SEARCH] expansions={len(search['expansions'])}/32 continuations={4*len(search['expansions'])}/128")
        data = read_json(root/"data/manifest.json")
        if data:
            lines.append(f"[SFT_DATA] unique_responses={data['samples']} successes={data['successes_by_task']} missing_tasks={data['missing_tasks']}")
        for node in ("original", "after_warmup", "after_sft"):
            results = [read_json(p) for p in (root/"evaluation"/node).glob("*_eval_*/result.json") if '.incomplete-' not in str(p)]
            if results:
                lines.append(f"[EVAL] {node}: completed={len(results)}/120 successes={sum(bool(r.get('success')) for r in results)} errors={sum(bool(r.get('error')) for r in results)}")
        return lines
    if config.get("pose_probe"):
        stages["pose_probe"]  # Show 0/12 before the first episode completes.
    lines = []
    for stage, tasks in sorted(stages.items()):
        done = sum(v["episodes"] for v in tasks.values())
        successes = sum(v["successes"] for v in tasks.values())
        total = "/12" if stage == "pose_probe" else ""
        details = " ".join(f"{task}={v['successes']}/{v['episodes']}" for task, v in sorted(tasks.items()))
        lines.append(f"[PROGRESS] {stage}: finished={done}{total} successes={successes} {details}".rstrip())
    for stage in ("warmup", "rl_fixed", "rl_diverse"):
        tasks = defaultdict(lambda: {"episodes": 0, "successes": 0})
        updates = 0
        for path in sorted((root/stage/"metrics").glob("step_*.json")):
            row = read_json(path)
            if not row or not (root/stage/f"step_{row.get('step', 0):05d}"/"COMPLETE.json").is_file():
                continue
            updates += 1
            tasks[row.get("task", "unknown")]["episodes"] += row.get("group_size", 0)
            tasks[row.get("task", "unknown")]["successes"] += row.get("successes", 0)
        if updates:
            details = " ".join(f"{task}={v['successes']}/{v['episodes']}" for task, v in sorted(tasks.items()))
            lines.append(f"[TRAINING_YIELD] {stage}: committed_updates={updates} {details}")
    gate = read_json(root/"recipe_success_gate.json")
    if gate:
        details = " ".join(f"{t}={n}/{gate.get('targets', {}).get(t, '?')}" for t, n in gate.get("unique_verified_successes", {}).items())
        lines.append(f"[SFT_DATA] verified_unique {details} ready={gate.get('passed')}")
    return lines


def watch(root, pid, since, interval=1., emit=None):
    emit = emit or (lambda line: print(line, flush=True))
    tail = LogTail()
    last_stage = None
    last_progress = []
    last_output = time.monotonic()
    gone_since = None
    emit(f"[WATCH] run={root} pid={pid}; existing log tails first, then live updates")
    emit("[WATCH] Ctrl+C exits this monitor; no stop signal is sent to the experiment")
    while True:
        changed = False
        paths = [root / "launcher.log", *sorted((root / "logs").glob("*.log"))]
        for path in paths:
            for line in tail.poll(path):
                if display_line(path, line, emit):
                    changed = True
        status_path = root / "status.json"
        status = read_json(status_path)
        try:
            fresh = status_path.stat().st_mtime >= since
        except FileNotFoundError:
            fresh = False
        if not fresh:
            status = {}
        stage = status.get("stage", "starting")
        if stage != last_stage:
            emit(f"[STAGE] {stage} time={status.get('time', 'pending')}")
            if status.get("error"):
                emit(f"[ERROR] {status['error']}")
            last_stage = stage
            changed = True
        current_progress = progress(root)
        if current_progress != last_progress:
            for line in current_progress:
                emit(line)
            last_progress = current_progress
            changed = True
        alive = launcher_alive(pid, root)
        now = time.monotonic()
        if alive:
            gone_since = None
        else:
            if gone_since is None:
                gone_since = now
            # Drain final log lines/status, and tolerate the launcher's initial exec transition.
            if now - gone_since >= 2:
                if stage in TERMINAL:
                    for task, video in read_json(root / "videos/index.json").items():
                        emit(f"[VIDEO] {task}: {video.get('status')} {video.get('video', '')}".rstrip())
                    code = TERMINAL[stage]
                    emit(f"[END] stage={stage} exit_code={code}; task success is reported separately")
                    if (root / "report.json").is_file():
                        emit(f"[REPORT] {root / 'report.json'}")
                    return code
                emit("[ERROR] Launcher exited without a fresh final status. Inspect the log lines above.")
                return 1
        if changed:
            last_output = now
        elif now - last_output >= 30:
            emit(f"[WAIT] stage={stage} launcher_alive={alive}; no new action log in 30 seconds")
            last_output = now
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--since", type=float, help="Ignore terminal status older than this Unix timestamp")
    parser.add_argument("--interval", type=float, default=1.)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run the monitor on the Linux GPU instance executing this experiment")
    root = args.run.resolve()
    if not root.is_dir() or args.interval <= 0:
        parser.error("Run directory must exist and interval must be positive")
    try:
        pid = args.pid if args.pid is not None else int((root / "launcher.pid").read_text().strip())
        since = args.since if args.since is not None else (root / "launcher.pid").stat().st_mtime - 1.
    except (OSError, ValueError):
        parser.error("Missing or invalid launcher.pid; use the output directory printed by start_lab_all.sh")
    if pid <= 1:
        parser.error("Invalid launcher PID")
    try:
        return watch(root, pid, since, args.interval)
    except KeyboardInterrupt:
        print("[WATCH] Monitoring stopped; no signal was sent to the experiment", flush=True)
        return 130
    except (OSError, RuntimeError) as error:
        print(f"[WATCH_ERROR] {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
