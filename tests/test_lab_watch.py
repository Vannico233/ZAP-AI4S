import importlib.util
import json
import os
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("watch_lab_run", Path(__file__).resolve().parents[1] / "scripts/watch_lab_run.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_clock(monkeypatch, on_sleep=lambda tick: None):
    clock = {"value": 0.}
    monkeypatch.setattr(monitor.time, "monotonic", lambda: clock["value"])
    def sleep(seconds):
        clock["value"] += seconds
        on_sleep(clock["value"])
    monkeypatch.setattr(monitor.time, "sleep", sleep)
    return clock


def test_log_tail_reads_each_complete_line_once_and_handles_truncation(tmp_path):
    path = tmp_path / "worker.log"
    tail = monitor.LogTail()
    assert tail.poll(path) == []
    path.write_bytes(b'first\n{"event":"act')
    assert tail.poll(path) == ["first"]
    with path.open("ab") as stream:
        stream.write(b'ion"}\nlast\n')
    assert tail.poll(path) == ['{"event":"action"}', "last"]
    assert tail.poll(path) == []
    path.write_bytes(b"new\n")
    assert tail.poll(path) == ["new"]


def test_tail_limits_initial_history(tmp_path):
    path = tmp_path / "worker.log"
    path.write_bytes(b"old line\n" * 10000 + b"newest line\n")
    lines = monitor.LogTail().poll(path)
    assert lines[-1] == "newest line" and len(lines) < 2000


def test_progress_separates_stages_and_ignores_partial_files(tmp_path):
    put(tmp_path / "config.json", {"pose_probe": True})
    put(tmp_path / "pose_probe/pick_probe_0/summary.json", {"job": {"task": "pick"}, "episodes": 1, "successes": 1})
    put(tmp_path / "pose_probe/transport_probe_0/summary.json", {"job": {"task": "transport"}, "episodes": 1, "successes": 0})
    put(tmp_path / "pose_probe/old.incomplete-123/summary.json", {"job": {"task": "pick"}, "episodes": 99, "successes": 99})
    put(tmp_path / "base_fixed/pick/summary.json", {"job": {"task": "pick"}, "episodes": 3, "successes": 0})
    assert monitor.progress(tmp_path) == [
        "[PROGRESS] base_fixed: finished=3 successes=0 pick=0/3",
        "[PROGRESS] pose_probe: finished=2/12 successes=1 pick=1/1 transport=0/1"]


def test_follow_streams_live_action_and_exits_after_completion(tmp_path, monkeypatch):
    log = tmp_path / "logs/pose_probe_gpu0.log"
    log.parent.mkdir()
    log.write_text("Loading weights\n")
    put(tmp_path / "config.json", {"pose_probe": True})
    put(tmp_path / "status.json", {"stage": "pose_probe"})
    def update(tick):
        if tick == 1:
            with log.open("a") as stream:
                stream.write(json.dumps({"event": "action", "task": "pick", "attempt": 300000, "step": 1,
                                         "action": {"action": "move_to"}, "feedback": {"accepted": True}, "reward": .1}) + "\n")
            put(tmp_path / "pose_probe/pick_probe_0/summary.json", {"job": {"task": "pick"}, "episodes": 1, "successes": 1})
            put(tmp_path / "status.json", {"stage": "pose_probe_complete", "training_started": False})
            put(tmp_path / "videos/index.json", {"pick": {"status": "verified_success", "video": "example.mp4"}})
    clock = fake_clock(monkeypatch, update)
    monkeypatch.setattr(monitor, "launcher_alive", lambda pid, root: clock["value"] < 1)
    output = []
    assert monitor.watch(tmp_path, 5, 0, emit=output.append) == 0
    assert sum("ACTION task=pick" in line for line in output) == 1
    assert any("finished=1/12" in line for line in output)
    assert any("[VIDEO] pick: verified_success example.mp4" == line for line in output)
    assert output[-1].startswith("[END] stage=pose_probe_complete")


@pytest.mark.parametrize("stage,expected", [("failed", 1), ("interrupted", 130), ("awaiting_success", 2), ("completed", 0)])
def test_terminal_status_exit_codes(tmp_path, monkeypatch, stage, expected):
    put(tmp_path / "status.json", {"stage": stage, "error": "example" if stage == "failed" else None})
    fake_clock(monkeypatch)
    monkeypatch.setattr(monitor, "launcher_alive", lambda *args: False)
    output = []
    assert monitor.watch(tmp_path, 5, 0, emit=output.append) == expected
    assert output[-1].startswith(f"[END] stage={stage}")


def test_old_terminal_status_does_not_hide_failed_restart(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    put(path, {"stage": "pose_probe_complete"})
    os.utime(path, (100, 100))
    fake_clock(monkeypatch)
    monkeypatch.setattr(monitor, "launcher_alive", lambda *args: False)
    output = []
    assert monitor.watch(tmp_path, 5, 200, emit=output.append) == 1
    assert "without a fresh final status" in output[-1]


def test_terminal_status_waits_for_launcher_cleanup(tmp_path, monkeypatch):
    put(tmp_path / "status.json", {"stage": "pose_probe_complete"})
    clock = fake_clock(monkeypatch)
    monkeypatch.setattr(monitor, "launcher_alive", lambda *args: clock["value"] < 4)
    assert monitor.watch(tmp_path, 5, 0, emit=lambda line: None) == 0
    assert clock["value"] >= 6


def test_crash_without_status_exits_instead_of_waiting_forever(tmp_path, monkeypatch):
    fake_clock(monkeypatch)
    monkeypatch.setattr(monitor, "launcher_alive", lambda *args: False)
    output = []
    assert monitor.watch(tmp_path, 5, 0, emit=output.append) == 1
    assert output[-1].startswith("[ERROR] Launcher exited")


def test_quiet_worker_prints_wait_even_when_launcher_heartbeats_continue(tmp_path, monkeypatch):
    put(tmp_path / "status.json", {"stage": "pose_probe"})
    def update(tick):
        with (tmp_path / "launcher.log").open("a") as stream:
            stream.write(json.dumps({"stage": "pose_probe", "time": tick}) + "\n")
        if tick == 35:
            put(tmp_path / "status.json", {"stage": "pose_probe_complete"})
    clock = fake_clock(monkeypatch, update)
    monkeypatch.setattr(monitor, "launcher_alive", lambda *args: clock["value"] < 35)
    output = []
    assert monitor.watch(tmp_path, 5, 0, emit=output.append) == 0
    assert any(line.startswith("[WAIT] stage=pose_probe launcher_alive=True") for line in output)


def test_rejected_action_and_episode_outcomes_are_distinct():
    output = []
    for event in [
        {"event": "action", "feedback": {"accepted": False, "reason": "Outside workspace"}},
        {"event": "episode", "success": True},
        {"event": "episode", "success": False},
        {"event": "episode", "success": False, "error": "SAM unavailable"},
    ]:
        monitor.display_line(Path("worker.log"), json.dumps(event), output.append)
    assert any("REJECTED: Outside workspace" in line for line in output)
    assert any("EPISODE SUCCESS" in line for line in output)
    assert any("EPISODE FAIL" in line for line in output)
    assert any("EPISODE ERROR" in line for line in output)


@pytest.mark.parametrize("kind,expected", [("pipeline", True), ("wrapper", True), ("other", False), ("zombie", False), ("other_run", False)])
def test_process_check_matches_output_and_launcher_identity(tmp_path, monkeypatch, kind, expected):
    run = (tmp_path / "run").resolve()
    proc = tmp_path / "proc"; folder = proc / "7"; folder.mkdir(parents=True)
    monkeypatch.setattr(monitor.os, "geteuid", lambda: folder.stat().st_uid, raising=False)
    args = ["python", "-m", "tube_agent.lab.pipeline", "--output", str(run)]
    if kind == "wrapper": args = ["bash", "scripts/lab.sh", "all", "--output", str(run)]
    if kind == "other": args[2] = "unrelated.module"
    if kind == "other_run": args[-1] = str(tmp_path / "another_run")
    (folder / "cmdline").write_bytes("\0".join(args).encode())
    (folder / "stat").write_text("7 (python (worker)) " + ("Z" if kind == "zombie" else "S") + " 1")
    assert monitor.launcher_alive(7, run, proc) is expected
