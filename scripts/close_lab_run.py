"""Stop one identified lab pipeline and archive existing evidence (stdlib only).

No model loading, rendering, training, package installation or source deletion.
Use --stop on the GPU host. --include adds read-only, already stopped runs.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import time

TASKS = ("pick", "transport", "insert", "unscrew", "rack_transfer")
MODULES = {"tube_agent.lab.pipeline", "tube_agent.lab.worker",
           "tube_agent.lab.distributed_train", "tube_agent.lab.run",
           "tube_agent.lab.replay", "tube_agent.lab.doctor", "tube_agent.lab.zap_check"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True)+"\n", encoding="utf-8", newline="\n")


def modules(args):
    return {args[i+1] for i, arg in enumerate(args[:-1]) if arg in ("-m", "--module")}


def read_process(pid, proc_root=Path("/proc")):
    folder = proc_root / str(pid)
    try:
        if folder.stat().st_uid != os.geteuid(): return None
        args = (folder/"cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
        # comm may contain spaces or ')'; fields after its final ')' start at state.
        fields = (folder/"stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"): return None
        return {"pid":int(pid), "ppid":int(fields[1]), "start":fields[19],
                "args":args, "cwd":(folder/"cwd").resolve()}
    except (OSError, ValueError, IndexError):
        return None


def process_snapshot():
    return {int(p.name): value for p in Path("/proc").iterdir()
            if p.name.isdigit() and (value := read_process(int(p.name))) is not None}


def output_path(process):
    args = process["args"]
    for index, arg in enumerate(args):
        value = args[index+1] if arg == "--output" and index+1 < len(args) else None
        if arg.startswith("--output="): value = arg.split("=", 1)[1]
        if value:
            path = Path(value)
            return (path if path.is_absolute() else process["cwd"]/path).resolve()
    return None


def scoped_processes(snapshot, run):
    """Exact pipeline output or known lab children writing within that run."""
    found = {}
    for pid, process in snapshot.items():
        kinds = modules(process["args"])
        if not kinds & MODULES: continue
        output = output_path(process)
        if output is None: continue
        if "tube_agent.lab.pipeline" in kinds:
            matches = output == run
        else:
            matches = output.is_relative_to(run)
        if matches: found[pid] = process
    return found


def descendants(snapshot, seeds):
    found = dict(seeds)
    while True:
        extra = {pid:p for pid,p in snapshot.items() if p["ppid"] in found and pid not in found}
        if not extra: return found
        found.update(extra)


def same_process(left, right):
    return right is not None and all(left[k] == right[k] for k in ("pid", "start", "args"))


def terminate_identified(process):
    """A pidfd binds the signal to the process identity, protecting PID reuse."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Stopping requires Linux pidfd support (available on the supplied Ubuntu/Python 3.11 host)")
    try:
        fd = os.pidfd_open(process["pid"])
    except ProcessLookupError:
        return False
    try:
        if not same_process(process, read_process(process["pid"])): return False
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            return False  # The checked process may have exited just before signalling.
        return True
    finally:
        os.close(fd)


def stop_run(run, stop=False, timeout=60):
    if sys.platform != "linux":
        if stop: raise RuntimeError("Run --stop on the Linux GPU instance")
        return {"mode":"offline_snapshot", "processes_checked":False}
    snapshot = process_snapshot()
    scoped = scoped_processes(snapshot, run)
    known = descendants(snapshot, scoped)
    # launcher.pid is diagnostic only; a stale/reused PID is never trusted.
    launcher = (run/"launcher.pid").read_text().strip() if (run/"launcher.pid").is_file() else None
    report = {"requested_run":str(run), "launcher_pid":launcher,
              "matched_pids":sorted(scoped), "tracked_descendants":sorted(known), "signalled_pids":[]}
    if known and not stop:
        raise RuntimeError("This run is active. Use --stop on its GPU instance; no signal was sent.")
    roots = {pid:p for pid,p in scoped.items() if "tube_agent.lab.pipeline" in modules(p["args"])}
    targets = roots or scoped  # Orphaned output-scoped workers can be identified directly.
    for pid, process in targets.items():
        if terminate_identified(process): report["signalled_pids"].append(pid)
    start = time.monotonic(); last = -10.
    while known:
        current = process_snapshot()
        # Track children started during shutdown, but only under already identified parents.
        active_parents = {pid:p for pid,p in known.items() if same_process(p,current.get(pid))}
        known.update(descendants(current, active_parents))
        known.update(scoped_processes(current, run))
        live = {pid:p for pid,p in known.items() if same_process(p,current.get(pid))}
        if not live: break
        elapsed = time.monotonic()-start
        if elapsed >= timeout:
            raise RuntimeError("Shutdown timeout; original data preserved. Remaining identified PIDs: "+str(sorted(live)))
        if elapsed-last >= 5:
            print("WAITING_FOR_SHUTDOWN:",sorted(live),flush=True);last=elapsed
        # Give the pipeline time to write status and reap its own worker groups.
        if elapsed >= 15:
            for pid, process in live.items():
                if pid not in report["signalled_pids"] and terminate_identified(process):
                    report["signalled_pids"].append(pid)
        time.sleep(.5)
    report["status"] = "identified_processes_stopped" if scoped else "no_matching_process"
    report["scope"] = "Specified run and identified descendants only; unrelated or unidentifiable services are not stopped"
    return report


@contextmanager
def stopped_lock(run):
    """Keep the snapshot stable and refuse a run active on another instance."""
    if sys.platform != "linux":
        yield; return
    import fcntl
    path = run/".lock"
    if not path.exists():
        yield; return
    with path.open("r+b") as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Run is still locked, possibly on another instance: "+str(run)) from None
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def collect(runs, output, stop_report, max_file_mb=64):
    """Copy bounded existing evidence. Never load code or weights from a run."""
    output.mkdir(parents=True, exist_ok=False)
    entries = []; skipped = []; videos = []; totals = defaultdict(lambda:dict(episodes=0,successes=0,eligible_successes=0,errors=0,invalid_actions=0))
    budget = 1024*2**20; used = 0
    def add(path, root, label, tail=False):
        nonlocal used
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file(): return None
        relative = Path("evidence")/label/path.relative_to(root)
        if tail: relative = relative.with_name(relative.name+".tail.txt")
        target = output/relative
        if target.exists(): return relative.as_posix()
        size = path.stat().st_size; amount = min(size,2*2**20) if tail else size
        if amount > max_file_mb*2**20 or used+amount > budget:
            skipped.append({"path":str(path),"bytes":size,"reason":"archive size limit; original preserved"});return None
        with path.open("rb") as stream:
            if tail: stream.seek(size-amount)
            data = stream.read(amount)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);used+=len(data)
        entries.append({"file":relative.as_posix(),"sha256":hashlib.sha256(data).hexdigest(),
                        "bytes":len(data),"source":str(path),"source_bytes":size,"tail_only":tail and size>amount})
        return relative.as_posix()
    metadata = ("config.json","status.json","report.json","REPORT.md","success_gate.json",
                "gate_replays.json","acquisition_diagnostics.json","selection.json","sam_health.json",
                "launcher.pid","videos/index.json","imported_probe/IMPORT.json")
    for index, root in enumerate(runs):
        label = f"run_{index:02d}"; selected = set(); successes=defaultdict(list); failures=defaultdict(list)
        for name in metadata: add(root/name,root,label)
        add(root/"launcher.log",root,label,tail=True)
        for path in (root/"logs").glob("*.log"): add(path,root,label,tail=True)
        for pattern in ("summary.json","DONE.json"):
            for path in root.rglob(pattern):
                if not any(".incomplete-" in part for part in path.parts):add(path,root,label)
        for path in sorted(root.rglob("result.json")):
            if not path.resolve().is_relative_to(root) or any(".incomplete-" in part for part in path.parts):continue
            value=read_json(path,{})
            if value.get("policy")!="model" or value.get("task") not in TASKS:continue
            add(path,root,label)
            stage=path.relative_to(root).parts[0];task=value["task"]
            count=totals[(label,stage,task)]
            count["episodes"]+=1;count["successes"]+=int(value.get("success") is True)
            count["errors"]+=int(bool(value.get("error")));count["invalid_actions"]+=value.get("invalid_actions",0)
            valid=(value.get("success") is True and not value.get("error") and value.get("invalid_actions")==0
                   and value.get("split")=="train" and value.get("demonstration") is None
                   and value.get("observation_mode")=="rgbd_sam3.1_detector")
            count["eligible_successes"]+=int(valid)
            (successes if value.get("success") else failures)[task].append((path.parent,value))
        for candidates in successes.values():
            selected.update(p for p,_ in candidates[:10])
        for candidates in failures.values():
            selected.add(candidates[0][0])
            selected.add(max(candidates,key=lambda pair:pair[1].get("reward",-100))[0])
        for episode in selected:
            add(episode/"trajectory.jsonl",root,label)
        for path in root.rglob("episode.mp4"):
            replay=read_json(path.parent/"report.json",{})
            states=replay.get("state_matches",[]);original=replay.get("original_result",{})
            if (not states or not all(s is True for s in states) or not replay.get("replay_result",{}).get("success")
                or original.get("policy")!="model" or original.get("success") is not True
                or original.get("error")):continue
            stored=add(path,root,label)
            add(path.parent/"report.json",root,label);add(path.parent/"poster.jpg",root,label)
            videos.append({"run":label,"task":original.get("task"),"seed":original.get("seed"),
                           "file":stored,"original":str(path.resolve()),"kind":"existing_verified_model_action_replay"})
    stats=[{"run":key[0],"stage":key[1],"task":key[2],**value} for key,value in sorted(totals.items())]
    offline = stop_report.get("mode") == "offline_snapshot"
    summary={"closeout":"offline_snapshot" if offline else "terminated_by_user","created_utc":datetime.now(timezone.utc).isoformat(),
        "runs":[{"id":f"run_{i:02d}","path":str(p)} for i,p in enumerate(runs)],
        "shutdown":stop_report,"stages":stats,"videos":videos,"skipped":skipped,
        "new_experiments_started":False,"new_training_started":False,
        "interpretation":"Stage counts are separate. Imported/search/independent trials must not be pooled as a policy success rate. Eligible counts are filters, not replay gate admission.",
        "training_evidence":[{"run":f"run_{i:02d}","directories":[name for name in
            ("training_smoke","warmup","sft","rl_fixed","rl_diverse") if (p/name).is_dir()],
            "gate":read_json(p/"success_gate.json")} for i,p in enumerate(runs)]}
    write_json(output/"summary.json",summary);write_json(output/"MANIFEST.json",entries)
    status_text = ("Offline evidence snapshot; server process state was not checked." if offline
                   else "Specified run closed out at the user's request; shutdown scope is recorded in summary.json.")
    lines=["# Experiment closeout","", status_text,
           "No new sampling, training or rendering was started.",
           "Original run directories and environments are preserved.","",
           "Counts below are per stage. Search yield and imported traces are not independent success rates.","",
           "Counts use completed model result.json files present in the input. Original job summaries are also included.","",
           "| Run | Stage | Task | Completed | Success | Eligible filter | Errors | Invalid actions |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in stats:
        lines.append("| "+" | ".join(str(r[k]) for k in ("run","stage","task","episodes","successes","eligible_successes","errors","invalid_actions"))+" |")
    lines.extend(["","## Existing verified videos",""])
    for video in videos:
        if video["file"]:lines.append(f"- {video['run']} / {video['task']} / seed {video['seed']}: [video]({video['file']})")
    if not videos:lines.append("No existing replay-verified model success video was found in these runs.")
    elif not any(v["file"] for v in videos):lines.append("Verified videos were found but exceeded archive limits; see original paths in summary.json.")
    lines.extend(["","See summary.json for original paths, prior training evidence and shutdown scope.",
                  f"Skipped oversized files: {len(skipped)}. Originals remain in their run directories."])
    (output/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    archive=output.with_name(output.name+".tar.gz")
    with tarfile.open(archive,"x:gz") as tar:
        tar.add(output,arcname=output.name,recursive=True)
    with archive.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    archive.with_name(archive.name+".sha256").write_text(digest+"  "+archive.name+"\n",encoding="ascii",newline="\n")
    print("CLOSEOUT:",output,flush=True);print("ARCHIVE:",archive,flush=True)
    print("SHA256:",digest,flush=True);print("VERIFIED_VIDEOS_FOUND:",len(videos),flush=True)
    print("VERIFIED_VIDEOS_ARCHIVED:",sum(v["file"] is not None for v in videos),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--include",type=Path,action="append",default=[])
    p.add_argument("--stop",action="store_true");p.add_argument("--timeout",type=float,default=60.)
    p.add_argument("--output",type=Path);p.add_argument("--max-file-mb",type=int,default=64)
    a=p.parse_args()
    runs=list(dict.fromkeys(path.resolve() for path in [a.run,*a.include]))
    for run in runs:
        if not (run/"config.json").is_file():p.error("No pipeline config found: "+str(run))
    if a.timeout<=0 or a.max_file_mb<1:p.error("Timeout and file limit must be positive")
    output=(a.output or Path("closeouts")/("lab_closeout_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    if any(output.is_relative_to(run) or run.is_relative_to(output) for run in runs):
        p.error("Closeout directory must be separate from the experiment directories")
    if output.exists() or output.with_name(output.name+".tar.gz").exists():p.error("Output already exists; choose a new path")
    shutdown=stop_run(runs[0],a.stop,a.timeout)
    with ExitStack() as stack:
        for run in runs:
            if run!=runs[0]:stop_run(run,False,a.timeout)
            stack.enter_context(stopped_lock(run))
        collect(runs,output,shutdown,a.max_file_mb)


if __name__=="__main__":main()
