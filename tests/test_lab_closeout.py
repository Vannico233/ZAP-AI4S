import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import pytest

spec=importlib.util.spec_from_file_location("close_lab_run",Path(__file__).resolve().parents[1]/"scripts/close_lab_run.py")
close=importlib.util.module_from_spec(spec);spec.loader.exec_module(close)


def process(pid,output=None,module="tube_agent.lab.pipeline",ppid=1,start="123"):
    args=["python","-m",module]
    if output is not None:args.extend(["--output",str(output)])
    return {"pid":pid,"ppid":ppid,"start":start,"args":args,"cwd":Path.cwd()}


def test_scope_uses_exact_pipeline_and_bounded_worker_paths(tmp_path):
    run=(tmp_path/"run").resolve()
    values=[process(11,run),process(12,run/"mine",module="tube_agent.lab.worker"),
            process(13,tmp_path/"run_other"),process(14,run,module="unrelated"),
            process(15,run/"subrun"),process(16,None,module="tube_agent.lab.sam_server",ppid=11)]
    snapshot={p["pid"]:p for p in values}
    scoped=close.scoped_processes(snapshot,run)
    assert set(scoped)=={11,12}
    assert set(close.descendants(snapshot,scoped))=={11,12,16}


def test_stale_launcher_pid_does_not_kill_another_run(tmp_path,monkeypatch):
    (tmp_path/"launcher.pid").write_text("44\n")
    monkeypatch.setattr(close.sys,"platform","linux")
    monkeypatch.setattr(close,"process_snapshot",lambda:{44:process(44,tmp_path/"other")})
    monkeypatch.setattr(close,"terminate_identified",lambda p:pytest.fail("Unrelated process must not be signalled"))
    report=close.stop_run(tmp_path,True)
    assert report["status"]=="no_matching_process" and report["signalled_pids"]==[]


def test_stop_signals_parent_and_waits_for_its_children(tmp_path,monkeypatch):
    parent=process(51,tmp_path);child=process(52,None,module="tube_agent.lab.sam_server",ppid=51)
    other=process(53,None,module="tube_agent.lab.sam_server")
    snapshots=iter([{51:parent,52:child,53:other},{53:other}]);sent=[]
    monkeypatch.setattr(close.sys,"platform","linux")
    monkeypatch.setattr(close,"process_snapshot",lambda:next(snapshots))
    monkeypatch.setattr(close,"terminate_identified",lambda p:sent.append(p["pid"]) or True)
    report=close.stop_run(tmp_path,True)
    assert sent==[51] and report["tracked_descendants"]==[51,52]
    assert report["status"]=="identified_processes_stopped"


def test_collect_without_stop_refuses_active_process(tmp_path,monkeypatch):
    monkeypatch.setattr(close.sys,"platform","linux")
    monkeypatch.setattr(close,"process_snapshot",lambda:{4:process(4,tmp_path)})
    monkeypatch.setattr(close,"terminate_identified",lambda p:pytest.fail("No stop authorization"))
    with pytest.raises(RuntimeError,match="active"):
        close.stop_run(tmp_path,False)


def test_timeout_preserves_uncooperative_process_without_sigkill(tmp_path,monkeypatch):
    monkeypatch.setattr(close.sys,"platform","linux")
    monkeypatch.setattr(close,"process_snapshot",lambda:{4:process(4,tmp_path)})
    sent=[];times=iter([0.,2.])
    monkeypatch.setattr(close.time,"monotonic",lambda:next(times))
    monkeypatch.setattr(close,"terminate_identified",lambda p:sent.append(p["pid"]) or True)
    with pytest.raises(RuntimeError,match="Shutdown timeout"):
        close.stop_run(tmp_path,True,1.)
    assert sent==[4]


def test_pidfd_rechecks_identity_and_never_signals_reused_pid(tmp_path,monkeypatch):
    original=process(7,tmp_path);closed=[];sent=[]
    monkeypatch.setattr(close.os,"pidfd_open",lambda pid:99,raising=False)
    monkeypatch.setattr(close.signal,"pidfd_send_signal",lambda fd,sig:sent.append((fd,sig)),raising=False)
    monkeypatch.setattr(close.os,"close",lambda fd:closed.append(fd))
    monkeypatch.setattr(close,"read_process",lambda pid:{**original,"start":"different"})
    assert close.terminate_identified(original) is False and sent==[] and closed==[99]


def test_process_exiting_before_signal_is_harmless(tmp_path,monkeypatch):
    original=process(7,tmp_path);closed=[]
    def gone(*args):raise ProcessLookupError()
    monkeypatch.setattr(close.os,"pidfd_open",lambda pid:99,raising=False)
    monkeypatch.setattr(close.signal,"pidfd_send_signal",gone,raising=False)
    monkeypatch.setattr(close.os,"close",lambda fd:closed.append(fd))
    monkeypatch.setattr(close,"read_process",lambda pid:original)
    assert close.terminate_identified(original) is False and closed==[99]


def test_proc_stat_identity_with_spaces_and_parentheses(tmp_path,monkeypatch):
    folder=tmp_path/"7";folder.mkdir()
    monkeypatch.setattr(close.os,"geteuid",lambda:folder.stat().st_uid,raising=False)
    (folder/"cmdline").write_bytes(b"python\0-m\0tube_agent.lab.pipeline\0")
    fields=["S","123",*(["0"]*17),"9876", "0"]
    (folder/"stat").write_text("7 (python ) child) "+" ".join(fields))
    result=close.read_process(7,tmp_path)
    assert result["ppid"]==123 and result["start"]=="9876"


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def fixture_result(**changes):
    return {"policy":"model","task":"pick","split":"train","seed":0,"success":True,
            "invalid_actions":0,"error":None,"reward":4.9,"demonstration":None,
            "observation_mode":"rgbd_sam3.1_detector",**changes}


def test_closeout_preserves_evidence_and_separates_stages(tmp_path):
    run=tmp_path/"run";probe=tmp_path/"probe";out=tmp_path/"closed"
    put(run/"config.json",{});put(probe/"config.json",{})
    cases=[("base_fixed/pick/try_0",fixture_result(success=False,reward=0.)),
           ("mine_round_000/pick/try_1",fixture_result(invalid_actions=2)),
           ("imported_probe/pick_copy",fixture_result()),
           ("mine_round_000/pick/try_2.incomplete-dead",fixture_result())]
    for directory,result in cases:
        put(run/directory/"result.json",result)
        (run/directory/"trajectory.jsonl").write_text('{"step":1}\n')
    put(probe/"pick_probe/pick_0/try_200000/result.json",fixture_result())
    video=probe/"videos/pick_0";video.mkdir(parents=True)
    (video/"episode.mp4").write_bytes(b"fixture video bytes")
    put(video/"report.json",{"original_result":fixture_result(),"replay_result":{"success":True},"state_matches":[True,True]})
    bad=run/"videos/failed";bad.mkdir(parents=True);(bad/"episode.mp4").write_bytes(b"not verified")
    put(bad/"report.json",{"original_result":fixture_result(),"replay_result":{"success":True},"state_matches":[False]})
    before={str(p):p.read_bytes() for root in (run,probe) for p in root.rglob("*") if p.is_file()}
    summary=close.collect([run,probe],out,{"status":"no_matching_process"})
    assert before=={str(p):p.read_bytes() for root in (run,probe) for p in root.rglob("*") if p.is_file()}
    assert not summary["new_experiments_started"] and not summary["new_training_started"]
    counts={(r["run"],r["stage"]):r for r in summary["stages"]}
    assert counts["run_00","mine_round_000"]["successes"]==1
    assert counts["run_00","mine_round_000"]["eligible_successes"]==0
    assert counts["run_00","imported_probe"]["episodes"]==1
    assert len(summary["videos"])==1 and summary["videos"][0]["run"]=="run_01"
    archive=tmp_path/"closed.tar.gz"
    with archive.open("rb") as stream:actual=hashlib.file_digest(stream,"sha256").hexdigest()
    checksum=(tmp_path/"closed.tar.gz.sha256").read_bytes()
    assert actual.encode() in checksum and b"\r" not in checksum
    with tarfile.open(archive) as tar:
        names=tar.getnames()
        assert "closed/SUMMARY.md" in names and any(n.endswith("episode.mp4") for n in names)
        assert not any("incomplete" in n or "videos/failed/episode.mp4" in n for n in names)


def test_large_files_are_reported_and_logs_are_explicitly_tailed(tmp_path):
    run=tmp_path/"run";run.mkdir();put(run/"config.json",{})
    (run/"launcher.log").write_bytes(b"x"*(3*2**20))
    put(run/"base_fixed/pick/try_0/result.json",fixture_result())
    (run/"base_fixed/pick/try_0/trajectory.jsonl").write_bytes(b"y"*(3*2**20))
    summary=close.collect([run],tmp_path/"closed",{},max_file_mb=2)
    assert len(summary["skipped"])==1
    manifest=json.loads((tmp_path/"closed/MANIFEST.json").read_text())
    log=next(row for row in manifest if row["source"].endswith("launcher.log"))
    assert log["tail_only"] and log["bytes"]==2*2**20 and log["file"].endswith(".tail.txt")


def test_existing_training_is_not_claimed_absent(tmp_path):
    run=tmp_path/"run";(run/"rl_fixed").mkdir(parents=True);put(run/"config.json",{})
    summary=close.collect([run],tmp_path/"closed",{})
    assert summary["training_evidence"][0]["directories"]==["rl_fixed"]
    assert summary["new_training_started"] is False


def test_offline_archive_does_not_claim_to_have_stopped_server(tmp_path):
    run=tmp_path/"run";put(run/"config.json",{})
    summary=close.collect([run],tmp_path/"closed",{"mode":"offline_snapshot","processes_checked":False})
    assert summary["closeout"]=="offline_snapshot"
    assert "server process state was not checked" in (tmp_path/"closed/SUMMARY.md").read_text()
