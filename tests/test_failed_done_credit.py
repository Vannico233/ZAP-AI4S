import copy
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from tube_agent.lab.credit import turn_advantage, credit_metrics
from tube_agent.lab.pick_ablation import evaluation_jobs, training_command, execute, diagnostics
from tube_agent.lab.experiment import write_json


def row(action="done", success=False, accepted=True):
    return {"action":{"action":action},"feedback":{"accepted":accepted},
            "judge_after":{"success":success}}


@pytest.mark.parametrize("adv", [-1.,0.,.086277])
def test_only_positive_failed_done_credit_is_removed(adv):
    r=row();before=copy.deepcopy(r)
    assert turn_advantage(r,adv,"nonpositive")==min(adv,0.)
    assert turn_advantage(r,adv,"legacy")==adv
    assert turn_advantage(row(success=True),adv,"nonpositive")==adv
    assert turn_advantage(row("move_to"),adv,"nonpositive")==adv
    assert turn_advantage(row(accepted=False),adv,"nonpositive")==-1.
    assert r==before


def test_credit_gradient_direction_and_audit():
    import torch
    from tube_agent.lab.train import dr_grpo_loss
    for advantage,expected in ((.086277,0.),(-.2,.2/10240)):
        x=torch.tensor([-.5],requires_grad=True)
        dr_grpo_loss(x,x.detach(),turn_advantage(row(),advantage,"nonpositive"),10240).backward()
        assert x.grad.item()==pytest.approx(expected)
    assert credit_metrics([[row("move_to"),row()],[row(success=True)]],[.086277,1.],"nonpositive")==dict(
        failed_done_turns=1,positive_failed_done_before=1,positive_failed_done_after=0)


def test_matched_arms_and_disjoint_evaluation_rng(tmp_path):
    a=SimpleNamespace(output=tmp_path,model="model",sam_url="http://localhost:8002")
    commands=[training_command(a,m,Path("step_00038")) for m in ("legacy","nonpositive")]
    for command in commands:
        assert command[command.index("--tasks")+1]=="pick"
        assert command[command.index("--updates")+1]=="8"
        assert "rack_transfer" not in command
    normalized=[[str(v).replace("nonpositive","MODE").replace("legacy","MODE") for v in c] for c in commands]
    assert normalized[0]==normalized[1]
    ranges=[{j["attempt_start"]+i for j in evaluation_jobs(s) for i in range(j["attempts"])} for s in (80000,81000,82000)]
    assert all(len(r)==12 for r in ranges)
    assert not ranges[0]&ranges[1] and not ranges[1]&ranges[2]


def test_orchestrator_is_bounded_and_uses_committed_checkpoints(tmp_path,monkeypatch):
    import tube_agent.lab.pick_ablation as module
    calls=[]
    class FakeRunner:
        a=SimpleNamespace(output=tmp_path,model="model",sam_url="http://localhost:8002")
        def ensure_sam(self):pass
        def sample(self,name,adapter,jobs):
            calls.append((name,adapter,jobs));return tmp_path/name
        def step(self,name,command):
            assert "--failed-done-credit" in command
            write_json(tmp_path/name/"step_00008/COMPLETE.json",{"step":8})
        def status(self,*args,**kwargs):pass
    monkeypatch.setattr(module,"diagnostics",lambda p:{"episodes":12})
    adapters={f"step_{i:05d}":Path(f"step_{i:05d}") for i in (38,39,68)}
    execute(FakeRunner(),adapters)
    assert len(calls)==7
    assert sum(sum(j["attempts"] for j in c[2]) for c in calls)==84
    assert all(j["task"]=="pick" and j["mode"]=="independent" for c in calls for j in c[2])
    report=json.loads((tmp_path/"report.json").read_text())
    assert report["status"]=="completed" and not report["automatic_four_task_restart"]


def test_diagnostics_rejects_manual_and_counts_states_separately(tmp_path):
    for i in range(12):
        folder=tmp_path/f"pick_shard_{i//4}"/f"try_{i:03d}"
        write_json(folder/"result.json",dict(policy="model",demonstration=None,error=None,steps=1,
                   success=False,invalid_actions=0,reward=.17))
        r=row();r["judge_after"].update(held=True,lifted_once=False)
        (folder/"trajectory.jsonl").write_text(json.dumps(r)+"\n")
    report=diagnostics(tmp_path)
    assert report["failed_done"]==report["grasped"]==12
    assert report["lifted"]==report["success"]==0
    path=tmp_path/"pick_shard_0/try_000/result.json"
    value=json.loads(path.read_text());value["demonstration"]={"manual":True};write_json(path,value)
    with pytest.raises(ValueError,match="Invalid evaluation evidence"):diagnostics(tmp_path)
