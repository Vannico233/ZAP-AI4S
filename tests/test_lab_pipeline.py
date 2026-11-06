import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tube_agent.lab.experiment import eligible,write_json,read_json,checkpoint
from tube_agent.lab.pipeline import jobs,select_candidate,export_sources,report

def success(**patch):
    return {"policy":"model","split":"train","task":"pick","success":True,"error":None,
            "invalid_actions":0,"observation_mode":"rgbd_sam3.1_detector","demonstration":None,**patch}

def test_plan_uses_all_four_cards_and_bounded_budget():
    output=subprocess.check_output([sys.executable,"-m","tube_agent.lab.pipeline","--plan"],text=True)
    plan=json.loads(output)
    assert plan["policy_gpus"]==[0,2,3] and plan["sam_gpu"]==1
    assert plan["max_total_episodes"]==sum(plan["episode_budgets"].values())
    assert plan["held_out"]=="rack_transfer"
    with pytest.raises(ValueError):jobs(["rack_transfer"],[0],"train",1)

def test_export_excludes_test_demos_failures_and_interrupted_updates(tmp_path):
    variants=[success(),success(split="test"),success(demonstration={"manual":True}),
              success(task="rack_transfer"),success(success=False),success(invalid_actions=1),success()]
    row={"messages":[{"role":"user","content":"image test"}],"raw_response":'{"action":"finish"}'}
    for i,result in enumerate(variants):
        folder=tmp_path/(f"try_{i}" if i!=6 else "rank_00.incomplete-123")
        write_json(folder/"result.json",result)
        (folder/"trajectory.jsonl").write_text(json.dumps(row)+"\n")
    output=tmp_path/"data/success.jsonl"
    info=export_sources([tmp_path],output)
    assert info["episodes_by_task"]=={"pick":1} and info["actions"]==1
    assert eligible(json.loads(output.read_text())["source"])

def test_validation_selection_requires_all_primitives_and_never_rack(tmp_path):
    from tube_agent.lab.experiment import TRAIN_TASK_NAMES
    candidates=[]
    for name,wins in (("base",0),("rl",1)):
        directory=tmp_path/name
        for task in TRAIN_TASK_NAMES:
            write_json(directory/task/"summary.json",{"job":{"task":task},"episodes":2,"successes":wins,
                        "errors":0,"mean_reward":wins,"invalid_actions":0})
        candidates.append((name,None,directory))
    assert select_candidate(candidates)[0]["name"]=="rl"
    write_json(tmp_path/"rl/rack/summary.json",{"job":{"task":"rack_transfer"},"episodes":2,"successes":2,
                                             "errors":0,"mean_reward":5,"invalid_actions":0})
    with pytest.raises(ValueError,match="contaminated"):select_candidate(candidates)

def test_partial_checkpoint_is_not_resumable(tmp_path):
    (tmp_path/"step_00002").mkdir()
    write_json(tmp_path/"step_00001/COMPLETE.json",{"step":1})
    assert checkpoint(tmp_path).name=="step_00001"

def test_worker_resumes_completed_episodes_and_stops_at_unique_successes(tmp_path,monkeypatch):
    from tube_agent.lab.worker import run_jobs
    import tube_agent.lab.run as run
    calls=[]
    def episode(task,split,seed,attempt,directory,*args,**kwargs):
        calls.append(attempt)
        result=success(task=task,steps=2,reward=5.)
        trajectory=[{"action":{"action":"gripper","opening":attempt*.01},"feedback":{"accepted":True},
                     "judge_after":{"reward":.2}}]
        write_json(Path(directory)/"result.json",result)
        (Path(directory)/"trajectory.jsonl").write_text(json.dumps(trajectory[0])+"\n")
        return result,trajectory
    monkeypatch.setattr(run,"episode",episode)
    tasks=jobs(["pick"],[0],"train",10,"search",2)
    run_jobs(tasks,tmp_path,None,None)
    assert calls==[0,1]
    # Remove only the job completion marker to simulate interruption after its
    # final episode, then ensure finished actions are not generated again.
    (tmp_path/tasks[0]["id"]/"DONE.json").unlink()
    run_jobs(tasks,tmp_path,None,None)
    assert calls==[0,1]

def test_no_success_report_never_invents_a_video(tmp_path):
    value=report(tmp_path,"completed")
    assert value["videos"]=={} and not value["all_five_have_success_videos"]
    assert "not run" in (tmp_path/"REPORT.md").read_text()

@pytest.mark.skipif(sys.platform=="win32",reason="Local Windows Torch build has no working Gloo transport; Linux NCCL is checked by pipeline training_smoke")
def test_gradient_sum_matches_global_objective_on_three_processes(tmp_path):
    pytest.importorskip("torch")
    script=tmp_path/"distributed_check.py"
    script.write_text('''
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from tube_agent.lab.distributed_train import synchronize_gradients

def worker(rank,uri,output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=uri,rank=rank,world_size=3)
    weight=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.SGD([weight],lr=.1)
    # Unequal local work: rank 2 has no backward, others have different sizes.
    if rank<2:
        for x in ([1.,2.] if rank==0 else [3.]):
            ((weight*x).square().sum()/3).backward()
    assert synchronize_gradients([weight],dist)
    assert torch.allclose(weight.grad,torch.tensor([28./3]),atol=1e-6)
    optimizer.step()
    assert torch.allclose(weight,torch.tensor([1.-.1*28./3]),atol=1e-6)
    optimizer.zero_grad(set_to_none=True)
    assert not synchronize_gradients([weight],dist)
    Path(output,f"rank_{rank}.ok").write_text("OK")
    dist.destroy_process_group()

if __name__=="__main__":
    mp.spawn(worker,args=(sys.argv[1],sys.argv[2]),nprocs=3,join=True)
''',encoding="utf-8")
    rendezvous="file:///"+(tmp_path/"rendezvous").resolve().as_posix() if os.name=="nt" else "file://"+str((tmp_path/"rendezvous").resolve())
    environment=os.environ.copy();environment["PYTHONPATH"]=str(Path(__file__).resolve().parents[1])
    subprocess.run([sys.executable,str(script),rendezvous,str(tmp_path)],env=environment,check=True,timeout=120)
    assert len(list(tmp_path.glob("rank_*.ok")))==3

def test_gradient_reduction_math_with_unequal_rank_work():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from tube_agent.lab.distributed_train import synchronize_gradients
    barrier=threading.Barrier(3,timeout=15);values={};total={}
    class Collective:
        def __init__(self,rank):self.rank=rank
        def all_reduce(self,tensor):
            values[self.rank]=tensor.detach().clone();barrier.wait()
            if self.rank==0:total["tensor"]=sum(values.values())
            barrier.wait();tensor.copy_(total["tensor"]);barrier.wait()
    def worker(rank):
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        if rank<2:
            for x in ([1.,2.] if rank==0 else [3.]):((parameter*x).square().sum()/3).backward()
        assert synchronize_gradients([parameter],Collective(rank))
        gradient=parameter.grad.clone();parameter.grad=None
        assert not synchronize_gradients([parameter],Collective(rank))
        return gradient
    with ThreadPoolExecutor(3) as pool:gradients=list(pool.map(worker,range(3)))
    assert all(torch.allclose(g,torch.tensor([28./3]),atol=1e-6) for g in gradients)

@pytest.mark.parametrize("have_success",[True,False])
def test_pipeline_sequences_training_and_holds_out_composition(tmp_path,monkeypatch,have_success):
    import tube_agent.lab.pipeline as pipeline
    from types import SimpleNamespace
    monkeypatch.setattr(pipeline,"gpu_check",lambda a:{})
    monkeypatch.setattr(pipeline.shutil,"disk_usage",lambda path:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(pipeline,"make_videos",lambda runner:{})
    a=pipeline.parser().parse_args([]);a.output=tmp_path
    class FakeRunner:
        def __init__(self):self.a=a;self.output=tmp_path;self.training=[];self.samples=[]
        def status(self,*args,**kwargs):pass
        def ensure_sam(self):pass
        def step(self,*args,**kwargs):pass
        def train(self,name,stage,**kwargs):
            assert "rack_transfer" not in kwargs.get("tasks",[])
            self.training.append((name,stage,kwargs))
            return tmp_path/name/"step_00001" if stage!="smoke" else None
        def sample(self,name,adapter,job_list):
            self.samples.append((name,job_list))
            directory=tmp_path/name
            for job in job_list:
                write_json(directory/job["id"]/"summary.json",{"job":job,"episodes":1,"successes":int(have_success),
                           "errors":0,"mean_reward":float(have_success),"invalid_actions":0,"unique_successes":0})
                result=success(task=job["task"],split=job["split"],success=have_success)
                episode=directory/job["id"]/"try_000"
                write_json(episode/"result.json",result)
                (episode/"trajectory.jsonl").write_text(json.dumps({"messages":[{"role":"user","content":job["task"]}],
                                                                   "raw_response":'{"action":"finish"}'})+"\n")
            return directory
    runner=FakeRunner();pipeline.execute_pipeline(runner)
    names=[name for name,_,_ in runner.training]
    assert names[:2]==["training_smoke","warmup"]
    assert ("sft" in names)==have_success
    assert names[-2:]==["rl_fixed","rl_diverse"]
    assert runner.training[-1][2]["seeds"]==[0,1,2,3]
    for name,job_list in runner.samples:
        if name not in ("base_test","final_test"):
            assert all(j["task"]!="rack_transfer" for j in job_list)
    assert read_json(tmp_path/"report.json")["status"]=="completed"
