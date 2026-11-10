"""Bounded pick-only failed-done ablation. Never starts SFT or four-task RL."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
from .experiment import read_json, write_json, rows, checkpoint
from .pipeline import Runner, ROOT, gpu_check, source_digest
from .credit import failed_done


def evaluation_jobs(start, split="train"):
    # Same scene and sampling seeds for each checkpoint/arm; no prefix search.
    return [dict(id=f"pick_shard_{i}", task="pick", seed=0, split=split,
                 mode="independent", attempts=4, attempt_start=start+4*i,
                 temperature=.8) for i in range(3)]


def diagnostics(directory):
    values=[]
    for path in sorted(Path(directory).glob("*/try_*/result.json")):
        if any(".incomplete-" in p for p in path.parts):
            continue
        result=read_json(path)
        if result.get("error") or result.get("policy")!="model" or result.get("demonstration") is not None:
            raise ValueError(f"Invalid evaluation evidence: {path}")
        trace=rows(path.parent/"trajectory.jsonl")
        if len(trace)!=result["steps"]:
            raise ValueError(f"Incomplete evaluation trajectory: {path}")
        values.append(dict(success=int(result["success"]), steps=result["steps"],
            failed_done=int(any(failed_done(r) for r in trace)),
            grasped=int(any(r["judge_after"].get("held",False) for r in trace)),
            lifted=int(any(r["judge_after"].get("lifted_once",False) for r in trace)),
            invalid_actions=result["invalid_actions"], reward=result["reward"]))
    if len(values)!=12:
        raise ValueError(f"Expected 12 independent episodes, got {len(values)} in {directory}")
    return {"episodes":len(values), **{k:sum(v[k] for v in values) for k in values[0]},
            "interpretation":"Counts over 12 independent rollouts; steps/reward are totals. Grasp/lift are judge states, not success."}


def training_command(a, mode, adapter):
    return [sys.executable,"-m","torch.distributed.run","--standalone",
        "--nproc_per_node=3","--max_restarts=0","-m","tube_agent.lab.distributed_train",
        "grpo","--output",str(a.output/f"train_{mode}"),"--model",a.model,
        "--adapter",str(adapter),"--tasks","pick","--scene-seeds","0","--seed","0",
        "--updates","8","--group-size","6","--rank","16","--learning-rate","1e-5",
        "--max-steps","40","--max-context","16384","--loss-normalization","dr_grpo",
        "--failed-done-credit",mode,"--action-protocol","zap_pose_v1","--sam-url",a.sam_url]


def execute(runner, adapters):
    a=runner.a
    runner.ensure_sam()
    report={"status":"running", "checkpoint_comparison":{}, "arms":{},
            "automatic_four_task_restart":False, "training_start":"step_00038",
            "optimizer":"Fresh AdamW in both arms; not a continuation of v5 optimizer state"}
    for step,adapter in adapters.items():
        directory=runner.sample(f"compare_{step}",adapter,evaluation_jobs(80000))
        report["checkpoint_comparison"][step]=diagnostics(directory)
        write_json(a.output/"report.json",report)
    # Predeclared start, not selected by reward or by these evaluation results.
    for mode in ("legacy","nonpositive"):
        runner.step(f"train_{mode}",training_command(a,mode,adapters["step_00038"]))
        adapter=checkpoint(a.output/f"train_{mode}")
        if adapter is None or adapter.name!="step_00008":
            raise RuntimeError("Missing completed eight-update ablation checkpoint")
        fixed=runner.sample(f"fixed_{mode}",adapter,evaluation_jobs(81000))
        validation=runner.sample(f"validation_{mode}",adapter,evaluation_jobs(82000,"validation"))
        report["arms"][mode]={"fixed":diagnostics(fixed),"validation":diagnostics(validation)}
        write_json(a.output/"report.json",report)
    report.update(status="completed",decision="Stop for evidence review; no automatic scaling or success claim")
    write_json(a.output/"report.json",report)
    runner.status("completed",report=str(a.output/"report.json"))


def main():
    from tube_agent.model_utils import DEFAULT_MODEL, resolve_model
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("runs/lab_pick_ablation/v1"))
    p.add_argument("--source-run",type=Path,default=Path("runs/lab_zap_recipe/seed0_v5"))
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--sam-url",default="http://127.0.0.1:8002")
    p.add_argument("--plan",action="store_true")
    a=p.parse_args()
    plan=dict(checkpoints=[38,39,68],checkpoint_rollouts=36,training_updates_per_arm=8,
              training_rollouts=96,fixed_rollouts=24,validation_rollouts=24,total_episodes=180,
              training_tasks=["pick"],training_start="step_00038",modes=["legacy","nonpositive"],
              automatic_four_task_restart=False)
    if a.plan:
        print(json.dumps(plan,indent=2));return
    if sys.platform!="linux":p.error("Run on the existing Linux GPU instance; --plan works locally")
    import fcntl
    a.output=a.output.resolve();a.source_run=a.source_run.resolve()
    if a.output==a.source_run or a.source_run in a.output.parents or a.output in a.source_run.parents:
        p.error("Ablation output must be separate from the source run")
    a.model=str(resolve_model(a.model));a.policy_gpus=[0,2,3];a.sam_gpu=1
    a.sam_checkpoint=ROOT/"assets/sam3.1/sam3.1_multiplex.pt"
    a.cpu_threads=4;a.max_steps=40;a.action_protocol="zap_pose_v1"
    a.min_free_gpu_gb=48;a.startup_timeout=900
    adapters={f"step_{step:05d}":a.source_run/"warmup"/f"step_{step:05d}" for step in (38,39,68)}
    hashes={}
    for name,path in adapters.items():
        if not (path/"COMPLETE.json").is_file() or not (path/"adapter_model.safetensors").is_file():
            p.error(f"Missing committed source checkpoint: {path}")
        hashes[name]=hashlib.sha256((path/"adapter_model.safetensors").read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/".lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("This ablation output already has a running process")
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!="plan"}
    config.update(plan=plan,source_sha256=source_digest(),adapter_sha256=hashes)
    if (a.output/"config.json").exists() and read_json(a.output/"config.json")!=config:
        raise SystemExit("Code, checkpoint or settings changed; use a new output directory")
    write_json(a.output/"config.json",config)
    runner=Runner(a)
    def interrupted(signum,frame):raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if (a.output/"report.json").exists() and read_json(a.output/"report.json").get("status")=="completed":
            runner.status("completed",message="Ablation already completed");return
        write_json(a.output/"gpu_check.json",gpu_check(a))
        execute(runner,adapters)
    except BaseException as exc:
        runner.status("interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=str(exc))
        raise
    finally:runner.close();lock.close()


if __name__=="__main__":main()
