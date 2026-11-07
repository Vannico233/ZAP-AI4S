"""One resumable four-GPU experiment: self-exploration -> SFT -> GRPO -> held-out evaluation -> videos."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse
from .experiment import (TRAIN_TASK_NAMES,TASK_NAMES,checkpoint,eligible,fingerprint,read_json,
                         rows,successful_results,write_json)

ROOT=Path(__file__).resolve().parents[2]
PIPELINE_VERSION="lab_success_first_v2"

def jobs(tasks,seeds,split,attempts,mode="independent",min_successes=10,attempt_start=0):
    if split=="train" and "rack_transfer" in tasks:raise ValueError("Held-out task cannot enter training")
    return [{"id":f"{task}_seed_{seed:05d}","task":task,"seed":seed,"split":split,
             "mode":mode,"attempts":attempts,"attempt_start":attempt_start,"min_successes":min_successes,"temperature":.8}
            for task in tasks for seed in seeds]

def summaries(root):
    grouped=defaultdict(list)
    for path in sorted(Path(root).glob("*/summary.json")):
        value=read_json(path);grouped[value["job"]["task"]].append(value)
    result={}
    for task,values in grouped.items():
        n=sum(v["episodes"] for v in values);wins=sum(v["successes"] for v in values)
        result[task]={"episodes":n,"successes":wins,"success_rate":wins/n,
                      "errors":sum(v["errors"] for v in values),
                      "mean_reward":sum(v["mean_reward"]*v["episodes"] for v in values)/n,
                      "invalid_actions":sum(v["invalid_actions"] for v in values)}
    return result

def select_candidate(candidates):
    """Select by four-task validation macro success, then reward. Never use test."""
    scored=[]
    for name,adapter,directory in candidates:
        values=summaries(directory)
        if set(values)!=set(TRAIN_TASK_NAMES) or any(v["errors"] for v in values.values()):
            raise ValueError(f"Incomplete or contaminated validation for {name}")
        score=(sum(v["success_rate"] for v in values.values())/4,
               sum(v["mean_reward"] for v in values.values())/4)
        scored.append({"name":name,"adapter":str(adapter) if adapter else None,"score":list(score),"tasks":values})
    winner=max(scored,key=lambda value:tuple(value["score"]))
    return winner,scored

def export_sources(sources,output):
    """Stream only eligible self-success traces, including unique search suffixes."""
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(".partial.jsonl");seen=set();counts=Counter();actions=0
    with temporary.open("w",encoding="utf-8") as stream:
        for source in sources:
            for path,result in successful_results(source):
                if not eligible(result):continue
                added=False
                for row in rows(path.parent/"trajectory.jsonl"):
                    key=fingerprint([row["messages"],row["raw_response"]])
                    if key in seen:continue
                    seen.add(key);stream.write(json.dumps({**row,"source":result})+"\n")
                    actions+=1;added=True
                counts[result["task"]]+=int(added)
    if actions:temporary.replace(output)
    else:temporary.unlink()
    return {"episodes_by_task":dict(counts),"actions":actions,"data":str(output) if actions else None}

def source_digest():
    digest=hashlib.sha256()
    for path in sorted((ROOT/"tube_agent").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode());digest.update(path.read_bytes())
    return digest.hexdigest()

class Runner:
    def __init__(self,a):
        self.a=a;self.output=a.output;self.children=[];self.sam=None
        self.output.mkdir(parents=True,exist_ok=True);(self.output/"logs").mkdir(exist_ok=True)
    def status(self,stage,**extra):
        value={"stage":stage,"time":datetime.now(timezone.utc).isoformat(),**extra}
        write_json(self.output/"status.json",value);print(json.dumps(value),flush=True)
    def start(self,name,command,gpus):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=",".join(map(str,gpus)),CUDA_DEVICE_ORDER="PCI_BUS_ID",PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS=str(self.a.cpu_threads),
            PYOPENGL_PLATFORM="egl",MUJOCO_GL="egl")
        # EGL device enumeration is independent of CUDA_VISIBLE_DEVICES. Keep
        # the known working default/explicit EGL selection, never a CUDA rank.
        path=self.output/"logs"/(name+".log")
        stream=path.open("a",encoding="utf-8")
        stream.write("\nCOMMAND: "+json.dumps(command)+"\n");stream.flush()
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.children.append((name,process,stream));return process
    def wait(self,processes,stage):
        last=0
        while any(process.poll() is None for process in processes):
            for process in processes:
                if process.poll() not in (None,0):raise RuntimeError(f"{stage} child exited {process.returncode}; see {self.output/'logs'}")
            if self.sam is not None and self.sam.poll() is not None:raise RuntimeError("Managed SAM service exited; see logs/sam.log")
            if time.monotonic()-last>=30:
                if self.sam is not None:
                    import httpx
                    try:
                        with httpx.Client(timeout=3,trust_env=False) as client:
                            health=client.get(self.a.sam_url+"/health");health.raise_for_status()
                            write_json(self.output/"sam_health.json",health.json())
                    except httpx.HTTPError as exc:
                        raise RuntimeError("Managed SAM health check failed; see logs/sam.log") from exc
                self.status(stage,running_processes=sum(p.poll() is None for p in processes),logs=str(self.output/"logs"));last=time.monotonic()
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError(f"{stage} failed; see {self.output/'logs'}")
    def step(self,name,command,gpus=None):
        marker=self.output/"steps"/(name+".json")
        if marker.exists():return
        process=self.start(name,command,gpus or self.a.policy_gpus);self.wait([process],name)
        write_json(marker,{"complete":True})
    def sample(self,name,adapter,job_list):
        output=self.output/name
        if (output/"DONE.json").is_file():return output
        workers=[]
        for index,gpu in enumerate(self.a.policy_gpus):
            partition=job_list[index::len(self.a.policy_gpus)]
            if not partition:continue
            job_path=self.output/"plans"/f"{name.replace('/','_')}_gpu{gpu}.json";write_json(job_path,partition)
            command=[sys.executable,"-m","tube_agent.lab.worker","--jobs",str(job_path),"--output",str(output),
                     "--model",self.a.model,"--sam-url",self.a.sam_url,"--max-steps",str(self.a.max_steps)]
            if adapter:command.extend(["--adapter",str(adapter)])
            workers.append(self.start(name.replace('/','_')+f"_gpu{gpu}",command,[gpu]))
        self.wait(workers,name);write_json(output/"DONE.json",{"jobs":job_list,"adapter":str(adapter) if adapter else None})
        return output
    def train(self,name,stage,adapter=None,data=None,updates=1,seeds=None,tasks=None):
        from .success_gate import require_gate
        require_gate(self.output)
        output=self.output/name
        if not (output/"DONE.json").is_file():
            command=[sys.executable,"-m","torch.distributed.run","--standalone",f"--nproc_per_node={len(self.a.policy_gpus)}",
                     "--max_restarts=0","-m","tube_agent.lab.distributed_train",stage,"--output",str(output),
                     "--model",self.a.model,"--sam-url",self.a.sam_url,"--updates",str(updates),
                     "--group-size",str(self.a.group_size),"--max-steps",str(self.a.max_steps),
                     "--epochs",str(self.a.sft_epochs),"--scene-seeds",*[str(s) for s in (seeds or [self.a.seed])]]
            if adapter:command.extend(["--adapter",str(adapter)])
            if data:command.extend(["--data",str(data)])
            if tasks:command.extend(["--tasks",*tasks])
            process=self.start(name,command,self.a.policy_gpus);self.wait([process],name)
        if stage=="smoke":return None
        value=checkpoint(output)
        if value is None:raise RuntimeError(f"No complete checkpoint in {output}")
        return value
    def ensure_sam(self):
        import httpx
        from .vision import SamClient
        with httpx.Client(timeout=5,trust_env=False) as client:
            try:
                response=client.get(self.a.sam_url+"/health")
            except httpx.ConnectError:response=None
            if response is not None:
                response.raise_for_status();health=response.json()
                if health.get("backend")!="sam3.1_detector" or not health.get("ready"):raise RuntimeError("Port is occupied by a service other than ready native SAM")
                raise RuntimeError("SAM port is already occupied. Stop your old SAM terminal before starting this run, or select an unused --sam-url. This pipeline requires its own logged SAM process; no external service was stopped.")
        parsed=urlparse(self.a.sam_url)
        if parsed.hostname not in ("127.0.0.1","localhost") or parsed.scheme!="http":raise ValueError("Auto-start SAM requires a local HTTP URL")
        python=ROOT/".venv-sam/bin/python"
        if not python.exists():raise RuntimeError("Missing .venv-sam; run bash scripts/lab.sh prepare first")
        self.sam=self.start("sam",[str(python),"-m","tube_agent.lab.sam_server","--checkpoint",str(self.a.sam_checkpoint),
                                  "--port",str(parsed.port or 8001)],[self.a.sam_gpu])
        deadline=time.monotonic()+self.a.startup_timeout;last=0
        with httpx.Client(timeout=5,trust_env=False) as client:
            while time.monotonic()<deadline:
                if self.sam.poll() is not None:raise RuntimeError("SAM startup failed; see logs/sam.log")
                try:
                    response=client.get(self.a.sam_url+"/health")
                    if response.is_success and response.json().get("ready"):
                        segmenter=SamClient(self.a.sam_url)
                        try:segmenter.health()
                        finally:segmenter.close()
                        return
                except httpx.TransportError:pass
                if time.monotonic()-last>30:self.status("sam_loading",log=str(self.output/"logs/sam.log"));last=time.monotonic()
                time.sleep(2)
        raise TimeoutError("SAM startup timeout; inspect logs/sam.log")
    def close(self):
        for name,process,stream in reversed(self.children):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
            stream.close()

def gpu_check(a):
    output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,memory.total","--format=csv,noheader,nounits"],text=True)
    cards={int(parts[0]):{"free_mb":int(parts[1]),"total_mb":int(parts[2])}
           for line in output.splitlines() if (parts:=line.split(','))}
    for gpu in [*a.policy_gpus,a.sam_gpu]:
        if gpu not in cards:raise RuntimeError(f"GPU {gpu} not found")
    for gpu in a.policy_gpus:
        if cards[gpu]["free_mb"]<a.min_free_gpu_gb*1024:
            raise RuntimeError(f"GPU {gpu} has only {cards[gpu]['free_mb']} MiB free. Stop your previous Qwen server/experiments on this GPU, then resume. No external process was stopped.")
    return cards

def make_videos(runner):
    output=runner.output;videos={}
    # Prefer a fresh final-policy evaluation, then other model successes.
    sources=["final_fixed","final_test","base_test",*[p.name for p in sorted(output.glob("mine_round_*"))],"qualification","mine_warm","mine_base","warmup/rollouts","rl_fixed/rollouts","rl_diverse/rollouts","base_fixed"]
    for task in TASK_NAMES:
        marker=output/"videos"/(task+".json")
        if marker.exists() and read_json(marker).get("status")=="verified_success":
            videos[task]=read_json(marker);continue
        candidates=[]
        for stage in sources:
            group=[(path,r) for path,r in successful_results(output/stage) if r.get("task")==task]
            candidates.extend((stage,path,r) for path,r in sorted(group,key=lambda pair:(pair[1].get("seed")!=runner.a.seed,pair[1].get("invalid_actions",0),pair[1].get("steps",1000))))
        attempts=[]
        for index,(stage,path,result) in enumerate(candidates[:3]):
            directory=output/"videos"/f"{task}_{index:02d}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
            command=[sys.executable,"-m","tube_agent.lab.replay","--episode",str(path.parent),"--output",str(directory)]
            process=runner.start(f"video_{task}_{index}",command,[runner.a.policy_gpus[0]])
            try:
                runner.wait([process],"video_"+task)
                report=read_json(directory/"report.json")
                if not all(report["state_matches"]) or not report["replay_result"]["success"]:raise RuntimeError("Replay mismatch")
                videos[task]={"status":"verified_success","video":str(directory/"episode.mp4"),
                              "source":str(path.parent),"stage":stage,"seed":result.get("seed"),"split":result.get("split"),
                              "generator_provenance":result.get("generator_provenance"),"search_prefix_length":result.get("search_prefix_length",0)}
                break
            except RuntimeError as error:attempts.append({"source":str(path.parent),"error":str(error)})
        if task not in videos:
            videos[task]={"status":"no_success" if not candidates else "replay_failed","attempts":attempts}
        write_json(marker,videos[task])
    write_json(output/"videos/index.json",videos)
    return videos

def report(output,status,error=None):
    output=Path(output)
    value={"status":status,"error":error,"pipeline_version":PIPELINE_VERSION,
           "selection":read_json(output/"selection.json") if (output/"selection.json").exists() else None,
           "base_fixed":summaries(output/"base_fixed"),"final_fixed":summaries(output/"final_fixed"),
           "base_test":summaries(output/"base_test"),"final_test":summaries(output/"final_test"),
           "videos":read_json(output/"videos/index.json") if (output/"videos/index.json").exists() else {}}
    value["success_gate"]=read_json(output/"success_gate.json") if (output/"success_gate.json").exists() else None
    value["all_five_have_success_videos"]=all(value["videos"].get(t,{}).get("status")=="verified_success" for t in TASK_NAMES)
    value["mining"]={name:summaries(output/name) for name in ("mine_base","mine_warm")}
    write_json(output/"report.json",value)
    lines=["# Visual lab experiment",f"\nStatus: {status}","\nCompleted execution does not mean every task succeeded.",
           "\n| Task | Base fixed seed | Final fixed seed | Base held-out scenes | Final held-out scenes | Video |",
           "|---|---|---|---|---|---|"]
    def cell(stage,task):
        item=value[stage].get(task);return f"{item['successes']}/{item['episodes']}" if item else "not run"
    for task in TASK_NAMES:
        video=value["videos"].get(task,{})
        lines.append("| "+" | ".join([task,*[cell(s,task) for s in ("base_fixed","final_fixed","base_test","final_test")],video.get("video",video.get("status","pending"))])+" |")
    lines.extend(["\nSearch traces are data-mining results, not independent evaluation success rates.",
                  "rack_transfer is held out from all training and checkpoint selection.",
                  "Videos may come from mining or earlier checkpoints; videos/index.json records the source."])
    if error:lines.append("\nError: "+error)
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return value

def execute_pipeline(runner):
    a=runner.a;out=runner.output
    runner.status("preflight",gpus=gpu_check(a))
    if shutil.disk_usage(out).free<30*2**30:raise RuntimeError("At least 30 GiB free disk space is required for trajectories/checkpoints")
    runner.ensure_sam()
    # Failed diagnostic directories are preserved; a new one is used on retry.
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    runner.step("render_check",[sys.executable,"scripts/check_render.py","--output",str(out/"preflight"/stamp/"render")],[a.policy_gpus[0]])
    runner.step("physics_cameras",[sys.executable,"-m","tube_agent.lab.doctor","--output",str(out/"preflight"/stamp/"physics")],[a.policy_gpus[0]])
    # No optimizer steps (including smoke) until acquisition AND qualification pass.
    from .success_gate import inspect_gate, diagnostics, require_gate
    runner.sample("base_fixed",None,jobs(TRAIN_TASK_NAMES,[a.seed],"train",a.fixed_attempts))
    sources=[out/"base_fixed"]
    for round_index in range(a.search_rounds):
        current=inspect_gate(out,sources,out/"qualification",a.seed,a.min_successes,
                             a.qualification_attempts,a.qualification_successes)
        if current["acquisition_ready"]:break
        missing=[t for t,n in current["unique_verified_successes"].items() if n<a.min_successes]
        name=f"mine_round_{round_index:03d}"
        runner.sample(name,None,jobs(missing,[a.seed],"train",a.search_attempts,"search",
                                    a.min_successes,attempt_start=1000+round_index*a.search_attempts))
        sources.append(out/name)
        current=inspect_gate(out,sources,out/"qualification",a.seed,a.min_successes,
                             a.qualification_attempts,a.qualification_successes)
        diagnostics(out,sources,a.seed)
        runner.status("acquisition_progress",round=round_index+1,
                      verified_successes=current["unique_verified_successes"])
        if not any(current["unique_verified_successes"].values()):
            runner.status("acquisition_needs_debug",message="No verified success in the first batch. Inspect saved images/actions before spending more GPU time.")
            break
    current=inspect_gate(out,sources,out/"qualification",a.seed,a.min_successes,
                         a.qualification_attempts,a.qualification_successes)
    if current["acquisition_ready"]:
        # No prefix replay; new sampling RNG, same initial scene and base policy.
        runner.sample("qualification",None,jobs(TRAIN_TASK_NAMES,[a.seed],"train",
                      a.qualification_attempts,attempt_start=100000))
        current=inspect_gate(out,sources,out/"qualification",a.seed,a.min_successes,
                             a.qualification_attempts,a.qualification_successes)
    if not current["passed"]:
        diagnostics(out,sources,a.seed)
        make_videos(runner)
        report(out,"awaiting_success")
        runner.status("awaiting_success",training_started=False,gate=str(out/"success_gate.json"),
                      diagnostics=str(out/"acquisition_diagnostics.json"),
                      message="Success gate not met. No SFT, RL or optimizer smoke ran. Inspect evidence before more sampling.")
        return
    require_gate(out)
    runner.train("training_smoke","smoke")
    data=out/"data/self_success.jsonl";export_report=out/"data/export.json"
    # Only replay-verified admitted episodes enter SFT.
    admitted=[Path(record["result"]).parent for records in current["approved"].values() for record in records]
    info=export_sources(admitted,data);write_json(export_report,info)
    if not info["actions"] or set(info["episodes_by_task"])!=set(TRAIN_TASK_NAMES):
        raise RuntimeError("Verified successes did not export for every task; training stopped")
    candidates=[("base",None,out/"validation_base")]
    adapter=runner.train("sft","sft",data=data)
    candidates.append(("sft",adapter,out/"validation_sft"))
    adapter=runner.train("rl_fixed","grpo",adapter=adapter,updates=a.fixed_updates)
    candidates.append(("rl_fixed",adapter,out/"validation_rl_fixed"))
    adapter=runner.train("rl_diverse","grpo",adapter=adapter,updates=a.diverse_updates,seeds=[a.seed+i for i in range(4)])
    candidates.append(("rl_diverse",adapter,out/"validation_rl_diverse"))
    for name,candidate,directory in candidates:
        runner.sample(directory.name,candidate,jobs(TRAIN_TASK_NAMES,[0,1],"validation",a.validation_attempts))
    selected,scores=select_candidate(candidates)
    write_json(out/"selection.json",{"selected":selected,"candidates":scores,"criterion":"validation macro success, then mean reward; no test or rack_transfer selection"})
    final=Path(selected["adapter"]) if selected["adapter"] else None
    runner.sample("base_test",None,jobs(TASK_NAMES,a.test_seeds,"test",a.test_attempts))
    runner.sample("final_test",final,jobs(TASK_NAMES,a.test_seeds,"test",a.test_attempts))
    runner.sample("final_fixed",final,jobs(TRAIN_TASK_NAMES,[a.seed],"train",a.fixed_attempts))
    make_videos(runner)
    summary=report(out,"completed")
    runner.status("completed",report=str(out/"report.json"),videos=str(out/"videos/index.json"),
                  all_five_have_success_videos=summary["all_five_have_success_videos"])

def parser():
    from tube_agent.model_utils import DEFAULT_MODEL
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("runs/lab_success_first/fixed_seed0_v2"));p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--policy-gpus",nargs="+",type=int,default=[0,2,3]);p.add_argument("--sam-gpu",type=int,default=1)
    p.add_argument("--sam-url",default="http://127.0.0.1:8001")
    p.add_argument("--sam-checkpoint",type=Path,default=ROOT/"assets/sam3.1/sam3.1_multiplex.pt")
    p.add_argument("--seed",type=int,default=0);p.add_argument("--max-steps",type=int,default=40)
    p.add_argument("--search-attempts",type=int,default=16);p.add_argument("--min-successes",type=int,default=10)
    p.add_argument("--search-rounds",type=int,default=4)
    p.add_argument("--qualification-attempts",type=int,default=10)
    p.add_argument("--qualification-successes",type=int,default=3)
    p.add_argument("--fixed-updates",type=int,default=12)
    p.add_argument("--diverse-updates",type=int,default=16);p.add_argument("--group-size",type=int,default=6)
    p.add_argument("--sft-epochs",type=int,default=1);p.add_argument("--fixed-attempts",type=int,default=3)
    p.add_argument("--validation-attempts",type=int,default=2);p.add_argument("--test-attempts",type=int,default=2)
    p.add_argument("--test-seeds",nargs="+",type=int,default=[0,1,2]);p.add_argument("--cpu-threads",type=int,default=4)
    p.add_argument("--min-free-gpu-gb",type=int,default=48);p.add_argument("--startup-timeout",type=int,default=900)
    p.add_argument("--plan",action="store_true",help="Print budgets and GPU assignment without loading models")
    return p

def main():
    p=parser();a=p.parse_args()
    if len(set(a.policy_gpus))!=len(a.policy_gpus) or a.sam_gpu in a.policy_gpus:p.error("Policy GPU IDs must be unique and exclude SAM GPU")
    if a.group_size<2 or a.group_size%len(a.policy_gpus):p.error("Group size must be >=2 and divisible by policy GPU count")
    if not 0<=a.seed<99997 or any(not 0<=s<100000 for s in a.test_seeds):p.error("Seed out of range")
    if any(getattr(a,k)<1 for k in ("max_steps","search_attempts","min_successes","search_rounds","qualification_attempts","qualification_successes","fixed_updates","diverse_updates","sft_epochs","fixed_attempts","validation_attempts","test_attempts","cpu_threads")):p.error("Budgets must be positive")
    if a.qualification_successes>a.qualification_attempts:p.error("Qualification successes exceed attempts")
    if a.search_rounds*a.search_attempts>=99000:p.error("Acquisition RNG range would overlap qualification")
    budget={"base_and_final_fixed":8*a.fixed_attempts,"search_max":4*a.search_attempts*a.search_rounds,
            "qualification":4*a.qualification_attempts,
            "rl_max":(a.fixed_updates+a.diverse_updates)*a.group_size,
            "validation_max":4*4*2*a.validation_attempts,"base_and_final_test":2*5*len(a.test_seeds)*a.test_attempts,
            "training_smoke":len(a.policy_gpus)}
    if a.plan:
        print(json.dumps({"version":PIPELINE_VERSION,"policy_gpus":a.policy_gpus,"sam_gpu":a.sam_gpu,
                          "episode_budgets":budget,"max_total_episodes":sum(budget.values()),
                          "fixed_scene_seed":a.seed,"no_training_before_gate":True,
                          "gate":{"unique_successes_per_task":a.min_successes,"independent_successes":a.qualification_successes,"independent_attempts":a.qualification_attempts},"held_out":"rack_transfer","resume":"Repeat identical command and output path"},indent=2));return
    if sys.platform!="linux":p.error("Run the full pipeline on the Linux GPU instance; --plan works locally")
    import fcntl
    from tube_agent.model_utils import resolve_model
    a.output=a.output.resolve();a.model=str(resolve_model(a.model));a.sam_checkpoint=a.sam_checkpoint.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/".lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("This output directory already has a running pipeline")
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!="plan"}
    config.update(version=PIPELINE_VERSION,source_sha256=source_digest(),episode_budgets=budget)
    if (a.output/"config.json").exists() and read_json(a.output/"config.json")!=config:
        raise SystemExit("Code or settings changed since this run. Choose a new --output; old results are preserved.")
    write_json(a.output/"config.json",config)
    runner=Runner(a)
    def interrupted(signum,frame):raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if (a.output/"report.json").exists() and read_json(a.output/"report.json").get("status")=="completed":
            runner.status("already_completed",report=str(a.output/"report.json"));return
        execute_pipeline(runner)
    except BaseException as error:
        runner.status("interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=f"{type(error).__name__}: {error}")
        report(a.output,"interrupted" if isinstance(error,KeyboardInterrupt) else "failed",str(error));raise
    finally:runner.close();lock.close()

if __name__=="__main__":main()
