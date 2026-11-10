"""Replicated LoRA SFT/GRPO with explicit synchronous gradient sums.

Each rank owns a policy replica and different rollouts from the SAME scene and
policy version. Variable-length trajectories use local backward followed by one
gradient reduction, avoiding mismatched DDP backward counts. SAM stays external.
This is a small ZAP-method adaptation (LoRA, beta=0), not the original trainer.
"""
import argparse
from datetime import timedelta,datetime,timezone
import json
import math
import os
from pathlib import Path
import random
from .experiment import checkpoint,eligible,read_json,write_json,rows
from .credit import turn_advantage, credit_metrics


def task_schedule(path,tasks,updates):
    schedule=read_json(path) if path else [tasks[i%len(tasks)] for i in range(updates)]
    if (not isinstance(schedule,list) or len(schedule)!=updates
        or any(t not in tasks for t in schedule)):
        raise ValueError("Task schedule must contain exactly --updates training task names")
    return schedule

def synchronize_gradients(parameters,distributed=None):
    """SUM gradients already normalized by global batch/group size."""
    import torch
    parameters=list(parameters)
    active=torch.tensor([int(any(p.grad is not None for p in parameters))],device=parameters[0].device)
    if distributed is not None:distributed.all_reduce(active)
    if not active.item():return False
    # PEFT normally uses FP32 adapters, but group by dtype defensively.
    groups={}
    for p in parameters:groups.setdefault((p.device,p.dtype),[]).append(p)
    for group in groups.values():
        flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in group])
        if distributed is not None:distributed.all_reduce(flat)
        offset=0
        for p in group:
            p.grad=flat[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
    return True

def commit_checkpoint(model,processor,optimizer,output,step,metadata):
    import torch
    output=Path(output);final=output/f"step_{step:05d}"
    temporary=output/f".step_{step:05d}.partial-{os.getpid()}-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
    temporary.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(temporary,safe_serialization=True);processor.save_pretrained(temporary)
    torch.save({"optimizer":optimizer.state_dict(),"step":step},temporary/"optimizer.pt")
    write_json(temporary/"experiment.json",metadata)
    write_json(temporary/"COMPLETE.json",{"step":step})
    temporary.rename(final)
    return final

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("smoke","sft","grpo"));p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",required=True);p.add_argument("--adapter",type=Path);p.add_argument("--data",type=Path)
    p.add_argument("--tasks",nargs="+",choices=("pick","transport","insert","unscrew"),default=["pick","transport","insert","unscrew"])
    p.add_argument("--task-schedule",type=Path)
    p.add_argument("--loss-normalization",choices=("grpo","dr_grpo"),default="grpo")
    p.add_argument("--failed-done-credit", choices=("legacy", "nonpositive"), default="legacy")
    p.add_argument("--scene-seeds",nargs="+",type=int,default=[0]);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--updates",type=int,default=24);p.add_argument("--group-size",type=int,default=6)
    p.add_argument("--epochs",type=int,default=1);p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--rank",type=int,default=16);p.add_argument("--learning-rate",type=float,default=1e-5)
    p.add_argument("--max-steps",type=int,default=40);p.add_argument("--max-context",type=int,default=16384)
    p.add_argument("--sam-url",default="http://127.0.0.1:8001")
    p.add_argument("--action-protocol",choices=("rpy_rounding_v2","zap_pose_v1"),default="rpy_rounding_v2")
    a=p.parse_args()
    world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"));local=int(os.environ.get("LOCAL_RANK","0"))
    if a.group_size<2 or a.group_size%world:p.error("group-size must be >=2 and divisible by policy GPU count")
    if min(a.updates,a.epochs,a.batch_size,a.max_steps)<1 or any(not 0<=s<100000 for s in a.scene_seeds):p.error("Invalid budgets/seeds")
    if a.stage=="sft" and a.data is None:p.error("SFT requires --data")
    schedule=task_schedule(a.task_schedule,a.tasks,a.updates)
    import torch
    import torch.distributed as dist
    from .train import load,LocalGenerator,advantages,sample_inputs,token_logps,clipped_loss,dr_grpo_loss
    from .run import episode
    from .vision import SamClient
    torch.cuda.set_device(local);device=f"cuda:{local}"
    distributed=dist if world>1 else None
    if distributed:dist.init_process_group("nccl",timeout=timedelta(hours=2))
    def barrier():
        if distributed:dist.barrier()
    def gather(value):
        if not distributed:return [value]
        values=[None]*world;dist.all_gather_object(values,value);return values
    torch.manual_seed(a.seed);random.seed(a.seed)
    a.output.mkdir(parents=True,exist_ok=True)
    metadata={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    metadata.update(world_size=world,gradient_reduction="sum_global_normalized",kl_beta=0.,clip=.2,
                    rollout_temperature=1.,top_p=1.,top_k=0,vision_encoder_trained=False,
                    held_out_tasks=["rack_transfer"],training_split="train",weight_decay=.01,
                    task_schedule_contents=schedule,max_response_tokens=256,
                    advantage_std_normalization=a.loss_normalization!="dr_grpo",
                    episode_token_denominator=a.max_steps*256 if a.loss_normalization=="dr_grpo" else "actual_turns_and_turn_tokens",
                    stapo=False,async_policy_staleness=0)
    if (a.output/"config.json").exists() and read_json(a.output/"config.json")!=metadata:
        raise ValueError("Training configuration changed; use a new output directory")
    if rank==0:write_json(a.output/"config.json",metadata)
    barrier();resume=checkpoint(a.output)
    model,processor=load(a.model,resume or a.adapter,a.rank,device=device)
    parameters=[v for v in model.parameters() if v.requires_grad]
    if distributed:
        for value in parameters:dist.broadcast(value.data,src=0)
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate)
    completed=0
    if resume:
        restored=torch.load(resume/"optimizer.pt",map_location=device,weights_only=True)
        optimizer.load_state_dict(restored["optimizer"]);completed=restored["step"]
    def optimize():
        updated=synchronize_gradients(parameters,distributed)
        if updated:
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return updated
    def train_mode():
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.eval()
    def record(step,report,total):
        if rank==0:
            write_json(a.output/"metrics"/f"step_{step:05d}.json",report)
            print(json.dumps(report),flush=True)
            if a.stage=="grpo" or step%4==0 or step==total:
                commit_checkpoint(model,processor,optimizer,a.output,step,metadata)
        barrier()
    segmenter=None
    try:
        if a.stage=="smoke":
            segmenter=SamClient(a.sam_url);segmenter.health()
            generator=LocalGenerator(model,processor,a.max_context)
            probe=a.output/f"probe_rank_{rank}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
            result,trajectory=episode("pick","train",0,rank,probe,generator,segmenter,1.,1,save_frames=False,log_progress=True,
                                      action_protocol=a.action_protocol)
            if result.get("error") or not trajectory:raise RuntimeError("Visual training probe failed: "+str(result))
            train_mode();loss=-token_logps(model,*sample_inputs(processor,trajectory[0],a.max_context)).mean()/world
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training probe loss")
            loss.backward()
            if not any(v.grad is not None and bool(v.grad.abs().sum()>0) for v in parameters):
                raise RuntimeError("No nonzero LoRA gradients from visual input")
            optimize()
            # Deliberately discard these parameters: this tests execution, not a
            # self-success training stage. Smoke trajectories never enter SFT.
            report={"rank":rank,"loss":float(loss.detach())*world,"cuda":torch.cuda.get_device_name(local),
                    "peak_allocated_gb":torch.cuda.max_memory_allocated(local)/2**30}
            reports=gather(report)
            if rank==0:write_json(a.output/"smoke_report.json",{"passed":True,"ranks":reports,"parameters_discarded":True})
        elif a.stage=="sft":
            data=rows(a.data)
            if not data or any(not eligible(r.get("source",{})) for r in data):
                raise ValueError("SFT accepts only genuine visual model TRAIN successes")
            per_epoch=math.ceil(len(data)/a.batch_size);total=per_epoch*a.epochs
            for epoch in range(a.epochs):
                order=list(range(len(data)));random.Random(a.seed+epoch).shuffle(order)
                for batch in range(per_epoch):
                    step=epoch*per_epoch+batch+1
                    if step<=completed:continue
                    indices=order[batch*a.batch_size:(batch+1)*a.batch_size]
                    train_mode();optimizer.zero_grad(set_to_none=True);loss_sum=0.
                    for index in indices[rank::world]:
                        loss=-token_logps(model,*sample_inputs(processor,data[index],a.max_context)).mean()/len(indices)
                        loss.backward();loss_sum+=float(loss.detach())
                    optimize();losses=gather(loss_sum)
                    record(step,{"stage":"sft","step":step,"epoch":epoch,"loss":sum(losses),"actions":len(indices)},total)
        else:
            segmenter=SamClient(a.sam_url);segmenter.health()
            generator=LocalGenerator(model,processor,a.max_context)
            for update in range(completed,a.updates):
                task=schedule[update];seed=a.scene_seeds[(update//len(a.tasks))%len(a.scene_seeds)]
                if rank==0:print(json.dumps({"event":"training_group_start","stage":a.output.name,
                    "update":update+1,"total_updates":a.updates,"task":task,"scene_seed":seed,
                    "group_size":a.group_size,"loss_normalization":a.loss_normalization}),flush=True)
                generator.provenance={"model":a.model,"training_stage":str(a.output),"policy_update":update,
                                      "rank":rank,"temperature":1.,"top_p":1.,"top_k":0,"on_policy":True,
                                      "training":True,"initial_adapter":str(a.adapter) if a.adapter else None}
                # An interrupted update is regenerated from the last complete optimizer
                # checkpoint; partial samples are preserved but never mixed with new ones.
                batch_dir=a.output/"rollouts"/f"update_{update:04d}"/f"rank_{rank:02d}"
                if batch_dir.exists():batch_dir.rename(batch_dir.with_name(batch_dir.name+".incomplete-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
                results=[];trajectories=[]
                for local_attempt in range(a.group_size//world):
                    attempt=rank*(a.group_size//world)+local_attempt
                    result,trajectory=episode(task,"train",seed,update*a.group_size+attempt,batch_dir/f"try_{attempt:03d}",
                        generator,segmenter,1.,a.max_steps,save_frames=False,log_progress=True,action_protocol=a.action_protocol)
                    if result.get("error"):raise RuntimeError(result["error"])
                    results.append(result);trajectories.append(trajectory)
                    print(json.dumps({"event":"training_episode","update":update+1,"rank":rank,
                                      "local_attempt":local_attempt+1,**result}),flush=True)
                all_results=[r for batch in gather(results) for r in batch]
                adv=advantages([r["reward"] for r in all_results],normalize_std=a.loss_normalization!="dr_grpo");old=[]
                model.eval()
                with torch.no_grad():
                    for trajectory in trajectories:
                        old.append([token_logps(model,*sample_inputs(processor,row,a.max_context)).cpu() for row in trajectory])
                train_mode();optimizer.zero_grad(set_to_none=True);loss_sum=0.;invalid_turns=0
                for i,trajectory in enumerate(trajectories):
                    for j,row in enumerate(trajectory):
                        value=float(adv[rank*(a.group_size//world)+i])
                        invalid_turns+=int(row["feedback"].get("accepted") is False)
                        value=turn_advantage(row,value,a.failed_done_credit)
                        if value==0:continue
                        current=token_logps(model,*sample_inputs(processor,row,a.max_context))
                        if a.loss_normalization=="dr_grpo":
                            loss=dr_grpo_loss(current,old[i][j].to(device),value,a.max_steps*256)/a.group_size
                        else:loss=clipped_loss(current,old[i][j].to(device),value)/len(trajectory)/a.group_size
                        loss.backward();loss_sum+=float(loss.detach())
                credit=credit_metrics(trajectories,adv[rank*(a.group_size//world):(rank+1)*(a.group_size//world)],a.failed_done_credit)
                updated=optimize();stats=gather({"loss":loss_sum,"invalid_turns":invalid_turns,**credit})
                record(update+1,{"stage":"grpo","step":update+1,"task":task,"scene_seed":seed,
                    "total_updates":a.updates,"loss_normalization":a.loss_normalization,
                    "successes":sum(r["success"] for r in all_results),"group_size":a.group_size,
                    "rewards":[r["reward"] for r in all_results],"advantages":adv.tolist(),
                    "optimizer_step_taken":updated,
                    "failed_done_credit":a.failed_done_credit,
                    **{key:sum(r[key] for r in stats) for key in credit},
                    "loss":sum(r["loss"] for r in stats),"invalid_turns":sum(r["invalid_turns"] for r in stats),
                    "reward_terms":[r["reward_terms"] for r in all_results]},a.updates)
        barrier()
        if rank==0:
            last=checkpoint(a.output)
            write_json(a.output/"DONE.json",{"checkpoint":str(last) if last else None,"config":metadata})
    finally:
        if segmenter:segmenter.close()
        if distributed:dist.destroy_process_group()

if __name__=="__main__":main()
