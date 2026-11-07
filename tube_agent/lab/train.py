"""Single-GPU multimodal LoRA SFT and online GRPO, without a serving dependency.

GRPO uses groups on one identical initial scene, terminal ladder rewards,
per-turn invalid-action negative advantages, clipped token likelihood ratios,
and no reference KL (beta=0). This is an explicit small-scale implementation,
not a port of the paper's full distributed optimizer or hyperparameters.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import os
import numpy as np
from tube_agent.model_utils import DEFAULT_MODEL,resolve_model
from tube_agent.serve import build_inputs
from .env import TRAIN_TASKS
from .run import episode
from .vision import SamClient

def advantages(rewards,normalize_std=True):
    values=np.asarray(rewards,dtype=float)
    if not np.isfinite(values).all():raise ValueError("Nonfinite reward")
    centered=values-values.mean()
    return centered/(values.std()+1e-6) if normalize_std else centered

def clipped_loss(new_logp,old_logp,advantage,clip=.2):
    import torch
    ratio=torch.exp(new_logp-old_logp)
    return -torch.minimum(ratio*advantage,torch.clamp(ratio,1-clip,1+clip)*advantage).mean()


def dr_grpo_loss(new_logp,old_logp,advantage,token_budget,clip=.2):
    """Fixed maximum-episode token denominator; no sampled-length normalization."""
    import torch
    if token_budget<1:raise ValueError("Positive token budget required")
    ratio=torch.exp(new_logp-old_logp)
    return -torch.minimum(ratio*advantage,torch.clamp(ratio,1-clip,1+clip)*advantage).sum()/token_budget

def load(model_path,adapter=None,rank=16,device="cuda:0",trainable=True):
    import torch
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model,PeftModel
    if not torch.cuda.is_available():raise RuntimeError("Run training on a GPU instance")
    os.environ["HF_HUB_OFFLINE"]="1"
    path=str(resolve_model(model_path))
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
                         attn_implementation="sdpa",local_files_only=True).to(device)
    if adapter:model=PeftModel.from_pretrained(model,adapter,is_trainable=trainable,local_files_only=True)
    elif trainable:
        targets=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear) and ".language_model." in name]
        if not targets:raise RuntimeError("No Qwen language-model LoRA targets found")
        model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.,
                            target_modules=targets,task_type="CAUSAL_LM"))
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        model.config.use_cache=False
        model.config.text_config.use_cache=False
    else:model.eval()
    return model,processor

def sample_inputs(processor,row,max_context=16384):
    import torch
    inputs=build_inputs(processor,row["messages"])
    prompt_length=inputs["input_ids"].shape[1]
    response=row.get("response_ids")
    if response is None:
        response=processor.tokenizer.encode(row["raw_response"],add_special_tokens=False)+[processor.tokenizer.eos_token_id]
    if not response or prompt_length+len(response)>max_context:raise ValueError("Empty or overlong training response")
    ids=torch.tensor([response],dtype=torch.long)
    for key,value in list(inputs.items()):
        if key=="input_ids":inputs[key]=torch.cat([value,ids],dim=1)
        elif key=="attention_mask":inputs[key]=torch.cat([value,torch.ones_like(ids)],dim=1)
        elif key in ("token_type_ids","mm_token_type_ids"):
            inputs[key]=torch.cat([value,torch.zeros_like(ids)],dim=1)
    return inputs,prompt_length

def token_logps(model,inputs,prompt_length):
    import torch
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    response_length=inputs["input_ids"].shape[1]-prompt_length
    output=model(**inputs,use_cache=False,logits_to_keep=response_length+1)
    # Only response predictions are retained. FP32 logsumexp avoids BF16 loss.
    logits=output.logits[0,:-1].float()
    target=inputs["input_ids"][0,prompt_length:]
    return logits.gather(-1,target[:,None]).squeeze(-1)-torch.logsumexp(logits,dim=-1)

class LocalGenerator:
    def __init__(self,model,processor,max_context=16384):self.model,self.processor,self.max_context=model,processor,max_context
    def __call__(self,messages,seed,temperature):
        import torch
        inputs=build_inputs(self.processor,messages).to(self.model.device)
        length=inputs["input_ids"].shape[1]
        if length+256>self.max_context:raise ValueError("Vision conversation exceeds configured context")
        torch.manual_seed(seed);self.model.eval()
        options={"max_new_tokens":256,"do_sample":temperature>0,"use_cache":True}
        if temperature>0:options.update(temperature=temperature,top_p=1.,top_k=0)
        with torch.inference_mode():generated=self.model.generate(**inputs,**options)
        self.last_response_ids=generated[0,length:].tolist()
        return self.processor.decode(generated[0,length:],skip_special_tokens=True)

def save(model,processor,path,metadata):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(path,safe_serialization=True);processor.save_pretrained(path)
    (path/"experiment.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

def export(source,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    rows=[];seen=set();episode_count=0
    for result_path in sorted(Path(source).rglob("result.json")):
        result=json.loads(result_path.read_text(encoding="utf-8"))
        if not (result.get("policy")=="model" and result.get("split")=="train" and result.get("task") in TRAIN_TASKS
                and result.get("observation_mode")=="rgbd_sam3.1_detector" and result.get("demonstration") is None
                and result.get("success") and not result.get("error") and result.get("invalid_actions")==0):continue
        trajectory=[json.loads(line) for line in (result_path.parent/"trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        added=False
        for row in trajectory:
            # Deduplicate identical state/prompt + response, including repeated search prefixes.
            signature=hashlib.sha256(json.dumps([row["messages"],row["raw_response"]],sort_keys=True).encode()).hexdigest()
            if signature in seen:continue
            seen.add(signature);rows.append({**row,"source":result});added=True
        episode_count+=int(added)
    if not rows:raise ValueError("No eligible visual self-success training actions found; nothing written")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row)+"\n")
    print(json.dumps({"contributing_episodes":episode_count,"unique_actions":len(rows),"output":str(output)}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("export","sft","grpo"));p.add_argument("--source",type=Path)
    p.add_argument("--data",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",default=DEFAULT_MODEL);p.add_argument("--adapter",type=Path)
    p.add_argument("--rank",type=int,default=16);p.add_argument("--learning-rate",type=float,default=1e-5)
    p.add_argument("--epochs",type=int,default=1);p.add_argument("--group-size",type=int,default=8)
    p.add_argument("--updates",type=int,default=20);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--tasks",nargs="+",choices=TRAIN_TASKS,default=list(TRAIN_TASKS))
    p.add_argument("--scene-seeds",nargs="+",type=int,default=[0]);p.add_argument("--max-steps",type=int,default=40)
    p.add_argument("--max-context",type=int,default=16384);p.add_argument("--sam-url",default="http://127.0.0.1:8001")
    a=p.parse_args()
    if a.stage=="export":
        if a.source is None:p.error("export requires --source")
        export(a.source,a.output);return
    if a.output.exists():p.error("Output directory already exists; choose a new one to preserve checkpoints")
    if a.group_size<2 or a.updates<1 or a.epochs<1 or a.max_steps<1 or any(not 0<=s<100000 for s in a.scene_seeds):p.error("Invalid training settings")
    if a.stage=="sft" and not a.data:p.error("sft requires --data")
    import torch
    torch.manual_seed(a.seed);random.seed(a.seed)
    segmenter=None
    if a.stage=="grpo":segmenter=SamClient(a.sam_url);sam_health=segmenter.health()
    else:sam_health=None
    a.output.mkdir(parents=True,exist_ok=False)
    metadata={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    metadata.update(vision_encoder_trained=False,grpo=a.stage=="grpo",kl_beta=0.,clip=.2,
                    rollout_temperature=1.,sam_service=sam_health,training_split="train",held_out_tasks=["rack_transfer"])
    (a.output/"config.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    try:
        model,processor=load(a.model,a.adapter,a.rank)
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=a.learning_rate)
        if a.stage=="sft":
            data=[json.loads(s) for s in a.data.read_text(encoding="utf-8").splitlines() if s.strip()]
            if not data:raise ValueError("Empty SFT data")
            for row in data:
                src=row["source"]
                if src.get("task") not in TRAIN_TASKS or src.get("split")!="train" or src.get("policy")!="model" or src.get("demonstration") is not None or src.get("observation_mode")!="rgbd_sam3.1_detector":
                    raise ValueError("Only visual, self-generated TRAIN successes may train this stage")
                if not src.get("success") or src.get("error") or src.get("invalid_actions"):raise ValueError("Invalid SFT success provenance")
            for epoch in range(a.epochs):
                random.shuffle(data);model.train();optimizer.zero_grad()
                for index,row in enumerate(data):
                    inputs,length=sample_inputs(processor,row,a.max_context)
                    loss=-token_logps(model,inputs,length).mean()
                    window=min(8,len(data)-(index//8)*8)
                    (loss/window).backward()
                    if (index+1)%8==0 or index+1==len(data):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad()
                    print(json.dumps({"epoch":epoch,"action":index,"loss":float(loss.detach())}),flush=True)
                save(model,processor,a.output/f"epoch_{epoch+1:03d}",metadata)
        else:
            generator=LocalGenerator(model,processor,a.max_context)
            for update in range(a.updates):
                task=a.tasks[update%len(a.tasks)];seed=a.scene_seeds[(update//len(a.tasks))%len(a.scene_seeds)]
                results=[];trajectories=[]
                for attempt in range(a.group_size):
                    result,rows=episode(task,"train",seed,update*a.group_size+attempt,a.output/f"rollouts/update_{update:04d}/try_{attempt:03d}",
                                        generator,segmenter,1.,a.max_steps,save_frames=False)
                    if result["error"]:raise RuntimeError(result["error"])
                    results.append(result);trajectories.append(rows)
                adv=advantages([r["reward"] for r in results]);old=[]
                model.eval()
                with torch.no_grad():
                    for rows in trajectories:
                        old.append([token_logps(model,*sample_inputs(processor,row,a.max_context)).cpu() for row in rows])
                # Train mode enables checkpointing; keep dropout off so policy
                # likelihoods match generation. Qwen uses layer norms, not BN.
                model.train()
                for module in model.modules():
                    if isinstance(module,torch.nn.Dropout):module.eval()
                optimizer.zero_grad();loss_sum=0.;invalid_turns=0
                for i,rows in enumerate(trajectories):
                    for j,row in enumerate(rows):
                        value=float(adv[i])
                        if row["feedback"].get("accepted") is False:value=-1.;invalid_turns+=1
                        if value==0:continue
                        current=token_logps(model,*sample_inputs(processor,row,a.max_context))
                        loss=clipped_loss(current,old[i][j].to(model.device),value)/len(rows)/a.group_size
                        loss.backward();loss_sum+=float(loss.detach())
                if any(v.grad is not None for v in model.parameters()):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                optimizer.zero_grad()
                report={"update":update,"task":task,"scene_seed":seed,"successes":sum(r["success"] for r in results),
                        "group_size":a.group_size,"rewards":[r["reward"] for r in results],"advantages":adv.tolist(),
                        "invalid_turns":invalid_turns,"loss":loss_sum}
                with (a.output/"training.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(report)+"\n")
                print(json.dumps(report),flush=True)
                save(model,processor,a.output/f"update_{update+1:04d}",{**metadata,"completed_updates":update+1})
    finally:
        if segmenter:segmenter.close()

if __name__=="__main__":main()
