"""One GPU replica: independent evaluation or model-only prefix exploration."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from .experiment import eligible,read_json,write_json,rows,fingerprint

def run_jobs(jobs,output,generator,segmenter,max_steps=40):
    # Imports stay here so plan/report tools do not require graphics libraries.
    from .run import episode,PrefixTree
    from tube_agent.run import summarize
    output=Path(output)
    for job in jobs:
        directory=output/job["id"];directory.mkdir(parents=True,exist_ok=True)
        if (directory/"DONE.json").is_file():continue
        tree=PrefixTree(max_steps);results=[];unique_successes=set()
        for attempt in range(job["attempts"]):
            location=directory/f"try_{attempt:03d}"
            prefix=tree.select() if job["mode"]=="search" else None
            if (location/"result.json").is_file() and not read_json(location/"result.json").get("error"):
                result=read_json(location/"result.json")
                trajectory=rows(location/"trajectory.jsonl") if job["mode"]=="search" else []
            else:
                if location.exists():
                    location.rename(location.with_name(location.name+".incomplete-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
                result,trajectory=episode(job["task"],job["split"],job["seed"],attempt,location,generator,segmenter,
                    temperature=job.get("temperature",.8),max_steps=max_steps,prefix=prefix,save_frames=False,log_progress=True)
                print(json.dumps({"event":"episode","job":job["id"],**result}),flush=True)
            if result.get("error"):raise RuntimeError(result["error"]+"; fix infrastructure and resume the same output directory")
            results.append(result)
            if job["mode"]=="search":tree.update(trajectory,result)
            if eligible(result):
                if not trajectory:trajectory=rows(location/"trajectory.jsonl")
                unique_successes.add(fingerprint([r["action"] for r in trajectory]))
            summary={**summarize(results),"job":job,"eligible_successes":sum(eligible(r) for r in results),
                     "unique_successes":len(unique_successes),
                     "interpretation":"Search yield, not independent policy success rate" if job["mode"]=="search" else "Independent rollouts"}
            write_json(directory/"summary.json",summary)
            if job["mode"]=="search" and summary["unique_successes"]>=job.get("min_successes",3):break
        write_json(directory/"DONE.json",summary)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",required=True);p.add_argument("--adapter",type=Path)
    p.add_argument("--sam-url",default="http://127.0.0.1:8001");p.add_argument("--max-steps",type=int,default=40)
    a=p.parse_args()
    from .train import load,LocalGenerator
    from .vision import SamClient
    model,processor=load(a.model,a.adapter,trainable=False)
    generator=LocalGenerator(model,processor)
    generator.provenance={"model":a.model,"adapter":str(a.adapter) if a.adapter else None,
                          "temperature_from_job":True,"top_p":1.,"top_k":0,"training":False}
    segmenter=SamClient(a.sam_url)
    try:
        segmenter.health();run_jobs(read_json(a.jobs),a.output,generator,segmenter,a.max_steps)
    finally:segmenter.close()

if __name__=="__main__":main()
