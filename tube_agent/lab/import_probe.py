"""Copy compatible completed probe successes for normal replay admission."""
import hashlib
import json
from pathlib import Path
import shutil
from .experiment import eligible, read_json, rows, write_json
from .success_gate import sha256


def import_probe(source, output, model, seed):
    from .env import LabEnv
    from .policy import SYSTEM, POLICY_VERSION, parse_visual_action
    from .vision import PERCEPTION_VERSION
    source=Path(source).resolve();output=Path(output)
    config=read_json(source/"config.json");status=read_json(source/"status.json")
    if not config.get("pick_probe") or status.get("stage")!="probe_complete":
        raise ValueError("Only a completed independent pick probe may be imported")
    if config.get("seed")!=seed or Path(config["model"]).resolve()!=Path(model).resolve():
        raise ValueError("Probe seed/model differs from this acquisition")
    signature=hashlib.sha256(SYSTEM.encode()).hexdigest();records=[];skipped=[]
    for path in sorted((source/"pick_probe").glob("*/try_*/result.json")):
        if ".incomplete-" in str(path):continue
        result=read_json(path)
        if not eligible(result):continue
        provenance=result.get("generator_provenance") or {}
        expected={"seed":seed,"policy_version":POLICY_VERSION,"prompt_sha256":signature,
                  "environment_version":LabEnv.VERSION,"action_protocol":LabEnv.ACTION_PROTOCOL,
                  "perception_version":PERCEPTION_VERSION,"search_prefix_length":0}
        if (any(result.get(k)!=v for k,v in expected.items()) or provenance.get("adapter")
            or provenance.get("training") is not False or not 200000<=result.get("attempt",-1)<203000
            or Path(provenance.get("model","")).resolve()!=Path(model).resolve()):
            skipped.append({"source":str(path),"reason":"incompatible policy or provenance"});continue
        trajectory=path.parent/"trajectory.jsonl";trace=rows(trajectory)
        if not trace or len(trace)!=result["steps"]:raise ValueError("Incomplete probe trajectory: "+str(path))
        for row in trace:
            if row["messages"][0]!={"role":"system","content":SYSTEM}:
                raise ValueError("Probe system prompt differs: "+str(path))
            if parse_visual_action(row["raw_response"])[0]!=row["action"]:
                raise ValueError("Recorded action differs from model output: "+str(path))
        identity=hashlib.sha256((sha256(path)+sha256(trajectory)).encode()).hexdigest()
        destination=output/(result["task"]+"_"+identity[:24])
        destination.mkdir(parents=True,exist_ok=True)
        for file in (path,trajectory):
            target=destination/file.name
            if target.exists() and sha256(target)!=sha256(file):
                raise ValueError("Imported evidence changed; choose a new output")
            if not target.exists():
                temporary=target.with_suffix(target.suffix+".partial")
                shutil.copyfile(file,temporary);temporary.replace(target)
        records.append({"source":str(path.parent),"copy":str(destination),"sha256":identity})
    manifest={"source":str(source),"candidates":len(records),"records":records,"skipped":skipped,
              "admission":"Candidates only. Gate must physically replay and deduplicate before counting."}
    previous=output/"IMPORT.json"
    if previous.exists() and read_json(previous)!=manifest:
        raise ValueError("Probe source changed since import; preserve this run and choose a new output")
    write_json(previous,manifest)
    return manifest
