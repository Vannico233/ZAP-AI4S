"""Copy compatible completed probe successes for normal replay admission."""
import hashlib
import json
from pathlib import Path
import shutil
from .experiment import eligible, read_json, rows, write_json
from .success_gate import sha256


def import_probe(source, output, model, seed, action_protocol="rpy_rounding_v2"):
    from .env import LabEnv
    from .zap_env import ZapLabEnv
    from .policy import VisualPolicy, parse_visual_action
    from .vision import PERCEPTION_VERSION
    source=Path(source).resolve();output=Path(output)
    config=read_json(source/"config.json");status=read_json(source/"status.json")
    pose_probe=config.get("pose_probe",False)
    if not ((config.get("pick_probe") and status.get("stage")=="probe_complete")
            or (pose_probe and status.get("stage")=="pose_probe_complete")):
        raise ValueError("Only a completed independent pick/pose probe may be imported")
    if config.get("action_protocol","rpy_rounding_v2")!=action_protocol:
        raise ValueError("Probe action protocol differs; legacy RPY successes cannot enter a ZAP pose campaign")
    if config.get("seed")!=seed or Path(config["model"]).resolve()!=Path(model).resolve():
        raise ValueError("Probe seed/model differs from this acquisition")
    policy=VisualPolicy(None,action_protocol=action_protocol)
    environment=ZapLabEnv if action_protocol=="zap_pose_v1" else LabEnv
    signature=hashlib.sha256(policy.system.encode()).hexdigest();records=[];skipped=[]
    low,high=(300000,312000) if pose_probe else (200000,203000)
    for path in sorted((source/("pose_probe" if pose_probe else "pick_probe")).glob("*/try_*/result.json")):
        if ".incomplete-" in str(path):continue
        result=read_json(path)
        if not eligible(result):continue
        provenance=result.get("generator_provenance") or {}
        expected={"seed":seed,"policy_version":policy.version,"prompt_sha256":signature,
                  "environment_version":environment.VERSION,"action_protocol":action_protocol,
                  "perception_version":PERCEPTION_VERSION,"search_prefix_length":0}
        if (any(result.get(k)!=v for k,v in expected.items()) or provenance.get("adapter")
            or provenance.get("training") is not False or not low<=result.get("attempt",-1)<high
            or Path(provenance.get("model","")).resolve()!=Path(model).resolve()):
            skipped.append({"source":str(path),"reason":"incompatible policy or provenance"});continue
        trajectory=path.parent/"trajectory.jsonl";trace=rows(trajectory)
        if not trace or len(trace)!=result["steps"]:raise ValueError("Incomplete probe trajectory: "+str(path))
        for row in trace:
            if row["messages"][0]!={"role":"system","content":policy.system}:
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
