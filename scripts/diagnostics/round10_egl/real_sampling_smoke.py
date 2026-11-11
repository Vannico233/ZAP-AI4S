"""Run two real policy decisions per rank on round-10 scenes; diagnostic outputs only."""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import warnings

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from tube_agent.curriculum import rollout
from tube_agent.curriculum.config import digest, file_digest, load
from tube_agent.curriculum.learning import Generator, load_model
from tube_agent.lab.experiment import write_json as save
from tube_agent.lab.vision import Observer, SamClient, decode_image
from egl_devices import active_egl_device


class SampleLimit(BaseException):
    """End this diagnostic after complete, persisted turns, preserving episode cleanup."""


def checksum(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def inspect_frames(first, repeat, env):
    if [f["calibration"]["name"] for f in first] != ["top", "side"] or len(repeat) != 2:
        raise RuntimeError("Expected the two production cameras")
    audit = {}
    for a, b in zip(first, repeat):
        name = a["calibration"]["name"]
        rgb, depth = a["rgb"], a["depth"]
        if b["calibration"]["name"] != name:
            raise RuntimeError("Repeat capture changed camera")
        for frame in (a, b):
            color, z = frame["rgb"], frame["depth"]
            far = env.model.stat.extent * env.model.vis.map.zfar
            if color.shape != (480,640,3) or color.dtype != np.uint8 or color.std() < 1:
                raise RuntimeError(f"Invalid RGB image: {name}")
            if (z.shape != (480,640) or not np.issubdtype(z.dtype,np.floating)
                    or not np.isfinite(z).all() or z.min() <= 0 or z.max() > far * 1.01
                    or np.ptp(z) < .01 or not np.any((z > .01) & (z < 3))):
                raise RuntimeError(f"Invalid metric depth: {name}")
        rgb_drift = float((np.max(np.abs(rgb.astype(np.int16)-b["rgb"].astype(np.int16)),axis=2) > 8).mean())
        depth_drift = float((np.abs(depth-b["depth"]) > .001).mean())
        if rgb_drift > .001 or depth_drift > .001:
            raise RuntimeError(f"Unstable fixed pose: {name}: RGB={rgb_drift}, depth={depth_drift}")
        audit[name] = {"rgb_sha256":checksum(rgb), "depth_sha256":checksum(depth),
                       "rgb_std":float(rgb.std()), "depth_range_m":[float(depth.min()),float(depth.max())],
                       "rgb_repeat_fraction_over_8_levels":rgb_drift,
                       "depth_repeat_fraction_over_1mm":depth_drift}
    return audit


class AuditedSam(SamClient):
    def __init__(self, url):
        super().__init__(url)
        self.requests = Counter()

    def segment(self, rgb, queries):
        detected = super().segment(rgb, queries)
        self.requests[checksum(rgb)] += 1
        return detected


def run_rank(args, folder, rank):
    root = args.run.resolve()
    config = load(root/"config.json")
    adapter = root/"warmup/checkpoints/step_00009"
    ckpt = json.loads((adapter/"COMPLETE.json").read_text())
    if (ckpt["counters"]["round"] != 9 or ckpt["config_sha256"] != digest(config)
            or ckpt["adapter_sha256"] != file_digest(adapter/"adapter_model.safetensors")):
        raise ValueError("Round-9 adapter/config checksum mismatch")
    metrics = json.loads((root/"warmup/metrics/round_09.json").read_text())
    if Path(args.model).resolve() != Path(json.loads((root/"provenance.json").read_text())["model"]["resolved_path"]).resolve():
        raise ValueError("Model path differs from frozen experiment")
    seed = (10009012, 10009019, 10009026)[rank]
    recorded = json.loads((root/"schedule.json").read_text())[9]["groups"]
    [scene_id] = [g["scene_id"] for g in recorded for r in g["rollouts"] if r["policy_seed"] == seed]
    [scene] = [s for s in json.loads((root/"scenes.json").read_text())["scenes"] if s["id"] == scene_id]
    device = f"cuda:{rank}"
    print(f"Loading production sampler on {device}, scene {scene_id}",flush=True)
    model, processor = load_model(args.model,config,adapter,device)
    generator = Generator(model,processor,config)
    loaded = generator.bind("warmup_before_round_10",metrics["adapter_after"])
    segmenter = AuditedSam(args.sam_url)
    captures = []
    class SmokeObserver(Observer):
        def __init__(self, env, sam):
            super().__init__(env,sam)
            self.cameras.renderer._gl_context.make_current()
            if active_egl_device() != rank or "MUJOCO_EGL_DEVICE_ID" not in os.environ:
                raise RuntimeError("Active EGL context does not match the policy CUDA rank")
            production_capture = self.cameras.capture
            def checked_capture():
                with warnings.catch_warnings(record=True) as messages:
                    warnings.simplefilter("always")
                    frames = production_capture()
                    repeat = production_capture()
                if messages:
                    raise RuntimeError(f"Camera capture warning: {messages[0].message}")
                captures.append(inspect_frames(frames, repeat, self.env))
                return frames
            self.cameras.capture = checked_capture

        def observe(self, output=None):
            if len(captures) >= 2:
                raise SampleLimit()
            print(f"Rank {rank} capturing validated RGB-D for turn {len(captures)+1}",flush=True)
            return super().observe(output)

    path = folder/"sample"
    try:
        health = segmenter.health()
        dist.barrier(device_ids=[rank])
        with patch.object(rollout,"Observer",SmokeObserver):
            try:
                result = rollout.episode(scene,config,path,generator,segmenter,seed,
                                         "warmup_before_round_10",origin="diagnostic",save_frames=True)
            except SampleLimit:
                pass
            else:
                if result["error"]:
                    raise RuntimeError(result["error"])
    finally:
        segmenter.close()

    rows = list(rollout.read_rows(path))
    if not 1 <= len(rows) <= 2 or len(rows) != len(captures):
        raise RuntimeError("Missing or extra sampled turns")
    turns = []
    for i, (row, raw_audit) in enumerate(zip(rows,captures)):
        view = path/f"view_{i:03d}"
        inputs = [x["image_url"]["url"] for x in row["messages"][-1]["content"] if x["type"]=="image_url"]
        if len(inputs) != 2 or not row["response_ids"] or not row["raw_response"].strip() or not row["model_input_sha256"]:
            raise RuntimeError("Qwen did not consume both images and return a decoded response")
        masks = {}
        for name, value in zip(("top","side"),inputs):
            if not segmenter.requests[raw_audit[name]["rgb_sha256"]]:
                raise RuntimeError(f"SAM did not receive the raw {name} camera pixels")
            with Image.open(view/f"{name}_annotated.png") as file:
                if not np.array_equal(np.asarray(decode_image(value)),np.asarray(file)):
                    raise RuntimeError(f"Qwen image disagrees with saved {name} overlay")
            detected = json.loads((view/f"{name}_masks.json").read_text())
            masks[name] = [d["query"] for d in detected]
        turns.append({"step":row["step"],"frames":raw_audit,"sam_3d_queries":masks,
                      "qwen_input_sha256":row["model_input_sha256"],"generated_tokens":len(row["response_ids"]),
                      "action":row["action"]["action"],"accepted":row["feedback"]["accepted"]})
    return {"passed":True,"rank":rank,"scene":scene_id,"adapter_tensor_sha256":loaded,
            "sam_backend":health["backend"],"sam_requests":sum(segmenter.requests.values()),
            "real_decisions":len(turns),"egl_index":int(os.environ["MUJOCO_EGL_DEVICE_ID"]),
            "egl_cuda_ordinal":rank,"origin":"diagnostic","training_eligible":False,"turns":turns}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--model",required=True)
    p.add_argument("--sam-url",required=True)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if os.environ["WORLD_SIZE"] != "3" or int(os.environ["RANK"]) != rank:
        raise ValueError("Run on one node with three torchrun processes")
    if args.out.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Diagnostic output must be outside the training run")
    folder = args.out/f"rank-{rank}"
    folder.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl",timeout=timedelta(minutes=30))
    try:
        try:
            report = run_rank(args,folder,rank)
        except Exception as exc:
            save(folder/"report.json",{"passed":False,"rank":rank,"error":f"{type(exc).__name__}: {exc}"})
            raise
        save(folder/"report.json",report)
        reports = [None]*3
        dist.all_gather_object(reports,report)
        masks = sum(len(q) for r in reports for t in r["turns"] for q in t["sam_3d_queries"].values())
        actions = sum(t["action"] != "invalid_model_output" and t["accepted"] for r in reports for t in r["turns"])
        passed = bool(masks and actions)
        if rank == 0:
            summary = {"passed":passed,"sam_3d_detections":masks,"accepted_model_actions":actions,
                       "real_decisions":sum(r["real_decisions"] for r in reports),
                       "origin":"diagnostic","training_eligible":False,"ranks":reports}
            save(args.out/"summary.json",summary)
            print(json.dumps({k:v for k,v in summary.items() if k != "ranks"}),flush=True)
        if not passed:
            raise RuntimeError("Real sampler produced no usable SAM detection or accepted model action across the three ranks")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
