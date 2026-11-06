"""Native SAM 3.1 checkpoint detector, isolated from simulator/Qwen dependencies.

This uses the detector branch of the official multiplex checkpoint, recomputed
at action boundaries. It does not claim asynchronous video tracking parity.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import os
from pathlib import Path
import inspect
import traceback


def adapt_session_init(predictor):
    """Bridge the pinned base predictor's extra flag to SAM 3.1's init API.

    The native base always sends offload_state_to_cpu=False, but the multiplex
    model does not accept that argument. Drop only this unsupported false flag;
    never silently ignore a requested CPU offload or another unknown argument.
    """
    original = predictor.model.init_state
    if getattr(original, "_lab_init_compat", False):
        return True
    parameters = inspect.signature(original).parameters
    if "offload_state_to_cpu" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return False

    def compatible_init(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu:
            raise ValueError("This SAM 3.1 model does not support CPU state offloading")
        return original(*args, **kwargs)

    compatible_init._lab_init_compat = True
    predictor.model.init_state = compatible_init
    return True

def segment_snapshot(predictor,image,queries,threshold):
    """Official request schema, one fresh single-frame session per text query."""
    from .vision import encode_mask
    detections=[]
    for query in queries:
        session=predictor.handle_request({"type":"start_session","resource_path":[image]})["session_id"]
        try:
            output=predictor.handle_request({"type":"add_prompt","session_id":session,
                                            "frame_index":0,"text":query,
                                            "output_prob_thresh":threshold})["outputs"]
            for mask,score in zip(output["out_binary_masks"],output["out_probs"]):
                detections.append({"query":query,"score":float(score),"mask":encode_mask(mask)})
        finally:predictor.handle_request({"type":"close_session","session_id":session})
    return detections

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--port",type=int,default=8001)
    p.add_argument("--threshold",type=float,default=.35)
    a=p.parse_args()
    if not a.checkpoint.is_file():p.error("SAM checkpoint is missing; download on the CPU instance first")
    manifest_path=a.checkpoint.parent/"SAM_MANIFEST.json"
    if not manifest_path.is_file():p.error("Missing SAM_MANIFEST.json; use scripts/download_sam.py and upload the whole directory")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repo_id")!="facebook/sam3.1":p.error("Expected official facebook/sam3.1 weights")
    digest=hashlib.sha256()
    with a.checkpoint.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):digest.update(chunk)
    if digest.hexdigest()!=manifest.get("sha256") or a.checkpoint.stat().st_size!=manifest.get("bytes"):
        p.error("Checkpoint checksum mismatch; upload or download again")
    os.environ["HF_HUB_OFFLINE"]="1"
    import torch
    from sam3.model_builder import build_sam3_predictor
    from .vision import decode_image,encode_mask
    if not torch.cuda.is_available():raise RuntimeError("SAM service requires a CUDA GPU")
    predictor=build_sam3_predictor(checkpoint_path=str(a.checkpoint),version="sam3.1",
                                   compile=False,warm_up=False,use_fa3=False,use_rope_real=False,
                                   async_loading_frames=False,default_output_prob_thresh=a.threshold)
    checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    state=checkpoint.get("model",checkpoint)
    if not any(k.startswith("detector.") for k in state):
        raise ValueError("Expected official SAM 3.1 multiplex checkpoint with detector.* weights")
    # Do not accept the upstream loader's warning-only partial initialization.
    predictor.model.load_state_dict(state,strict=True)
    del checkpoint,state
    predictor.model.eval()
    session_compat=adapt_session_init(predictor)
    health={"ready":True,"backend":"sam3.1_detector","checkpoint_sha256":digest.hexdigest(),
            "checkpoint_revision":manifest.get("revision"),"checkpoint_source":manifest.get("source","huggingface"),
            "temporal_tracking":False,"threshold":a.threshold,"session_init_compat":session_compat}
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,body):
            payload=json.dumps(body).encode()
            try:
                self.send_response(status);self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):self.reply(200,health) if self.path=="/health" else self.reply(404,{"error":"not found"})
        def do_POST(self):
            if self.path!="/segment":return self.reply(404,{"error":"not found"})
            try:
                size=int(self.headers.get("Content-Length",0))
                if not 0<size<12*1024*1024:raise ValueError("Invalid request size")
                request=json.loads(self.rfile.read(size));queries=request["queries"]
                if not isinstance(queries,list) or not 1<=len(queries)<=8 or any(not isinstance(q,str) or not 1<=len(q)<=120 for q in queries):
                    raise ValueError("Expected 1-8 short text queries")
                img=decode_image(request["image"]);detections=[]
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    detections=segment_snapshot(predictor,img,queries,a.threshold)
                self.reply(200,{"detections":detections})
            except Exception as exc:
                traceback.print_exc()
                self.reply(400,{"error":f"{type(exc).__name__}: {exc}"})
    server=HTTPServer(("127.0.0.1",a.port),Handler)
    print(json.dumps(health),flush=True);print(f"Ready: http://127.0.0.1:{a.port}",flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=="__main__":main()
