"""Run the registered Qwen checkpoint and a temporary native SAM service on the original node."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "run", "out", "model"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument("--port", type=int, default=18302)
    args = p.parse_args()
    root, run, out, model = [getattr(args, key).resolve() for key in ("root","run","out","model")]
    if out == run or out.is_relative_to(run):
        p.error("Smoke results must be outside the training run")
    here = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=False)
    def status(stage, **fields):
        value = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **fields)
        temp = out/"status.tmp"
        temp.write_text(json.dumps(value, indent=2)+"\n")
        temp.replace(out/"status.json")
        print(json.dumps(value),flush=True)

    def interrupted(sig, frame):
        raise InterruptedError(f"Received signal {sig}")
    signal.signal(signal.SIGTERM, interrupted)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
               PYOPENGL_PLATFORM="egl", MUJOCO_GL="egl", PYTHONFAULTHANDLER="1")
    env["PYTHONPATH"] = str(root)+os.pathsep+env.get("PYTHONPATH","")
    env["LD_LIBRARY_PATH"] = str(root/".runtime/egl-jammy/lib")+os.pathsep+env.get("LD_LIBRARY_PATH","")
    url = f"http://127.0.0.1:{args.port}"
    opener = build_opener(ProxyHandler({}))
    with (out/"sam.log").open("w") as samlog, (out/"sampling.log").open("w") as policylog:
        sam = policy = None
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1",args.port))
            status("starting_sam",url=url)
            sam = subprocess.Popen([str(root/".venv-sam/bin/python"),"-u","-m","tube_agent.lab.sam_server",
                                    "--checkpoint",str(root/"assets/sam3.1/sam3.1_multiplex.pt"),
                                    "--port",str(args.port)],cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES="1"),
                                   stdout=samlog,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                if sam.poll() is not None:
                    raise RuntimeError(f"SAM exited with {sam.returncode}")
                try:
                    with opener.open(url+"/health",timeout=3) as res:
                        health = json.load(res)
                except (URLError,OSError):
                    time.sleep(2)
                    continue
                if health.get("ready") and health.get("backend")=="sam3.1_detector":
                    break
                time.sleep(2)
            (out/"sam-health.json").write_text(json.dumps(health,indent=2)+"\n")
            status("sampling",url=url)
            command = ["bash",str(here/"with_mapped_egl.sh"),str(out/"bindings"),
                       str(root/".venv-linux/bin/python"),"-m","torch.distributed.run","--standalone",
                       "--nproc_per_node=3","--max_restarts=0","--",str(here/"real_sampling_smoke.py"),
                       "--run",str(run),"--out",str(out),"--model",str(model),"--sam-url",url]
            policy = subprocess.Popen(command,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES="0,2,3"),
                                      stdout=policylog,stderr=subprocess.STDOUT,start_new_session=True)
            while policy.poll() is None:
                if sam.poll() is not None:
                    raise RuntimeError(f"SAM exited with {sam.returncode} during sampling")
                time.sleep(1)
            if policy.returncode:
                raise RuntimeError(f"Sampling workers exited with {policy.returncode}")
            status("workers_passed")
        except BaseException as exc:
            status("failed",error=str(exc))
            raise
        finally:
            for proc in (policy,sam):
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()


if __name__ == "__main__":
    main()
