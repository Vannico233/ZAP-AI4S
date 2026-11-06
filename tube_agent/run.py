"""python -m tube_agent.run --policy scripted|model --task pick|transport"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from .env import TubeEnv, SPLIT_OFFSETS
from .policy import ModelPolicy, ScriptedPolicy, SYSTEM_PROMPTS, DEMONSTRATION_METADATA


def policy_metadata(args):
    variant = args.prompt_variant if args.policy == "model" else None
    result = {"prompt_variant": variant,
              "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPTS[variant].encode()).hexdigest() if variant else None,
              "feedback_version": "detailed_v1" if variant == "geometry_v1" else "basic_v1"}
    if variant == "manual_demo_v1":
        result["demonstration"] = dict(DEMONSTRATION_METADATA)
    return result


def summarize(records):
    n = len(records)
    successes = sum(bool(r["success"]) for r in records)
    p = successes/n if n else 0.
    z = 1.96
    denominator = 1 + z*z/n if n else 1
    center = (p + z*z/(2*n))/denominator if n else 0.
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator if n else 0.
    return {"episodes": n, "successes": successes, "success_rate": p,
            "wilson_95": [max(0., center-half), min(1., center+half)],
            "mean_steps": float(np.mean([r["steps"] for r in records])) if n else 0.,
            "invalid_actions": sum(r["invalid_actions"] for r in records),
            "errors": sum(r.get("error") is not None for r in records),
            "mean_reward": float(np.mean([r["reward"] for r in records])) if n else 0.}


def run_episode(args, seed, attempt, directory):
    directory.mkdir(parents=True, exist_ok=False)
    writer = None
    env = policy = None
    started = time.monotonic()
    error = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(directory/"episode.mp4"), fps=12.5, codec="libx264", quality=7)
    try:
        env = TubeEnv(args.task, args.split, seed, args.max_steps,
                      frame_callback=writer.append_data if writer else None,
                      detailed_feedback=args.prompt_variant == "geometry_v1")
        policy = (ScriptedPolicy(args.task) if args.policy == "scripted" else
                  ModelPolicy(args.base_url, args.model, args.temperature, seed*1009+attempt,
                              args.max_tokens, args.vision, args.prompt_variant))
        observation, feedback = env.observe(), None
        with (directory/"trajectory.jsonl").open("w", encoding="utf-8") as log:
            while not env.ended:
                before = observation
                frame = env.render() if args.vision else None
                action, raw = policy.act(env.instruction, before, feedback, frame)
                observation, feedback, ended = env.step(action)
                log.write(json.dumps({"step": env.steps, "observation": before,
                                      "action": action, "raw_response": raw, "feedback": feedback}, ensure_ascii=False)+"\n")
                log.flush()
        result = env.result()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = env.result() if env else {"reward": 0., "steps": 0, "invalid_actions": 0}
        result["success"] = False
        result["reward"] = 0.
    finally:
        if policy is not None:
            (directory/"messages.json").write_text(json.dumps(policy.history, ensure_ascii=False), encoding="utf-8")
            if hasattr(policy, "close"):
                policy.close()
        if env:
            env.close()
        if writer:
            writer.close()
    result.update(policy_metadata(args))
    result.update({"task": args.task, "split": args.split, "seed": seed, "attempt": attempt,
                   "policy": args.policy, "model": args.model if args.policy == "model" else None,
                   "observation_mode": "rgb_plus_oracle_state" if args.vision else "oracle_object_state",
                   "seconds": round(time.monotonic()-started, 2), "error": error,
                   "episode_dir": directory.name})
    (directory/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["scripted", "model"], default="scripted")
    p.add_argument("--prompt-variant", choices=list(SYSTEM_PROMPTS), default="base",
                   help="Optional geometry guidance or a manual demonstration in the system prompt; model policy only")
    p.add_argument("--task", choices=["pick", "transport"], default="transport")
    p.add_argument("--split", choices=list(SPLIT_OFFSETS), default="test")
    p.add_argument("--episodes", type=int, default=10, help="Number of distinct scene seeds")
    p.add_argument("--attempts", type=int, default=1, help="Independent model attempts per scene, useful for self-sampling")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen-tube")
    p.add_argument("--temperature", type=float, default=0.)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--vision", action="store_true", help="Attach images in addition to oracle object geometry")
    p.add_argument("--video", action="store_true", help="Requires EGL or OSMesa on headless Linux")
    p.add_argument("--output", type=Path, default=Path("runs"))
    args = p.parse_args()
    if args.policy != "model" and args.prompt_variant != "base":
        p.error("--prompt-variant requires --policy model")
    if args.prompt_variant == "manual_demo_v1" and args.task != "transport":
        p.error("manual_demo_v1 is a transport demonstration; use --task transport")
    if min(args.episodes, args.attempts, args.max_steps, args.max_tokens) < 1:
        p.error("episode, attempt, step and token counts must be positive")
    if not 0 <= args.seed_start < args.seed_start + args.episodes <= 100_000:
        p.error("scene seeds must stay in [0, 100000)")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        p.error("temperature must be finite and nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.output/f"{stamp}_{args.policy}_{args.task}_{args.split}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(policy_metadata(args))
    # Do not persist any endpoint credentials or query parameters.
    from urllib.parse import urlsplit, urlunsplit
    u = urlsplit(config["base_url"])
    config["base_url"] = urlunsplit((u.scheme, u.netloc.split("@")[-1], u.path, "", ""))
    config["versions"] = {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "httpx")}
    (directory/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = []
    for seed in range(args.seed_start, args.seed_start+args.episodes):
        for attempt in range(args.attempts):
            result = run_episode(args, seed, attempt, directory/f"seed_{seed:05d}_try_{attempt:02d}")
            records.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            (directory/"summary.json").write_text(json.dumps(summarize(records), indent=2), encoding="utf-8")
    summary = summarize(records)
    print(f"Saved: {directory}\n" + json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
