"""Execute a saved, hand-authored command list in one reproducible scene."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .env import TubeEnv, SPLIT_OFFSETS
from .policy import SYSTEM, observation_message


def run_commands(spec, output, *, verbose=False):
    """Preserve every attempted action; this is never a model-generated rollout."""
    if not isinstance(spec, dict) or spec.get("task") not in ("pick", "transport"):
        raise ValueError("Commands require task=pick or task=transport")
    if spec.get("split") not in SPLIT_OFFSETS:
        raise ValueError("Commands require a documented split")
    seed, max_steps = spec.get("seed"), spec.get("max_steps", 20)
    if type(seed) is not int or not 0 <= seed < 100_000:
        raise ValueError("Commands require an integer seed in [0, 100000)")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(spec.get("actions"), list):
        raise ValueError("Commands require an actions list")
    # Work from a copy and preserve invalid action fields for the environment to judge.
    spec = json.loads(json.dumps(spec))
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    source = {"task": spec["task"], "split": spec["split"], "seed": seed,
              "attempt": 0, "policy": "manual", "model": None,
              "observation_mode": "oracle_object_state", "feedback_version": "basic_v1",
              "provenance": "hand_authored", "prompt_variant": None,
              "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()}
    config = {"commands": spec, "source": source, "max_steps": max_steps,
              "versions": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy")},
              "judge_after_is_diagnostic_only": True}
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    messages = [{"role": "system", "content": SYSTEM}]
    env, error, executed = None, None, 0
    started = time.monotonic()
    try:
        env = TubeEnv(spec["task"], spec["split"], seed, max_steps)
        observation, feedback = env.observe(), None
        with (directory / "trajectory.jsonl").open("w", encoding="utf-8") as log:
            for action in spec["actions"]:
                if env.ended:
                    raise RuntimeError("Environment already ended; trailing commands were not executed")
                before = observation
                raw = json.dumps(action, ensure_ascii=True)
                messages.append({"role": "user", "content": observation_message(env.instruction, before, feedback)})
                messages.append({"role": "assistant", "content": raw})
                observation, feedback, _ = env.step(action)
                executed += 1
                row = {"step": env.steps, "observation": before, "action": action,
                       "raw_response": raw, "feedback": feedback,
                       "observation_after": observation, "judge_after": env.result()}
                log.write(json.dumps(row, ensure_ascii=True) + "\n")
                log.flush()
                if verbose:
                    print(json.dumps({k: row[k] for k in ("step", "action", "feedback", "judge_after")}), flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        result = (env.result() if env is not None else
                  {"success": False, "reward": 0., "steps": 0, "invalid_actions": 0})
        incomplete = env is None or not env.ended or executed != len(spec["actions"])
        if env is not None:
            env.close()
    result["final_judge"] = dict(result)
    if error or incomplete:
        # Keep the physical verdict for diagnosis, but never label an incomplete
        # or interrupted command sequence as a completed successful trajectory.
        result.update(success=False, reward=0.)
    result.update(source)
    result.update({"error": error, "incomplete": incomplete,
                   "commands_planned": len(spec["actions"]), "commands_executed": executed,
                   "seconds": round(time.monotonic() - started, 3), "episode_dir": directory.name})
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (directory / "messages.json").write_text(json.dumps(messages, ensure_ascii=True, indent=2), encoding="utf-8")
    (directory / "replay.jsonl").write_text(json.dumps({"messages": messages, "source": source}) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, required=True,
                        help="JSON object with task, split, seed, actions and optional max_steps")
    parser.add_argument("--output", type=Path, required=True, help="A new directory for this attempt")
    args = parser.parse_args()
    spec = json.loads(args.commands.read_text(encoding="utf-8-sig"))
    result = run_commands(spec, args.output, verbose=True)
    print(f"Saved: {args.output}\n" + json.dumps(result, indent=2), flush=True)
    if result["error"] or result["invalid_actions"] or result["incomplete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
