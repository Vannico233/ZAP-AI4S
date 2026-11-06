"""Reproducible scripted environment check; this never calls a language model."""
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from tube_agent.env import TubeEnv
from tube_agent.policy import ScriptedPolicy
from tube_agent.run import summarize


def main():
    results = {}
    for split in ("test", "ood"):
        for task in ("pick", "transport"):
            records = []
            for seed in range(20):
                env = TubeEnv(task=task, split=split, seed=seed)
                try:
                    policy = ScriptedPolicy(task)
                    observation, feedback = env.observe(), None
                    while not env.ended:
                        action, _ = policy.act(env.instruction, observation, feedback)
                        observation, feedback, _ = env.step(action)
                    result = env.result()
                    result["seed"] = seed
                    records.append(result)
                finally:
                    env.close()
            results[f"{task}/{split}"] = {"summary": summarize(records), "episodes": records}
            print(task, split, summarize(records), flush=True)
    output = Path("runs/physics_validation.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"policy": "scripted", "is_qwen_result": False,
                                 "python": platform.python_version(), "platform": platform.platform(),
                                 "mujoco": mujoco.__version__, "numpy": np.__version__,
                                 "results": results}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
