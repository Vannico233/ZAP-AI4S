"""One visual episode, with exact model contexts and deterministic prefix replay."""
import copy
import json
from pathlib import Path
import time
import numpy as np
from tube_agent import interface
from tube_agent.lab.policy import VisualPolicy
from tube_agent.interface import execute_action as execute
from tube_agent.lab.zap_policy import SYSTEM
from tube_agent.lab.experiment import write_json
from .config import digest


def system_prompt(c):
    g = c["geometry"]
    if c["action_protocol"] == interface.PROTOCOL:
        layout = ""
        if "middle_camera" in c["observation"]:
            layout = ("\nThe middle camera is mounted above the arm bases, looking down toward the racks. "
                      "The left and right arm names refer to the robots, even when their positions "
                      "in the image change. table_up_middle gives the tabletop's upward unit normal "
                      "in the same optical frame as both TCPs.")
        return interface.SYSTEM + layout + (
            f"\nThe red tube radius is {g['tube_radius']} metres. The square rack hole width is "
            f"{2*g['hole_half_width']} metres. The blue rack is the source, and the yellow rack is the target.")
    return SYSTEM.split("Known apparatus dimensions:")[0] + (
        f"Known apparatus dimensions: tube radius {g['tube_radius']} m, total length "
        f"{2*(g['tube_radius']+g['tube_half_segment'])} m. Rack hole width {2*g['hole_half_width']} m. "
        f"Both rack floors z={g['floor_z']} m. Blue source rim z={g['source_rim_z']} m; "
        f"yellow target rim z={g['target_rim_z']} m. The single hole is at the visible rack center. "
        "The task may start with the tube already physically held. TCP is not the tube bottom. "
        "Use RGB and depth to estimate the held-object offset. Keep the tube cap-up. "
        "There is no wait tool: update_todo changes memory only and consumes a turn. "
        "done ends the episode and allows 0.5 seconds of physical settling. "
        "Only finish when the object satisfies the instruction. An empty closed gripper is not a grasp."
    )


def context(policy, observer):
    return copy.deepcopy({"history": policy.history, "plan": policy.plan,
        "action_memory": policy.action_memory, "initial_visual_reference": policy.initial_visual_reference,
        "queries": observer.queries})


def restore_context(policy, observer, value):
    for key in ("history", "plan", "action_memory", "initial_visual_reference"):
        setattr(policy, key, copy.deepcopy(value[key]))
    observer.queries = list(value["queries"])


def same_state(first, second, atol=1e-5):
    return all(np.allclose(first[k], second[k], atol=atol, rtol=0) for k in ("physics", "judge"))


def read_rows(directory):
    with (Path(directory)/"trajectory.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _mount_hash(config):
    mount = config["observation"].get("middle_camera")
    return digest(mount) if mount is not None else None


def _check_action_frame(row, protocol, mount_hash):
    if row.get("action_protocol", "zap_pose_v1") != protocol:
        raise ValueError("Saved action protocol differs from this environment")
    if row.get("middle_camera_sha256") != mount_hash:
        raise ValueError("Saved action uses a different middle camera mount")


def episode(scene, config, directory, generator, segmenter, policy_seed, policy_version,
            temperature=None, prefix=None, origin="warmup", save_frames=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    policy = VisualPolicy(generator, config["policy"]["temperature"] if temperature is None else temperature,
                          policy_seed, config["action_protocol"])
    policy.system = system_prompt(config)
    policy.version = config["observation"]["version"]
    mount_hash = _mount_hash(config)
    env = observer = None
    feedback = None
    count = 0
    error = None
    started = time.monotonic()
    provenance = copy.deepcopy(getattr(generator, "provenance", {}))
    provenance.update(temperature=policy.temperature,
                      top_p=config["policy"]["top_p"] if policy.temperature > 0 else None,
                      deterministic=policy.temperature == 0)
    try:
        env = interface.make_env(scene, config)
        observer = interface.make_observer(env, segmenter)
        write_json(directory/"initialization.json", {"scene": scene, "state": env.snapshot(),
            "physical_setup_actions": env.initialization_actions, "training_eligible": False})
        with (directory/"trajectory.jsonl").open("w", encoding="utf-8") as stream:
            for saved in prefix or []:
                _check_action_frame(saved, config["action_protocol"], mount_hash)
                feedback = execute(env, observer, policy, saved["action"])
                if (feedback != saved["feedback"] or
                        not same_state(env.snapshot(), saved["state_after"], config["search"]["replay_atol"])):
                    raise RuntimeError("Prefix replay diverged; no continuation is allowed")
                restore_context(policy, observer, saved["context_after"])
                stream.write(json.dumps(saved)+"\n")
                count += 1
            stream.flush()
            while not env.ended:
                observation, images = observer.observe(directory/f"view_{env.steps:03d}" if save_frames else None)
                observation["observation_version"] = policy.version
                action, raw, messages = policy.act(env.instruction, observation, images, feedback, env.steps)
                feedback = execute(env, observer, policy, action)
                policy.remember(action, feedback, env.steps)
                row = {"step": env.steps, "task": scene["task"], "scene_id": scene["id"],
                       "policy_version": policy_version, "messages": messages, "raw_response": raw,
                       "action": action, "feedback": feedback, "judge_after": env.result(),
                       "state_after": env.snapshot(), "context_after": context(policy, observer),
                       "perception": observer.last_audit, "response_format": policy.last_parse,
                       "source_origin": origin, "policy_seed": policy_seed,
                       "camera_calibration": observation["cameras"], "action_protocol": config["action_protocol"]}
                if mount_hash is not None:
                    row["middle_camera_sha256"] = mount_hash
                speeds = env.arm_speeds() if config["action_protocol"] == interface.PROTOCOL else env.data.qvel[:7]
                row["node_eligible"] = (action.get("action") in ("move_to", "set_gripper") and
                    feedback.get("accepted") is True and
                    (action["action"] != "move_to" or feedback.get("reached") is True) and
                    row["judge_after"]["geometry"]["still"] and
                    float(np.max(np.abs(speeds))) <= config["search"]["settled_joint_speed_rad_s"])
                if hasattr(generator, "last_response_ids"):
                    row["response_ids"] = list(generator.last_response_ids)
                if hasattr(generator, "last_input_sha256"):
                    row["model_input_sha256"] = generator.last_input_sha256
                stream.write(json.dumps(row)+"\n"); stream.flush()
                count += 1
                print(json.dumps({"event": "action", "task": scene["task"], "scene": scene["id"],
                    "attempt": policy_seed, "policy_version": policy_version,
                    "step": env.steps, "action": action, "feedback": feedback,
                    "reward": row["judge_after"]["reward"], "success": row["judge_after"]["success"]}), flush=True)
        result = env.result()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = env.result() if env else {}
        result.update(success=False)
    finally:
        if observer: observer.close()
        if env: env.close()
    result.update(task=scene["task"], scene=scene, split=scene["split"], seed=scene["seed"],
        policy="model", policy_version=policy_version, policy_seed=policy_seed, origin=origin,
        config_sha256=digest(config), observation_mode="rgbd_sam3.1_detector",
        prompt_sha256=digest(policy.system), generator_provenance=provenance,
        search_prefix_length=len(prefix or []), replayed_actions=len(prefix or []),
        search_prefix_sha256=digest([r["action"] for r in prefix or []]),
        generated_actions=count-len(prefix or []), error=error, seconds=time.monotonic()-started,
        demonstration=None, termination="error" if error else "catastrophic" if result.get("dropped") or result.get("severe_collision") else
            "max_steps" if result.get("steps", 0) >= config["policy"]["max_steps"] else "done")
    write_json(directory/"result.json", result)
    print(json.dumps({"event": "episode", "task": scene["task"], "scene": scene["id"],
                      "attempt": policy_seed, "policy_version": policy_version,
                      "success": result["success"], "steps": result.get("steps"),
                      "reward": result.get("reward"), "error": error}), flush=True)
    return result


def replay(scene, config, rows):
    """No perception/inference; verify one complete root-to-leaf action sequence."""
    env = interface.make_env(scene, config)
    class ObserverStub:
        queries = []
    observer = ObserverStub()
    policy = VisualPolicy(None, action_protocol=config["action_protocol"])
    mount_hash = _mount_hash(config)
    try:
        for row in rows:
            _check_action_frame(row, config["action_protocol"], mount_hash)
            feedback = execute(env, observer, policy, row["action"])
            if feedback != row["feedback"] or not same_state(env.snapshot(), row["state_after"], config["search"]["replay_atol"]):
                raise RuntimeError("Saved successful episode does not physically replay")
        return env.result()
    finally:
        env.close()
