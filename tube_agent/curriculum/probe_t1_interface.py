"""Compare first post-grasp model actions on identical archived T1 observations."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tube_agent import interface
from tube_agent.lab.experiment import read_json, write_json
from tube_agent.lab.policy import parse_visual_action
from tube_agent.model_utils import DEFAULT_MODEL, resolve_model
from .collect_t1 import EMPTY_ADAPTER, ORIGIN, check_original_policy, lock, row_evidence
from .config import digest, file_digest
from .rollout import read_rows
from .t1_relative_probe import GENERAL, resolve_relative


DIRECTED = ("For this diagnostic turn, translate the {arm} fingertip midpoint upward by exactly "
            "0.08 metres along the tabletop normal, from its current measured position. "
            "Keep the current orientation. Return one action.")
QUESTIONS = ("autonomous", "directed")
INTERFACES = ("absolute", "relative")


def cases(source, config, plan):
    """Choose every first nonterminal held endpoint with a subsequent archived decision."""
    for attempt in plan["attempts"]:
        scene, seed = attempt["scene"], attempt["policy_seed"]
        directory = source / attempt["directory"]
        record = read_json(directory / "verified.json")
        result = read_json(directory / "result.json")
        if (scene["task"] != "T1" or scene["split"] != "train" or result["error"]
                or result["policy"] != "model" or result["origin"] != ORIGIN
                or result["demonstration"] is not None or result["replayed_actions"] != 0
                or result["config_sha256"] != digest(config) or result["scene"] != scene
                or record["scene_id"] != scene["id"] or record["policy_seed"] != seed):
            raise ValueError("Expected completed, model-generated T1 baseline episodes")
        check_original_policy(result, config)
        if not result["ever_grasped"]:
            continue
        rows = iter(read_rows(directory))
        grasp = next((r for r in rows if r["judge_after"]["held"]), None)
        if grasp is None:
            continue  # A transient grasp can end between decisions.
        row = next(rows, None)
        if row is None:
            continue
        if not all(row_evidence(r, scene, seed, config, step, ORIGIN)
                   for r, step in ((grasp, grasp["step"]), (row, grasp["step"] + 1))):
            raise ValueError("Mismatched archived prefix evidence")
        close = grasp["action"]
        if (close.get("action") != "set_gripper" or close.get("opening") != 0
                or close.get("arm") not in interface.ARMS or not grasp["feedback"]["accepted"]):
            raise ValueError("Cannot identify the grasping arm from the recorded closure")
        yield row, close["arm"]


def messages_for(original, arm, question, api):
    messages = copy.deepcopy(original)
    if api == "relative":
        messages[0]["content"] += "\n" + GENERAL
    if question == "directed":
        block = messages[-1]["content"][0]
        task = json.loads(block["text"])
        task["instruction"] = DIRECTED.format(arm=arm)
        block["text"] = json.dumps(task)
    return messages


def score(raw, observation, holding_arm, api):
    """Score the requested pose; neither IK feasibility nor physical outcome is inferred."""
    result = dict(action=None, resolved_action=None, same_arm=False, signed_up_m=None,
                  position_error_m=None, orientation_error_deg=None, clear_up=False,
                  directed_match=False, invalid_reason=None)
    try:
        action, _ = parse_visual_action(raw)
        result["action"] = action
        kind = action.get("action")
        if kind == "translate":
            if api != "relative":
                raise ValueError("translate is unavailable on the absolute interface")
            move = resolve_relative(action, observation)
        elif kind == "move_to":
            move = action
        else:
            return result
        if set(move) != {"action", "arm", "position", "quaternion_wxyz"} or move["arm"] not in interface.ARMS:
            raise ValueError("Invalid move_to fields or arm")
        position = interface._vector(move["position"], 3, "position")
        quaternion = interface._vector(move["quaternion_wxyz"], 4, "quaternion_wxyz")
        length = np.linalg.norm(quaternion)
        if abs(length - 1.) > .01:
            raise ValueError("Command quaternion is outside the unit tolerance")
        arms = observation["arms"]
        current = arms[move["arm"]]["ee_pose"]
        desired = arms[holding_arm]["ee_pose"]
        normal = np.asarray(observation["table_up_middle"])
        delta = position - current["position"]
        target = np.asarray(desired["position"]) + .08 * normal
        distance = float(np.linalg.norm(position - target))
        q = np.asarray(desired["quaternion_wxyz"])
        cosine = abs(float(np.dot(quaternion / length, q / np.linalg.norm(q))))
        angle = float(np.degrees(2 * np.arccos(np.clip(cosine, 0., 1.))))
        up = float(np.dot(delta, normal))
        if not np.isfinite([up, distance, angle]).all():
            raise ValueError("Nonfinite command geometry")
        same = move["arm"] == holding_arm
        result.update(resolved_action=move, same_arm=same, signed_up_m=up,
                      position_error_m=distance, orientation_error_deg=angle,
                      clear_up=same and up >= .01, directed_match=same and distance <= .01 and angle <= 5.)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["invalid_reason"] = str(exc)
    return result


def write_rows(path, rows):
    temporary = path.with_suffix(".jsonl.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def run(source, model_path, output):
    from .learning import Generator, load_model
    from .pipeline import model_reference

    collection = read_json(source / "collection.json")
    plan = collection["plan"]
    config = read_json(source / "config.json")
    if (plan["config_sha256"] != digest(config) or plan["policy"] != "original"
            or config["version"] != "tube_curriculum_bimanual_v2" or config["policy"].get("kinematic_note")
            or (model_path / "adapter_config.json").exists()):
        raise ValueError("Expected the original bimanual baseline without an adapter or kinematic note")
    model = model_reference(model_path)
    if model["files"] != collection["model"]["files"]:
        raise ValueError("Probe weights or processor files differ from the source collection")
    eligible = list(cases(source, config, plan))
    if not eligible:
        raise ValueError("No eligible post-grasp contexts found")
    manifest = dict(kind="frozen_context_t1_interface_probe", training_eligible=False,
                    source=str(source), source_collection_sha256=file_digest(source / "collection.json"),
                    config_sha256=digest(config), model_file_hashes=model["files"], policy=config["policy"],
                    directed_instruction=DIRECTED, extra_tool_prompt=GENERAL, displacement_m=.08,
                    clear_up_m=.01, directed_position_tolerance_m=.01, directed_angle_tolerance_deg=5.,
                    probe_sha256=file_digest(Path(__file__)),
                    resolver_sha256=file_digest(Path(__file__).with_name("t1_relative_probe.py")),
                    cases=[dict(scene_id=r["scene_id"], policy_seed=r["policy_seed"], arm=arm,
                                grasp_step=r["step"]-1, input_sha256=r["model_input_sha256"],
                                archived_response_sha256=digest(r["response_ids"])) for r, arm in eligible])
    manifest_file = output.with_suffix(".manifest.json")
    if manifest_file.exists() and read_json(manifest_file) != manifest:
        raise ValueError("Existing output is from another probe or collection")
    if output.exists() and not manifest_file.is_file():
        raise ValueError("Incomplete probe provenance")
    write_json(manifest_file, manifest)
    rows = []
    if output.exists():
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    keys = [(r["policy_seed"], question, api) for r, _ in eligible
            for question in QUESTIONS for api in INTERFACES]
    done = [(r["policy_seed"], r["question"], r["interface"]) for r in rows]
    if done != keys[:len(done)] or any(r["manifest_sha256"] != digest(manifest) for r in rows):
        raise ValueError("Inconsistent saved response records")
    generator = None
    for row, arm in eligible:
        original = row["messages"]
        block = original[-1]["content"]
        if original[0]["role"] != "system" or [b["type"] for b in block] != ["text"] + ["image_url"] * 3:
            raise ValueError("Expected original system and three-view policy context")
        observation = json.loads(block[0]["text"])["observation"]
        grasp_step = row["step"] - 1
        seed = row["policy_seed"] + grasp_step * 100003
        for question in QUESTIONS:
            for api in INTERFACES:
                if (row["policy_seed"], question, api) in done:
                    continue
                if generator is None:
                    base, processor = load_model(model_path, config, trainable=False)
                    generator = Generator(base, processor, config)
                    generator.bind("original", EMPTY_ADAPTER)
                raw = generator(messages_for(original, arm, question, api), seed, config["policy"]["temperature"])
                original_condition = question == "autonomous" and api == "absolute"
                if original_condition and generator.last_input_sha256 != row["model_input_sha256"]:
                    raise RuntimeError("Reconstructed original model input differs from archived input")
                record = dict(scene_id=row["scene_id"], policy_seed=row["policy_seed"], generation_seed=seed,
                              holding_arm=arm, grasp_step=grasp_step, interface=api, question=question,
                              training_eligible=False, manifest_sha256=digest(manifest),
                              source_input_sha256=row["model_input_sha256"], input_sha256=generator.last_input_sha256,
                              response_sha256=digest(generator.last_response_ids),
                              historical_response_sha256=digest(row["response_ids"]),
                              historical_action=row["action"] if original_condition else None,
                              baseline_tokens_match=(row["response_ids"] == generator.last_response_ids
                                                     if original_condition else None),
                              raw_response=raw, **score(raw, observation, arm, api))
                rows.append(record)
                write_rows(output, rows)
                print(f"Completed {len(rows)}/{len(keys)}: {row['scene_id']} {question} {api}", flush=True)
    summary = []
    for question in QUESTIONS:
        for api in INTERFACES:
            selected = [r for r in rows if r["question"] == question and r["interface"] == api]
            summary.append(dict(question=question, interface=api, cases=len(selected),
                                valid_moves=sum(r["resolved_action"] is not None for r in selected),
                                clear_up=sum(r["clear_up"] for r in selected),
                                positive_up=sum(r["same_arm"] and r["signed_up_m"] > 0 for r in selected),
                                directed_match=sum(r["directed_match"] for r in selected)))
    report = dict(cases=len(eligible), responses=len(rows), counts=summary, training_eligible=False,
                  baseline_tokens_match=sum(r["baseline_tokens_match"] is True for r in rows),
                  output_sha256=file_digest(output), manifest_sha256=digest(manifest))
    write_json(output.with_suffix(".summary.json"), report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Completed original T1 collection")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True, type=Path, help="Compact output JSONL; resumes in place")
    args = parser.parse_args()
    if args.out.suffix != ".jsonl" or args.out.resolve().is_relative_to(args.source.resolve()):
        parser.error("Use a .jsonl output outside the original collection directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with lock(args.out.with_suffix(".lock"), wait=False):
        run(args.source.resolve(), resolve_model(args.model), args.out)


if __name__ == "__main__":
    main()
