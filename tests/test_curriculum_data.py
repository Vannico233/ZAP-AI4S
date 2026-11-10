import json
from pathlib import Path
import pytest
from tube_agent.curriculum.config import load, scene_manifest, digest
from tube_agent.curriculum.data import freeze, read_sample
from tube_agent.lab.experiment import write_json, read_json


def root(tmp_path):
    c = load(); manifest = scene_manifest(c)
    write_json(tmp_path/"config.json", c); write_json(tmp_path/"scenes.json", manifest)
    write_json(tmp_path/"warmup/checkpoints/step_00001/COMPLETE.json", {})
    return c, manifest["scenes"][0]


def candidate(path, c, scene, final_token=2, origin="warmup", **patch):
    path.mkdir(parents=True)
    rows = [{"action": {"action": "move_to"}, "messages": [{"role": "user", "content": "exact input"}],
             "response_ids": [1], "model_input_sha256": "prefix", "judge_after": {"success": False}},
            {"action": {"action": "done"}, "messages": [{"role": "user", "content": "exact terminal input"}],
             "response_ids": [final_token], "model_input_sha256": "terminal", "judge_after": {"success": True}}]
    (path/"trajectory.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    result = {"success": True, "error": None, "policy": "model", "demonstration": None,
              "scene": scene, "task": scene["task"], "config_sha256": digest(c), "origin": origin, "invalid_actions": 0, **patch}
    write_json(path/"result.json", result)


def test_only_complete_successes_deduplicate_traces_and_shared_prefixes(tmp_path, monkeypatch):
    import tube_agent.curriculum.data as data
    c, scene = root(tmp_path); checks = []
    monkeypatch.setattr(data, "replay", lambda *args: checks.append(1) or {"success": True})
    base = tmp_path/"warmup/rollouts/round_01/group"
    candidate(base/"try_00", c, scene)
    candidate(base/"try_01", c, scene)  # Exact duplicate trajectory.
    candidate(base/"try_02", c, scene, success=False)  # High partial success is not admitted.
    candidate(tmp_path/"search/rollouts/expansion_000/try_00", c, scene, final_token=3, origin="search")
    manifest = freeze(tmp_path)
    assert len(checks) == 2
    assert manifest["successes_by_task"] == {"T1": 2, "T2": 0, "T3": 0, "T4": 0}
    assert manifest["duplicates_removed"] == 1 and manifest["samples"] == 3
    assert manifest["missing_tasks"] == ["T2", "T3", "T4"] and manifest["includes_tree_search"]
    samples = [json.loads(line) for line in (tmp_path/"data/samples.jsonl").read_text().splitlines()]
    rows = [read_sample(sample, set()) for sample in samples]
    assert [r["response_ids"] for r in rows] == [[1], [2], [3]]
    assert freeze(tmp_path) == manifest


@pytest.mark.parametrize("bad", ["evaluation", "manual", "missing_context"])
def test_data_rejects_eval_manual_or_unrecorded_context(tmp_path, monkeypatch, bad):
    import tube_agent.curriculum.data as data
    c, scene = root(tmp_path)
    monkeypatch.setattr(data, "replay", lambda *args: {"success": True})
    if bad == "evaluation": scene = next(s for s in scene_manifest(c)["scenes"] if s["split"] == "eval")
    path = tmp_path/"warmup/rollouts/round_01/group/try_00"
    candidate(path, c, scene, **({"policy": "manual"} if bad == "manual" else {}))
    if bad == "missing_context":
        rows = [json.loads(line) for line in (path/"trajectory.jsonl").read_text().splitlines()]
        del rows[0]["model_input_sha256"]
        (path/"trajectory.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError): freeze(tmp_path)


def test_empty_dataset_and_replay_divergence_are_not_faked(tmp_path, monkeypatch):
    import tube_agent.curriculum.data as data
    c, scene = root(tmp_path)
    path = tmp_path/"warmup/rollouts/round_01/group/try_00"
    candidate(path, c, scene)
    monkeypatch.setattr(data, "replay", lambda *args: {"success": False})
    with pytest.raises(ValueError, match="replay"): freeze(tmp_path)
    result = read_json(path/"result.json"); result["success"] = False; write_json(path/"result.json", result)
    manifest = freeze(tmp_path)
    assert manifest["samples"] == 0 and manifest["missing_tasks"] == list(("T1", "T2", "T3", "T4"))
