import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading

import pytest

from tube_agent.env import TubeEnv
from tube_agent.export_sft import export
from tube_agent.model_utils import resolve_model
from tube_agent.policy import ModelPolicy, ScriptedPolicy, SYSTEM_PROMPTS, parse_action
from tube_agent.train_sft import make_examples
from tube_agent.serve import convert_messages


def test_processor_receives_typed_content_even_in_text_mode():
    original = [{"role": "system", "content": "rules"}, {"role": "user", "content": "state"}]
    result = convert_messages(original)
    assert result[0]["content"] == [{"type": "text", "text": "rules"}]
    assert result[1]["content"] == [{"type": "text", "text": "state"}]
    assert original[0]["content"] == "rules"


@pytest.mark.parametrize("prompt_variant", ["base", "geometry_v1", "manual_demo_v1"])
def test_model_http_loop_uses_real_physics(prompt_variant):
    controller = ScriptedPolicy("transport")
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(request)
            current = json.loads(request["messages"][-1]["content"])
            _, raw = controller.act(current["instruction"], current["observation"])
            body = json.dumps({"choices": [{"message": {"content": raw}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = TubeEnv(detailed_feedback=prompt_variant == "geometry_v1")
    policy = ModelPolicy(f"http://127.0.0.1:{server.server_port}/v1", "test", prompt_variant=prompt_variant)
    try:
        observation, feedback = env.observe(), None
        while not env.ended:
            action, _ = policy.act(env.instruction, observation, feedback)
            observation, feedback, _ = env.step(action)
        assert env.result()["success"]
        assert len(received) == 9
        assert len(policy.history) == 19
        assert received[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert received[0]["messages"][0]["content"] == SYSTEM_PROMPTS[prompt_variant]
    finally:
        policy.close()
        env.close()
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("text", ['{"action":"finish"}', '```json\n{"action":"finish"}\n```',
                                  '<think>internal</think>{"action":"finish"}'])
def test_parse_action(text):
    assert parse_action(text) == {"action": "finish"}


@pytest.mark.parametrize("text", ["[]", '{"action":"finish"}{"action":"finish"}', "run python"])
def test_reject_ambiguous_output(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_manual_reference_is_valid_but_not_assistant_history():
    path = Path(__file__).resolve().parents[1]/"examples/manual_transport_demo_v1.json"
    demo = json.loads(path.read_text(encoding="utf-8"))
    env = TubeEnv(demo["task"], demo["split"], demo["seed"])
    policy = ModelPolicy("http://127.0.0.1:1/v1", "unused", prompt_variant="manual_demo_v1")
    try:
        assert demo["policy"] == "manual" and demo["split"] == "train"
        assert demo["initial_observation"] == env.observe()
        for action in demo["actions"]:
            _, feedback, _ = env.step(action)
            assert feedback["accepted"]
        assert env.ended and env.result()["success"]
        assert env.steps == len(demo["actions"]) == 9
        # The reference contributes no assistant labels or already-executed turns.
        assert len(policy.history) == 1 and policy.history[0]["role"] == "system"
        assert "success" not in demo and "judge_after" not in demo
    finally:
        policy.close()
        env.close()


def write_episode(root, name, policy="model", split="train", success=True):
    path = root/name
    path.mkdir()
    result = {"policy": policy, "split": split, "success": success, "error": None,
              "invalid_actions": 0, "task": "pick", "seed": 0, "attempt": 0,
              "model": "mock-for-test-only", "observation_mode": "oracle_object_state"}
    (path/"result.json").write_text(json.dumps(result), encoding="utf-8")
    messages = [{"role": "system", "content": "control"}, {"role": "user", "content": "observation"},
                {"role": "assistant", "content": '{"action":"finish"}'}]
    (path/"messages.json").write_text(json.dumps(messages), encoding="utf-8")


def test_export_excludes_scripted_test_failed_and_duplicates(tmp_path):
    write_episode(tmp_path, "good")
    write_episode(tmp_path, "duplicate")
    write_episode(tmp_path, "scripted", policy="scripted")
    write_episode(tmp_path, "test", split="test")
    write_episode(tmp_path, "failed", success=False)
    output = tmp_path/"data.jsonl"
    counts = export(tmp_path, output)
    assert counts["exported_episodes"] == 1
    assert counts["excluded_policy_or_split"] == 2
    assert counts["duplicates"] == 1
    with pytest.raises(FileExistsError):
        export(tmp_path, output)


def test_empty_export_does_not_create_data(tmp_path):
    write_episode(tmp_path, "scripted", policy="scripted")
    with pytest.raises(ValueError):
        export(tmp_path, tmp_path/"data.jsonl")
    assert not (tmp_path/"data.jsonl").exists()


@pytest.mark.parametrize("variant", ["geometry_v1", "manual_demo_v1"])
def test_export_preserves_prompt_and_feedback_provenance(tmp_path, variant):
    write_episode(tmp_path, "good")
    path = tmp_path/"good"/"result.json"
    result = json.loads(path.read_text())
    provenance = {"prompt_variant": variant, "system_prompt_sha256": "test-digest",
                  "feedback_version": "detailed_v1" if variant == "geometry_v1" else "basic_v1"}
    if variant == "manual_demo_v1":
        provenance["demonstration"] = {"policy": "manual", "split": "train", "seed": 0,
                                       "content_sha256": "test-demo-digest"}
    result.update(provenance)
    path.write_text(json.dumps(result))
    output = tmp_path/"data.jsonl"
    export(tmp_path, output)
    source = json.loads(output.read_text())["source"]
    assert all(source[key] == value for key, value in provenance.items())


@pytest.mark.parametrize("as_dict", [False, True])
def test_sft_masks_observations_and_all_previous_actions(as_dict):
    class Tokenizer:
        eos_token = "!"
        def apply_chat_template(self, messages, **kwargs):
            ids = list(range(10 + 3*len(messages)))
            return {"input_ids": ids} if as_dict else ids
        def encode(self, text, **kwargs):
            return [100+ord(char) for char in text]
    record = {"source": {"split": "train", "policy": "model"}, "messages": [
        {"role": "user", "content": "obs1"}, {"role": "assistant", "content": "A"},
        {"role": "user", "content": "obs2"}, {"role": "assistant", "content": "B"}]}
    examples, skipped = make_examples([record], Tokenizer(), 100)
    assert skipped == 0 and len(examples) == 2
    assert examples[0]["labels"] == [-100]*13 + [165, 133]
    assert examples[1]["labels"] == [-100]*19 + [166, 133]


def test_model_directory_resolution(tmp_path):
    nested = tmp_path/"Qwen3.5-4B"
    nested.mkdir()
    (nested/"config.json").write_text('{"model_type":"qwen3_5"}')
    assert resolve_model(tmp_path) == nested
    assert resolve_model(nested) == nested
    other = tmp_path/"second-model"
    other.mkdir()
    (other/"config.json").write_text('{"model_type":"qwen3_5"}')
    with pytest.raises(ValueError):
        resolve_model(tmp_path)
