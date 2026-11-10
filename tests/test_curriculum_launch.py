"""Run with Python 3.11 as well as 3.13: their argparse behavior differs here."""
import argparse
import sys
from types import SimpleNamespace
import pytest
from tube_agent.curriculum import pipeline
from tube_agent.curriculum.config import load, scene_manifest
from tube_agent.lab.experiment import write_json


def minimal_torchrun_parser():
    # Relevant torchrun definitions; usable without installing Torch on Python 3.11.
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--nproc-per-node", "--nproc_per_node")
    parser.add_argument("--max-restarts", "--max_restarts")
    parser.add_argument("-m", "--module", action="store_true")
    parser.add_argument("--run-path", "--run_path", action="store_true")
    parser.add_argument("training_script")
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)
    return parser


def captured_commands(tmp_path, monkeypatch):
    config = load()
    write_json(tmp_path/"scenes.json", scene_manifest(config))
    write_json(tmp_path/"data/manifest.json", {"samples": 1})
    monkeypatch.setattr(pipeline, "gpu_check", lambda _: {})
    monkeypatch.setattr(pipeline, "report", lambda *args: {"status": "completed"})
    commands = {}
    runner = SimpleNamespace(output=tmp_path, a=SimpleNamespace(
        policy_gpus=[0, 2, 3], model="/shared/model with spaces", sam_url="http://127.0.0.1:8002"),
        status=lambda *args, **kwargs: None, ensure_sam=lambda: None,
        evaluate=lambda _: None,
        step=lambda name, command, *args: commands.setdefault(name, command))
    runner.train_stage = lambda stage: pipeline.CurriculumRunner.train_stage(runner, stage)
    pipeline.execute(runner, config)
    return commands


@pytest.mark.parametrize("stage", ["smoke", "warmup", "sft", "search"])
@pytest.mark.parametrize("parser_kind", ["stdlib", "torch"])
def test_actual_pipeline_commands_preserve_all_child_arguments(stage, parser_kind, tmp_path, monkeypatch):
    if parser_kind == "torch":
        torchrun = pytest.importorskip("torch.distributed.run")
        parser = torchrun.get_args_parser()
    else:
        parser = minimal_torchrun_parser()
    root = tmp_path/"run with spaces"
    command = captured_commands(root, monkeypatch)[stage]
    parsed = parser.parse_args(command[3:])
    module = "tube_agent.curriculum.search" if stage == "search" else "tube_agent.curriculum.train"
    expected = ([] if stage == "search" else [stage])+[
        "--run", str(root), "--model", "/shared/model with spaces"]
    assert parsed.module and parsed.training_script == module
    assert parsed.training_script_args == expected
    assert str(parsed.nproc_per_node) == "3"


@pytest.mark.skipif(sys.version_info[:2] != (3, 11), reason="Regression occurred on Python 3.11")
def test_python311_reproduces_original_failure_without_separator(capsys):
    command = pipeline.distributed_command("tube_agent.curriculum.train", "smoke", "--run", "/shared/run")
    command.remove("--")
    with pytest.raises(SystemExit) as exc:
        minimal_torchrun_parser().parse_args(command[3:])
    assert exc.value.code == 2
    assert "ambiguous option: --run" in capsys.readouterr().err


def test_runtime_preflight_exercises_installed_torchrun_parser(tmp_path, capsys):
    pytest.importorskip("torch.distributed.run")
    pipeline.validate_launch_arguments(tmp_path/"run with spaces", "/shared/model with spaces")
    assert '"commands": 4' in capsys.readouterr().out
