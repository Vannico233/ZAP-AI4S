import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from tube_agent.lab import zap_pose
from tube_agent.lab.zap_env import ZapLabEnv
from tube_agent.lab.zap_check import convert
from tube_agent.lab.sanity import commands
from tube_agent.lab.experiment import eligible, write_json


# Numerical outputs independently evaluated with Node from the official widget
# https://zero-real.github.io/assets/js/orientation-widget.js (2026-11-07).
@pytest.mark.parametrize("triple,slot,tool",[
    ([0,0,0],[1,0,0],[0,0,-1]),
    ([90,0,123],[0,1,0],[0,0,-1]),
    ([0,90,0],[0,0,1],[1,0,0]),
    ([0,90,90],[1,0,0],[0,1,0]),
    ([45,60,30],[.28884862931764366,.46562532461428047,.8365163037378078],[.75,.43301270189221924,-.5]),
    ([-20,89.999,125],[.46985451075804896,.3289782170021017,-.8191520441642277],[-.5735764362636852,.819152044164228,-.000017453292519072936]),
    ([-20,90.001,125],[.46983811002786036,.3290016396722298,-.8191520441642277],[-.5735764362636852,.819152044164228,.000017453292519172517]),
    ([170,180,60],[-.6427876096865391,.7660444431189782,0],[0,0,1]),
])
def test_physical_axes_match_original_zap_widget(triple,slot,tool):
    matrix=zap_pose.rotation(*triple)
    assert np.allclose(matrix[:,0],slot,atol=1e-12)
    assert np.allclose(matrix[:,2],tool,atol=1e-12)
    assert np.allclose(zap_pose.rotation(**zap_pose.angles(matrix)),matrix,atol=1e-10)


def test_semantic_inverse_has_no_horizontal_euler_jump():
    for tilt in [0.,.001,45.,89.999,90.,90.001,179.999,180.]:
        for yaw in [-180.,-13.,0.,123.,359.]:
            m=zap_pose.rotation(yaw,tilt,35.)
            recovered=zap_pose.angles(m)
            assert np.allclose(zap_pose.rotation(**recovered),m,atol=1e-8)
            if 0.<tilt<180.:assert recovered["tilt_direction"]==pytest.approx(35.)


@pytest.mark.parametrize("task",["pick","transport","insert","unscrew","rack_transfer"])
def test_semantic_interface_solves_seed_zero_physics_diagnostic(task):
    env=ZapLabEnv(task,"test" if task=="rack_transfer" else "train",0,60)
    try:
        for action in commands(env):
            feedback=env.step(convert(action))[1]
            assert feedback["accepted"],feedback
        result=env.result()
        assert result["success"] and not result["invalid_actions"]
        assert result["reward_version"]=="zap_terminal_ladder_v1"
        assert result["reward_terms"]["terminal_progress"]<=.9
        assert result["reward"]>4.
    finally:env.close()


def test_bad_pose_is_refused_without_partial_motion_or_clipping():
    env=ZapLabEnv()
    try:
        start=env.data.qpos.copy();timestamp=env.data.time
        for action in [
            {"action":"move","xyz":[.4,0,.2],"rpy":[math.pi,0,0]},
            {"action":"move_to","xyz":[.4,0,.2],"slot_yaw":0,"tilt":90},
            {"action":"move_to","xyz":[.4,0,.2],"slot_yaw":True,"tilt":0},
            {"action":"move_to","xyz":[.4,0,.2],"slot_yaw":0,"tilt":181,"tilt_direction":0},
            {"action":"move_to","xyz":[.4,0,.004],"slot_yaw":0,"tilt":0},
            {"action":"set_gripper","opening":2},
        ]:
            assert not env.step(action)[1]["accepted"]
            assert np.array_equal(env.data.qpos,start) and env.data.time==timestamp
        assert "tcp_rpy" not in env.proprioception()
    finally:env.close()


def test_later_ik_failure_does_not_execute_first_waypoint(monkeypatch):
    env=ZapLabEnv();calls=[]
    try:
        initial=env.data.qpos.copy();timestamp=env.data.time
        def ik(*args):
            calls.append(1)
            return initial[:7].copy() if len(calls)==1 else None
        monkeypatch.setattr(env,"_solve_ik",ik)
        feedback=env.step({"action":"move_to","xyz":[.45,0,.2],"slot_yaw":0,"tilt":0})[1]
        assert not feedback["accepted"] and feedback["outcome"]=="not_attempted"
        assert np.array_equal(env.data.qpos,initial) and env.data.time==timestamp
    finally:env.close()


def test_reward_ladder_uses_final_state_and_cannot_accumulate_by_waiting():
    env=ZapLabEnv()
    try:
        for action in list(commands(env))[:3]:env.step(convert(action))
        assert env.result()["reward_terms"]["grasp"]==.2
        pos=env.tcp.copy();pos[2]+=.04
        env.step({"action":"move_to","xyz":pos.tolist(),"slot_yaw":0,"tilt":0})
        r=env.result();terms=r["reward_terms"]
        assert terms["lift"]==pytest.approx(.55*r["lift_height_m"]/.08,abs=1e-5)
        assert terms["terminal_progress"]==pytest.approx(.2+terms["lift"],abs=1e-5)
        assert not r["success"]
        prior=env.result()["reward"]
        env.steps+=1
        assert env.result()["reward"]==pytest.approx(prior-.01,abs=1e-5)
        env.step({"action":"set_gripper","opening":1.})
        assert env.result()["reward_terms"]["lift"]==0.
    finally:env.close()


def test_search_reuses_only_matching_visual_model_prefixes(tmp_path):
    from tube_agent.lab.run import PrefixTree
    from tube_agent.lab.worker import seed_tree
    from tube_agent.lab.zap_policy import SYSTEM
    job={"task":"pick","seed":0}
    result={"task":"pick","seed":0,"split":"train","policy":"model","success":False,
        "observation_mode":"rgbd_sam3.1_detector","action_protocol":"zap_pose_v1",
        "prompt_sha256":hashlib.sha256(SYSTEM.encode()).hexdigest(),"steps":1,
        "generator_provenance":{"adapter":None,"training":False,"model":"model"}}
    action={"action":"set_gripper","opening":0.}
    row={"action":action,"raw_response":json.dumps(action),"messages":[{"role":"system","content":SYSTEM}],
         "feedback":{"accepted":True},"judge_after":{"reward":.2}}
    for name,patch in [("good",{}),("other_task",{"task":"insert"}),
                       ("manual",{"policy":"manual_command_debug"}),("old",{"action_protocol":"rpy_rounding_v2"})]:
        write_json(tmp_path/name/"result.json",{**result,**patch})
        (tmp_path/name/"trajectory.jsonl").write_text(json.dumps(row)+"\n")
    tree=PrefixTree()
    assert seed_tree(tree,job,[tmp_path],"zap_pose_v1","model")==1
    assert len(tree.nodes)==2 and len(tree.select())==1


def test_search_rolls_back_before_reported_empty_grasp():
    from tube_agent.lab.run import PrefixTree
    trace=[]
    for kind,params in [("move_to",{"xyz":[.46,-.18,.04],"slot_yaw":0,"tilt":0}),
                        ("set_gripper",{"opening":0.}),
                        ("move_to",{"xyz":[.46,-.18,.15],"slot_yaw":0,"tilt":0})]:
        trace.append({"action":{"action":kind,**params},"feedback":{"accepted":True,"reached":True},
                      "judge_after":{"held":False,"reward":.1}})
    tree=PrefixTree();tree.update(trace,{"success":False,"search_prefix_length":0})
    assert tree.empty_close_cutoffs==1
    assert max(len(n["prefix"]) for n in tree.nodes.values())==1
    # Repeatedly failing this branch eventually forces a return to the root.
    for _ in range(3):
        prefix=tree.select();assert len(prefix)==1
        tree.update(trace,{"success":False,"search_prefix_length":1})
    assert tree.select()==[] and len(tree.blocked)==1


def test_search_keeps_a_confirmed_grasp_prefix():
    from tube_agent.lab.run import PrefixTree
    row={"action":{"action":"set_gripper","opening":0.},"feedback":{"accepted":True},
         "judge_after":{"held":True,"reward":.2}}
    tree=PrefixTree();tree.update([row],{"success":False})
    assert len(tree.nodes)==2 and tree.empty_close_cutoffs==0


def test_manual_commands_are_audited_and_never_model_training_data(tmp_path):
    source=tmp_path/"actions.json";source.write_text('[{"action":"done"}]')
    output=tmp_path/"debug"
    subprocess.run([sys.executable,"-m","tube_agent.lab.command_probe","--task","pick",
                    "--commands",str(source),"--output",str(output),"--no-render"],check=True,capture_output=True)
    result=json.loads((output/"result.json").read_text())
    assert result["policy"]=="manual_command_debug" and not eligible({**result,"success":True})
    assert result["seed"]==0


def test_plan_exposes_protocol_and_probe_has_zero_training():
    args=[sys.executable,"-m","tube_agent.lab.pipeline","--plan","--action-protocol","zap_pose_v1"]
    plan=json.loads(subprocess.check_output(args,text=True))
    assert plan["action_protocol"]=="zap_pose_v1" and plan["no_training_before_gate"]
    plan=json.loads(subprocess.check_output(args+["--pose-probe"],text=True))
    assert plan["max_total_episodes"]==12 and plan["episode_budgets"]["training"]==0


def test_import_refuses_legacy_data_in_semantic_campaign(tmp_path):
    from tube_agent.lab.import_probe import import_probe
    write_json(tmp_path/"config.json",{"pick_probe":True,"seed":0,"model":"model"})
    write_json(tmp_path/"status.json",{"stage":"probe_complete"})
    with pytest.raises(ValueError,match="protocol differs"):
        import_probe(tmp_path,tmp_path/"copies","model",0,"zap_pose_v1")


def test_four_task_probe_never_starts_training(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tube_agent.lab.pipeline as pipeline
    monkeypatch.setattr(pipeline,"gpu_check",lambda a:{})
    monkeypatch.setattr(pipeline.shutil,"disk_usage",lambda p:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(pipeline,"make_videos",lambda runner:{})
    args=pipeline.parser().parse_args(["--action-protocol","zap_pose_v1","--pose-probe"])
    args.output=tmp_path
    class Runner:
        a=args;output=tmp_path
        def status(self,*args,**kwargs):pass
        def ensure_sam(self):pass
        def step(self,*args,**kwargs):pass
        def sample(self,name,adapter,jobs):self.jobs=jobs;assert name=="pose_probe"
        def train(self,*args,**kwargs):raise AssertionError("Probe must never train")
    runner=Runner();pipeline.execute_pipeline(runner)
    assert len(runner.jobs)==12 and all(j["seed"]==0 and j["attempts"]==1 for j in runner.jobs)
    assert {j["task"] for j in runner.jobs}=={"pick","transport","insert","unscrew"}
    assert all(j["mode"]=="independent" for j in runner.jobs)


def test_new_probe_import_is_physically_verified_before_counting(tmp_path):
    from tube_agent.lab.import_probe import import_probe
    from tube_agent.lab.run import state
    from tube_agent.lab.success_gate import inspect_gate
    from tube_agent.lab.zap_policy import SYSTEM, POLICY_VERSION
    from tube_agent.lab.vision import PERCEPTION_VERSION
    source=tmp_path/"probe";location=source/"pose_probe/pick_probe_0/try_300000"
    model=str(tmp_path/"model")
    write_json(source/"config.json",{"pose_probe":True,"action_protocol":"zap_pose_v1","seed":0,"model":model})
    write_json(source/"status.json",{"stage":"pose_probe_complete"})
    env=ZapLabEnv();trace=[]
    try:
        for action in commands(env):
            action=convert(action);feedback=env.step(action)[1]
            trace.append({"step":env.steps,"action":action,"feedback":feedback,"state_after":state(env),
                          "messages":[{"role":"system","content":SYSTEM}],"raw_response":json.dumps(action)})
        result={**env.result(),"task":"pick","split":"train","seed":0,"attempt":300000,"max_steps":40,
            "policy":"model","policy_version":POLICY_VERSION,"observation_mode":"rgbd_sam3.1_detector",
            "prompt_sha256":hashlib.sha256(SYSTEM.encode()).hexdigest(),"environment_version":env.VERSION,
            "perception_version":PERCEPTION_VERSION,"search_prefix_length":0,"demonstration":None,
            "generator_provenance":{"model":model,"adapter":None,"training":False}}
    finally:env.close()
    write_json(location/"result.json",result)
    (location/"trajectory.jsonl").write_text("".join(json.dumps(r)+"\n" for r in trace))
    copied=tmp_path/"acquisition/imported_probe"
    assert import_probe(source,copied,model,0,"zap_pose_v1")["candidates"]==1
    gate=inspect_gate(tmp_path/"acquisition",[copied],tmp_path/"qualification",0)
    assert gate["unique_verified_successes"]["pick"]==1 and not gate["passed"]


@pytest.mark.requires_render
def test_semantic_visual_loop_and_prefix_restore_use_same_protocol(tmp_path):
    from tube_agent.lab.run import episode
    from tube_agent.lab.zap_policy import SYSTEM
    class PixelMask:
        def segment(self,rgb,queries):return []  # Missing masks must still allow RGB observation.
    calls=[]
    actions=iter([{"action":"move_to","xyz":[.46,-.18,.2],"slot_yaw":0,"tilt":0},
                  {"action":"done"}])
    def generate(messages,seed,temperature):
        assert messages[0]["content"]==SYSTEM
        observation=json.loads(messages[-1]["content"][0]["text"])["observation"]
        assert "tcp_orientation" in observation and "tcp_rpy" not in observation
        assert "reward" not in observation and "degrees" in observation["units"]
        calls.append(seed);return json.dumps(next(actions))
    first,trace=episode("pick","train",0,0,tmp_path/"first",generate,PixelMask(),
                        save_frames=False,action_protocol="zap_pose_v1")
    assert first["error"] is None and first["environment_version"]==ZapLabEnv.VERSION
    assert trace[0]["feedback"]["accepted"]
    def done(*args):return '{"action":"done"}'
    second,replayed=episode("pick","train",0,1,tmp_path/"second",done,PixelMask(),prefix=trace[:1],
                            save_frames=False,action_protocol="zap_pose_v1")
    assert second["error"] is None and second["search_prefix_length"]==1
    assert replayed[0]["state_after"]==trace[0]["state_after"]
