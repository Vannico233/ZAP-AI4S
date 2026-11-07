import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
import pytest
from tube_agent.lab.policy import VisualPolicy,parse_visual_action


@pytest.mark.parametrize("text",[
    '{"action":"gripper","opening":0}',
    '```json\n{"action":"gripper","opening":0}\n```',
    'I will close the gripper now.\n\n{"action":"gripper","opening":0}\n',
])
def test_one_action_is_preserved_without_rewriting_parameters(text):
    action,audit=parse_visual_action(text)
    assert action=={"action":"gripper","opening":0}
    assert audit["mode"] in ("json","leading_prose")


@pytest.mark.parametrize("text",[
    '{"action":"finish"}{"action":"gripper","opening":0}',
    '[{"action":"finish"}]',
    'Choose {"action":"finish"} or {"action":"gripper","opening":0}',
    '{"outer":{"action":"finish"}',
    '{"action":"finish"} then do another action',
    '{"action":"gripper","opening":NaN}',
    '{"action":"finish","action":"gripper"}',
    'No JSON at all',
])
def test_ambiguous_truncated_and_nonfinite_outputs_are_rejected(text):
    with pytest.raises(ValueError):parse_visual_action(text)


def test_hallucinated_grasp_does_not_enter_history_and_reference_is_frozen():
    captured=[]
    def generator(messages,*args):
        captured.append(messages)
        return 'The gripper is holding the tube.\n{"action":"move","xyz":[.4,0,.2],"rpy":[3.14,0,0]}'.replace('[.4,0,.2]','[0.4,0,0.2]')
    policy=VisualPolicy(generator)
    obs={"gripper_opening":.08,"detections":[{"visible_surface_center":[.46,-.18,.016]}],"cameras":[{"position":[1,2,3]}]}
    action,raw,_=policy.act('Pick',obs,[np.zeros((12,12,3),dtype=np.uint8)],None,0)
    assert policy.last_parse["mode"]=="leading_prose" and 'holding' in raw
    policy.remember(action,{"accepted":True,"reached":True},1)
    obs['detections'][0]['visible_surface_center'][2]=.15
    policy.act('Pick',obs,[],{"accepted":True},1)
    text=json.loads(captured[-1][-1]['content'][0]['text'])
    assert text['initial_visual_reference'][0]['visible_surface_center'][2]==.016
    assert text['observation']['detections'][0]['visible_surface_center'][2]==.15
    assert 'cameras' not in text['observation']
    assert text['gripper_interpretation']=='wide_open_not_grasp_evidence'
    assert len(text['action_history'])==1
    assert all('holding the tube' not in m['content'] for m in policy.history)


def test_collector_includes_failed_trajectory_before_diagnostics_exist(tmp_path):
    root=tmp_path/'run';episode=root/'base_fixed/pick_seed_00000/try_000';episode.mkdir(parents=True)
    (root/'config.json').write_text('{}')
    (episode/'result.json').write_text(json.dumps({"task":"pick","policy":"model","error":None,"success":False,"reward":0}))
    from tube_agent.lab.vision import image_url
    row={"action":{"action":"invalid_model_output"},"feedback":{"accepted":False},
         "messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":image_url(np.zeros((12,12,3),dtype=np.uint8))}}]}]}
    (episode/'trajectory.jsonl').write_text(json.dumps(row)+'\n')
    script=Path(__file__).resolve().parents[1]/'scripts/collect_success_first.py'
    subprocess.run([sys.executable,str(script),'--run',str(root)],cwd=tmp_path,check=True)
    with tarfile.open(next(tmp_path.glob('success_first_review_*.tar.gz'))) as archive:
        assert 'base_fixed/pick_seed_00000/try_000/trajectory.jsonl' in archive.getnames()
        assert 'base_fixed/pick_seed_00000/try_000/first_0.png' in archive.getnames()
        assert json.load(archive.extractfile('COLLECTION.json'))['selected_episodes']==1


def test_pick_probe_cannot_call_any_training_stage(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tube_agent.lab.pipeline as pipeline
    monkeypatch.setattr(pipeline,'gpu_check',lambda a:{})
    monkeypatch.setattr(pipeline.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(pipeline,'make_videos',lambda runner:{})
    a=pipeline.parser().parse_args(['--pick-probe']);a.output=tmp_path
    class Runner:
        def __init__(self):self.a=a;self.output=tmp_path;self.job_list=[]
        def status(self,*args,**kwargs):pass
        def ensure_sam(self):pass
        def step(self,*args):pass
        def sample(self,name,adapter,jobs):
            assert adapter is None
            self.job_list=jobs
        def train(self,*args,**kwargs):raise AssertionError('Probe must never train')
    runner=Runner();pipeline.execute_pipeline(runner)
    assert sum(j['attempts'] for j in runner.job_list)==9
    assert len({j['id'] for j in runner.job_list})==3
    assert all(j['task']=='pick' and j['seed']==0 and j['mode']=='independent' for j in runner.job_list)
    assert json.loads((tmp_path/'report.json').read_text())['status']=='probe_complete'
