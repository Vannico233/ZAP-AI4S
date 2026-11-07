import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
import pytest
from tube_agent.lab.import_probe import import_probe
from tube_agent.lab.experiment import write_json
from tube_agent.lab.policy import SYSTEM, POLICY_VERSION
from tube_agent.lab.env import LabEnv
from tube_agent.lab.vision import PERCEPTION_VERSION


def probe(tmp_path):
    source=tmp_path/'probe';episode=source/'pick_probe/pick_probe_0/try_200000'
    model=str(tmp_path/'model')
    write_json(source/'config.json',{'pick_probe':True,'model':model,'seed':0})
    write_json(source/'status.json',{'stage':'probe_complete'})
    result={'policy':'model','split':'train','task':'pick','success':True,'error':None,'invalid_actions':0,
            'demonstration':None,'observation_mode':'rgbd_sam3.1_detector','seed':0,'steps':1,'attempt':200000,
            'policy_version':POLICY_VERSION,'prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
            'environment_version':LabEnv.VERSION,'action_protocol':LabEnv.ACTION_PROTOCOL,
            'perception_version':PERCEPTION_VERSION,'search_prefix_length':0,
            'generator_provenance':{'model':model,'adapter':None,'training':False}}
    write_json(episode/'result.json',result)
    row={'messages':[{'role':'system','content':SYSTEM}], 'raw_response':'{"action":"finish"}',
         'action':{'action':'finish'}}
    (episode/'trajectory.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    return source,episode,model,result


def test_probe_import_is_idempotent_and_keeps_source_bytes(tmp_path):
    source,episode,model,result=probe(tmp_path);destination=tmp_path/'import'
    first=import_probe(source,destination,model,0)
    assert first['candidates']==1 and 'replay' in first['admission']
    assert import_probe(source,destination,model,0)==first
    copied=Path(first['records'][0]['copy'])
    assert (copied/'trajectory.jsonl').read_bytes()==(episode/'trajectory.jsonl').read_bytes()
    assert not (destination/'success_gate.json').exists()


@pytest.mark.parametrize('patch',[
    {'prompt_sha256':'old'}, {'policy_version':'old'}, {'seed':1},
    {'attempt':100000}, {'search_prefix_length':1},
    {'generator_provenance':{'training':False,'adapter':'trained','model':'other'}},
])
def test_incompatible_probe_successes_are_not_imported(tmp_path,patch):
    source,episode,model,result=probe(tmp_path)
    write_json(episode/'result.json',{**result,**patch})
    assert import_probe(source,tmp_path/'import',model,0)['candidates']==0


def test_active_probe_and_changed_action_are_rejected(tmp_path):
    source,episode,model,result=probe(tmp_path)
    write_json(source/'status.json',{'stage':'pick_probe'})
    with pytest.raises(ValueError,match='completed'):import_probe(source,tmp_path/'import',model,0)
    write_json(source/'status.json',{'stage':'probe_complete'})
    trace=episode/'trajectory.jsonl';row=json.loads(trace.read_text())
    row['action']={'action':'gripper','opening':0}
    trace.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='differs from model'):import_probe(source,tmp_path/'import',model,0)


def linker():
    path=Path(__file__).resolve().parents[1]/'scripts/link_shared_runtime.py'
    spec=importlib.util.spec_from_file_location('link_shared_runtime',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def runtime(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    for path in ('scripts/lab.sh','.venv-linux/bin/python','.venv-sam/bin/python',
                 '.runtime/egl-jammy/EGL_RUNTIME.json','assets/sam3.1/sam3.1_multiplex.pt',
                 'assets/sam3.1/SAM_MANIFEST.json'):
        p=old/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    (old/'runs').mkdir();(new/'scripts').mkdir(parents=True);(new/'scripts/lab.sh').write_text('fixture')
    return old,new


def test_shared_runtime_plan_preserves_existing_files(tmp_path,monkeypatch):
    old,new=runtime(tmp_path);(new/'.runtime').mkdir();called=[]
    monkeypatch.setattr(Path,'symlink_to',lambda *a,**kw:called.append(a))
    with pytest.raises(ValueError,match='already exists'):linker().link_runtime(old,new)
    assert not called and (old/'.venv-linux/bin/python').read_text()=='fixture'


def test_shared_runtime_links_only_expected_directories(tmp_path,monkeypatch):
    old,new=runtime(tmp_path);calls=[]
    monkeypatch.setattr(Path,'symlink_to',lambda target,source,**kw:calls.append((target,source)))
    linker().link_runtime(old,new)
    assert {str(t.relative_to(new)).replace('\\','/') for t,s in calls}=={
        '.venv-linux','.venv-sam','.runtime','runs','assets/sam3.1'}
    assert all(s.is_relative_to(old) for t,s in calls)


def test_review_collects_successes_failure_and_verified_video(tmp_path):
    source,episode,model,result=probe(tmp_path)
    for index,won in ((1,True),(2,True),(3,True),(4,False)):
        path=episode.with_name(f'try_{200000+index}')
        write_json(path/'result.json',{**result,'success':won})
        (path/'trajectory.jsonl').write_bytes((episode/'trajectory.jsonl').read_bytes())
    video=source/'videos/pick/episode.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'fixture')
    write_json(source/'videos/index.json',{'pick':{'status':'verified_success','video':str(video)}})
    script=Path(__file__).resolve().parents[1]/'scripts/collect_success_first.py'
    subprocess.run([sys.executable,str(script),'--run',str(source)],cwd=tmp_path,check=True)
    with tarfile.open(next(tmp_path.glob('success_first_review_*.tar.gz'))) as t:
        names=t.getnames()
        assert len([n for n in names if n.endswith('/trajectory.jsonl')])==5
        assert 'videos/pick/episode.mp4' in names
