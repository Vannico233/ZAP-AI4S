"""Test retry semantics and honest missing detections, not SAM accuracy."""
import json
import sys
import numpy as np
import pytest
from tube_agent.lab.vision import detect_views


def frames():
    return [{"rgb":np.zeros((16,16,3),dtype=np.uint8),"depth":np.full((16,16),.5),
             "calibration":{"name":name,"cx":7.5,"cy":7.5,"fx":20.,"fy":20.,
                            "position":[0,0,1],"rotation_camera_to_world":np.eye(3).tolist()}}
            for name in ("top","side")]


def mask():
    result=np.zeros((16,16),dtype=bool);result[4:8,4:8]=True;return result


def test_alias_recovery_preserves_actual_query_and_sam_mask():
    class Segmenter:
        def segment(self,rgb,queries):
            return [{"query":"cyan rectangle","score":.8,"mask":mask()}] if queries==["cyan rectangle"] else []
    detected,audit=detect_views(frames(),Segmenter(),["teal tray"])
    assert not audit["missing_queries"]
    assert all(d[0]["query"]=="teal tray" and d[0]["segmentation_query"]=="cyan rectangle" for d in detected)
    assert np.array_equal(detected[0][0]["mask"],mask())
    assert len(audit["query_attempts"])==4
    assert [a["valid_mask_count"] for a in audit["query_attempts"]]==[0,0,1,1]


def test_empty_results_stay_missing_and_retry_budget_is_bounded():
    class Empty:
        def segment(self,rgb,queries):return []
    detected,audit=detect_views(frames(),Empty(),["teal tray"])
    assert detected==[[],[]] and audit["missing_queries"]==["teal tray"]
    assert audit["status"]=="missing_detections" and len(audit["query_attempts"])==6
    _,custom=detect_views(frames(),Empty(),["custom user description"])
    assert len(custom["query_attempts"])==2


def test_existing_detection_does_not_trigger_retries():
    class Segmenter:
        def segment(self,rgb,queries):return [{"query":"teal tray","score":.8,"mask":mask()}]
    _,audit=detect_views(frames(),Segmenter(),["teal tray"])
    assert len(audit["query_attempts"])==2


def test_service_failure_is_not_converted_to_missing_detection():
    class Broken:
        def segment(self,rgb,queries):raise RuntimeError("SAM service failed")
    with pytest.raises(RuntimeError,match="SAM service failed"):
        detect_views(frames(),Broken(),["teal tray"])


@pytest.mark.requires_render
def test_sampling_records_missing_masks_without_fabricating_success(tmp_path,monkeypatch):
    from tube_agent.lab import run
    class Segmenter:
        def __init__(self,*args):pass
        def health(self):return {"ready":True}
        def segment(self,rgb,queries):return []
        def close(self):pass
    class Generator:
        def __init__(self,*args):pass
        def health(self):return {"ready":True}
        def __call__(self,messages,seed,temperature):
            content=json.loads(messages[-1]["content"][0]["text"])
            assert content["observation"]["missing_queries"]==["red test tube"]
            assert content["observation"]["detections"]==[]
            assert len(messages[-1]["content"])==3
            return '{"action":"finish"}'
        def close(self):pass
    monkeypatch.setattr(run,"SamClient",Segmenter);monkeypatch.setattr(run,"HttpGenerator",Generator)
    monkeypatch.setattr(sys,"argv",["run","--task","pick","--attempts","1","--output",str(tmp_path)])
    run.main()
    summary=json.loads(next(tmp_path.rglob("summary.json")).read_text())
    result=json.loads(next(tmp_path.rglob("result.json")).read_text())
    assert summary["successes"]==0 and summary["errors"]==0
    assert summary["missing_detection_observations"]==1
    assert result["success"] is False and result["missing_detection_observations"]==1
    assert result["perception_version"]=="sam_query_retry_v1"


@pytest.mark.requires_render
@pytest.mark.parametrize("strict",[False,True])
def test_service_preflight_reports_all_tasks_without_faking_perception_success(tmp_path,monkeypatch,capsys,strict):
    from tube_agent.lab import doctor,vision,policy
    class Segmenter:
        def __init__(self,*args):pass
        def health(self):return {"ready":True}
        def segment(self,rgb,queries):return []
        def close(self):pass
    class Generator:
        def __init__(self,*args):pass
        def health(self):return {"ready":True}
        def __call__(self,*args):return '{"action":"finish"}'
        def close(self):pass
    monkeypatch.setattr(vision,"SamClient",Segmenter)
    monkeypatch.setattr(policy,"HttpGenerator",Generator)
    output=tmp_path/"check"
    monkeypatch.setattr(sys,"argv",["doctor","--services-only","--output",str(output)]+(["--strict-perception"] if strict else []))
    if strict:
        with pytest.raises(SystemExit) as error:doctor.main()
        assert error.value.code==2
    else:doctor.main()
    report=json.loads((output/"report.json").read_text())
    assert report["services_ready"] is True and report["perception_complete"] is False
    assert len(report["perception_warnings"])==5 and len(report["perception_audit"])==5
    assert report["physical_checks"]=="not_run" and report["checks"]==[]
    assert report["status"]=="services_ready_with_perception_warnings"
    assert "PASS:" not in capsys.readouterr().out
    assert (output/"live_perception/transport/perception.json").exists()
