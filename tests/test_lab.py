import json
import math
from pathlib import Path
import numpy as np
import pytest
from tube_agent.lab.env import LabEnv,TASKS,rpy_matrix,matrix_rpy
from tube_agent.lab.sanity import commands
from tube_agent.lab.vision import Cameras,Observer,encode_mask,decode_mask,unproject
from tube_agent.lab.run import episode,PrefixTree
from tube_agent.lab.train import advantages,export

@pytest.mark.parametrize("task",TASKS)
def test_contact_physics_has_a_successful_seed_zero(task):
    env=LabEnv(task,"test" if task=="rack_transfer" else "train",max_steps=60)
    try:
        for action in commands(env):
            _,feedback,_=env.step(action)
            assert feedback["accepted"]
        r=env.result()
        assert r["success"],r
        assert r["reward"]>4 and r["invalid_actions"]==0
        if task=="unscrew":assert r["thread_released"] and r["cap_turns"]>=1 and r["cap_held"]
        elif task in ("insert","rack_transfer"):assert r["insertion_stable_seconds"]>=.4
    finally:env.close()

@pytest.mark.parametrize("task",TASKS)
def test_finish_without_manipulation_is_never_success(task):
    e=LabEnv(task,"test" if task=="rack_transfer" else "train")
    try:
        e.step({"action":"finish"});assert not e.result()["success"]
        assert e.result()["reward"]<1
    finally:e.close()

def test_cap_cannot_be_pulled_off_without_unscrewing():
    e=LabEnv("unscrew")
    try:
        p=e.cap_initial
        for a in ({"action":"move","xyz":[*p[:2],.23],"rpy":[math.pi,0,0]},
                  {"action":"move","xyz":p.tolist(),"rpy":[math.pi,0,0]},
                  {"action":"gripper","opening":0},
                  {"action":"move","xyz":[*p[:2],.25],"rpy":[math.pi,0,0]},
                  {"action":"finish"}):e.step(a)
        assert not e.thread_released and not e.result()["success"]
        assert e.data.body("cap").xpos[2]<.15
    finally:e.close()

def test_full_pose_validation_and_no_reward_farming():
    e=LabEnv("insert")
    try:
        assert not e.step({"action":"move","xyz":[.4,0,.2],"yaw":0})[1]["accepted"]
        assert not e.step({"action":"move","xyz":[.4,0,.2],"rpy":[math.pi,float("nan"),0]})[1]["accepted"]
        before=e.result()["reward"]
        for _ in range(3):e.step({"action":"gripper","opening":.08})
        assert e.result()["reward"]<=before
    finally:e.close()

def test_generalization_cannot_enter_training_split():
    with pytest.raises(ValueError,match="held out"):LabEnv("rack_transfer","train")

def test_rotation_roundtrip_including_vertical():
    for angles in ([math.pi,0,.2],[math.pi,math.pi/2,2.8],[.2,-.4,.8]):
        rotation=rpy_matrix(angles)
        assert np.allclose(rpy_matrix(matrix_rpy(rotation)),rotation)

def test_mask_wire_roundtrip_and_calibrated_depth():
    mask=np.array([[True,False,True],[False,True,False]])
    assert np.array_equal(decode_mask(encode_mask(mask)),mask)
    calibration={"cx":1.,"cy":1.,"fx":2.,"fy":2.,"position":[0,0,1],"rotation_camera_to_world":np.eye(3).tolist()}
    points=unproject(np.ones((3,3)),np.eye(3,dtype=bool),calibration)
    assert np.allclose(points[:,2],0.)
    assert np.allclose(points[1],[0,0,0])

class ImageOnlySegmenter:
    """Test double using pixels only, not SAM accuracy evidence."""
    def segment(self,rgb,queries):
        mask=(rgb[:,:,0]>100)&(rgb[:,:,0]>rgb[:,:,1].astype(float)*1.6)&(rgb[:,:,0]>rgb[:,:,2].astype(float)*1.6)
        return [{"query":"red test tube","score":1.,"mask":mask}] if mask.sum()>12 else []

@pytest.mark.requires_render
def test_observer_contains_no_oracle_geometry_or_judge(tmp_path):
    e=LabEnv("pick");o=Observer(e,ImageOnlySegmenter())
    try:
        e.observe=lambda: (_ for _ in ()).throw(AssertionError("Oracle observation was accessed"))
        obs,images=o.observe(tmp_path/"frames")
        assert obs["observation_mode"]=="rgbd_sam3.1_detector"
        assert obs["detections"] and len(images)==2
        assert not ({"reward","success","tray_center","objects","tube_position"}&set(obs))
        assert all("visible_surface_center" in d and "center" not in d for d in obs["detections"])
        assert len(list((tmp_path/"frames").glob("*_depth.npy")))==2
    finally:o.close();e.close()

@pytest.mark.requires_render
def test_visual_rollout_prefix_replays_exact_state_and_context(tmp_path):
    actions=iter([{"action":"move","xyz":[.46,-.18,.20],"rpy":[math.pi,0,0]}, {"action":"finish"}])
    def generate(messages,seed,temperature):return json.dumps(next(actions))
    first,rows=episode("pick","train",0,0,tmp_path/"first",generate,ImageOnlySegmenter(),save_frames=False)
    assert first["error"] is None
    def finish(messages,seed,temperature):return '{"action":"finish"}'
    second,replayed=episode("pick","train",0,1,tmp_path/"second",finish,ImageOnlySegmenter(),prefix=rows[:1],save_frames=False)
    assert second["error"] is None and second["search_prefix_length"]==1
    assert replayed[0]["state_after"]==rows[0]["state_after"]
    rows[0]["state_after"][0]+=.01
    failed,_=episode("pick","train",0,2,tmp_path/"bad",finish,ImageOnlySegmenter(),prefix=rows[:1],save_frames=False)
    assert "diverged" in failed["error"] and not failed["success"]

def test_equal_group_rewards_have_zero_advantage():
    assert np.allclose(advantages([.2,.2,.2]),0)
    a=advantages([0,0,5,0]);assert a[2]>0 and a[0]<0 and abs(a.mean())<1e-6


def test_prefix_tree_counts_each_visited_ancestor_once():
    from tube_agent.lab.run import PrefixTree
    tree=PrefixTree()
    rows=[{"action":{"action":"gripper","opening":0.},"feedback":{"accepted":True},"judge_after":{"reward":.2}}]
    for success in (False,True,False):
        tree.select();tree.update(rows,{"success":success,"error":None})
    assert tree.nodes["root"]["visits"]==3 and tree.nodes["root"]["wins"]==1
    child=tree.nodes[tree.key(rows)]
    assert child["visits"]==3 and child["wins"]==1

def test_export_refuses_manual_or_held_out_or_assisted_successes(tmp_path):
    for i,patch in enumerate(({"policy":"manual"},{"split":"test"},{"task":"rack_transfer"},{"demonstration":{"seed":0}})):
        path=tmp_path/f"episode_{i}";path.mkdir()
        result={"policy":"model","split":"train","task":"pick","success":True,"error":None,"invalid_actions":0,
                "observation_mode":"rgbd_sam3.1_detector","demonstration":None,**patch}
        (path/"result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError,match="No eligible"):export(tmp_path,tmp_path/"sft.jsonl")
    assert not (tmp_path/"sft.jsonl").exists()

@pytest.mark.parametrize("loss_type",["grpo","dr_grpo"])
def test_qwen_vision_lora_sft_and_grpo_backward(tmp_path,loss_type):
    torch=pytest.importorskip("torch")
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from peft import LoraConfig,get_peft_model
    from tube_agent.lab.train import token_logps,clipped_loss,dr_grpo_loss
    torch.set_num_threads(2);torch.manual_seed(4)
    config=Qwen3_5Config(text_config={"vocab_size":256,"hidden_size":64,"intermediate_size":128,
        "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
        "linear_key_head_dim":16,"linear_value_head_dim":16,"linear_num_key_heads":4,"linear_num_value_heads":4,
        "layer_types":["linear_attention","full_attention"],"max_position_embeddings":128,
        "rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":.5,"mrope_section":[1,1,2]},
        "pad_token_id":0,"eos_token_id":2},
        vision_config={"depth":1,"hidden_size":32,"intermediate_size":64,"num_heads":2,"out_hidden_size":64,
                       "num_position_embeddings":16,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2},
        image_token_id=253,video_token_id=254,vision_start_token_id=252,vision_end_token_id=251)
    base=Qwen3_5ForConditionalGeneration(config)
    original_state={k:v.detach().clone() for k,v in base.state_dict().items()}
    targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and ".language_model." in n]
    model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=targets,task_type="CAUSAL_LM"))
    ids=torch.tensor([[12,252,253,251,13,14,15,2]])
    inputs={"input_ids":ids,"attention_mask":torch.ones_like(ids),"mm_token_type_ids":torch.tensor([[0,0,1,0,0,0,0,0]]),
            "pixel_values":torch.randn(4,1176),"image_grid_thw":torch.tensor([[1,2,2]])}
    model.eval()
    old=token_logps(model,inputs,6).detach()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    current=token_logps(model,inputs,6)
    loss=clipped_loss(current,old,1.) if loss_type=="grpo" else dr_grpo_loss(current,old,1.,40*256)
    loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
    optimizer.step();optimizer.zero_grad()
    assert token_logps(model,inputs,6).mean()>old.mean()
    sft=-token_logps(model,inputs,6).mean();sft.backward();assert torch.isfinite(sft)
    optimizer.step();optimizer.zero_grad();model.eval()
    # A committed adapter + optimizer must produce the same *next* update as
    # uninterrupted training, not just reload without an exception.
    from tube_agent.lab.distributed_train import commit_checkpoint
    from peft import PeftModel
    class Processor:
        def save_pretrained(self,path):
            (path/"processor_test.json").write_text("{}")
    saved=commit_checkpoint(model,Processor(),optimizer,tmp_path,2,{"test":"tiny_multimodal"})
    restored_base=Qwen3_5ForConditionalGeneration(config);restored_base.load_state_dict(original_state)
    restored=PeftModel.from_pretrained(restored_base,saved,is_trainable=True,local_files_only=True);restored.eval()
    resumed_optimizer=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.01)
    resumed_optimizer.load_state_dict(torch.load(saved/"optimizer.pt",weights_only=True)["optimizer"])
    assert torch.allclose(token_logps(model,inputs,6),token_logps(restored,inputs,6),atol=1e-6,rtol=0)
    for net,opt in ((model,optimizer),(restored,resumed_optimizer)):
        (-token_logps(net,inputs,6).mean()).backward();opt.step();opt.zero_grad()
    assert torch.allclose(token_logps(model,inputs,6),token_logps(restored,inputs,6),atol=1e-6,rtol=0)

def test_sam_native_session_contract_and_cleanup():
    from tube_agent.lab.sam_server import segment_snapshot
    from PIL import Image
    class Predictor:
        def __init__(self):self.requests=[]
        def handle_request(self,r):
            self.requests.append(r)
            if r["type"]=="start_session":return {"session_id":"unit-test"}
            if r["type"]=="add_prompt":
                assert r["frame_index"]==0 and r["output_prob_thresh"]==.35
                return {"outputs":{"out_binary_masks":[np.ones((12,12),dtype=bool)],"out_probs":[.9]}}
            return {}
    predictor=Predictor()
    output=segment_snapshot(predictor,Image.new("RGB",(12,12)),["red tube","white cap"],.35)
    assert [r["query"] for r in output]==["red tube","white cap"]
    assert [r["type"] for r in predictor.requests]==["start_session","add_prompt","close_session"]*2
    class Broken(Predictor):
        def handle_request(self,r):
            if r["type"]=="add_prompt":raise RuntimeError("inference failed")
            return super().handle_request(r)
    broken=Broken()
    with pytest.raises(RuntimeError):segment_snapshot(broken,Image.new("RGB",(12,12)),["tube"],.35)
    assert broken.requests[-1]["type"]=="close_session"


def test_sam31_init_adapter_only_consumes_unsupported_false_flag():
    from types import SimpleNamespace
    from tube_agent.lab.sam_server import adapt_session_init
    class Multiplex:
        def init_state(self,resource_path,offload_video_to_cpu=False,async_loading_frames=False):
            return resource_path,offload_video_to_cpu,async_loading_frames
    predictor=SimpleNamespace(model=Multiplex())
    with pytest.raises(TypeError,match="offload_state_to_cpu"):
        predictor.model.init_state(["image"],offload_state_to_cpu=False)
    assert adapt_session_init(predictor)
    assert predictor.model.init_state(["image"],offload_state_to_cpu=False,async_loading_frames=True)==(["image"],False,True)
    assert adapt_session_init(predictor)
    with pytest.raises(ValueError,match="CPU state offloading"):
        predictor.model.init_state(["image"],offload_state_to_cpu=True)
    with pytest.raises(TypeError):predictor.model.init_state(["image"],unknown_flag=False)
    class Supported:
        def init_state(self,resource_path,offload_state_to_cpu=False):return offload_state_to_cpu
    native=SimpleNamespace(model=Supported());assert not adapt_session_init(native)
    assert native.model.init_state(["image"],offload_state_to_cpu=True)


def test_sam_client_preserves_remote_error_body():
    import httpx
    from tube_agent.lab.vision import SamClient
    client=SamClient();client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(400,json={"error":"TypeError: unexpected keyword argument 'offload_state_to_cpu'"})))
    try:
        with pytest.raises(RuntimeError,match="offload_state_to_cpu"):
            client.segment(np.zeros((12,12,3),dtype=np.uint8),["red tube"])
    finally:client.close()

@pytest.mark.requires_render
def test_visual_http_services_contract(tmp_path):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    from tube_agent.lab.policy import HttpGenerator
    from tube_agent.lab.vision import SamClient,decode_image
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,obj):
            payload=json.dumps(obj).encode();self.send_response(200)
            self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):self.reply({"ready":True,"backend":"sam3.1_detector","adapter":None})
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path=="/segment":
                rgb=np.array(decode_image(request["image"]))
                detections=ImageOnlySegmenter().segment(rgb,request["queries"])
                self.reply({"detections":[{**d,"mask":encode_mask(d["mask"])} for d in detections]})
            else:
                assert self.path=="/v1/chat/completions"
                assert sum(p["type"]=="image_url" for p in request["messages"][-1]["content"])==2
                self.reply({"choices":[{"message":{"content":'{"action":"finish"}'}}]})
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f"http://127.0.0.1:{server.server_port}";sam=SamClient(url);model=HttpGenerator(url+"/v1")
    try:
        sam.health();model.health()
        result,_=episode("pick","train",0,0,tmp_path/"http_episode",model,sam,save_frames=False)
        assert result["error"] is None and result["steps"]==1 and not result["success"]
    finally:model.close();sam.close();server.shutdown();server.server_close();thread.join()
