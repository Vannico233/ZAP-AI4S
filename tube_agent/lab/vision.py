"""Calibrated RGB-D sensing and mask-derived 3D measurements. No object state."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import httpx
import numpy as np
from PIL import Image, ImageDraw

def image_url(image):
    stream=io.BytesIO(); Image.fromarray(image).save(stream,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode()

def decode_image(url):
    if not url.startswith("data:image/"): raise ValueError("Embedded image required")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",",1)[1],validate=True))).convert("RGB")

def encode_mask(mask):
    mask=np.asarray(mask,dtype=bool)
    return {"shape":list(mask.shape),"bits":base64.b64encode(np.packbits(mask).tobytes()).decode()}

def decode_mask(value):
    shape=tuple(value["shape"])
    if len(shape)!=2 or min(shape)<1 or max(shape)>2048: raise ValueError("Invalid mask shape")
    bits=np.unpackbits(np.frombuffer(base64.b64decode(value["bits"],validate=True),dtype=np.uint8))
    if len(bits)<np.prod(shape): raise ValueError("Truncated mask")
    return bits[:np.prod(shape)].reshape(shape).astype(bool)

class Cameras:
    def __init__(self, env, names=("top","side"), width=640, height=480):
        import mujoco
        self.env,self.names,self.width,self.height=env,tuple(names),width,height
        self.renderer=mujoco.Renderer(env.model,height=height,width=width)

    def capture(self):
        import mujoco
        frames=[]
        options=mujoco.MjvOption(); options.geomgroup[3]=1
        for name in self.names:
            cam=self.env.model.camera(name); cid=cam.id
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.env.data,camera=name,scene_option=options)
            rgb=self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            depth=self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            focal=.5*self.height/np.tan(np.deg2rad(float(cam.fovy[0]))/2)
            calibration={"name":name,"width":self.width,"height":self.height,
                         "fx":float(focal),"fy":float(focal),"cx":(self.width-1)/2,"cy":(self.height-1)/2,
                         "position":self.env.data.cam_xpos[cid].tolist(),
                         "rotation_camera_to_world":self.env.data.cam_xmat[cid].reshape(3,3).tolist(),
                         "convention":"OpenGL: x right, y up, optical direction -z; metric axial depth"}
            frames.append({"rgb":rgb,"depth":depth,"calibration":calibration})
        return frames

    def close(self): self.renderer.close()

def unproject(depth, mask, calibration):
    valid=mask & np.isfinite(depth) & (depth>.01) & (depth<3.)
    v,u=np.nonzero(valid)
    if not len(v): return np.empty((0,3))
    d=depth[v,u]
    camera=np.column_stack(((u-calibration["cx"])*d/calibration["fx"],
                            -(v-calibration["cy"])*d/calibration["fy"],-d))
    return camera@np.asarray(calibration["rotation_camera_to_world"]).T+calibration["position"]

def measure(points):
    points=np.asarray(points)
    if len(points)<12: return None
    lower,upper=np.quantile(points,[.02,.98],axis=0)
    center=(lower+upper)/2
    covariance=np.cov(points.T)
    values,axes=np.linalg.eigh(covariance); axis=axes[:,np.argmax(values)]
    # Principal axes have sign ambiguity. This convention does NOT identify a cap.
    pivot=2 if abs(axis[2])>.7 else int(np.argmax(np.abs(axis)))
    if axis[pivot]<0: axis=-axis
    return {"visible_surface_center":np.round(center,4).tolist(),
            "visible_bounds_min":np.round(lower,4).tolist(),"visible_bounds_max":np.round(upper,4).tolist(),
            "visible_height":round(float(upper[2]-lower[2]),4),"principal_axis_unsigned":np.round(axis,4).tolist(),
            "point_count":len(points),"measurement_warning":"Visible surface only; occlusion biases center and size. Axis sign is not cap direction."}

class SamClient:
    def __init__(self,url="http://127.0.0.1:8001"):
        self.url=url.rstrip("/"); self.client=httpx.Client(timeout=180.,trust_env=False)
    def health(self):
        r=self.client.get(self.url+"/health");r.raise_for_status();h=r.json()
        if h.get("backend")!="sam3.1_detector" or not h.get("ready"):
            raise RuntimeError("A ready native SAM 3.1 detector service is required")
        return h
    def segment(self,rgb,queries):
        r=self.client.post(self.url+"/segment",json={"image":image_url(rgb),"queries":queries})
        r.raise_for_status()
        return [{**d,"mask":decode_mask(d["mask"])} for d in r.json()["detections"]]
    def close(self):self.client.close()

class Observer:
    def __init__(self,env,segmenter):
        self.env,self.segmenter=env,segmenter;self.cameras=Cameras(env)
        self.queries={"pick":["red test tube"],"transport":["red test tube","teal tray"],
                      "insert":["red test tube","yellow test tube rack"],
                      "unscrew":["white cap","red test tube"],
                      "rack_transfer":["red test tube","blue test tube rack","yellow test tube rack"]}[env.lab_task]
    def observe(self,output=None):
        frames=self.cameras.capture(); objects=[]; images=[]
        if output: Path(output).mkdir(parents=True,exist_ok=False)
        for frame in frames:
            c=frame["calibration"];name=c["name"]
            detections=self.segmenter.segment(frame["rgb"],self.queries)
            overlay=Image.fromarray(frame["rgb"]);draw=ImageDraw.Draw(overlay)
            serial=[]
            for index,d in enumerate(detections):
                mask=d["mask"]
                if mask.shape!=frame["depth"].shape:raise ValueError("Mask/depth shape mismatch")
                measurement=measure(unproject(frame["depth"],mask,c))
                if measurement is None:continue
                identity=f"{name}:{index}"
                objects.append({"id":identity,"query":d["query"],"score":d["score"],"camera":name,**measurement})
                v,u=np.nonzero(mask)
                tinted=np.array(overlay)
                tinted[mask]=(.75*tinted[mask]+.25*np.array([255,190,20])).astype(np.uint8)
                overlay=Image.fromarray(tinted);draw=ImageDraw.Draw(overlay)
                box=[int(u.min()),int(v.min()),int(u.max()),int(v.max())]
                draw.rectangle(box,outline=(255,190,20),width=2)
                draw.text((box[0],max(0,box[1]-12)),identity+" "+d["query"],fill=(255,255,0))
                serial.append({"id":identity,"query":d["query"],"score":d["score"],"mask":encode_mask(mask)})
            images.append(np.array(overlay))
            if output:
                directory=Path(output)
                Image.fromarray(frame["rgb"]).save(directory/f"{name}_rgb.png")
                overlay.save(directory/f"{name}_annotated.png")
                np.save(directory/f"{name}_depth.npy",frame["depth"],allow_pickle=False)
                (directory/f"{name}_calibration.json").write_text(json.dumps(c,indent=2),encoding="utf-8")
                (directory/f"{name}_masks.json").write_text(json.dumps(serial),encoding="utf-8")
        return {"observation_mode":"rgbd_sam3.1_detector","units":"metres and radians",
                **self.env.proprioception(),"queries":self.queries,"detections":objects,
                "cameras":[f["calibration"] for f in frames]},images
    def close(self):self.cameras.close()
