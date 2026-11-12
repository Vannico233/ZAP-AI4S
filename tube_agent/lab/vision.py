"""Calibrated RGB-D sensing and mask-derived 3D measurements. No object state."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import httpx
import numpy as np
from PIL import Image, ImageDraw
from tube_agent import interface

PERCEPTION_VERSION = "sam_query_retry_v1"
# Bounded, image-description retries; these contain no poses or simulator IDs.
QUERY_ALIASES = {
    "red test tube": ("red tube", "red cylinder"),
    "teal tray": ("cyan rectangle", "teal rectangular plate"),
    "yellow test tube rack": ("yellow block", "yellow container"),
    "blue test tube rack": ("blue block", "blue container"),
    "white cap": ("white cylinder", "white lid"),
}

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
    def __init__(self, env, names=None, width=None, height=None):
        import mujoco
        self.middle_frame = getattr(env, "ACTION_PROTOCOL", None) == interface.PROTOCOL
        spec = getattr(env, "spec", {}).get("observation", {})
        if names is None:
            names = interface.CAMERA_NAMES if self.middle_frame else ("top", "side")
        width = spec.get("width", 640) if width is None else width
        height = spec.get("height", 480) if height is None else height
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
                         "fx":float(focal),"fy":float(focal),"cx":(self.width-1)/2,"cy":(self.height-1)/2}
            if self.middle_frame:
                calibration.update(coordinate_frame=interface.FRAME,
                                   T_middle_from_view=interface.view_extrinsics(self.env, name).tolist(),
                                   convention="Optical: x right, y down, z forward; metric axial depth")
            else:
                calibration.update(position=self.env.data.cam_xpos[cid].tolist(),
                                   rotation_camera_to_world=self.env.data.cam_xmat[cid].reshape(3,3).tolist(),
                                   convention="OpenGL: x right, y up, optical direction -z; metric axial depth")
            frames.append({"rgb":rgb,"depth":depth,"calibration":calibration})
        return frames

    def close(self): self.renderer.close()

def unproject(depth, mask, calibration):
    valid=mask & np.isfinite(depth) & (depth>.01) & (depth<3.)
    v,u=np.nonzero(valid)
    if not len(v): return np.empty((0,3))
    d=depth[v,u]
    optical=np.column_stack(((u-calibration["cx"])*d/calibration["fx"],
                             (v-calibration["cy"])*d/calibration["fy"],d))
    if "T_middle_from_view" in calibration:
        transform = np.asarray(calibration["T_middle_from_view"])
        return optical @ transform[:3,:3].T + transform[:3,3]
    return (optical @ interface.GL_FROM_OPTICAL) @ np.asarray(calibration["rotation_camera_to_world"]).T + calibration["position"]

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
        self.url=url.rstrip("/"); self.client=httpx.Client(timeout=httpx.Timeout(660.,connect=10.),trust_env=False)
    def health(self):
        r=self.client.get(self.url+"/health");r.raise_for_status();h=r.json()
        if h.get("backend")!="sam3.1_detector" or not h.get("ready"):
            raise RuntimeError("A ready native SAM 3.1 detector service is required")
        return h
    def segment(self,rgb,queries):
        r=self.client.post(self.url+"/segment",json={"image":image_url(rgb),"queries":queries})
        if not r.is_success:
            try:detail=r.json().get("error",r.text)
            except (ValueError,AttributeError):detail=r.text
            raise RuntimeError(f"SAM /segment HTTP {r.status_code}: {str(detail)[:6000]}")
        return [{**d,"mask":decode_mask(d["mask"])} for d in r.json()["detections"]]
    def close(self):self.client.close()


def detect_views(frames, segmenter, queries):
    """Retry known missing concepts in both images; never fabricate a mask."""
    detections = [[] for _ in frames]
    attempts = []

    def request(view, actual_queries, canonical_names):
        frame = frames[view]
        raw = segmenter.segment(frame["rgb"], actual_queries)
        counts = {q: 0 for q in actual_queries}
        for detection in raw:
            query = detection["query"]
            if query not in canonical_names:
                continue
            mask = detection["mask"]
            if mask.shape != frame["depth"].shape:
                raise ValueError("Mask/depth shape mismatch")
            measurement = measure(unproject(frame["depth"], mask, frame["calibration"]))
            if measurement is None:
                continue
            if frame["calibration"].get("coordinate_frame") == interface.FRAME:
                measurement["coordinate_frame"] = interface.FRAME
                measurement["visible_depth_extent_m"] = measurement.pop("visible_height")
            detections[view].append({**detection, "query": canonical_names[query],
                                    "segmentation_query": query, "measurement": measurement})
            counts[query] += 1
        for query in actual_queries:
            attempts.append({"camera": frame["calibration"]["name"],
                             "query": canonical_names[query], "segmentation_query": query,
                             "valid_mask_count": counts[query]})

    def found():
        return {d["query"] for view in detections for d in view}

    for view in range(len(frames)):
        request(view, list(queries), {q: q for q in queries})
    for canonical in queries:
        if canonical in found():
            continue
        for alias in QUERY_ALIASES.get(canonical, ()):
            for view in range(len(frames)):
                request(view, [alias], {alias: canonical})
            if canonical in found():
                break
    missing = [q for q in queries if q not in found()]
    return detections, {"version": PERCEPTION_VERSION, "missing_queries": missing,
                        "status": "missing_detections" if missing else "candidate_masks_available",
                        "query_attempts": attempts,
                        "note": "Mask availability is not a guarantee of correct object identity."}

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
            if output:
                c=frame["calibration"];name=c["name"];directory=Path(output)
                Image.fromarray(frame["rgb"]).save(directory/f"{name}_rgb.png")
                np.save(directory/f"{name}_depth.npy",frame["depth"],allow_pickle=False)
                (directory/f"{name}_calibration.json").write_text(json.dumps(c,indent=2),encoding="utf-8")
        view_detections,self.last_audit=detect_views(frames,self.segmenter,self.queries)
        if output:
            (Path(output)/"perception.json").write_text(json.dumps(self.last_audit,indent=2),encoding="utf-8")
        for frame,detections in zip(frames,view_detections):
            c=frame["calibration"];name=c["name"]
            overlay=Image.fromarray(frame["rgb"]);draw=ImageDraw.Draw(overlay)
            serial=[]
            for index,d in enumerate(detections):
                mask=d["mask"]
                measurement=d["measurement"]
                identity=f"{name}:{index}"
                objects.append({"id":identity,"query":d["query"],"segmentation_query":d["segmentation_query"],
                                "score":d["score"],"camera":name,**measurement})
                v,u=np.nonzero(mask)
                tinted=np.array(overlay)
                tinted[mask]=(.75*tinted[mask]+.25*np.array([255,190,20])).astype(np.uint8)
                overlay=Image.fromarray(tinted);draw=ImageDraw.Draw(overlay)
                box=[int(u.min()),int(v.min()),int(u.max()),int(v.max())]
                draw.rectangle(box,outline=(255,190,20),width=2)
                draw.text((box[0],max(0,box[1]-12)),identity+" "+d["segmentation_query"],fill=(255,255,0))
                serial.append({"id":identity,"query":d["query"],"segmentation_query":d["segmentation_query"],
                               "score":d["score"],"mask":encode_mask(mask)})
            images.append(np.array(overlay))
            if output:
                directory=Path(output)
                overlay.save(directory/f"{name}_annotated.png")
                (directory/f"{name}_masks.json").write_text(json.dumps(serial),encoding="utf-8")
        return {"observation_mode":"rgbd_sam3.1_detector","units":"metres and radians",
                "perception_version":PERCEPTION_VERSION,"perception_status":self.last_audit["status"],
                "missing_queries":self.last_audit["missing_queries"],
                **self.env.proprioception(),"queries":self.queries,"detections":objects,
                "cameras":[f["calibration"] for f in frames]},images
    def close(self):self.cameras.close()
