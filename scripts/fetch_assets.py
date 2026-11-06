"""Fetch only the Panda collision meshes; render those meshes as the robot.

The robot kinematics, dynamics, collision model and actuators remain upstream's.
No simulator services, full benchmark downloads or pretrained action policies.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import urllib.request
import xml.etree.ElementTree as ET

REVISION = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/franka_emika_panda/"
DEST = Path(__file__).resolve().parents[1] / "assets" / "panda"


def download(name):
    with urllib.request.urlopen(BASE + name, timeout=60) as response:
        return response.read()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "assets").mkdir(exist_ok=True)
    root = ET.fromstring(download("panda.xml"))
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("class") == "visual":
                parent.remove(child)
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    asset = root.find("asset")
    meshes = []
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name", Path(mesh.get("file")).stem) not in used:
            asset.remove(mesh)
        else:
            meshes.append(mesh.get("file"))

    def save_mesh(name):
        target = DEST / "assets" / name
        target.write_bytes(download("assets/" + name))
        return name

    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in pool.map(save_mesh, meshes):
            print(name)
    (DEST / "LICENSE").write_bytes(download("LICENSE"))
    ET.indent(root)
    ET.ElementTree(root).write(DEST / "panda.xml", encoding="utf-8")
    (DEST / "SOURCE.json").write_text(json.dumps({
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "revision": REVISION,
        "path": "franka_emika_panda",
        "modification": "Removed visual-only meshes; collision geometry retained for rendering."
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(meshes)} meshes to {DEST}")


if __name__ == "__main__":
    main()
