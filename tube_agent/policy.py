"""Only three JSON actions. No code execution, tool framework or grasp macro."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re

import httpx
from PIL import Image

SYSTEM = """You control a simulated Franka Panda parallel gripper on a horizontal table.
Choose exactly ONE action per response, as a JSON object. No prose or markdown.
Actions:
{"action":"move","xyz":[x,y,z],"yaw":radians}
{"action":"gripper","opening":metres}
{"action":"finish"}
Coordinates are absolute world metres. Z is up; table top is z=0.
The TCP is the midpoint of the fingertip contact pads, NOT the palm.
The gripper points down. yaw is rotation about world Z; at yaw=0 its fingers
close along world Y and its long direction is world X. move preserves opening.
Workspace: x=[0.20,0.65], y=[-0.40,0.40], z=[0.009,0.50], yaw=[-pi,pi].
Opening is the distance between fingers: 0 closes, 0.08 fully opens.
Actions are executed with physical contacts and gravity. Objects are not attached
automatically. A successful move is not proof of a successful grasp or task.
Observations contain exact simulator object geometry to isolate action learning;
they are not results of a learned perception model. Use feedback after each action.
Budget: at most 20 actions unless steps_left says otherwise. Submit finish when done.
"""

# Opt-in prompt assistance, kept separate from the original baseline.
SYSTEM_PROMPTS = {"base": SYSTEM, "geometry_v1": SYSTEM + """
Geometry and execution reminders:
- move.xyz is a TCP target, not an object center or a support surface point.
  tray_center gives the center of the tray's TOP SUPPORT SURFACE. Its z is not
  a placement target for the TCP. Account for the tube's radius, orientation
  and the observed offset between its center and the TCP when placing it.
- Workspace limits apply to commanded TCP coordinates. A scene surface may lie
  below the minimum allowed TCP z. Do not copy such a point into move.xyz.
- Motions have tracking error and a grasp can settle. For a required minimum
  lift, aim above the minimum with a margin and check the observed object pose.
  reached=true only describes motion tolerance; it does not mean task success.
- For transport, lift the tube clear of the support before moving sideways;
  lower it at the destination, release it, and withdraw the gripper.
- A rejected action did not execute. Correct the offending parameter using
  the feedback and workspace bounds instead of repeating the rejected action.
"""}


def parse_action(text):
    # Some compatible providers still emit a think section or fenced JSON.
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if clean.startswith("```") and clean.endswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean).strip()
    action = json.loads(clean)
    if not isinstance(action, dict):
        raise ValueError("Response must be one JSON object")
    return action


def observation_message(instruction, observation, feedback=None):
    return json.dumps({"instruction": instruction, "observation": observation,
                       "last_action_feedback": feedback}, ensure_ascii=False)


class ScriptedPolicy:
    """Environment sanity baseline; NEVER label these trajectories self-generated."""
    def __init__(self, task):
        self.task = task
        self.index = 0
        self.history = []

    def act(self, instruction, observation, feedback=None, image=None):
        obj = next(o for o in observation["objects"] if o["name"] == "tube_red")
        if self.index == 0:
            x, y, z = obj["center"]
            yaw = math.atan2(obj["axis"][1], obj["axis"][0])
            tray = observation["tray_center"]
            def move(a, b, c):
                return {"action": "move", "xyz": [a, b, c], "yaw": yaw}
            self.actions = [move(x, y, .20), move(x, y, z+.003),
                            {"action": "gripper", "opening": 0}, move(x, y, .19)]
            if self.task == "transport":
                self.actions += [move(tray[0], tray[1], .19),
                                 move(tray[0], tray[1], tray[2]+obj["radius"]+.006),
                                 {"action": "gripper", "opening": .08}, move(tray[0], tray[1], .20)]
            self.actions += [{"action": "finish"}]
        action = self.actions[min(self.index, len(self.actions)-1)]
        self.index += 1
        return action, json.dumps(action)


class ModelPolicy:
    def __init__(self, base_url, model, temperature=0., seed=0, max_tokens=192, vision=False,
                 prompt_variant="base"):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.temperature, self.seed, self.max_tokens = temperature, seed, max_tokens
        self.vision = vision
        self.history = [{"role": "system", "content": SYSTEM_PROMPTS[prompt_variant]}]
        key = os.environ.get("TUBE_API_KEY", "local")
        self.client = httpx.Client(timeout=180., trust_env=False,
                                   headers={"Authorization": f"Bearer {key}"})

    def act(self, instruction, observation, feedback=None, image=None):
        content = observation_message(instruction, observation, feedback)
        if self.vision:
            if image is None:
                raise ValueError("Vision policy requires an image")
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=85)
            content = [{"type": "text", "text": content},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}]
        self.history.append({"role": "user", "content": content})
        response = self.client.post(self.base_url + "/chat/completions", json={
            "model": self.model, "messages": self.history, "temperature": self.temperature,
            "seed": self.seed, "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}})
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        if not isinstance(raw, str):
            raise ValueError("Endpoint did not return textual assistant content")
        self.history.append({"role": "assistant", "content": raw})
        try:
            return parse_action(raw), raw
        except (ValueError, TypeError):
            # Count an invalid action and allow a bounded correction on the next turn.
            return {"action": "invalid_model_output"}, raw

    def close(self):
        self.client.close()
