"""Render saved model actions with unchanged physics; no model inference is run."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tube_agent.env import TubeEnv
from tube_agent.policy import parse_action

FPS = 25
LEAD_FRAMES, TAIL_FRAMES = 16, 50
BACKGROUND = (15, 23, 35)
WHITE, MUTED, GREEN = (239, 245, 251), (165, 183, 202), (87, 222, 166)


def font(size):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def writer(path):
    return imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8,
                             pixelformat="yuv420p", macro_block_size=16,
                             ffmpeg_params=["-movflags", "+faststart", "-threads", "2"])


def render_episode(record, output, index):
    source = record["source"]
    if source["policy"] != "model" or source["task"] not in ("pick", "transport"):
        raise ValueError("Expected saved model actions for a tube task")
    messages = record["messages"]
    actions = [parse_action(m["content"]) for m in messages if m["role"] == "assistant"]
    observations = [json.loads(m["content"])["observation"]
                    for m in messages if m["role"] == "user"]
    if not actions or len(actions) != len(observations):
        raise ValueError("Each saved action must have its preceding state observation")
    stem = f"{index:02d}_seed_{source['seed']:05d}_try_{source['attempt']:02d}"
    path = output / f"{stem}.mp4"
    if path.exists():
        raise FileExistsError(path)
    env = TubeEnv(source["task"], source["split"], source["seed"],
                  max_steps=observations[0]["steps_left"],
                  detailed_feedback=source.get("feedback_version") == "detailed_v1")
    detail = None
    frames, times, observations_match = [], [], []
    stage = {"text": "READY", "command": "Saved model actions / same scene seed"}
    title_font, text_font, small_font = font(27), font(22), font(18)
    try:
        origin = float(env.data.time)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [*env.tube_pos[:2], .065]
        camera.distance, camera.azimuth, camera.elevation = .36, -50, -15
        detail = mujoco.Renderer(env.model, height=480, width=640)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 1

        def capture(overview):
            detail.update_scene(env.data, camera=camera, scene_option=options)
            near = detail.render().copy()
            canvas = Image.new("RGB", (1280, 640), BACKGROUND)
            canvas.paste(Image.fromarray(overview), (0, 64))
            canvas.paste(Image.fromarray(near), (640, 64))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 10), f"QWEN3.5-4B / {source['task'].upper()}", font=title_font, fill=WHITE)
            draw.text((690, 10), f"SEED {source['seed']}   ATTEMPT {source['attempt']}", font=title_font, fill=WHITE)
            draw.text((20, 42), "RECORDED ACTION REPLAY", font=small_font, fill=MUTED)
            draw.text((690, 42), "TRAIN SPLIT / NO NEW MODEL INFERENCE", font=small_font, fill=MUTED)
            for x, label in ((16, "OVERVIEW"), (656, "GRASP DETAIL")):
                draw.rectangle((x-5, 76, x+160, 103), fill=BACKGROUND)
                draw.text((x, 77), label, font=small_font, fill=WHITE)
            result = env.result()
            elapsed = max(0., float(env.data.time)-origin)
            draw.text((20, 552), stage["text"], font=text_font, fill=WHITE)
            draw.text((20, 583), stage["command"], font=small_font, fill=MUTED)
            draw.text((680, 552), f"Lift: {result['lift_height_m']*100:.2f} cm   Contact: {'YES' if result['held'] else 'NO'}",
                      font=text_font, fill=GREEN if result["held"] else WHITE)
            status = ("PASS" if result["success"] else "FAIL") if env.ended else "IN PROGRESS"
            draw.text((680, 583), f"{status}   |   sim {elapsed:.2f}s   |   1x simulation time",
                      font=small_font, fill=GREEN if env.ended and result["success"] else MUTED)
            draw.text((20, 614), "Original actions and physics. Closing freeze frame adds no simulation steps.",
                      font=small_font, fill=MUTED)
            times.append(elapsed)
            frames.append(np.asarray(canvas))

        env.frame_callback = capture
        capture(env.render())
        for step, (action, expected) in enumerate(zip(actions, observations), 1):
            observations_match.append(env.observe() == expected)
            stage["text"] = f"ACTION {step}/{len(actions)} / {action['action'].upper()}"
            stage["command"] = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
            _, feedback, _ = env.step(action)
            if not feedback["accepted"]:
                raise ValueError(f"Saved action was rejected: {feedback}")
        if not env.ended:
            raise ValueError("Saved trajectory ended before the environment terminated")
        result = env.result()
        capture(env.render())
        elapsed = times[-1]
        with writer(path) as video:
            for _ in range(LEAD_FRAMES):
                video.append_data(frames[0])
            for frame_index in range(math.ceil(elapsed*FPS)):
                sample = min(frame_index/FPS, elapsed)
                chosen = max(0, bisect_right(times, sample)-1)
                video.append_data(frames[chosen])
            for _ in range(TAIL_FRAMES):
                video.append_data(frames[-1])
        Image.fromarray(frames[-1]).save(output / f"{stem}.jpg", quality=94)
        return {"video": path.name, "poster": f"{stem}.jpg", "source": source,
                "simulation_seconds": round(elapsed, 4), "result": result,
                "recorded_action_count": len(actions),
                "exact_rounded_observation_matches": observations_match,
                "mode": "saved_action_replay", "new_model_inference": False}
    finally:
        if detail is not None:
            detail.close()
        env.close()


def montage(episodes, output):
    columns = 3
    rows = math.ceil(len(episodes)/columns)
    height = math.ceil((64 + rows*332 + 44)/16)*16
    readers = [imageio.get_reader(str(output/e["video"]), input_params=["-threads", "1"])
               for e in episodes]
    frames, exhausted = [None]*len(readers), [False]*len(readers)
    heading, label, note = font(25), font(18), font(17)
    count, last = 0, None
    try:
        with writer(output / "all_successes_grid.mp4") as video:
            while True:
                updated = False
                for i, reader in enumerate(readers):
                    if not exhausted[i]:
                        try:
                            frames[i] = reader.get_next_data()
                            updated = True
                        except (IndexError, StopIteration):
                            exhausted[i] = True
                if not updated:
                    break
                canvas = Image.new("RGB", (1200, height), BACKGROUND)
                draw = ImageDraw.Draw(canvas)
                tasks = ", ".join(sorted({e["source"]["task"].upper() for e in episodes}))
                draw.text((16, 7), f"{len(episodes)} SAVED SUCCESS TRAJECTORIES / {tasks}", font=heading, fill=WHITE)
                draw.text((16, 38), "Synchronized recorded-action replay. Shorter clips hold their final frame.", font=note, fill=MUTED)
                for i, (frame, episode) in enumerate(zip(frames, episodes)):
                    if frame is None:
                        raise RuntimeError("Video contains no readable frames")
                    x, y = (i % columns)*400, 64 + (i//columns)*332
                    closeup = Image.fromarray(frame).crop((640, 64, 1280, 544)).resize((400, 300), Image.Resampling.LANCZOS)
                    canvas.paste(closeup, (x, y+32))
                    src = episode["source"]
                    draw.text((x+10, y+5), f"{i+1:02d}   seed {src['seed']} / attempt {src['attempt']}", font=label, fill=WHITE)
                draw.text((16, height-32), "Same scene seeds and physical judge / 1x simulation time / no new model inference", font=note, fill=MUTED)
                last = np.asarray(canvas)
                video.append_data(last)
                count += 1
        Image.fromarray(last).save(output / "all_successes_grid.jpg", quality=94)
        return count
    finally:
        for reader in readers:
            reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(s) for s in args.data.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not records:
        parser.error("No saved trajectories found")
    episodes = []
    for index, record in enumerate(records, 1):
        item = render_episode(record, args.output, index)
        episodes.append(item)
        print(json.dumps({"video": item["video"], "result": item["result"],
                          "observation_matches": item["exact_rounded_observation_matches"]}), flush=True)
    frames = montage(episodes, args.output)
    report = {"mode": "saved_action_replay", "new_model_inference": False,
              "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "mujoco": mujoco.__version__, "fps": FPS, "physics_capture_fps": 12.5,
              "leading_freeze_seconds": LEAD_FRAMES/FPS, "closing_freeze_seconds": TAIL_FRAMES/FPS,
              "grid_frames": frames, "episodes": episodes}
    (args.output / "replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    note = ("Saved Qwen model action replays in the original MuJoCo environment.\n"
            "No new model inference or training is performed. Camera views and captions are for visualization only.\n"
            "Videos play at 1x simulation speed (inference waiting time is absent), with static intro/outro holds.\n"
            "The JSON report records actual replay results and comparisons to original pre-action observations.\n"
            "These are selected successful pick episodes, not a new success-rate evaluation or transport demonstration.\n")
    (args.output / "README.txt").write_text(note, encoding="ascii")
    bundle = args.output / "success_replay_videos.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [args.output / name for episode in episodes for name in (episode["video"], episode["poster"])]
        paths += [args.output / name for name in ("all_successes_grid.mp4", "all_successes_grid.jpg",
                                                 "replay_report.json", "README.txt")]
        for path in paths:
            archive.write(path, path.name)
    print(f"Saved grid, individual videos and archive: {args.output}", flush=True)


if __name__ == "__main__":
    main()
