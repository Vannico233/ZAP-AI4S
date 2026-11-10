"""One global budget: at most 32 node expansions, four continuations per node."""
import argparse
from collections import Counter
from datetime import timedelta, datetime, timezone
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from tube_agent.lab.experiment import read_json, write_json
from tube_agent.lab.vision import SamClient
from .config import load, digest, TASKS
from .learning import load_model, Generator
from .rollout import episode, read_rows
from .data import warmup_episodes


def node_key(scene, rows):
    return digest([scene["sha256"], [r["action"] for r in rows]])


class Forest:
    def __init__(self, config):
        self.config = config
        self.nodes = {}
        self.task_expansions = Counter()
        self.solved = set()
        self.expansions = []
        self.blocked = set()

    def add(self, directory, limit=None):
        result = read_json(Path(directory)/"result.json")
        if result.get("error") or result["scene"]["split"] != "train" or result.get("policy") != "model":
            return
        rows = list(read_rows(directory)); prefix = []
        parent = None
        for row in rows[:limit]:
            prefix.append(row)
            if not row["feedback"].get("accepted"): break
            if not row.get("node_eligible", False): continue
            if row["action"]["action"] not in ("move_to", "set_gripper"): continue
            if row["action"]["action"] == "move_to" and not row["feedback"].get("reached"): continue
            judge = row["judge_after"]
            if judge["success"] or judge["severe_collision"] or judge["dropped"]: break
            if len(prefix) >= self.config["policy"]["max_steps"]: break
            if judge["reward_terms"]["terminal_progress"] <= .001: continue
            key = node_key(result["scene"], prefix)
            if key in self.blocked: break
            self.nodes.setdefault(key, {"key": key, "parent": parent, "task": result["task"],
                "scene": result["scene"], "directory": str(Path(directory).resolve()),
                "length": len(prefix), "progress": judge["reward_terms"]["terminal_progress"],
                "visits": 0, "wins": 0, "failures": {}})
            parent = key
        for task in TASKS:
            active = [n for n in self.nodes.values() if n["task"] == task]
            cap = self.config["search"]["max_nodes_per_task"]
            for node in sorted(active, key=lambda n: (n["progress"], -n["length"]))[:max(0, len(active)-cap)]:
                self.nodes.pop(node["key"], None)

    def select(self):
        active = [n for n in self.nodes.values() if n["task"] not in self.solved and n["key"] not in self.blocked]
        if not active: return None
        tasks = {n["task"] for n in active}
        task = min(tasks, key=lambda t: (self.task_expansions[t], t))
        def score(n):
            return n["wins"]/(1+n["visits"])+n["progress"]+self.config["search"]["exploration"]*math.sqrt(
                math.log(2+sum(self.task_expansions.values()))/(1+n["visits"]))
        return dict(max((n for n in active if n["task"] == task), key=score))

    def update(self, node, results):
        self.task_expansions[node["task"]] += 1
        stored = self.nodes[node["key"]]
        for directory, result in results:
            stored["visits"] += 1; stored["wins"] += int(result["success"])
            if result["success"]: self.solved.add(node["task"])
            else:
                reason = result["failure_type"]
                stored["failures"][reason] = stored["failures"].get(reason, 0)+1
            self.add(directory)
        prune = any(n >= self.config["search"]["same_failure_prune"] for n in stored["failures"].values())
        if prune:
            self.blocked.add(node["key"])
            changed = True
            while changed:
                changed = False
                for key, child in self.nodes.items():
                    if child["parent"] in self.blocked and key not in self.blocked:
                        self.blocked.add(key); changed = True
        self.expansions.append({"selected_node": node, "attempts": len(results), "pruned": prune,
            "results": [{"directory": str(d), "success": r["success"], "failure_type": r["failure_type"],
                         "replayed_actions": r["replayed_actions"], "generated_actions": r["generated_actions"]}
                        for d, r in results]})

    def state(self):
        return {"nodes": self.nodes, "task_expansions": dict(self.task_expansions),
                "solved": sorted(self.solved), "blocked": sorted(self.blocked), "expansions": self.expansions}

    def restore(self, state):
        self.nodes = state["nodes"]; self.task_expansions = Counter(state["task_expansions"])
        self.solved = set(state["solved"]); self.blocked = set(state["blocked"]); self.expansions = state["expansions"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--model", required=True)
    a = p.parse_args(); root = a.run.resolve(); c = load(root/"config.json")
    output = root/"search"; output.mkdir(exist_ok=True)
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group("nccl", timeout=timedelta(hours=24))
    def broadcast(value):
        values = [value]
        if world > 1: dist.broadcast_object_list(values, src=0)
        return values[0]
    forest = Forest(c); candidates = []
    if rank == 0:
        successes = Counter(); paths = list(warmup_episodes(root))
        for path in paths:
            result = read_json(path); successes[result["task"]] += int(result["success"])
        for path in paths:
            if not successes[read_json(path)["task"]]: forest.add(path.parent)
        candidates = sorted({n["task"] for n in forest.nodes.values()})
        if (output/"state.json").exists(): forest.restore(read_json(output/"state.json"))
        write_json(output/"plan.json", {"candidate_tasks": candidates, "warmup_successes": dict(successes),
            "budget_global": 128, "continuations_per_node": 4, "model": "after_warmup",
            "selection": "round-robin least expanded task; wins/(visits+1) + terminal_progress + 0.45*sqrt(log(2+total_expansions)/(visits+1))",
            "pruning": "3 continuations with the same failure type prune node and descendants",
            "root_replay_actions_count_separately": True, "weight_updates": 0})
    candidates = broadcast(candidates)
    if not candidates:
        if rank == 0: write_json(output/"DONE.json", {"continuations": 0, "reason": "no_zero_success_task_with_valid_prefix"})
        if world > 1: dist.destroy_process_group()
        return
    model, processor = load_model(a.model, c, root/"checkpoints/after_warmup", f"cuda:{local}", False)
    generator = Generator(model, processor, c); frozen = generator.bind("after_warmup")
    segmenter = SamClient(c["execution"]["sam_url"])
    try:
        segmenter.health()
        while True:
            selection = None
            if rank == 0 and len(forest.expansions) < 32:
                node = forest.select()
                if node: selection = {"node": node, "expansion": len(forest.expansions)}
            selection = broadcast(selection)
            if selection is None: break
            node, number = selection["node"], selection["expansion"]
            prefix = list(read_rows(node["directory"]))[:node["length"]]
            results = []
            for sample in range(rank, 4, world):
                directory = output/"rollouts"/f"expansion_{number:03d}"/f"try_{sample:02d}"
                if (directory/"result.json").exists() and read_json(directory/"result.json").get("error"):
                    directory.rename(directory.with_name(directory.name+".incomplete-"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
                if (directory/"result.json").exists():
                    result = read_json(directory/"result.json")
                    if (result["scene"] != node["scene"] or result["generator_provenance"]["adapter_sha256"] != frozen or
                            result["search_prefix_sha256"] != digest([r["action"] for r in prefix])):
                        raise ValueError("Search continuation belongs to another node or policy")
                else:
                    if directory.exists(): directory.rename(directory.with_name(directory.name+".incomplete-"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
                    generator.bind("after_warmup", frozen)
                    result = episode(node["scene"], c, directory, generator, segmenter,
                        20000000+number*4+sample, "after_warmup", prefix=prefix, origin="search")
                if result["error"]: raise RuntimeError(result["error"])
                results.append((str(directory), result))
            gathered = [None]*world
            if world > 1: dist.all_gather_object(gathered, results)
            else: gathered = [results]
            if rank == 0:
                complete = sorted([item for batch in gathered for item in batch])
                if len(complete) != 4: raise RuntimeError("Incomplete four-continuation expansion")
                forest.update(node, complete); write_json(output/"state.json", forest.state())
                print(json_line({"event": "search_expansion_complete", "expansions": number+1,
                                 "continuations": (number+1)*4, "task": node["task"], "solved": sorted(forest.solved)}), flush=True)
        if rank == 0:
            write_json(output/"DONE.json", {"continuations": len(forest.expansions)*4,
                "solved_tasks": sorted(forest.solved), "candidate_tasks": candidates,
                "replayed_actions": sum(r["replayed_actions"] for e in forest.expansions for r in e["results"]),
                "new_generated_actions": sum(r["generated_actions"] for e in forest.expansions for r in e["results"]),
                "weight_updates": 0})
    finally:
        segmenter.close()
        if world > 1: dist.destroy_process_group()


def json_line(value):
    import json
    return json.dumps(value)


if __name__ == "__main__": main()
