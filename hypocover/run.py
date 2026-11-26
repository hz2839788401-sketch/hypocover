"""Run matched, sequential proposal experiments with round-based checkpoints."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .model import LanguageModel, PrefixFeedback
from .policies import METHODS, Policy
from .tasks import load_tasks


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_for(*parts):
    data = "\x1f".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


@dataclass
class Episode:
    task: object
    replicate: int
    method: str
    policy: Policy
    seen: set = field(default_factory=set)
    archive: list = field(default_factory=list)
    valid_archive: list = field(default_factory=list)

    def propose(self, step, memory):
        cue = self.policy.choose(step)
        use_memory = self.method not in {"iid", "grammar", "grammar_tilt"}
        examples = self.archive[-memory:] if memory and use_memory else []
        return cue, self.task.prompt(cue, examples)

    def decoder_feedback(self, strength):
        grammar = self.method in {"grammar", "grammar_tilt"}
        if self.task.domain != "voxel" or (not grammar and self.method not in {"tilt", "adaptive"}):
            return None
        return PrefixFeedback(
            tuple(self.valid_archive), len(self.task.positions), self.task.observation["height"],
            0.0 if self.method == "grammar" else strength, grammar,
        )

    def observe(self, cue, text, step):
        candidate = self.task.parse(text)
        valid = self.task.valid(candidate)
        novel = candidate is not None and candidate not in self.seen
        if novel:
            self.seen.add(candidate)
            self.archive.append(self.task.render(candidate))
            if valid:
                self.valid_archive.append(candidate)
        self.policy.update(cue, valid, novel, step)
        return {
            "candidate": self.task.render(candidate) if candidate is not None else None,
            "valid": valid,
            "novel": novel,
            "cue_match": self.task.matches(cue, candidate) if cue is not None else None,
        }


def run(args):
    tasks = [task for task in load_tasks(args.tasks) if task.domain in args.domains]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise ValueError("No selected tasks")
    if any(method in {"grammar", "grammar_tilt"} for method in args.methods):
        if any(task.domain != "voxel" for task in tasks):
            raise ValueError("Grammar decoding is implemented only for voxel tasks")
        if args.max_new_tokens < max(len(task.positions) for task in tasks) + 1:
            raise ValueError("Grammar decoding needs one token per height plus a stop token")
    if args.budget < 1 or args.repeats < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("Budget, repeats, batch size and token cap must be positive")
    if len(set(args.methods)) != len(args.methods) or args.memory < 0:
        raise ValueError("Methods must be distinct and memory must be nonnegative")
    if not 0 < args.decay <= 1 or args.temperature <= 0 or not 0 < args.top_p <= 1 or args.tilt < 0:
        raise ValueError("Invalid sampling parameters")
    oracle = {}
    if "oracle" in args.methods:
        if not args.oracle_key:
            raise ValueError("--oracle-key is required for the oracle comparator")
        oracle = {case: row["complexity"] for case, row in read(args.oracle_key).items()}
    episodes = [
        Episode(task, replication, method, Policy(
            task, method, random.Random(seed_for(args.seed, task.id, replication)),
            oracle.get(task.id) if method == "oracle" else None, args.decay
        ))
        for task in tasks for replication in range(args.repeats) for method in args.methods
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "protocol.json"
    protocol = {
        key: value for key, value in vars(args).items() if key not in {"out", "tasks", "oracle_key", "offline"}
    }
    protocol["task_ids"] = [task.id for task in tasks]
    protocol["observations_sha256"] = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
    if "oracle" in args.methods:
        protocol["oracle_histogram_sha256"] = hashlib.sha256(
            json.dumps(oracle, sort_keys=True).encode()
        ).hexdigest()
    if manifest.exists() and read(manifest)["protocol"] != protocol:
        raise ValueError("Output directory is for a different protocol")
    model = None
    for step in range(args.budget):
        checkpoint = args.out / f"round-{step + 1:03d}.json"
        proposals = [episode.propose(step, args.memory) for episode in episodes]
        if checkpoint.exists():
            rows = read(checkpoint)
            for episode, (cue, _), row in zip(episodes, proposals, rows, strict=True):
                replay = episode.observe(cue, row["response"], step)
                if (row["task"], row["replicate"], row["method"], row["cue"]) != (
                    episode.task.id, episode.replicate, episode.method, list(cue) if cue else None
                ) or any(row[key] != value for key, value in replay.items()):
                    raise ValueError(f"Invalid replay at {checkpoint}")
            continue
        if model is None:
            model = LanguageModel(
                args.model, args.device, args.dtype, args.threads, args.offline, args.batch_size,
                args.temperature, args.top_p,
            )
            environment = {"protocol": protocol, "model": model.metadata}
            if manifest.exists() and read(manifest) != environment:
                raise ValueError("Resume requires the original model environment")
            write(manifest, environment)
        start = time.perf_counter()
        requests = [prompt for cue, prompt in proposals]
        plans = [episode.decoder_feedback(args.tilt) for episode in episodes]
        outputs = model.generate(requests, plans, seed_for(args.seed, step), args.max_new_tokens)
        rows = []
        for episode, (cue, _), generation in zip(episodes, proposals, outputs, strict=True):
            rows.append({
                "task": episode.task.id, "domain": episode.task.domain, "method": episode.method,
                "replicate": episode.replicate, "cue": cue, "response": generation.text,
                "input_tokens": generation.input_tokens, "output_tokens": generation.output_tokens,
                **episode.observe(cue, generation.text, step),
            })
        write(checkpoint, rows)
        print(f"Round {step+1}/{args.budget} | {len(rows)} proposals | {time.perf_counter()-start:.1f}s",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--oracle-key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=["iid", "memory", "static", "adaptive"])
    parser.add_argument("--domains", choices=["voxel", "causal"], nargs="+", default=["voxel", "causal"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--memory", type=int, default=6)
    parser.add_argument("--decay", type=float, default=0.90)
    parser.add_argument("--tilt", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    from filelock import FileLock, Timeout

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(args.out / "run.lock"), timeout=0):
            run(args)
    except Timeout:
        parser.error(f"Another process is writing to {args.out}")
