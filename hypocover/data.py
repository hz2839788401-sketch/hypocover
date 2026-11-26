"""Build exact HypoSpace observation sets and a separate evaluation key."""

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations, permutations, product
from pathlib import Path

from .tasks import Task, reachability


def voxel_cases():
    for count in (2, 3, 4):
        for occupied in combinations(range(9), count):
            top = [int(i in occupied) for i in range(9)]
            case = Task("v-" + "".join(map(str, top)), "voxel",
                        {"grid": 3, "height": 3, "top": top})
            yield count, case, list(product(range(1, 4), repeat=count))


def causal_cases():
    for n, interventions in ((3, 1), (4, 2)):
        nodes = tuple("ABCD"[:n])
        edges = tuple(a + b for a, b in permutations(nodes, 2))
        groups = defaultdict(list)
        for mask in range(1 << len(edges)):
            candidate = tuple(edge for j, edge in enumerate(edges) if mask & (1 << j))
            closure = reachability(nodes, candidate)
            if any(node in closure[node] for node in nodes):
                continue
            for observed in combinations(nodes, interventions):
                key = tuple((node, tuple(sorted(closure[node]))) for node in observed)
                groups[key].append(candidate)
        for obs, answers in sorted(groups.items()):
            if len(answers) < 3:
                continue
            case = Task(f"c{n}-" + "-".join(node + "".join(reached) for node, reached in obs),
                        "causal", {"nodes": nodes, "reaches": obs})
            yield n, case, answers


def build(destination, seed, samples):
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for stratum, case, answers in (*voxel_cases(), *causal_cases()):
        groups[(case.domain, stratum)].append((case, answers))
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("pilot", "test", "format-test"):
        tasks, answer_key = [], {}
        for group in groups.values():
            if split == "pilot":
                rng.shuffle(group)
            if split == "pilot":
                part = group[:2]
            elif split == "test":
                part = group[2:2 + samples]
            else:
                part = [row for row in group[2 + samples:2 + samples * 2] if row[0].domain == "voxel"]
            for task, answers in part:
                tasks.append(asdict(task))
                answer_key[task.id] = {
                    "count": len(answers),
                    "complexity": dict(sorted(Counter(map(task.complexity, answers)).items())),
                    "accepted": sorted(task.render(answer) for answer in answers),
                }
        for name, value in ((split, tasks), (f"{split}-key", answer_key)):
            (destination / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=33550336)
    parser.add_argument("--per-stratum", type=int, default=6)
    args = parser.parse_args()
    build(args.out, args.seed, args.per_stratum)
