"""Check selected voxel answer keys against a local clone of the official generator."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main(upstream, data):
    script = upstream / "3d" / "generate_3d_dataset_complete.py"
    spec = importlib.util.spec_from_file_location("upstream_voxel", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = set()
    for split in ("pilot", "test", "format-test"):
        tasks = json.loads((data / f"{split}.json").read_text(encoding="utf-8"))
        key = json.loads((data / f"{split}-key.json").read_text(encoding="utf-8"))
        for task in tasks:
            if task["domain"] != "voxel":
                continue
            if task["id"] in seen:
                raise ValueError(f"Observation occurs in multiple splits: {task['id']}")
            seen.add(task["id"])
            obs = task["observation"]
            grid = np.asarray(obs["top"]).reshape((obs["grid"], obs["grid"]))
            structures = module.enumerate_all_structures_by_heights(grid, obs["height"])
            official = {
                "".join(map(str, np.stack(structure.layers).sum(axis=0)[grid.astype(bool)]))
                for structure in structures
            }
            if set(key[task["id"]]["accepted"]) != official:
                raise ValueError(f"Upstream answer mismatch for {task['id']}")
    print(f"Exact upstream agreement for {len(seen)} disjoint voxel observations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Local clone of CTT-Pavilion/_HypoSpace")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    main(args.upstream, args.data)
