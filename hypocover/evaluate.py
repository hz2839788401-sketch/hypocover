"""Score HypoSpace traces and compute paired instance-level bootstrap intervals."""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .run import read, write


def episode_summary(rows, key):
    seen, recovered = set(), set()
    curve, valid = [], 0
    for row in rows:
        candidate = row["candidate"]
        is_new = candidate is not None and candidate not in seen
        if candidate is not None:
            seen.add(candidate)
        in_key = candidate in key["accepted"]
        if row["valid"] != in_key or row["novel"] != is_new:
            raise ValueError(f"Independent evaluation disagrees for {row['task']}: {candidate}")
        if in_key:
            valid += 1
            recovered.add(candidate)
        curve.append(len(recovered) / key["count"])
    n = len(rows)
    cues = [row for row in rows if row["cue"] is not None]
    return {
        "validity": valid / n,
        "uniqueness": len(seen) / n,
        "recovery": curve[-1],
        "distinct_valid": len(recovered),
        "new_valid_rate": len(recovered) / n,
        "coverage_auc": mean(curve),
        "parse_rate": sum(row["candidate"] is not None for row in rows) / n,
        "cue_adherence": mean(r["cue_match"] for r in cues) if cues else None,
        "prompt_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "curve": curve,
    }


def avg(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return mean(values) if values else None


def interval(differences, samples=5000):
    rng = random.Random(117)
    estimates = sorted(
        mean(rng.choices(differences, k=len(differences))) for _ in range(samples)
    )
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]]


def evaluate(source, key, destination, partial=False):
    key = read(key)
    manifest = read(source / "protocol.json")
    protocol = manifest["protocol"]
    rounds = sorted(source.glob("round-*.json"))
    if not rounds:
        raise ValueError("No generation rounds")
    if len(rounds) != protocol["budget"] and not partial:
        raise ValueError("Run is incomplete. Pass --partial to inspect intermediate results.")
    episodes = defaultdict(list)
    for step, path in enumerate(rounds, 1):
        if path.name != f"round-{step:03d}.json":
            raise ValueError("Generation rounds must form a contiguous prefix")
        for row in read(path):
            episodes[(row["domain"], row["task"], row["method"], row["replicate"])].append(row)
    expected = len(protocol["task_ids"]) * len(protocol["methods"]) * protocol["repeats"]
    if len(episodes) != expected or any(len(rows) != len(rounds) for rows in episodes.values()):
        raise ValueError("Incomplete or duplicated episode in trace")
    metrics = []
    for (domain, task, method, replicate), rows in episodes.items():
        case_key = key[task]
        if case_key["count"] != len(set(case_key["accepted"])):
            raise ValueError(f"Invalid evaluation key for {task}")
        stats = episode_summary(rows, case_key)
        metrics.append({
            "domain": domain, "task": task, "method": method, "replicate": replicate,
            "hypothesis_count": case_key["count"], **stats,
        })
    groups = defaultdict(list)
    for row in metrics:
        groups[row["domain"], row["method"]].append(row)
        groups["all", row["method"]].append(row)
    report = {}
    for (domain, method), rows in groups.items():
        keys = ("validity", "uniqueness", "recovery", "distinct_valid", "new_valid_rate",
                "coverage_auc", "parse_rate", "cue_adherence", "prompt_tokens", "output_tokens")
        report.setdefault(domain, {})[method] = {
            "instances": len({r["task"] for r in rows}),
            "episodes": len(rows),
            **{metric: avg(rows, metric) for metric in keys},
            "curve": [mean(r["curve"][i] for r in rows) for i in range(len(rounds))],
        }
    effects = {}
    for domain, methods in report.items():
        treatment = "adaptive" if "adaptive" in methods else "grammar_tilt"
        if treatment not in methods:
            continue
        for other in methods:
            if other == treatment:
                continue
            grouped = defaultdict(list)
            for row in groups[domain, treatment] + groups[domain, other]:
                grouped[row["task"], row["method"]].append(row["recovery"])
            case_ids = sorted({task for task, _ in grouped})
            differences = [mean(grouped[task, treatment]) - mean(grouped[task, other])
                           for task in case_ids]
            effects.setdefault(domain, {})[other] = {
                "treatment": treatment, "delta_recovery": mean(differences),
                "paired_bootstrap_95": interval(differences),
                "wins": sum(d > 1e-12 for d in differences),
                "ties": sum(abs(d) <= 1e-12 for d in differences),
                "losses": sum(d < -1e-12 for d in differences),
            }
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / "summary.json", {
        "model": manifest["model"], "rounds": len(rounds), "complete": len(rounds) == protocol["budget"],
        "aggregate": report, "paired_effects": effects,
    })
    with (destination / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [k for k in metrics[0] if k != "curve"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    for domain, methods in report.items():
        print(f"\n{domain:<10} {'VR':>8} {'NR':>8} {'RR':>8} {'NEW':>8} {'TOKENS':>9}")
        for method, row in methods.items():
            print(f"{method:<10} {row['validity']:8.2%} {row['uniqueness']:8.2%} "
                  f"{row['recovery']:8.2%} {row['distinct_valid']:8.2f} "
                  f"{row['prompt_tokens'] + row['output_tokens']:9.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    evaluate(args.run, args.key, args.out, args.partial)
