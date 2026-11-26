"""Observation-only prompts, canonicalization and single-candidate validators."""

import json
import re
from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class Task:
    id: str
    domain: str
    observation: dict

    @property
    def nodes(self):
        return tuple(self.observation["nodes"])

    @property
    def positions(self):
        return tuple(
            f"({i // self.observation['grid'] + 1},{i % self.observation['grid'] + 1})"
            for i, active in enumerate(self.observation["top"]) if active
        )

    def prompt(self, cue, history):
        if self.domain == "voxel":
            positions = ", ".join(
                f"{chr(65 + i)}={position}" for i, position in enumerate(self.positions)
            )
            count = len(self.positions)
            question = (
                f"A top view of a voxel scene shows occupied columns {positions}. "
                f"Every column is solid from the floor and has height 1 to "
                f"{self.observation['height']}. Other columns are empty. "
                f"Assign a height to each occupied column in the listed order. "
                f"Output exactly {count} digits with no spaces, one height for each column."
            )
        else:
            observations = ". ".join(
                f"{source} reaches exactly {', '.join(reached) or 'no other node'}"
                for source, reached in self.observation["reaches"]
            )
            question = (
                f"Find one directed acyclic graph on nodes {', '.join(self.nodes)}. "
                "Reach means a directed path of one or more edges. "
                f"Observations: {observations}. "
                "Return one line containing the direct edges as pairs of letters "
                "separated by spaces. AB means A->B. Return NONE for an empty graph."
            )
        if history:
            question += "\nAlready proposed (choose a different answer): " + " | ".join(history)
        if cue:
            question += "\nAdditional requirement: " + self.cue_text(cue)
        return question + "\nAnswer with the requested line only."

    def parse(self, text):
        text = text.strip()
        text = re.sub(r"^\s*(?:answer|heights?|edges?)\s*:\s*", "", text, flags=re.I)
        text = re.sub(r"^\x60\x60\x60(?:\w+)?\s*|\s*\x60\x60\x60$", "", text)
        text = text.strip().split("\n", 1)[0].strip()
        if self.domain == "voxel":
            compact = text.strip("[]. ")
            if re.fullmatch(r"[0-9]+", compact) and len(compact) == len(self.positions):
                return tuple(map(int, compact))
            match = re.fullmatch(r"\[?(\d+(?:[ ,]+\d+)*)\]?\.?", text)
            if not match:
                return None
            value = tuple(map(int, re.findall(r"\d+", match[1])))
            return value if len(value) == len(self.positions) else None
        if text.upper().rstrip(".") in {"NONE", "[]", "NO EDGES"}:
            return ()
        values = re.split(r"[,;\s]+", text.strip("[]. ").upper())
        if not all(re.fullmatch(r"[A-Z](?:->|→)?[A-Z]", edge) for edge in values):
            return None
        return tuple(sorted({edge[0] + edge[-1] for edge in values}))

    def valid(self, candidate):
        if candidate is None:
            return False
        if self.domain == "voxel":
            return len(candidate) == len(self.positions) and all(
                1 <= x <= self.observation["height"] for x in candidate
            )
        if any(edge[0] not in self.nodes or edge[1] not in self.nodes for edge in candidate):
            return False
        closure = reachability(self.nodes, candidate)
        return all(node not in closure[node] for node in self.nodes) and all(
            closure[node] == set(reached) for node, reached in self.observation["reaches"]
        )

    def render(self, candidate):
        if self.domain == "voxel" and all(0 <= value <= 9 for value in candidate):
            return "".join(map(str, candidate))
        return " ".join(map(str, candidate)) or "NONE"

    def complexity(self, candidate):
        return sum(candidate) if self.domain == "voxel" else len(candidate)

    def cues(self):
        if self.domain == "voxel":
            return tuple(
                ("height", i, height)
                for i in range(len(self.positions))
                for height in range(1, self.observation["height"] + 1)
            )
        # An observed non-descendant cannot be a direct child.
        # All other probes come directly from the node labels and observations.
        observed = {source: set(reached) for source, reached in self.observation["reaches"]}
        return tuple(
            ("edge", a + b, present)
            for a, b in permutations(self.nodes, 2)
            if a not in observed or b in observed[a]
            for present in (True, False)
        )

    def cue_text(self, cue):
        kind, *value = cue
        if kind == "height":
            label = chr(65 + value[0])
            return f"Column {label} must have height {value[1]}."
        if kind == "edge":
            return f"The direct edge {value[0]} must be {'present' if value[1] else 'absent'}."
        if self.domain == "voxel":
            return f"The column heights must sum to {value[0]}."
        return f"The graph must have exactly {value[0]} direct edges."

    def matches(self, cue, candidate):
        if candidate is None:
            return False
        kind, *value = cue
        if kind == "height":
            return candidate[value[0]] == value[1]
        if kind == "edge":
            return (value[0] in candidate) == value[1]
        return self.complexity(candidate) == value[0]


def reachability(nodes, edges):
    closed = {node: set() for node in nodes}
    for edge in edges:
        closed[edge[0]].add(edge[1])
    for mid in nodes:
        for source in nodes:
            if mid in closed[source]:
                closed[source].update(closed[mid])
    return closed


def load_tasks(path):
    with open(path, encoding="utf-8") as stream:
        return [Task(**row) for row in json.load(stream)]
