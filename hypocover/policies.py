"""Sampling policies with identical candidate generation and validation."""

from dataclasses import dataclass


METHODS = ("iid", "memory", "static", "oracle", "joint", "bandit", "tilt", "adaptive",
           "grammar", "grammar_tilt")


@dataclass
class Evidence:
    valid: int = 0
    invalid: int = 0
    new: int = 0
    repeated: int = 0
    novelty_good: float = 0.0
    novelty_bad: float = 0.0
    updated: int = 0

    def novelty_at(self, step, decay):
        factor = decay ** (step - self.updated)
        return self.novelty_good * factor, self.novelty_bad * factor

    def record(self, valid, novel, step, decay):
        self.valid += int(valid)
        self.invalid += int(not valid)
        if valid:
            self.new += int(novel)
            self.repeated += int(not novel)
            good, bad = self.novelty_at(step, decay)
            self.novelty_good = good + int(novel)
            self.novelty_bad = bad + int(not novel)
            self.updated = step


class Policy:
    """Factorize marginal discovery into validity and freshness given validity."""

    def __init__(self, task, method, rng, oracle_histogram=None, decay=0.90):
        self.task = task
        self.method = method
        self.rng = rng
        self.decay = decay
        by_feature = {}
        for cue in task.cues():
            by_feature.setdefault(cue[1], []).append(cue)
        feature_count = 1 if task.domain == "voxel" else 2
        features = rng.sample(list(by_feature), min(feature_count, len(by_feature)))
        cues = [cue for feature in features for cue in by_feature[feature]]
        rng.shuffle(cues)
        self.arms = [None, *cues]
        self.state = {cue: Evidence() for cue in self.arms}
        self.oracle_histogram = oracle_histogram

    def choose(self, step):
        if self.method in {"iid", "memory", "tilt", "grammar", "grammar_tilt"}:
            return None
        if self.method == "static":
            return self.arms[step % len(self.arms)]
        if self.method == "oracle":
            levels = sorted(self.oracle_histogram)
            target = self.rng.choices(levels, [self.oracle_histogram[k] for k in levels])[0]
            return ("complexity", int(target))
        if step == 0:
            return None
        return max(self.arms, key=lambda cue: self.score(self.state[cue], step))

    def score(self, evidence, step):
        if self.method == "joint":
            return self.rng.betavariate(1 + evidence.new, 3 + evidence.repeated + evidence.invalid)
        valid = self.rng.betavariate(2 + evidence.valid, 2 + evidence.invalid)
        fresh, repeated = evidence.novelty_at(step, self.decay)
        return valid * self.rng.betavariate(1 + fresh, 1 + repeated)

    def update(self, cue, valid, novel, step):
        if self.method in {"joint", "bandit", "adaptive"}:
            self.state[cue].record(valid, novel, step, self.decay)
