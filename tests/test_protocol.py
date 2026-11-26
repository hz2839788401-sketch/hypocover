import random
import unittest
from itertools import product

from hypocover.data import causal_cases, voxel_cases
from hypocover.evaluate import episode_summary
from hypocover.policies import Evidence, Policy
from hypocover.run import Episode


class TaskTests(unittest.TestCase):
    def test_voxel_encoding_is_a_bijection(self):
        _, task, answers = next(voxel_cases())
        self.assertEqual(len(answers), 3 ** len(task.positions))
        for answer in product(range(1, 4), repeat=len(task.positions)):
            self.assertEqual(answer, task.parse(task.render(answer)))
            self.assertTrue(task.valid(answer))
        self.assertEqual(task.parse("1 3"), (1, 3))
        self.assertFalse(task.valid(task.parse("40")))
        self.assertIsNone(task.parse("123"))

    def test_causal_key_and_canonicalization(self):
        _, task, key = next(causal_cases())
        for candidate in key:
            self.assertEqual(task.parse(task.render(candidate)), candidate)
            self.assertTrue(task.valid(candidate))
        self.assertEqual(task.parse("B->C, A->B, AB"), ("AB", "BC"))
        self.assertFalse(task.valid(("AB", "BA")))

    def test_metrics_penalize_duplicate_and_invalid_attempts(self):
        _, task, answers = next(voxel_cases())
        ep = Episode(task, 0, "memory", Policy(task, "memory", random.Random(1)))
        rows = []
        for step, response in enumerate(("12", "1 2", "44", "unparseable", "23")):
            rows.append({"task": task.id, "cue": None, "input_tokens": 5, "output_tokens": 1,
                         **ep.observe(None, response, step)})
        stats = episode_summary(rows, {"count": 9, "accepted": list(map(task.render, answers))})
        self.assertEqual(stats["validity"], 3 / 5)
        self.assertEqual(stats["uniqueness"], 3 / 5)
        self.assertEqual(stats["recovery"], 2 / 9)
        self.assertEqual(ep.decoder_feedback(2.5), None)


class PolicyTests(unittest.TestCase):
    def test_freshness_decays_while_validity_accumulates(self):
        evidence = Evidence()
        evidence.record(True, True, 0, 0.9)
        evidence.record(True, False, 2, 0.9)
        evidence.record(False, False, 3, 0.9)
        self.assertEqual((evidence.valid, evidence.invalid), (2, 1))
        self.assertAlmostEqual(evidence.novelty_at(4, 0.9)[0], 0.9 ** 4)
        self.assertAlmostEqual(evidence.novelty_at(4, 0.9)[1], 0.9 ** 2)

    def test_comparison_policies_share_identical_arms(self):
        _, task, _ = next(voxel_cases())
        static = Policy(task, "static", random.Random(1))
        adaptive = Policy(task, "adaptive", random.Random(1))
        self.assertEqual(static.arms, adaptive.arms)
        self.assertEqual(static.choose(0), None)
        self.assertEqual(adaptive.choose(0), None)
        self.assertEqual(set(static.choose(i) for i in range(4)), set(adaptive.arms))


if __name__ == "__main__":
    unittest.main()
