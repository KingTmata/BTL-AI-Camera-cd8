import unittest

from classroom_ai.experiments.checkpoint07_matrix import build_matrix


class MatrixTests(unittest.TestCase):
    def test_every_trial_resets_same_initializer_and_has_budget(self):
        jobs = build_matrix("/initial.pt", "/data.yaml", "/output", list(range(11)))
        self.assertEqual(len(jobs), 6)
        for job in jobs:
            self.assertEqual(job["model"], "/initial.pt")
            self.assertFalse(job["resume"])
            self.assertEqual((job["epochs"], job["patience"]), (10, 3))
            self.assertEqual(job["seed"], 42)
            self.assertEqual(job["device"], "0,1")
            self.assertEqual(job["batch"], 32)

    def test_controlled_differences(self):
        jobs = build_matrix("/initial.pt", "/data.yaml", "/output", [0, 1, 2])

        def delta(left, right):
            return {key for key in left if left[key] != right[key]} - {"name"}

        self.assertEqual(delta(jobs[0], jobs[1]), {"lr0"})
        self.assertEqual(delta(jobs[0], jobs[2]), {"optimizer"})
        self.assertEqual(delta(jobs[0], jobs[4]), {"scale"})
        self.assertEqual(delta(jobs[0], jobs[5]), {"freeze"})
        self.assertEqual(jobs[5]["freeze"], [0, 1, 2])

    def test_reject_unverified_empty_backbone(self):
        with self.assertRaises(ValueError):
            build_matrix("/initial.pt", "/data.yaml", "/output", [])
