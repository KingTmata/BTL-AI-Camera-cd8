import copy
import unittest

import torch

from classroom_ai.experiments.cache_ensemble import ensemble_cached_predictions
from classroom_ai.experiments.validation_replay import replay_metrics


class CacheEnsembleTests(unittest.TestCase):
    def cache(self, predictions):
        return dict(
            manifest_sha256="same_manifest",
            protocol={"imgsz": 1280, "batch": 8},
            names={0: "chair"},
            samples=[
                dict(
                    im_name="chairs.jpg",
                    target_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]]),
                    target_cls=torch.tensor([0.0, 0.0]),
                    predictions=torch.tensor(predictions),
                )
            ],
        )

    def test_complementary_detections_improve_recall_without_changing_targets(self):
        first = self.cache([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]])
        second = self.cache([[20.0, 0.0, 30.0, 10.0, 0.8, 0.0]])
        merged = ensemble_cached_predictions([first, second])
        native = replay_metrics(first["samples"], first["names"], iou=0.6)
        ensemble = replay_metrics(merged, first["names"])
        self.assertGreater(ensemble.box.map50, native.box.map50)
        self.assertAlmostEqual(ensemble.box.map50, 0.995, places=6)
        self.assertEqual(ensemble.nt_per_class.tolist(), [2])
        self.assertEqual(len(first["samples"][0]["predictions"]), 1)

    def test_duplicate_model_keeps_nms_metric_identical(self):
        cache = self.cache([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0], [20.0, 0.0, 30.0, 10.0, 0.8, 0.0]])
        direct = replay_metrics(cache["samples"], cache["names"], iou=0.6)
        merged = replay_metrics(
            ensemble_cached_predictions([cache, copy.deepcopy(cache)]), cache["names"]
        )
        self.assertAlmostEqual(direct.box.map50, merged.box.map50, places=12)
        self.assertAlmostEqual(direct.box.map, merged.box.map, places=12)

    def test_rejects_different_identity_mapping_or_canvas_targets(self):
        first = self.cache([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]])
        for field, value in [
            ("manifest_sha256", "different"),
            ("protocol", {"imgsz": 640}),
            ("names", {0: "person"}),
        ]:
            second = copy.deepcopy(first)
            second[field] = value
            with self.assertRaises(ValueError):
                ensemble_cached_predictions([first, second])
        for field, value in [
            ("target_boxes", torch.tensor([[0.0, 0.0, 5.0, 5.0], [20.0, 0.0, 30.0, 10.0]])),
            ("target_cls", torch.tensor([0.0, 1.0])),
        ]:
            second = copy.deepcopy(first)
            second["samples"][0][field] = value
            with self.assertRaises(ValueError):
                ensemble_cached_predictions([first, second])

    def test_rejects_missing_duplicate_or_reordered_images(self):
        first = self.cache([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]])
        second = copy.deepcopy(first)
        second["samples"] = []
        with self.assertRaises(ValueError):
            ensemble_cached_predictions([first, second])
        first["samples"].append(copy.deepcopy(first["samples"][0]))
        with self.assertRaises(ValueError):
            ensemble_cached_predictions([first, copy.deepcopy(first)])
        first["samples"][1]["im_name"] = "other.jpg"
        second = copy.deepcopy(first)
        second["samples"].reverse()
        with self.assertRaises(ValueError):
            ensemble_cached_predictions([first, second])


if __name__ == "__main__":
    unittest.main()
