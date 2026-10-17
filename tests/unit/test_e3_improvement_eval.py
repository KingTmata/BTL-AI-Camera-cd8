import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from classroom_ai.experiments.e3_improvement_eval import (
    merge_boxes,
    summarize_cache,
    tile_starts,
)


class ImprovementEvaluationTests(unittest.TestCase):
    def test_all_serialized_augmentation_groups_preserve_image_coordinates(self):
        import albumentations as A
        from ultralytics.data.augment import Albumentations

        groups = [
            [],
            [
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.3),
                A.RandomGamma(gamma_limit=(85, 115), p=0.15),
                A.CLAHE(clip_limit=(1, 2), p=0.05),
            ],
            [],
            [A.ToGray(p=0.15)],
            [A.InvertImg(p=0.05)],
        ]
        for transforms in groups:
            probe = Albumentations(transforms=[A.to_dict(t) for t in transforms])
            self.assertIsNotNone(probe.transform)
            self.assertFalse(probe.contains_spatial)
            self.assertEqual(len(probe.transform.transforms), len(transforms))

    def test_tiles_cover_edges_and_small_images(self):
        self.assertEqual(tile_starts(100, 512), [0])
        starts = tile_starts(1300, 512)
        self.assertEqual(starts[-1] + 512, 1300)
        self.assertTrue(all(b - a <= 512 for a, b in zip(starts, starts[1:])))

    def test_merge_keeps_overlapping_different_classes(self):
        rows = np.array(
            [[0, 0, 100, 100, 0.9, 0], [1, 1, 101, 101, 0.8, 0], [1, 1, 101, 101, 0.7, 1]],
            dtype=np.float32,
        )
        merged = merge_boxes(rows)
        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged[:, 5]), {0, 1})

    def test_confidence_rematches_and_counts_false_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.npz"
            np.savez(
                path,
                pred=np.array(
                    [[0, 0, 100, 100, 0.9, 0], [200, 200, 300, 300, 0.05, 0]], dtype=np.float32
                ),
                classes=np.array([0]),
                boxes=np.array([[0, 0, 100, 100]], dtype=np.float32),
                seconds=0.1,
            )
            result = summarize_cache([path], ["person"], thresholds=(0.01, 0.1, 0.95))
            json.dumps(result, allow_nan=False)
            low, middle, high = result["confidence_sweep"]
            self.assertEqual(low["precision_micro"], 0.5)
            self.assertEqual(middle["f1_micro"], 1.0)
            self.assertEqual(high["per_class"][0]["FN"], 1)
            self.assertEqual(high["recall_micro"], 0.0)


if __name__ == "__main__":
    unittest.main()
