import unittest
from src.classes import COCO_NAMES, PROJECT_NAMES
from src.coco_subset import category_mapping, coverage_review, normalized_box, ordered_candidates


class CocoSubsetTests(unittest.TestCase):
    def setUp(self):
        self.categories = [{"id": 100+i, "name": name} for i, name in enumerate(COCO_NAMES.values())]
        self.mapping = category_mapping(self.categories)
        self.images = [{"id": i, "width": 100, "height": 100} for i in range(1, 10)]
        self.annotations = [{"id": i, "image_id": i, "category_id": 100, "bbox": [10, 10, 20, 20], "iscrowd": 0} for i in range(1, 10)]

    def test_mapping_uses_category_names_not_raw_coco80_ids(self):
        self.assertEqual(self.mapping[101], 1)
        self.assertNotIn(60, self.mapping)
        self.assertEqual(normalized_box({**self.annotations[0], "category_id": 101}, self.images[0], self.mapping), (1, .2, .2, .2, .2))

    def test_deterministic_order_crowd_invalid_duplicate_and_reference_excluded(self):
        self.annotations[0]["iscrowd"] = 1
        self.annotations[1]["bbox"] = [99, 99, 20, 20]
        self.annotations.append(dict(self.annotations[2]))
        first, stats = ordered_candidates(self.images, self.annotations, self.mapping, [4])
        second, _ = ordered_candidates(list(reversed(self.images)), list(reversed(self.annotations)), self.mapping, [4])
        self.assertEqual(first, second)
        self.assertEqual(set(first), {5, 6, 7, 8, 9})
        self.assertEqual(stats["excluded_bad_images"], 3)

    def test_other_class_crowd_does_not_remove_target_image(self):
        self.annotations.append({"image_id": 1, "category_id": 999, "iscrowd": 1, "bbox": [0, 0, 1, 1]})
        ids, _ = ordered_candidates(self.images, self.annotations, self.mapping)
        self.assertIn(1, ids)

    def test_review_exact_quotas_class_coverage_and_missing_class_rejected(self):
        rows = []
        for split, count in (("train", 100), ("val", 30)):
            for i in range(count):
                row = {"image_id": f"{split}_{i}", "split": split}
                row.update({name.replace(' ', '_')+'_count': int(i % 8 == j) for j, name in enumerate(PROJECT_NAMES)})
                rows.append(row)
        chosen = coverage_review(rows)
        self.assertEqual(sum(r["split"] == "train" for r in chosen), 80)
        self.assertEqual(sum(r["split"] == "val" for r in chosen), 20)
        for split in ("train", "val"):
            for name in PROJECT_NAMES:
                self.assertTrue(any(r["split"] == split and r[name.replace(' ', '_')+'_count'] for r in chosen))
        for r in rows:
            r["cup_count"] = 0
        with self.assertRaises(ValueError):
            coverage_review(rows)
