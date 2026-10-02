import unittest
from src.evaluation import map_predictions, score_predictions
from src.inference.detector import COCO_TO_PROJECT, PROJECT_NAMES, class_mapping


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gt = {"info": {}, "images": [{"id": i, "width": 100, "height": 100} for i in range(1, 10)],
                   "categories": [{"id": i, "name": name} for i, name in enumerate(PROJECT_NAMES)],
                   "annotations": [{"id": i+1, "image_id": i+1, "category_id": i,
                                    "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0} for i in range(8)]}
        self.perfect = [{"image_id": i+1, "category_id": i, "bbox": [10, 10, 20, 20], "score": .8} for i in range(8)]

    def test_perfect_boxes_all_eight_classes_and_negative_image(self):
        metrics = score_predictions(self.gt, self.perfect)
        self.assertAlmostEqual(metrics["mAP50_95"], 1)
        self.assertEqual(metrics["images"], 9)
        self.assertEqual(metrics["negative_images"], 1)
        self.assertTrue(all(v["GT_boxes"] == 1 for v in metrics["per_class"].values()))

    def test_wrong_classes_and_empty_predictions_have_zero_ap(self):
        wrong = [{**p, "category_id": (p["category_id"]+1)%8} for p in self.perfect]
        self.assertEqual(score_predictions(self.gt, wrong)["mAP50"], 0)
        self.assertEqual(score_predictions(self.gt, [])["mAP50"], 0)

    def test_false_positive_on_negative_image_reduces_ap(self):
        extra = {"image_id": 9, "category_id": 0, "bbox": [10, 10, 20, 20], "score": .99}
        self.assertLess(score_predictions(self.gt, self.perfect+[extra])["mAP50"], 1)

    def test_duplicate_does_not_add_true_positive(self):
        # Higher score wrong box, duplicate it, then the real box: precision must fall.
        false = {**self.perfect[0], "bbox": [60, 60, 20, 20], "score": .99}
        single = score_predictions(self.gt, [false]+self.perfect)
        duplicate = score_predictions(self.gt, [false, {**false, "score": .98}]+self.perfect)
        self.assertLess(duplicate["mAP50"], single["mAP50"])

    def test_mapping_coco80_and_project8_same_table(self):
        box = [[10, 10, 30, 30]]
        coco = map_predictions(1, box, [.9], [60], COCO_TO_PROJECT)
        project = map_predictions(1, box, [.9], [1], class_mapping(dict(enumerate(PROJECT_NAMES))))
        self.assertEqual(coco, project)
        self.assertEqual(coco[0]["bbox"], [10, 10, 20, 20])
        self.assertEqual(map_predictions(1, box, [.9], [1], COCO_TO_PROJECT), [])

    def test_missing_class_reports_null_and_unknown_prediction_rejected(self):
        self.gt["annotations"] = self.gt["annotations"][:1]
        metrics = score_predictions(self.gt, self.perfect[:1])
        self.assertIsNone(metrics["per_class"]["table"]["AP50"])
        with self.assertRaises(ValueError):
            score_predictions(self.gt, [{**self.perfect[0], "image_id": 99}])


if __name__ == "__main__":
    unittest.main()
