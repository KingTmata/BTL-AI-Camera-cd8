import io
import unittest

import numpy as np
from PIL import Image

from src.inference.detector import COCO_TO_PROJECT, PROJECT_NAMES, class_mapping, crop_box, decode_image, reference_boxes
from src.classes import COCO_NAMES


class InspectionTests(unittest.TestCase):
    def test_mapping_keeps_coco_and_project_ids_distinct(self):
        coco = {i: f"unused_{i}" for i in range(80)}
        coco.update(COCO_NAMES)
        self.assertEqual(class_mapping(coco)[60], 1)
        self.assertNotIn(1, class_mapping(coco))
        self.assertEqual(class_mapping(dict(enumerate(PROJECT_NAMES)))[1], 1)
        with self.assertRaises(ValueError):
            class_mapping({0: "person", 1: "cell phone", 2: "bottle", 3: "laptop"})

    def test_reference_mapping_and_original_pixel_coordinates(self):
        rows = reference_boxes("60 0.5 0.5 0.2 0.4\n1 0.5 0.5 0.1 0.1", 100, 200)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["class_id"], 1)
        self.assertEqual(rows[0]["bbox_xyxy"], [40, 60, 60, 140])
        self.assertEqual(reference_boxes("", 100, 200), [])

    def test_invalid_labels_are_not_silent_negatives(self):
        for text in ("39 0.5 0.5 -0.2 0.4", "39 nan 0.5 0.2 0.4", "39 0 0 0.5 0.5", "39 0.5 0.5"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                reference_boxes(text, 100, 100)

    def test_crop_clips_to_image_and_rejects_empty_box(self):
        frame = np.zeros((20, 30, 3), np.uint8)
        self.assertEqual(crop_box(frame, [-3, 5, 40, 25]).shape, (15, 30, 3))
        with self.assertRaises(ValueError):
            crop_box(frame, [25, 10, 10, 5])

    def test_uploaded_red_pixel_becomes_opencv_bgr(self):
        payload = io.BytesIO()
        Image.new("RGB", (4, 4), (255, 0, 0)).save(payload, format="PNG")
        self.assertEqual(decode_image(payload.getvalue())[0, 0].tolist(), [0, 0, 255])


if __name__ == "__main__":
    unittest.main()
