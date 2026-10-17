import threading
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from ultralytics.engine.results import Results

from classroom_ai.inference.detector import ImageInspector


class InspectorControlsTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Overlapping same-class boxes should merge; another class must survive.
        boxes = torch.tensor(
            [[0, 0, 40, 40, 0.9, 0], [1, 1, 41, 41, 0.8, 0], [0, 0, 40, 40, 0.85, 1]]
        )
        result = Results(self.frame, "test", {0: "person", 1: "table"}, boxes=boxes)
        self.engine = ImageInspector.__new__(ImageInspector)
        self.engine.model = Mock()
        self.engine.model.predict.return_value = [result]
        self.engine.mapping = {0: 0, 1: 1}
        self.engine.lock = threading.Lock()
        self.engine.device = "cpu"

    def test_selected_cuda_device_is_used_without_changing_model_settings(self):
        self.engine.device = "cuda:0"
        output = self.engine.predict(self.frame, 0.25)
        args = self.engine.model.predict.call_args.kwargs
        self.assertEqual(args["device"], "cuda:0")
        self.assertEqual(args["imgsz"], 640)
        self.assertFalse(args["nms"])
        self.assertFalse(args["augment"])
        self.assertEqual(output["device"], "cuda:0")

    def test_size_and_class_nms_use_one_prediction(self):
        output = self.engine.predict(self.frame, 0.25, imgsz=1280, extra_nms_iou=0.6)
        self.assertEqual(len(output["detections"]), 2)
        self.assertEqual({r["class_id"] for r in output["detections"]}, {0, 1})
        self.engine.model.predict.assert_called_once()
        args = self.engine.model.predict.call_args.kwargs
        self.assertEqual(args["imgsz"], 1280)
        self.assertFalse(args["nms"])
        self.assertFalse(args["augment"])

    def test_default_retains_native_predictions_at_640(self):
        output = self.engine.predict(self.frame, 0.25)
        self.assertEqual(len(output["detections"]), 3)
        self.assertEqual(self.engine.model.predict.call_args.kwargs["imgsz"], 640)

    def test_invalid_settings_fail_before_prediction(self):
        for kwargs in (
            {"imgsz": 0},
            {"extra_nms_iou": 1.1},
            {"extra_nms_iou": float("nan")},
            {"passes": 2},
            {"passes": 3},
        ):
            with self.assertRaises(ValueError):
                self.engine.predict(self.frame, 0.25, **kwargs)
        self.engine.model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
