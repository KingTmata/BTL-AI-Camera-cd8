import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from classroom_ai.inference.detector import ImageInspector


class Values:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class InspectorTimingTests(unittest.TestCase):
    def setUp(self):
        self.inspector = ImageInspector.__new__(ImageInspector)
        self.inspector.model = Mock()
        boxes = SimpleNamespace(
            xyxy=Values([[1.0, 2.0, 3.0, 4.0]]), conf=Values([0.9]), cls=Values([0])
        )
        self.inspector.model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        self.inspector.mapping = {0: 0}
        self.inspector.lock = threading.Lock()
        self.inspector.device = "cpu"

    def test_prediction_reports_wait_before_processing_timer(self):
        output = self.inspector.predict(np.zeros((16, 16, 3), dtype=np.uint8), 0.25)
        self.assertGreaterEqual(output["inspector_lock_wait_ms"], 0)
        self.assertGreaterEqual(output["processing_ms"], 0)
        self.assertEqual(len(output["detections"]), 1)


if __name__ == "__main__":
    unittest.main()
