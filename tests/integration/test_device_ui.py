import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from classroom_ai.inference.camera import CameraSession


class PreviewTests(unittest.TestCase):
    def test_fast_preview_reuses_jpeg_and_invalidates_on_new_frame(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("""
import cv2
import numpy as np
import streamlit as st
from unittest.mock import Mock, patch
from classroom_ai.interfaces.web.app import webcam_image
from classroom_ai.interfaces.web.preview import PreviewSettings
snapshot = {'status': 'running', 'starts': 1, 'frames_processed': 1,
            'frame': np.zeros((20, 20, 3), dtype=np.uint8), 'output': {'detections': []}}
camera = Mock()
camera.snapshot.return_value = snapshot
st.session_state.camera_session = camera
encoded = cv2.imencode('.jpg', snapshot['frame'])[1].tobytes()
with patch('classroom_ai.interfaces.web.app.preview_jpeg', return_value=encoded) as encode, patch('classroom_ai.interfaces.web.app.st.image') as image:
    webcam_image([], PreviewSettings())
    webcam_image([], PreviewSettings())
    assert encode.call_count == 1
    snapshot['frames_processed'] = 2
    webcam_image([], PreviewSettings())
    assert encode.call_count == 2
    assert image.call_count == 3
    assert all(call.args[0].startswith('data:image/jpeg;base64,') for call in image.call_args_list)
camera.predict.assert_not_called()
""").run()
        self.assertEqual(len(app.exception), 0)

    def test_preview_does_not_wait_for_training_lock_held_by_prediction(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("""
from pathlib import Path
from unittest.mock import Mock
from classroom_ai.interfaces.web.app import webcam_panel
manager = Mock()
manager.active.side_effect = AssertionError("Preview waited for the inference/training guard")
webcam_panel(Path("configs/app.yaml"), .25, [], manager)
manager.active.assert_not_called()
""").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("Bấm Bật webcam", app.info[0].value)

    def test_fragment_uses_configured_rate(self):
        from classroom_ai.interfaces.web.app import render_webcam
        from classroom_ai.interfaces.web.preview import PreviewSettings

        for fps in (5, 10, 20):
            with (
                self.subTest(fps=fps),
                patch("classroom_ai.interfaces.web.app.st.fragment") as fragment,
            ):
                settings = PreviewSettings(fps=fps)
                render_webcam(settings, "checkpoint", 0.25, [], None)
                self.assertEqual(
                    [call.kwargs["run_every"] for call in fragment.call_args_list], [1, 1 / fps]
                )
                self.assertEqual(fragment.return_value.return_value.call_args.args[-1], settings)

    def test_loads_only_preview_settings_and_rejects_invalid_values(self):
        from classroom_ai.interfaces.web.preview import load_preview_settings

        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "app.yaml"
            config.write_text(
                "model: {imgsz: 1280}\nruntime: {preview_fps: 20, preview_max_width: 320, preview_jpeg_quality: 60}",
                encoding="utf-8",
            )
            settings = load_preview_settings(config)
            self.assertEqual(
                (settings.fps, settings.max_width, settings.jpeg_quality), (20, 320, 60)
            )
            for runtime in (
                "{preview_fps: 0}",
                "{preview_fps: .nan}",
                "{preview_fps: true}",
                "{preview_max_width: -1}",
                "{preview_jpeg_quality: 101}",
            ):
                config.write_text("runtime: " + runtime, encoding="utf-8")
                with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                    load_preview_settings(config)

    def test_jpeg_resizes_without_mutating_camera_frame_or_boxes(self):
        from classroom_ai.interfaces.web.preview import PreviewSettings, preview_jpeg

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rows = [{"class_name": "person", "confidence": 0.9, "bbox_xyxy": [100, 100, 500, 500]}]
        original = list(rows[0]["bbox_xyxy"])
        encoded = preview_jpeg(frame, rows, PreviewSettings())
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (360, 640, 3))
        self.assertGreater(decoded.sum(), 0)
        self.assertEqual(frame.sum(), 0)
        self.assertEqual(rows[0]["bbox_xyxy"], original)
        small = np.zeros((120, 160, 3), dtype=np.uint8)
        decoded = cv2.imdecode(
            np.frombuffer(preview_jpeg(small, [], PreviewSettings()), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape, small.shape)

    def test_snapshot_percentile_does_not_hold_worker_lock(self):
        camera = CameraSession(0, None, 0.25)
        camera._latencies.append(10)
        acquired = threading.Event()

        def percentile(values, q):
            def publish():
                with camera._lock:
                    camera._frames += 1
                    acquired.set()

            worker = threading.Thread(target=publish, daemon=True)
            worker.start()
            self.assertTrue(acquired.wait(1), "UI percentile blocks the camera worker")
            worker.join(1)
            return 10

        with patch("classroom_ai.inference.camera.np.percentile", side_effect=percentile):
            snapshot = camera.snapshot()
        self.assertEqual(snapshot["frames_processed"], 0)
        self.assertEqual(snapshot["app_latency_p95_ms"], 10)


if __name__ == "__main__":
    unittest.main()
