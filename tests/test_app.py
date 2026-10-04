import unittest
import time
from unittest.mock import patch

import numpy as np

from streamlit.testing.v1 import AppTest

from src.inference.detector import ROOT
from src.inference.camera import CameraSession


@unittest.skipUnless((ROOT / "weights/yolo26n.pt").is_file() and
                     (ROOT / "data/manifest.csv").is_file(),
                     "Cần weights và dataset chính để chạy integration test")
class AppTests(unittest.TestCase):
    def test_real_inference_filters_and_stale_results(self):
        app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.button(key="analyze").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric), 9, "Tám lớp và thời gian xử lý phải được hiển thị")
        app.multiselect[0].set_value([]).run()
        self.assertFalse(app.error)
        self.assertTrue(all(int(m.value) == 0 for m in app.metric if m.label != "Xử lý ảnh"))
        app.slider[0].set_value(0.5).run()
        self.assertEqual(len(app.metric), 0, "Đổi confidence phải ẩn kết quả cũ")
        app.radio[0].set_value("Ảnh của bạn").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric), 0)


@unittest.skipUnless((ROOT / "weights/yolo26n.pt").is_file(), "Cần weights để kiểm webcam với YOLO thật")
class WebcamAppTests(unittest.TestCase):
    def test_webcam_buttons_reopen_and_switch_source_release_device(self):
        class Capture:
            released = False
            def read(self):
                time.sleep(0.01)
                return True, np.zeros((24, 32, 3), np.uint8)
            def getBackendName(self):
                return "TEST"
            def release(self):
                self.released = True
        captures = []
        def factory(index):
            capture = Capture()
            captures.append(capture)
            return capture
        def session(*args, **kwargs):
            return CameraSession(*args, capture_factory=factory, **kwargs)
        with patch("src.ui.app.CameraSession", side_effect=session):
            app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
            app.radio[0].set_value("Webcam trực tiếp").run()
            self.assertFalse(app.exception)
            self.assertEqual(captures, [], "Camera chỉ được mở sau khi bấm Bật")
            app.button(key="webcam_start").click().run()
            camera = app.session_state.camera_session
            self.addCleanup(camera.stop)
            deadline = time.monotonic() + 10
            while camera.snapshot()["frames_processed"] == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertGreater(camera.snapshot()["frames_processed"], 0)
            app.button(key="webcam_reopen").click().run()
            self.assertTrue(captures[0].released)
            self.assertEqual(camera.snapshot()["starts"], 2)
            app.button(key="webcam_stop").click().run()
            self.assertEqual(camera.snapshot()["status"], "stopped")
            self.assertIsNone(camera.snapshot()["frame"])
            app.button(key="webcam_start").click().run()
            app.radio[0].set_value("Ảnh của bạn").run()
            self.assertFalse(app.exception)
            self.assertEqual(camera.snapshot()["status"], "stopped")
            self.assertTrue(all(capture.released for capture in captures))


if __name__ == "__main__":
    unittest.main()
