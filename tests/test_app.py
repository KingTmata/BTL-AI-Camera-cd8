import unittest
import time
from unittest.mock import Mock, patch

import numpy as np

from streamlit.testing.v1 import AppTest

from src.inference.detector import ROOT
from src.inference.camera import CameraSession
from src.training.options import ADVANCED_DEFAULTS


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

    def test_training_form_defaults_and_new_run_submission(self):
        with patch("src.training.manager.RunManager.start", return_value=ROOT / "runs/train/ui_test") as start:
            app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
            camera = Mock()
            camera.stop.return_value = True
            app.session_state.camera_session = camera
            app.selectbox(key="workspace_page").set_value("Training").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            camera.stop.assert_called_once()
            self.assertEqual(app.number_input(key="train_epochs").value, 1)
            self.assertEqual(app.number_input(key="train_batch").value, 2)
            self.assertEqual(app.number_input(key="train_workers").value, 0)
            self.assertEqual(app.selectbox(key="train_imgsz").value, 640)
            self.assertFalse(app.checkbox(key="train_cache").value)
            for key, value in ADVANCED_DEFAULTS.items():
                self.assertEqual(app.number_input(key=f"train_{key}").value, value)
            self.assertTrue(any("tắt từ epoch đầu" in w.value for w in app.warning))
            app.text_input(key="train_name").set_value("reviewed_cpu")
            app.button(key="train_start").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            start.assert_called_once_with(epochs=1, batch=2, imgsz=640, workers=0, cache=False,
                                          run_name="reviewed_cpu", checkpoint=ROOT / "weights/yolo26n.pt",
                                          advanced=ADVANCED_DEFAULTS)

    def test_advanced_preview_does_not_train_and_submission_preserves_values(self):
        with patch("src.training.manager.RunManager.start", return_value=ROOT / "runs/train/ui_test") as start:
            app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
            app.selectbox(key="workspace_page").set_value("Training").run()
            app.number_input(key="train_cls_pw").set_value(.25)
            app.number_input(key="train_mixup").set_value(.1)
            app.number_input(key="train_close_mosaic").set_value(0)
            app.button(key="train_preview").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            start.assert_not_called()
            self.assertFalse(any("tắt từ epoch đầu" in w.value for w in app.warning))
            app.button(key="train_start").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(start.call_args.kwargs["advanced"],
                             {**ADVANCED_DEFAULTS, "cls_pw": .25, "mixup": .1, "close_mosaic": 0})

    def test_active_training_blocks_recognition_and_new_start(self):
        with patch("src.training.manager.RunManager.active", return_value={"run_id": "active_run"}), \
             patch("src.training.manager.RunManager.history", return_value=[]):
            app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.radio), 0, "Active training must hide inference controls")
            self.assertTrue(any("tạm dừng" in i.value for i in app.info))
            app.selectbox(key="workspace_page").set_value("Training").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.button(key="train_start").disabled)


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
