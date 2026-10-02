import time
import unittest
from unittest.mock import patch

import numpy as np

from src.inference.camera import CameraSession
from src.inference.capture import open_camera


class FakeCapture:
    def __init__(self, opened=True, readable=True):
        self.opened = opened
        self.readable = readable
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(0.005)
        return (True, np.zeros((24, 32, 3), np.uint8)) if self.readable else (False, None)

    def set(self, *args):
        return True

    def getBackendName(self):
        return "TEST"

    def release(self):
        self.released = True


def prediction(frame, confidence):
    return {"detections": [], "processing_ms": 1.0}


def wait_for(session, predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = session.snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"Camera did not reach expected state: {session.snapshot()['status']}")


class CaptureTests(unittest.TestCase):
    def test_windows_falls_back_when_opened_camera_has_no_frames(self):
        unreadable, good = FakeCapture(readable=False), FakeCapture()
        with patch("src.inference.capture.sys.platform", "win32"), patch(
            "src.inference.capture.cv2.VideoCapture", side_effect=[unreadable, good]
        ):
            self.assertIs(open_camera(0), good)
        self.assertTrue(unreadable.released)
        self.assertFalse(good.released)

    def test_non_windows_uses_native_backend_and_failed_devices_release(self):
        bad = FakeCapture(opened=False)
        with patch("src.inference.capture.sys.platform", "linux"), patch(
            "src.inference.capture.cv2.VideoCapture", return_value=bad
        ) as factory, self.assertRaisesRegex(RuntimeError, "webcam"):
            open_camera(1)
        factory.assert_called_once_with(1, 0)
        self.assertTrue(bad.released)


class CameraTests(unittest.TestCase):
    def session(self, capture, **kwargs):
        session = CameraSession(0, prediction, 0.25, capture_factory=lambda index: capture, **kwargs)
        self.addCleanup(session.stop)
        return session

    def test_start_stop_reopen_and_exclusive_camera(self):
        captures = [FakeCapture(), FakeCapture()]
        session = CameraSession(0, prediction, 0.25, capture_factory=lambda index: captures.pop(0))
        self.addCleanup(session.stop)
        self.assertTrue(session.start())
        self.assertFalse(session.start())
        frame = wait_for(session, lambda s: s["frames_processed"] > 0)
        self.assertEqual(frame["frame"].shape, (24, 32, 3))
        competitor = self.session(FakeCapture())
        self.assertFalse(competitor.start())
        self.assertIn("phiên", competitor.snapshot()["error"])
        self.assertTrue(session.stop())
        self.assertIsNone(session.snapshot()["frame"])
        self.assertTrue(session.start())
        wait_for(session, lambda s: s["frames_processed"] > 0)
        self.assertEqual(session.snapshot()["starts"], 2)

    def test_unplug_and_prediction_error_release_camera_and_clear_frame(self):
        capture = FakeCapture()
        session = self.session(capture)
        session.start()
        wait_for(session, lambda s: s["frames_processed"] > 0)
        capture.readable = False
        snapshot = wait_for(session, lambda s: s["status"] == "error")
        self.assertIsNone(snapshot["frame"])
        self.assertTrue(capture.released)
        def fail(frame, confidence):
            raise ValueError("model failed")
        another_capture = FakeCapture()
        broken = CameraSession(0, fail, 0.25, capture_factory=lambda index: another_capture)
        self.addCleanup(broken.stop)
        self.assertTrue(broken.start())
        wait_for(broken, lambda s: s["status"] == "error")
        self.assertTrue(another_capture.released)

    def test_disconnected_browser_expires_without_heartbeat(self):
        capture = FakeCapture()
        session = self.session(capture, heartbeat_timeout=0.08)
        session.start()
        time.sleep(0.15)
        snapshot = session.snapshot()
        self.assertEqual(snapshot["status"], "stopped")
        self.assertTrue(capture.released)
        self.assertIsNone(snapshot["frame"])


if __name__ == "__main__":
    unittest.main()
