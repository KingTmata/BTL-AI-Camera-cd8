import time
import unittest

import numpy as np

from classroom_ai.inference.camera import CameraSession


class FakeCapture:
    def __init__(self):
        self.released = False

    def read(self):
        time.sleep(0.002)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def getBackendName(self):
        return "FAKE"

    def release(self):
        self.released = True


class CameraDiagnosticsTests(unittest.TestCase):
    def test_snapshot_reports_bounded_stage_timings_and_recent_fps(self):
        capture = FakeCapture()

        def predict(frame, confidence):
            return {
                "detections": [],
                "processing_ms": 1.0,
                "manager_lock_wait_ms": 0.5,
                "manager_active_check_ms": 0.25,
                "inspector_lock_wait_ms": 0.1,
            }

        session = CameraSession(987654, predict, 0.25, capture_factory=lambda index: capture)
        percentile_lock_states = []
        session._percentile = lambda samples, percentile: (
            percentile_lock_states.append(session._lock.locked())
            or CameraSession._percentile(samples, percentile)
        )
        self.assertTrue(session.start())
        deadline = time.monotonic() + 2
        snapshot = session.snapshot()
        while snapshot["frames_processed"] < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = session.snapshot()
        stopped = session.stop()
        self.assertTrue(stopped)
        self.assertTrue(capture.released)
        self.assertGreaterEqual(snapshot["frames_processed"], 4)
        self.assertEqual(snapshot["frame_size"], {"width_px": 64, "height_px": 48})
        self.assertTrue(percentile_lock_states)
        self.assertFalse(any(percentile_lock_states))
        for name in (
            "capture_read_p50_ms",
            "capture_read_p95_ms",
            "callback_p50_ms",
            "callback_p95_ms",
            "processing_p50_ms",
            "processing_p95_ms",
            "manager_lock_wait_p95_ms",
            "manager_active_check_p95_ms",
            "inspector_lock_wait_p95_ms",
            "recent_processing_fps",
            "latest_result_age_ms",
        ):
            self.assertIsNotNone(snapshot[name], name)


if __name__ == "__main__":
    unittest.main()
