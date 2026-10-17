import threading
import unittest
from fractions import Fraction
from unittest.mock import patch

import cv2
import numpy as np

from classroom_ai.interfaces.native_video import NativeDisplay
from classroom_ai.video.core import Frame, FrameBuffer, LatestMailbox, PlaybackClock, TimeSampler
from classroom_ai.video.media import validate_timestamps
from classroom_ai.video.pipeline import Results, VideoConfig


class VideoCoreTests(unittest.TestCase):
    def test_video_defaults_use_requested_e3_and_minimum_ui_confidence(self):
        config = VideoConfig(source="sample.mp4", mode="full")
        self.assertEqual(
            config.weights.name,
            "E3_YOLO26n_Project8_v05_from_CP07_v04_best19_AdamW_lr0p0001_bestEpoch10_20261012.pt",
        )
        self.assertEqual(config.confidence, 0.05)

    def test_window_manager_close_is_quit_and_cleanup_is_idempotent(self):
        display = NativeDisplay.__new__(NativeDisplay)
        display.title, display.opened = "closed-test-window", True
        with (
            patch("classroom_ai.interfaces.native_video.cv2.pollKey", return_value=-1),
            patch(
                "classroom_ai.interfaces.native_video.cv2.getWindowProperty",
                side_effect=cv2.error("window was destroyed"),
            ),
        ):
            self.assertEqual(display.poll(), "quit")
        with patch(
            "classroom_ai.interfaces.native_video.cv2.destroyWindow",
            side_effect=cv2.error("window was destroyed"),
        ):
            display.close()
            display.close()

    def test_sampling_uses_time_not_assumed_30fps(self):
        for fps in (Fraction(25), Fraction(30000, 1001), Fraction(30)):
            sampler = TimeSampler(5)
            times = [i / float(fps) for i in range(round(float(fps) * 2))]
            selected = [t for t in times if sampler.select(t)]
            self.assertEqual(len(selected), 10)
            self.assertAlmostEqual(selected[0], 0)
            for number, timestamp in enumerate(selected):
                self.assertLess(abs(timestamp - number / 5), 1 / float(fps) + 1e-6)

    def test_mailbox_replaces_only_pending_task(self):
        mailbox = LatestMailbox()
        mailbox.offer("active")
        self.assertEqual(mailbox.take(threading.Event()), "active")
        mailbox.offer("old")
        mailbox.offer("new")
        self.assertEqual(mailbox.replaced, 1)
        self.assertEqual(mailbox.take(threading.Event()), "new")
        mailbox.finish()
        self.assertIsNone(mailbox.take(threading.Event()))

    def test_frame_buffer_enforces_bytes_and_stop_unblocks_put(self):
        stop = threading.Event()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        buffer = FrameBuffer(max_frames=12, max_bytes=12)
        self.assertTrue(buffer.put(Frame(0, 0, image), stop))
        results = []
        worker = threading.Thread(
            target=lambda: results.append(buffer.put(Frame(1, 0.1, image), stop))
        )
        worker.start()
        stop.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [False])
        self.assertEqual(buffer.bytes_used, 12)
        self.assertEqual(buffer.get(0.01).frame_index, 0)
        self.assertEqual(buffer.bytes_used, 0)

    def test_oversized_frame_is_explicit_error(self):
        buffer = FrameBuffer(max_bytes=1)
        with self.assertRaisesRegex(ValueError, "128|frame|Frame"):
            buffer.put(Frame(0, 0, np.zeros((2, 2, 3), dtype=np.uint8)), threading.Event())

    def test_pause_and_rebuffer_shift_clock_without_slow_motion(self):
        clock = PlaybackClock()
        clock.start(10, 0)
        self.assertAlmostEqual(clock.deadline(1), 11)
        clock.suspend(10.4)
        clock.resume(15.4)
        self.assertAlmostEqual(clock.deadline(1), 16)
        clock.suspend(15.6)
        clock.resume(17.6)
        self.assertAlmostEqual(clock.deadline(2), 19)

    def test_pause_nested_inside_rebuffer_does_not_resume_clock_early(self):
        clock = PlaybackClock()
        clock.start(0, 0)
        clock.suspend(1, "buffer")
        clock.suspend(2, "pause")
        clock.resume(3, "buffer")
        self.assertIsNotNone(clock.suspended_at)
        clock.resume(5, "pause")
        self.assertAlmostEqual(clock.deadline(2), 6)

    def test_timestamp_validation_rejects_vfr_and_accepts_ntsc_rounding(self):
        fps = Fraction(30000, 1001)
        validate_timestamps((round(i / float(fps), 6) for i in range(100)), fps)
        with self.assertRaisesRegex(ValueError, "VFR"):
            validate_timestamps(iter([0, 0.04, 0.08, 0.16]), Fraction(25))
        with self.assertRaisesRegex(ValueError, "VFR"):
            validate_timestamps(iter([0, 0.04, 0.04, 0.12]), Fraction(25), tolerance=0.08)
        with self.assertRaisesRegex(ValueError, "timestamp|Timestamp"):
            validate_timestamps(iter([]), Fraction(30))

    def test_results_never_overlay_old_or_future_frame_and_are_bounded(self):
        results = Results()
        for index in (0, 5, 10):
            results.publish(
                {"frame_index": index, "timestamp_seconds": index / 30, "detections": []}
            )
        exact, latest = results.at(0)
        self.assertEqual(exact["frame_index"], 0)
        exact, latest = results.at(1)
        self.assertIsNone(exact)
        self.assertEqual(latest["frame_index"], 0)
        exact, latest = results.at(6)
        self.assertIsNone(exact)
        self.assertEqual(latest["frame_index"], 5)
        for index in range(1000):
            results.publish({"frame_index": index})
        self.assertLessEqual(len(results.rows), 64)


if __name__ == "__main__":
    unittest.main()
