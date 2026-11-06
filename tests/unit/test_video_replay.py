import threading
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from classroom_ai.interfaces.native_video import NativeDisplay
from classroom_ai.video.core import PlaybackClock
from classroom_ai.video.playback import Controls


def test_slow_first_decode_starts_clock_when_ready_and_later_lateness_still_drops():
    from classroom_ai.video.metrics import Metrics
    from classroom_ai.video.playback import play_completed

    now, shown = [0.0], []
    stop = threading.Event()

    class Capture:
        index = 0

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            now[0] += 0.1  # Three frame periods: startup must not consume media time.
            self.index += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    class Display:
        seek_request = None

        def poll(self):
            return None

        def take_seek(self):
            return None

        def enable_seek(self, *args):
            pass

        def show(self, frame, *args):
            shown.append(frame.frame_index)

        def update_position(self, index):
            pass

        def mark_end(self):
            stop.set()

    metrics = Metrics()
    display = Display()
    controls = Controls(display, stop, threading.Event(), PlaybackClock())
    session = SimpleNamespace(
        stop=stop,
        metrics=metrics,
        progress=SimpleNamespace(set=lambda *a: None, tick=lambda *a: None),
    )
    source = SimpleNamespace(path=Path("fixture.mp4"), fps=Fraction(30), frame_count=2)
    with (
        patch("classroom_ai.video.playback.cv2.VideoCapture", return_value=Capture()),
        patch("classroom_ai.video.playback.time.perf_counter", side_effect=lambda: now[0]),
    ):
        play_completed(session, source, controls, display)
    assert shown == [0]
    assert metrics.counts["frames_display_dropped"] == 1


def test_seek_preview_is_presented_even_when_input_poll_makes_it_late():
    from classroom_ai.video.metrics import Metrics
    from classroom_ai.video.playback import play_completed

    now, shown = [0.0], []
    stop = threading.Event()

    class Capture:
        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def get(self, *args):
            return 1

        def read(self):
            now[0] += 0.1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    class Display:
        seek_request = 1

        def poll(self):
            return None

        def take_seek(self):
            target, self.seek_request = self.seek_request, None
            if target is None:
                now[0] += 0.1  # A slow input poll after decode must not drop a seek preview.
            return target

        def enable_seek(self, *args):
            pass

        def show(self, frame, *args):
            shown.append(frame.frame_index)

        def update_position(self, index):
            pass

        def mark_end(self):
            stop.set()

    metrics, display = Metrics(), Display()
    controls = Controls(display, stop, threading.Event(), PlaybackClock())
    session = SimpleNamespace(
        stop=stop,
        metrics=metrics,
        progress=SimpleNamespace(set=lambda *a: None, tick=lambda *a: None),
    )
    source = SimpleNamespace(path=Path("fixture.mp4"), fps=Fraction(30), frame_count=2)
    with (
        patch("classroom_ai.video.playback.cv2.VideoCapture", return_value=Capture()),
        patch("classroom_ai.video.playback.time.perf_counter", side_effect=lambda: now[0]),
    ):
        play_completed(session, source, controls, display)
    assert shown == [1]
    assert metrics.counts.get("frames_display_dropped", 0) == 0


class ReplayControlsTests(unittest.TestCase):
    def test_seek_interrupts_frame_wait_even_when_paused(self):
        display = self.display()
        display.seek_request = None
        controls = Controls(display, threading.Event(), threading.Event(), PlaybackClock())
        controls.paused = True

        def poll():
            display.seek_request = 100
            return None

        with patch.object(display, "poll", side_effect=poll):
            self.assertTrue(
                controls.wait_until(
                    lambda: float("inf"), interrupted=lambda: display.seek_request is not None
                )
            )
        self.assertTrue(controls.paused)

    def display(self):
        display = NativeDisplay.__new__(NativeDisplay)
        display.title, display.opened = "test", False
        return display

    def test_programmatic_slider_update_does_not_request_seek(self):
        display = self.display()
        with patch("classroom_ai.interfaces.native_video.cv2.createTrackbar") as create:
            display.enable_seek(300, 30)
        callback = create.call_args.args[4]
        with patch(
            "classroom_ai.interfaces.native_video.cv2.setTrackbarPos",
            side_effect=lambda name, title, pos: callback(pos),
        ):
            display.update_position(60)
        self.assertIsNone(display.take_seek())
        callback(150)
        self.assertEqual(display.take_seek(), 150)
        self.assertIsNone(display.take_seek())

    def test_keyboard_jumps_are_clamped_and_space_restarts_at_end(self):
        display = self.display()
        with patch("classroom_ai.interfaces.native_video.cv2.createTrackbar"):
            display.enable_seek(300, 30)
        display.current_frame = 50
        with patch("classroom_ai.interfaces.native_video.cv2.pollKey", return_value=ord("a")):
            display.poll()
        self.assertEqual(display.take_seek(), 0)
        display.current_frame = 280
        with patch("classroom_ai.interfaces.native_video.cv2.pollKey", return_value=ord("d")):
            display.poll()
        self.assertEqual(display.take_seek(), 299)
        display.mark_end()
        with patch("classroom_ai.interfaces.native_video.cv2.pollKey", return_value=ord(" ")):
            self.assertEqual(display.poll(), "replay")
        self.assertEqual(display.take_seek(), 0)


if __name__ == "__main__":
    unittest.main()
