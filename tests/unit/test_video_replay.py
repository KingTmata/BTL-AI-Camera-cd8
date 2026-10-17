import threading
import unittest
from unittest.mock import patch

from classroom_ai.interfaces.native_video import NativeDisplay
from classroom_ai.video.core import PlaybackClock
from classroom_ai.video.playback import Controls


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
