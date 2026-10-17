import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from classroom_ai.interfaces.terminal_process import BackendJob, progress_fraction
from classroom_ai.paths import PROJECT_ROOT as ROOT


class RunnerTests(unittest.TestCase):
    def test_invalid_tui_tracking_does_not_open_ui(self):
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "classroom_ai.video",
                "--tui",
                "--mode",
                "fast",
                "--tracking",
                "bytetrack",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("full", result.stderr)

    def setUp(self):
        scratch = ROOT / ".temp" / "tui_tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "video có dấu & khoảng trắng.mp4"
        self.source.touch()

    def test_launch_is_subprocess_without_shell_and_stop_is_a_file(self):
        process = Mock()
        process.poll.return_value = None
        with patch(
            "classroom_ai.interfaces.terminal_process.subprocess.Popen", return_value=process
        ) as spawn:
            job = BackendJob(output_root=self.directory / "runs")
            job.start(self.source, "fast", {"device": "cpu", "imgsz": 1280})
        self.addCleanup(job.close)
        args, kwargs = spawn.call_args
        self.assertIn(str(self.source), args[0])
        self.assertNotIn("shell", kwargs)
        self.assertIn("--stop-file", args[0])
        self.assertIn("--device", args[0])
        job.request_stop()
        self.assertTrue(job.stop_file.is_file())
        process.poll.return_value = 130

    def test_progress_does_not_invent_percentage_while_loading(self):
        self.assertIsNone(progress_fraction("Loading model | stage 1.2s"))
        self.assertEqual(
            progress_fraction("Processing full video | 25/100 (25.0%) | ETA 4s"), (25, 100)
        )
        self.assertEqual(
            progress_fraction("Playing sampled video | played 50/100 | YOLO 9"), (50, 100)
        )
        self.assertEqual(progress_fraction("Detecting and tracking | 25/100 (25.0%)"), (25, 100))
        self.assertEqual(progress_fraction("Rendering tracked video | 40/100 (40.0%)"), (40, 100))

    def test_cannot_start_twice_or_launch_non_file(self):
        job = BackendJob(output_root=self.directory / "runs")
        with self.assertRaises(ValueError):
            job.start(self.directory, "fast")
        process = Mock()
        process.poll.return_value = None
        with patch(
            "classroom_ai.interfaces.terminal_process.subprocess.Popen", return_value=process
        ):
            job.start(self.source, "full")
        with self.assertRaises(ValueError):
            job.start(self.source, "full")
        process.poll.return_value = 0
        job.close()

    def test_real_backend_completes_and_stop_file_cancels_preflight(self):
        source = self.directory / "real video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=30:duration=0.6",
                "-c:v",
                "libx264",
                str(source),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                job = BackendJob(output_root=self.directory / "real_runs")
                self.addCleanup(job.close)
                job.start(source, "fast", {"baseline": True, "no_display": True})
                if cancel:
                    job.request_stop()
                deadline = time.monotonic() + 15
                while job.running and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(job.running)
                self.assertEqual(job.returncode, 130 if cancel else 0)
                report = json.loads((job.result_dir / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "stopped" if cancel else "completed")
                self.assertTrue(job.read_updates())


class TerminalUITests(unittest.IsolatedAsyncioTestCase):
    async def test_tracking_checkbox_mode_lock_and_subprocess_options(self):
        from textual.widgets import Checkbox, Input, Select

        from classroom_ai.interfaces.terminal import VideoTerminal

        job = Mock()
        job.running = False
        job.read_updates.return_value = []
        app = VideoTerminal(options={"tracking": "bytetrack", "track_gap_max_ms": 0}, job=job)
        async with app.run_test(size=(110, 34)) as pilot:
            checkbox = app.query_one("#tracking", Checkbox)
            self.assertTrue(checkbox.disabled)
            self.assertFalse(checkbox.value)
            app.query_one("#mode", Select).value = "full"
            await pilot.pause()
            self.assertFalse(checkbox.disabled)
            checkbox.value = True
            app.query_one("#source", Input).value = str(Path(__file__))
            app.action_start()
            self.assertEqual(job.start.call_args.args[2]["tracking"], "bytetrack")
            self.assertEqual(job.start.call_args.args[2]["track_gap_max_ms"], 0)
            self.assertTrue(checkbox.disabled)
            app.set_running(False)
            app.query_one("#mode", Select).value = "fast"
            await pilot.pause()
            self.assertTrue(checkbox.disabled)
            self.assertFalse(checkbox.value)
            app.action_start()
            self.assertNotIn("tracking", job.start.call_args.args[2])
            self.assertNotIn("track_gap_max_ms", job.start.call_args.args[2])

    async def test_buttons_start_stop_and_invalid_source(self):
        from textual.widgets import Button, Input, Static

        from classroom_ai.interfaces.terminal import VideoTerminal

        job = Mock()
        job.running = False
        job.read_updates.return_value = []
        job.returncode = None
        job.result_dir = None

        def start(*args, **kwargs):
            job.running = True

        job.start.side_effect = start
        app = VideoTerminal(job=job)
        scratch = ROOT / ".temp" / "tui_tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=scratch, delete=False) as stream:
            source = Path(stream.name)
        self.addCleanup(source.unlink)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#start")
            job.start.assert_not_called()
            app.query_one("#source", Input).value = str(source)
            await pilot.click("#start")
            self.assertEqual(job.start.call_count, 1)
            self.assertTrue(app.query_one("#start", Button).disabled)
            self.assertFalse(app.query_one("#stop", Button).disabled)
            await pilot.click("#stop")
            job.request_stop.assert_called_once()
            self.assertIn("dừng", str(app.query_one("#status", Static).render()))
            job.running = False
            job.returncode = 130
            app.poll_job()
            self.assertFalse(app.query_one("#start", Button).disabled)

    async def test_picker_and_exit_keyboard_are_available(self):
        from classroom_ai.interfaces.terminal import VideoPicker, VideoTerminal

        app = VideoTerminal()
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#browse")
            self.assertIsInstance(app.screen, VideoPicker)
            await pilot.press("escape")
            self.assertNotIsInstance(app.screen, VideoPicker)
            await pilot.press("ctrl+q")


if __name__ == "__main__":
    unittest.main()
