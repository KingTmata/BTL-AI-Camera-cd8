"""End-to-end media tests use real FFmpeg and a deterministic detector."""

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from classroom_ai.paths import PROJECT_ROOT as ROOT
from classroom_ai.video.media import executable, probe_source
from classroom_ai.video.pipeline import VideoConfig, run_video

SCRATCH = ROOT / ".temp" / "video_tests"


class Detector:
    def __init__(self, weights, device, delay=0):
        self.delay = delay

    def predict(self, image, confidence, **kwargs):
        time.sleep(self.delay)
        return {
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_xyxy": [1, 1, 10, 10],
                }
            ],
            "processing_ms": self.delay * 1000,
            "inspector_lock_wait_ms": 0,
        }


class Guard:
    from contextlib import contextmanager

    @contextmanager
    def inference_slot(self):
        yield {"manager_lock_wait_ms": 0, "manager_active_check_ms": 0}


class RecordingDisplay:
    def __init__(self):
        self.presented = []
        self.closed = False
        self.ended = False

    def enable_seek(self, frame_count, fps):
        pass

    def take_seek(self):
        return None

    def update_position(self, index):
        pass

    def mark_end(self):
        self.ended = True

    def status(self, text):
        pass

    def show(self, frame, result, latest, text):
        if result is not None:
            assert result["frame_index"] == frame.frame_index
        if latest is not None:
            assert latest["timestamp_seconds"] <= frame.timestamp_seconds
        self.presented.append((frame.frame_index, result))

    def poll(self):
        return "quit" if self.ended else None

    def close(self):
        self.closed = True


class VideoPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable("ffmpeg")
        executable("ffprobe")
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.weights = self.root / "fake.pt"
        self.weights.write_bytes(b"fake-test-weights")

    def fixture(self, fps="30", audio=True, seconds=0.6):
        source = self.root / f"video có dấu {fps.replace('/', '_')}.mp4"
        command = [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=64x48:rate={fps}",
        ]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
        command += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if audio:
            command += ["-c:a", "aac"]
        subprocess.run(command + [str(source)], check=True, capture_output=True, timeout=20)
        return source

    def run_config(self, source, mode, **kwargs):
        config = VideoConfig(
            source=source,
            mode=mode,
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
            no_display=True,
            **kwargs,
        )
        directory = run_video(
            config, predictor_factory=Detector, manager=Guard(), progress=lambda text: None
        )
        return directory, json.loads((directory / "report.json").read_text(encoding="utf-8"))

    def test_full_preserves_every_frame_fractional_fps_and_audio(self):
        for rate, audio in [("25", False), ("30000/1001", True), ("30", True)]:
            with self.subTest(rate=rate):
                source = self.fixture(rate, audio)
                directory, report = self.run_config(source, "full")
                original = probe_source(source)
                output = probe_source(directory / "annotated.mp4")
                self.assertEqual(output.fps, original.fps)
                self.assertEqual(output.frame_count, original.frame_count)
                self.assertEqual(output.audio, audio)
                rows = [
                    json.loads(line)
                    for line in (directory / "detections.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [r["frame_index"] for r in rows], list(range(original.frame_count))
                )
                self.assertEqual(report["status"], "completed")
                self.assertIsNone(report["first_result_display_warm_ms"])
                self.assertFalse((directory / "annotated.partial.mp4").exists())

    def test_fast_has_bounded_queues_and_no_claim_of_display_in_headless(self):
        directory, report = self.run_config(self.fixture(), "fast")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["counts"]["frames_decoded"], 18)
        self.assertLess(report["counts"]["frames_inferred"], 18)
        self.assertLessEqual(report["buffer_peak_frames"], 12)
        self.assertLessEqual(report["buffer_peak_bytes"], 128 * 1024 * 1024)
        self.assertEqual(report["counts"].get("frames_displayed", 0), 0)
        self.assertIsNone(report["first_result_display_warm_ms"])
        self.assertFalse((directory / "annotated.mp4").exists())

    def test_full_predictor_error_leaves_failure_report_and_no_final_video(self):
        class Broken(Detector):
            calls = 0

            def predict(self, *args, **kwargs):
                self.calls += 1
                if self.calls > 2:
                    raise RuntimeError("inference failed test")
                return super().predict(*args, **kwargs)

        config = VideoConfig(
            source=self.fixture(),
            mode="full",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
            no_display=True,
        )
        with self.assertRaisesRegex(RuntimeError, "inference failed test"):
            run_video(config, predictor_factory=Broken, manager=Guard(), progress=lambda text: None)
        directory = next(config.output_dir.iterdir())
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse((directory / "annotated.mp4").exists())

    def test_invalid_options_fail_before_creating_output(self):
        for option in ({"sample_fps": float("nan")}, {"buffer_ms": -1}, {"imgsz": 0}):
            config = VideoConfig(source=self.root / "missing.mp4", mode="fast", **option)
            with self.assertRaises(ValueError):
                config.validate()

    def test_tracking_options_invalid_in_fast_baseline_or_out_of_range(self):
        for option in (
            {"mode": "fast", "tracking": "bytetrack"},
            {"mode": "full", "tracking": "bytetrack", "baseline": True},
            {"mode": "full", "tracking": "other"},
            {"mode": "full", "track_smooth_alpha": 0},
            {"mode": "full", "track_gap_max_ms": -1},
            {"mode": "full", "track_max_lost_seconds": float("nan")},
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                VideoConfig(source="missing.mp4", **option).validate()

    def test_tracked_full_two_pass_raw_detections_preserved_and_no_second_inference(self):
        class Missing(Detector):
            calls = 0

            def predict(self, *args, **kwargs):
                self.calls += 1
                index = self.calls - 2  # First call is model warm-up, not a source frame.
                people = (
                    []
                    if index in (5, 6)
                    else [
                        dict(
                            class_id=0,
                            class_name="person",
                            confidence=0.05,
                            bbox_xyxy=[2 + index * 0.2, 2, 26 + index * 0.2, 44],
                        )
                    ]
                )
                table = dict(
                    class_id=1, class_name="table", confidence=0.8, bbox_xyxy=[30, 20, 60, 40]
                )
                return dict(detections=people + [table], processing_ms=0)

        detector = Missing(self.weights, "cpu")
        config = VideoConfig(
            source=self.fixture(),
            mode="full",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
            no_display=True,
            tracking="bytetrack",
        )
        directory = run_video(
            config, predictor_factory=lambda *a: detector, manager=Guard(), progress=lambda t: None
        )
        report = json.loads((directory / "report.json").read_text())
        raw = [json.loads(s) for s in (directory / "detections.jsonl").read_text().splitlines()]
        tracks = [json.loads(s) for s in (directory / "tracks.jsonl").read_text().splitlines()]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(detector.calls, 19)
        self.assertEqual(report["counts"]["frames_decoded"], 18)
        self.assertEqual(report["counts"]["frames_render_decoded"], 18)
        self.assertEqual(report["counts"]["frames_encoded"], 18)
        self.assertEqual(len(tracks), 18)
        self.assertEqual(tracks[0]["tracks"], [])
        self.assertEqual(
            len(raw[5]["detections"]), 1
        )  # Table unchanged, no person fabricated in raw YOLO.
        self.assertEqual(tracks[5]["tracks"][0]["state"], "interpolated")
        self.assertIsNone(tracks[5]["tracks"][0]["confidence"])
        self.assertEqual(report["counts"]["track_frames_interpolated"], 2)
        self.assertEqual(probe_source(directory / "annotated.mp4").frame_count, 18)

    def test_tracked_render_error_leaves_no_final_mp4(self):
        config = VideoConfig(
            source=self.fixture(),
            mode="full",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
            no_display=True,
            tracking="bytetrack",
        )
        with patch(
            "classroom_ai.video.tracked_processing.render_tracks",
            side_effect=RuntimeError("render broken"),
        ):
            with self.assertRaisesRegex(RuntimeError, "render broken"):
                run_video(
                    config, predictor_factory=Detector, manager=Guard(), progress=lambda t: None
                )
        directory = next(config.output_dir.iterdir())
        self.assertEqual(json.loads((directory / "report.json").read_text())["status"], "failed")
        self.assertFalse((directory / "annotated.mp4").exists())

    def test_tracked_single_frame_no_people_fractional_fps_and_audio(self):
        class OnlyTable(Detector):
            def predict(self, *args, **kwargs):
                return dict(
                    detections=[
                        dict(
                            class_id=1,
                            class_name="table",
                            confidence=0.9,
                            bbox_xyxy=[10, 10, 30, 30],
                        )
                    ],
                    processing_ms=0,
                )

        for rate, audio in (("25", False), ("30000/1001", True), ("30", True)):
            with self.subTest(rate=rate):
                source = self.fixture(fps=rate, audio=audio, seconds=0.03)
                config = VideoConfig(
                    source=source,
                    mode="full",
                    weights=self.weights,
                    device="cpu",
                    output_dir=self.root / "outputs",
                    no_display=True,
                    tracking="bytetrack",
                )
                directory = run_video(
                    config, predictor_factory=OnlyTable, manager=Guard(), progress=lambda t: None
                )
                original, output = probe_source(source), probe_source(directory / "annotated.mp4")
                self.assertEqual(original.frame_count, 1)
                self.assertEqual(output.frame_count, original.frame_count)
                self.assertEqual(output.fps, original.fps)
                self.assertEqual(output.audio, audio)
                row = json.loads((directory / "tracks.jsonl").read_text())
                self.assertEqual(row["tracks"], [])
                raw = json.loads((directory / "detections.jsonl").read_text())
                self.assertEqual(raw["detections"][0]["class_id"], 1)

    def test_fast_slow_detector_replaces_pending_but_keeps_frame_identity(self):
        display = RecordingDisplay()
        config = VideoConfig(
            source=self.fixture(seconds=1.5),
            mode="fast",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
            sample_fps=20,
        )
        directory = run_video(
            config,
            predictor_factory=lambda w, d: Detector(w, d, 0.15),
            manager=Guard(),
            display_factory=lambda: display,
            progress=lambda text: None,
        )
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertTrue(display.closed)
        self.assertGreater(report["inference_pending_replaced"], 0)
        self.assertIsNotNone(report["first_result_display_warm_ms"])
        self.assertEqual(display.presented[0][0], 0)
        self.assertEqual(display.presented[0][1]["frame_index"], 0)

    def test_baseline_does_not_construct_detector(self):
        config = VideoConfig(
            source=self.fixture(),
            mode="fast",
            output_dir=self.root / "outputs",
            no_display=True,
            baseline=True,
        )

        def forbidden(*args):
            self.fail("Baseline must not construct YOLO")

        directory = run_video(config, predictor_factory=forbidden, progress=lambda text: None)
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["counts"].get("frames_inferred", 0), 0)
        self.assertEqual(report["status"], "completed")

    def test_real_vfr_file_is_rejected(self):
        source = self.root / "vfr.mp4"
        command = [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=30:duration=1",
            "-vf",
            r"select=not(eq(mod(n\,3)\,0))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(source),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=20)
        with self.assertRaisesRegex(ValueError, "VFR"):
            probe_source(source)

    def test_full_playback_uses_finalized_output_and_can_close_cleanly(self):
        display = RecordingDisplay()
        config = VideoConfig(
            source=self.fixture(),
            mode="full",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
        )
        directory = run_video(
            config,
            predictor_factory=Detector,
            manager=Guard(),
            display_factory=lambda: display,
            progress=lambda text: None,
        )
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["counts"]["frames_decoded"], 18)
        self.assertEqual(report["counts"]["playback_frames_decoded"], 18)
        self.assertTrue(display.closed)
        self.assertGreater(len(display.presented), 0)

    def test_full_seek_forward_backward_while_paused_does_not_rerun_inference(self):
        class SeekingDisplay(RecordingDisplay):
            seek_request = None
            stage = 0
            key = None

            def show(self, frame, result, latest, text):
                super().show(frame, result, latest, text)
                if frame.frame_index == 3 and self.stage == 0:
                    self.stage, self.seek_request, self.key = 1, 12, "pause"
                elif frame.frame_index == 12 and self.stage == 1:
                    self.stage, self.seek_request = 2, 2
                elif frame.frame_index == 2 and self.stage == 2:
                    self.stage, self.key = 3, "pause"

            def take_seek(self):
                target, self.seek_request = self.seek_request, None
                return target

            def poll(self):
                key, self.key = self.key, None
                return key or super().poll()

        display = SeekingDisplay()
        config = VideoConfig(
            source=self.fixture(),
            mode="full",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
        )
        directory = run_video(
            config,
            predictor_factory=Detector,
            manager=Guard(),
            display_factory=lambda: display,
            progress=lambda text: None,
        )
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual([i for i, _ in display.presented[:6]], [0, 1, 2, 3, 12, 2])
        self.assertEqual(report["playback_seek_count"], 2)
        self.assertEqual(report["counts"]["frames_inferred"], 18)
        self.assertEqual(report["counts"]["frames_encoded"], 18)
        self.assertIsNone(report["display_fps_wall"])
        self.assertTrue(report["playback_completed"])

    def test_encoder_failure_records_failed_without_publishing_output(self):
        config = VideoConfig(
            source=self.fixture(),
            mode="full",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
            no_display=True,
        )
        with patch(
            "classroom_ai.video.pipeline.VideoWriter.write", side_effect=OSError("disk full test")
        ):
            with self.assertRaisesRegex(OSError, "disk full test"):
                run_video(
                    config, predictor_factory=Detector, manager=Guard(), progress=lambda text: None
                )
        directory = next(config.output_dir.iterdir())
        self.assertEqual(json.loads((directory / "report.json").read_text())["status"], "failed")
        self.assertFalse((directory / "annotated.mp4").exists())

    def test_quit_during_fast_loading_is_stopped_and_closes_display(self):
        class QuitDisplay(RecordingDisplay):
            def poll(self):
                return "quit"

        display = QuitDisplay()
        config = VideoConfig(
            source=self.fixture(),
            mode="fast",
            weights=self.weights,
            device="cpu",
            output_dir=self.root / "outputs",
        )
        directory = run_video(
            config,
            predictor_factory=lambda w, d: Detector(w, d, 0.05),
            manager=Guard(),
            display_factory=lambda: display,
            progress=lambda text: None,
        )
        self.assertEqual(json.loads((directory / "report.json").read_text())["status"], "stopped")
        self.assertTrue(display.closed)


if __name__ == "__main__":
    unittest.main()
