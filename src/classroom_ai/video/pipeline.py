"""Native full and sampled playback with bounded queues and durable reports."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from classroom_ai.paths import PROJECT_ROOT as ROOT
from classroom_ai.video.core import FrameBuffer, LatestMailbox, PlaybackClock, Results, TimeSampler
from classroom_ai.video.core import VideoConfig as VideoConfig
from classroom_ai.video.media import VideoWriter, frames, probe_source
from classroom_ai.video.metrics import Metrics, Progress
from classroom_ai.video.playback import Controls


class Session:
    def __init__(self, config, metrics, progress, directory, manager, predictor_factory):
        self.config, self.metrics, self.progress, self.directory = (
            config,
            metrics,
            progress,
            directory,
        )
        self.manager, self.predictor_factory = manager, predictor_factory
        self.stop, self.playing = threading.Event(), threading.Event()
        self.playing.set()
        self.ready, self.first_result = threading.Event(), threading.Event()
        self.results = Results()
        self.error = None
        self.warm_started = None
        self.predictor = None
        self.output = None

    def guarded(self, function):
        try:
            function()
        except BaseException as exc:
            self.error = exc
            self.stop.set()
        finally:
            self.ready.set()

    def initialize(self, source):
        if self.config.baseline:
            self.warm_started = time.perf_counter()
            self.ready.set()
            return
        self.progress.set("Checking device and loading runtime")
        started = time.perf_counter()
        from classroom_ai.inference.devices import cuda_devices, validate_device

        if self.config.device == "auto":
            devices, reason = cuda_devices()
            self.config.device = next(iter(devices), "cpu")
            if self.config.device == "cpu":
                self.metrics.values["device_note"] = reason or "No usable CUDA device; using CPU"
        else:
            self.config.device = validate_device(self.config.device)
        self.metrics.values["resolved_device"] = self.config.device
        if self.config.device.startswith("cuda:"):
            import torch

            self.metrics.values["gpu"] = torch.cuda.get_device_name(self.config.device)
            self.metrics.values["torch_cuda"] = torch.version.cuda
        if self.predictor_factory is None:
            from classroom_ai.inference.detector import ImageInspector

            self.predictor_factory = ImageInspector
        self.metrics.observe("runtime_device_setup", (time.perf_counter() - started) * 1000)
        if self.stop.is_set():
            return
        self.progress.set("Loading model")
        started = time.perf_counter()
        with self.manager.inference_slot():
            self.predictor = self.predictor_factory(self.config.weights, self.config.device)
        load_ms = (time.perf_counter() - started) * 1000
        self.metrics.observe("model_load", load_ms)
        if self.stop.is_set():
            return
        self.progress.set("Warming up YOLO")
        image = np.zeros((source.height, source.width, 3), dtype=np.uint8)
        started = time.perf_counter()
        with self.manager.inference_slot():
            self.predictor.predict(
                image,
                self.config.confidence,
                imgsz=self.config.imgsz,
                extra_nms_iou=self.config.nms_iou,
                passes=1,
            )
        warmup_ms = (time.perf_counter() - started) * 1000
        self.metrics.observe("warmup", warmup_ms)
        self.progress.model_ready(load_ms, warmup_ms)
        del image
        self.warm_started = time.perf_counter()
        self.metrics.values["model_ready_cold_ms"] = (
            self.warm_started - self.metrics.started
        ) * 1000
        self.ready.set()

    def infer(self, frame, stream):
        started = time.perf_counter()
        with self.manager.inference_slot() as timings:
            for key, value in timings.items():
                self.metrics.observe(key.removesuffix("_ms"), value)
            output = self.predictor.predict(
                frame.image,
                self.config.confidence,
                imgsz=self.config.imgsz,
                extra_nms_iou=self.config.nms_iou,
                passes=1,
            )
        elapsed = (time.perf_counter() - started) * 1000
        self.metrics.observe("inference_callback", elapsed)
        self.metrics.observe("model_processing", output["processing_ms"])
        self.metrics.observe("inspector_lock_wait", output.get("inspector_lock_wait_ms", 0))
        result = {
            "frame_index": frame.frame_index,
            "timestamp_seconds": frame.timestamp_seconds,
            "detections": output["detections"],
            "processing_ms": output["processing_ms"],
            "callback_ms": elapsed,
        }
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        self.metrics.count("frames_inferred")
        self.results.publish(result)
        if not self.first_result.is_set():
            now = time.perf_counter()
            self.metrics.values["first_result_available_warm_ms"] = (now - self.warm_started) * 1000
            self.metrics.values["first_result_available_cold_ms"] = (
                now - self.metrics.started
            ) * 1000
            self.first_result.set()
        return result


def _full(session, source):
    from classroom_ai.inference.detector import draw_boxes

    session.initialize(source)
    if session.stop.is_set():
        return
    if session.config.tracking == "bytetrack":
        from classroom_ai.video.tracked_processing import process_tracked

        process_tracked(session, source)
        return
    session.progress.set("Processing full video")
    writer = VideoWriter(source, session.directory)
    try:
        with (session.directory / "detections.jsonl").open("x", encoding="utf-8") as stream:
            for frame in frames(source, session.metrics, session.stop):
                result = session.infer(frame, stream)
                started = time.perf_counter()
                image = draw_boxes(frame.image, result["detections"])
                session.metrics.observe("draw", (time.perf_counter() - started) * 1000)
                started = time.perf_counter()
                writer.write(image)
                session.metrics.observe("encode_write", (time.perf_counter() - started) * 1000)
                session.metrics.count("frames_encoded")
        if not session.stop.is_set():
            session.progress.set("Finalizing and validating MP4")
            started = time.perf_counter()
            session.output = writer.finish()
            session.metrics.observe(
                "export_finalize_validate", (time.perf_counter() - started) * 1000
            )
    finally:
        writer.close()


def _consume(session, source, buffer, controls, display, *, results=True, first_gate=True):
    """PTS paced consumer, also used to measure headless scheduling."""
    frame_bytes = source.width * source.height * 3
    capacity = min(12, (128 * 1024 * 1024) // frame_bytes)
    target = min(
        capacity, max(1, math.ceil(session.config.buffer_ms * float(source.fps) / 1000) + 1)
    )
    if target < 1:
        raise ValueError("Một frame vượt giới hạn 128 MiB.")
    while not session.stop.is_set() and (
        controls.paused
        or not buffer.ready(target)
        or (first_gate and not session.first_result.is_set())
    ):
        controls.poll()
        session.progress.tick(source, display)
        time.sleep(0.005)
    if session.stop.is_set():
        return
    clock = controls.clock
    started = time.perf_counter()
    clock.start(started, 0)
    first_presented, last_presented = None, None
    interval_start = None
    paused_before = controls.pause_seconds
    while not session.stop.is_set():
        controls.poll()
        if controls.paused:
            session.progress.tick(source)
            time.sleep(0.005)
            interval_start = None
            continue
        wait_started = time.perf_counter()
        frame = buffer.get(0.005)
        session.metrics.observe("display_queue_wait", (time.perf_counter() - wait_started) * 1000)
        if frame is None:
            if buffer.finished:
                break
            stall = time.perf_counter()
            clock.suspend(stall, "buffer")
            session.metrics.count("rebuffer_events")
            if display:
                display.status("Buffering decoder... | Q/Esc: stop")
            while frame is None and not buffer.finished and not session.stop.is_set():
                controls.poll()
                session.progress.tick(source)
                frame = buffer.get(0.005) if not controls.paused else None
                time.sleep(0.002)
            clock.resume(time.perf_counter(), "buffer")
            session.metrics.observe("rebuffer", (time.perf_counter() - stall) * 1000)
            interval_start = None
            if frame is None:
                continue
        # If the UI is genuinely late, discard expired display frames, not inference results.
        if frame.frame_index > 0 and time.perf_counter() - clock.deadline(
            frame.timestamp_seconds
        ) > 1 / float(source.fps):
            session.metrics.count("frames_display_dropped")
            continue
        pause_before_wait = controls.pause_seconds
        if not controls.wait_until(lambda: clock.deadline(frame.timestamp_seconds)):
            break
        if controls.pause_seconds != pause_before_wait:
            interval_start = None
        result, latest = session.results.at(frame.frame_index) if results else (None, None)
        if display:
            before = time.perf_counter()
            display.show(
                frame, result, latest, f"{session.config.mode} | source {float(source.fps):.3f}fps"
            )
            # Pump events: first-result timing must include actual presentation, not enqueue.
            controls.poll()
            now = time.perf_counter()
            session.metrics.observe("draw_present", (now - before) * 1000)
            session.metrics.count("frames_displayed")
            if interval_start is not None and not controls.paused:
                session.metrics.observe("display_interval", (now - interval_start) * 1000)
            interval_start = now
            first_presented = first_presented if first_presented is not None else now
            last_presented = now
            if result and session.metrics.values["first_result_display_warm_ms"] is None:
                session.metrics.values["first_result_display_warm_ms"] = (
                    now - session.warm_started
                ) * 1000
                session.metrics.values["first_result_display_cold_ms"] = (
                    now - session.metrics.started
                ) * 1000
            if latest:
                session.metrics.observe(
                    "result_media_age",
                    max(0, frame.timestamp_seconds - latest["timestamp_seconds"]) * 1000,
                )
        session.metrics.count("frames_consumed")
        session.progress.tick(source)
    # Include the last frame's display duration in playback instead of ending one frame early.
    if not session.stop.is_set():
        controls.wait_until(lambda: clock.deadline(source.duration))
    session.metrics.values["playback_wall_seconds"] = time.perf_counter() - started
    session.metrics.values["playback_completed"] = not session.stop.is_set()
    session.metrics.values["user_pause_seconds"] = controls.pause_seconds - paused_before
    if first_presented is not None and last_presented > first_presented:
        count = session.metrics.counts.get("frames_displayed", 0)
        wall_elapsed = last_presented - first_presented
        session.metrics.values["display_fps_wall"] = (count - 1) / wall_elapsed
        active_elapsed = wall_elapsed - (controls.pause_seconds - paused_before)
        session.metrics.values["display_fps_excluding_user_pause"] = (
            (count - 1) / active_elapsed if active_elapsed > 0 else None
        )


def _fast(session, source, controls, display):
    buffer, mailbox = FrameBuffer(), LatestMailbox()

    def inference():
        session.initialize(source)
        if session.config.baseline:
            session.first_result.set()
            return
        with (session.directory / "detections.jsonl").open("x", encoding="utf-8") as stream:
            while not session.stop.is_set():
                task = mailbox.take(session.stop)
                if task is None:
                    break
                frame, queued_at = task
                session.metrics.observe(
                    "inference_queue_wait", (time.perf_counter() - queued_at) * 1000
                )
                session.infer(frame, stream)

    def reader():
        sampler = TimeSampler(session.config.sample_fps)
        try:
            for frame in frames(source, session.metrics, session.stop, session.playing):
                if not buffer.put(frame, session.stop, session.playing):
                    break
                if not session.config.baseline and sampler.select(frame.timestamp_seconds):
                    session.metrics.count("frames_sampled")
                    mailbox.offer((frame, time.perf_counter()))
                    if frame.frame_index == 0:
                        # Protect the first frame from replacement while startup is gated.
                        while not session.first_result.wait(0.02) and not session.stop.is_set():
                            pass
        finally:
            buffer.finish()
            mailbox.finish()

    worker = threading.Thread(target=lambda: session.guarded(inference), name="video-inference")
    decoder = threading.Thread(target=lambda: session.guarded(reader), name="video-decoder")
    worker.start()
    try:
        while not session.ready.wait(0.01) and not session.stop.is_set():
            controls.poll(allow_pause=False)
            session.progress.tick(source, display)
        if session.stop.is_set():
            return
        session.progress.set(
            "Playing sampled video"
            if not session.config.baseline
            else "Playback baseline (no YOLO)"
        )
        decoder.start()
        _consume(session, source, buffer, controls, display, results=not session.config.baseline)
        # Playback EOF is not permission to discard a pending inference/write.
        while worker.is_alive() and not session.stop.is_set():
            controls.poll(allow_pause=False)
            session.progress.tick(source)
            worker.join(0.02)
    finally:
        session.stop.set()
        session.playing.set()
        mailbox.finish()
        for thread in (decoder, worker):
            if thread.ident is not None:
                thread.join()
        session.metrics.values.update(
            buffer_peak_bytes=buffer.peak_bytes,
            buffer_peak_frames=buffer.peak_frames,
            inference_pending_replaced=mailbox.replaced,
        )


def _play_completed(session, source, controls, display):
    from classroom_ai.video.playback import play_completed

    play_completed(session, source, controls, display)


def run_video(
    config,
    *,
    predictor_factory=None,
    manager=None,
    display_factory=None,
    progress=print,
    program_started=None,
    stop_file=None,
):
    config.validate()
    metrics = Metrics()
    if program_started is not None:
        metrics.started = program_started
    metrics.values.update(
        first_result_display_warm_ms=None,
        first_result_display_cold_ms=None,
        first_result_available_warm_ms=None,
        first_result_available_cold_ms=None,
    )
    config.source = Path(config.source).resolve()
    config.weights = Path(config.weights).resolve()
    directory = Path(config.output_dir).resolve() / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:10]
    )
    directory.mkdir(parents=True, exist_ok=False)
    display, session, status, error = None, None, "failed", None
    state = Progress(metrics, progress)
    metadata = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("numpy", "opencv-python", "torch", "ultralytics", "psutil")
        },
    }
    try:
        state.set("Validating source timestamps (CFR scan)")
        progress(
            "Validating source timestamps (CFR scan); progress is indeterminate, cost reported separately."
        )
        started = time.perf_counter()
        next_control_check = 0.0

        def source_progress(count, total):
            nonlocal next_control_check
            if stop_file is not None and time.monotonic() >= next_control_check:
                next_control_check = time.monotonic() + 0.2
                if stop_file.exists():
                    raise KeyboardInterrupt
            state.preflight(count, total)

        source = probe_source(config.source, on_progress=source_progress)
        metrics.observe("source_preflight", (time.perf_counter() - started) * 1000)
        if source.width * source.height * 3 > 128 * 1024 * 1024:
            raise ValueError("Một frame vượt giới hạn 128 MiB.")
        metadata["source"] = source.metadata()
        if not config.baseline:
            if not config.weights.is_file():
                raise ValueError(f"Không thấy checkpoint: {config.weights}")
            with config.weights.open("rb") as stream:
                metadata["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            progress(
                f"Checkpoint: {config.weights.name} | imgsz {config.imgsz} | confidence {config.confidence:g}"
            )
            if manager is None:
                from classroom_ai.training.manager import RunManager

                manager = RunManager(ROOT)
        metadata["notes"] = {
            "preflight": "Strict streamed full-source timestamp scan; included in cold cost.",
            "preview": "Silent. Display FPS is not inference FPS. Headless presentation latency is null.",
            "percentiles": "At most 2000 recent samples per stage; lifetime count/mean/max retained.",
            "audio": "First audio track AAC in full MP4. Odd dimensions padded to even.",
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not config.no_display:
            if display_factory is None:
                from classroom_ai.interfaces.native_video import NativeDisplay

                display_factory = NativeDisplay
            display = display_factory()
        session = Session(config, metrics, state, directory, manager, predictor_factory)
        controls = Controls(display, session.stop, session.playing, PlaybackClock(), stop_file)
        if config.mode == "full":
            worker = threading.Thread(
                target=lambda: session.guarded(lambda: _full(session, source)),
                name="full-video-worker",
            )
            worker.start()
            try:
                while worker.is_alive():
                    controls.poll(allow_pause=False)
                    state.tick(source, display)
                    worker.join(0.02)
            finally:
                if worker.is_alive():
                    session.stop.set()
                    worker.join()
            if session.error:
                raise session.error
            if session.output:
                metrics.values["processing_complete_cold_ms"] = (
                    time.perf_counter() - metrics.started
                ) * 1000
            if session.output and display:
                _play_completed(session, probe_source(session.output), controls, display)
                if session.error:
                    raise session.error
            status = "completed" if session.output else "stopped"
        else:
            _fast(session, source, controls, display)
            if session.error:
                raise session.error
            # Normal cleanup also sets stop; distinguish EOF by consumed + dropped frames.
            total = metrics.counts.get("frames_consumed", 0) + metrics.counts.get(
                "frames_display_dropped", 0
            )
            status = "completed" if total == source.frame_count else "stopped"
    except KeyboardInterrupt:
        status = "stopped"
    except BaseException as exc:
        error = exc
    finally:
        if session:
            session.stop.set()
        if display:
            display.close()
        report = metrics.report()
        report.update(status=status, error=str(error) if error else None)
        report.update(
            rss_initial_bytes=state.initial_rss,
            rss_peak_sampled_bytes=state.peak_rss,
            rss_final_bytes=state.process.memory_info().rss,
        )
        inferred = report["counts"].get("frames_inferred", 0)
        decoded = report["counts"].get("frames_decoded", 0)
        report["inferred_fraction_of_source"] = inferred / metadata.get("source", {}).get(
            "frame_count", 1
        )
        report["inference_callback_fps"] = (
            (inferred * 1000 / report["stages"]["inference_callback"]["total_ms"])
            if inferred
            else None
        )
        report["frames_without_inference"] = max(
            0, metadata.get("source", {}).get("frame_count", decoded) - inferred
        )
        dropped = report["counts"].get("frames_display_dropped", 0)
        consumed = report["counts"].get("frames_consumed", 0)
        report["display_drop_fraction"] = (
            dropped / (consumed + dropped) if consumed + dropped else None
        )
        latency = report["first_result_display_warm_ms"]
        report["first_display_under_3s_warm"] = latency < 3000 if latency is not None else None
        metadata["resolved_device"] = report.get("resolved_device")
        metadata["device_note"] = report.get("device_note")
        if "tracking_settings" in report:
            metadata["tracking"] = report["tracking_settings"]
            metadata["versions"]["lap"] = version("lap")
        if "gpu" in report:
            metadata.update(gpu=report["gpu"], torch_cuda=report.get("torch_cuda"))
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        progress(f"{status.upper()} | {report['elapsed_seconds']:.2f}s | results: {directory}")
        progress(
            f"  device: {report.get('resolved_device', 'baseline')} | {report.get('device_note', '')}"
        )
        progress(
            f"  first YOLO result available warm: {report.get('first_result_available_warm_ms')}ms | "
            f"first displayed warm: {latency}ms | displayed cold: {report.get('first_result_display_cold_ms')}ms"
        )
        progress(
            f"  displayed FPS: {report.get('display_fps_wall')} | display drop fraction: {report['display_drop_fraction']}"
        )
        progress(
            f"  displayed FPS excluding explicit pause: {report.get('display_fps_excluding_user_pause')} | "
            f"user pause: {report.get('user_pause_seconds', 0):.3f}s"
        )
        progress(
            f"  RAM: initial {report['rss_initial_bytes'] / 1024**2:.1f} MiB, "
            f"peak sampled {report['rss_peak_sampled_bytes'] / 1024**2:.1f} MiB, "
            f"final {report['rss_final_bytes'] / 1024**2:.1f} MiB"
        )
        for stage, cost in report["stages"].items():
            progress(
                f"  {stage}: total {cost['total_ms']:.1f}ms | p50 {cost['p50_ms']:.1f}ms | p95 {cost['p95_ms']:.1f}ms"
            )
    if error:
        raise error
    return directory
