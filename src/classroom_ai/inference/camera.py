"""Webcam acquisition and one bounded session worker per camera."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque

import cv2
import numpy as np


def open_camera(index: int):
    if index < 0:
        raise ValueError("Chỉ số webcam phải từ 0 trở lên.")
    backends = (
        (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if sys.platform == "win32" else (cv2.CAP_ANY,)
    )
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Một số driver báo mở thành công nhưng không trả frame.
                for _ in range(3):
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size:
                        return capture
        except cv2.error:
            pass
        capture.release()
    raise RuntimeError(
        f"Không đọc được webcam {index}. Đóng Camera/Teams/Zoom, kiểm quyền camera cho ứng dụng desktop, "
        "kiểm nắp che/cáp USB hoặc thử chỉ số camera khác (0, 1, 2)."
    )


_CAMERAS_IN_USE = set()


_CAMERAS_LOCK = threading.Lock()


class CameraSession:
    def __init__(
        self, index, predict, confidence, *, capture_factory=open_camera, heartbeat_timeout=15
    ):
        self.index = index
        self.predict = predict
        self.confidence = confidence
        self.capture_factory = capture_factory
        self.heartbeat_timeout = heartbeat_timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._heartbeat = time.monotonic()
        self._status = "stopped"
        self._error = ""
        self._frame = None
        self._output = None
        self._frames = 0
        self._starts = 0
        self._backend = ""
        self._started = self._finished = 0.0
        self._latencies = deque(maxlen=300)
        self._capture_read_ms = deque(maxlen=300)
        self._callback_ms = deque(maxlen=300)
        self._processing_ms = deque(maxlen=300)
        self._manager_lock_wait_ms = deque(maxlen=300)
        self._manager_active_check_ms = deque(maxlen=300)
        self._inspector_lock_wait_ms = deque(maxlen=300)
        self._frame_times = deque(maxlen=300)
        self._published_at = 0.0

    @staticmethod
    def _percentile(samples, percentile):
        return round(float(np.percentile(samples, percentile)), 2) if samples else None

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            with _CAMERAS_LOCK:
                if self.index in _CAMERAS_IN_USE:
                    self._error = "Webcam đang được dùng ở một phiên khác. Dừng phiên đó hoặc chờ phiên đóng tự giải phóng."
                    self._status = "error"
                    return False
                _CAMERAS_IN_USE.add(self.index)
            self._stop.clear()
            self._heartbeat = self._started = time.monotonic()
            self._finished = 0.0
            self._frames = 0
            self._latencies.clear()
            self._capture_read_ms.clear()
            self._callback_ms.clear()
            self._processing_ms.clear()
            self._manager_lock_wait_ms.clear()
            self._manager_active_check_ms.clear()
            self._inspector_lock_wait_ms.clear()
            self._frame_times.clear()
            self._published_at = 0.0
            self._frame = self._output = None
            self._error = self._backend = ""
            self._status = "starting"
            self._starts += 1
            self._thread = threading.Thread(
                target=self._run, name=f"webcam-{self.index}", daemon=True
            )
            try:
                self._thread.start()
            except Exception:
                with _CAMERAS_LOCK:
                    _CAMERAS_IN_USE.discard(self.index)
                self._status = "error"
                raise
        return True

    def stop(self, timeout=2):
        self._stop.set()
        with self._lock:
            thread = self._thread
            if thread and thread.is_alive():
                self._status = "stopping"
            self._frame = self._output = None
        if thread:
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def snapshot(self):
        with self._lock:
            self._heartbeat = time.monotonic()
            end = self._finished or self._heartbeat
            elapsed = end - self._started if self._started else 0
            latencies = tuple(self._latencies)
            capture_read_ms = tuple(self._capture_read_ms)
            callback_ms = tuple(self._callback_ms)
            processing_ms = tuple(self._processing_ms)
            manager_lock_wait_ms = tuple(self._manager_lock_wait_ms)
            manager_active_check_ms = tuple(self._manager_active_check_ms)
            inspector_lock_wait_ms = tuple(self._inspector_lock_wait_ms)
            frame_times = tuple(self._frame_times)
            latest_result_age_ms = (
                (time.monotonic() - self._published_at) * 1000
                if self._published_at and self._status == "running"
                else None
            )
            frame_size = (
                {"width_px": int(self._frame.shape[1]), "height_px": int(self._frame.shape[0])}
                if self._frame is not None
                else None
            )
            recent_frame_times = tuple(
                timestamp for timestamp in frame_times if end - timestamp <= 5
            )
            recent_fps = (
                (len(recent_frame_times) - 1) / (recent_frame_times[-1] - recent_frame_times[0])
                if len(recent_frame_times) > 1 and recent_frame_times[-1] > recent_frame_times[0]
                else 0.0
            )
            snapshot = {
                "status": self._status,
                "error": self._error,
                "backend": self._backend,
                "frame": self._frame if self._status == "running" else None,
                "output": self._output if self._status == "running" else None,
                "camera_index": self.index,
                "starts": self._starts,
                "frames_processed": self._frames,
                "elapsed_s": round(elapsed, 2),
                "processing_fps": round(self._frames / elapsed, 2) if elapsed > 0 else 0,
                "latency_samples": len(latencies),
                "recent_processing_fps": round(recent_fps, 2),
                "latest_result_age_ms": (
                    round(latest_result_age_ms, 2) if latest_result_age_ms is not None else None
                ),
                "frame_size": frame_size,
            }
        # Copy bounded samples under the lock; calculate percentiles after releasing it.
        snapshot.update(
            {
                "capture_read_p50_ms": self._percentile(capture_read_ms, 50),
                "capture_read_p95_ms": self._percentile(capture_read_ms, 95),
                "callback_p50_ms": self._percentile(latencies, 50),
                "callback_p95_ms": self._percentile(latencies, 95),
                "callback_wall_p50_ms": self._percentile(callback_ms, 50),
                "callback_wall_p95_ms": self._percentile(callback_ms, 95),
                "processing_p50_ms": self._percentile(processing_ms, 50),
                "processing_p95_ms": self._percentile(processing_ms, 95),
                "manager_lock_wait_p50_ms": self._percentile(manager_lock_wait_ms, 50),
                "manager_lock_wait_p95_ms": self._percentile(manager_lock_wait_ms, 95),
                "manager_active_check_p50_ms": self._percentile(manager_active_check_ms, 50),
                "manager_active_check_p95_ms": self._percentile(manager_active_check_ms, 95),
                "inspector_lock_wait_p50_ms": self._percentile(inspector_lock_wait_ms, 50),
                "inspector_lock_wait_p95_ms": self._percentile(inspector_lock_wait_ms, 95),
                "app_latency_p95_ms": self._percentile(latencies, 95),
            }
        )
        return snapshot

    def _run(self):
        capture = None
        error = ""
        try:
            capture = self.capture_factory(self.index)
            with self._lock:
                self._backend = capture.getBackendName()
            failed_reads = 0
            while not self._stop.is_set():
                with self._lock:
                    expired = time.monotonic() - self._heartbeat > self.heartbeat_timeout
                if expired:
                    break
                read_started = time.perf_counter()
                ok, frame = capture.read()
                read_ms = (time.perf_counter() - read_started) * 1000
                with self._lock:
                    self._capture_read_ms.append(read_ms)
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 5:
                        raise RuntimeError(
                            "Webcam ngừng trả hình. Kiểm nắp che/cáp USB và bấm Bật webcam để thử lại."
                        )
                    self._stop.wait(0.05)
                    continue
                failed_reads = 0
                received = time.monotonic()
                callback_started = time.perf_counter()
                output = self.predict(frame, self.confidence)
                callback_ms = (time.perf_counter() - callback_started) * 1000
                latency_ms = (time.monotonic() - received) * 1000
                published_at = time.monotonic()
                with self._lock:
                    if self._stop.is_set():
                        break
                    self._frame = frame
                    self._output = output
                    self._frames += 1
                    self._latencies.append(latency_ms)
                    self._callback_ms.append(callback_ms)
                    self._frame_times.append(published_at)
                    self._published_at = published_at
                    if isinstance(output, dict):
                        for key, samples in (
                            ("processing_ms", self._processing_ms),
                            ("manager_lock_wait_ms", self._manager_lock_wait_ms),
                            ("manager_active_check_ms", self._manager_active_check_ms),
                            ("inspector_lock_wait_ms", self._inspector_lock_wait_ms),
                        ):
                            value = output.get(key)
                            if isinstance(value, (int, float)) and np.isfinite(value):
                                samples.append(float(value))
                    self._status = "running"
        except Exception as exc:
            error = str(exc)
        finally:
            try:
                if capture is not None:
                    capture.release()
            finally:
                with _CAMERAS_LOCK:
                    _CAMERAS_IN_USE.discard(self.index)
                with self._lock:
                    self._frame = self._output = None
                    self._published_at = 0.0
                    self._finished = time.monotonic()
                    self._error = error
                    self._status = "error" if error else "stopped"
