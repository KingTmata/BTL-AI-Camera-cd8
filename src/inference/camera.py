"""Một worker sở hữu webcam; giao diện chỉ lấy kết quả mới nhất, không xếp hàng."""

from collections import deque
import threading
import time

import numpy as np

from src.inference.capture import open_camera


_CAMERAS_IN_USE = set()
_CAMERAS_LOCK = threading.Lock()


class CameraSession:
    def __init__(self, index, predict, confidence, *, capture_factory=open_camera, heartbeat_timeout=15):
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
            self._frame = self._output = None
            self._error = self._backend = ""
            self._status = "starting"
            self._starts += 1
            self._thread = threading.Thread(target=self._run, name=f"webcam-{self.index}", daemon=True)
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
            return {
                "status": self._status, "error": self._error, "backend": self._backend,
                "frame": self._frame if self._status == "running" else None,
                "output": self._output if self._status == "running" else None,
                "camera_index": self.index, "starts": self._starts,
                "frames_processed": self._frames, "elapsed_s": round(elapsed, 2),
                "processing_fps": round(self._frames / elapsed, 2) if elapsed > 0 else 0,
                "app_latency_p95_ms": round(float(np.percentile(self._latencies, 95)), 2) if self._latencies else None,
                "latency_samples": len(self._latencies),
            }

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
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 5:
                        raise RuntimeError("Webcam ngừng trả hình. Kiểm nắp che/cáp USB và bấm Bật webcam để thử lại.")
                    self._stop.wait(0.05)
                    continue
                failed_reads = 0
                received = time.monotonic()
                output = self.predict(frame, self.confidence)
                latency_ms = (time.monotonic() - received) * 1000
                with self._lock:
                    if self._stop.is_set():
                        break
                    self._frame = frame
                    self._output = output
                    self._frames += 1
                    self._latencies.append(latency_ms)
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
                    self._finished = time.monotonic()
                    self._error = error
                    self._status = "error" if error else "stopped"
