"""Video configuration, bounded state and timing primitives; no model or GUI ownership."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from classroom_ai.paths import PROJECT_ROOT as ROOT


@dataclass(frozen=True)
class Frame:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


DEFAULT_WEIGHTS = (
    ROOT
    / "weights/checkpoint07_matrix_20261012/E3_YOLO26n_Project8_v05_from_CP07_v04_best19_AdamW_lr0p0001_bestEpoch10_20261012.pt"
)


@dataclass
class VideoConfig:
    source: Path
    mode: str
    weights: Path = DEFAULT_WEIGHTS
    device: str = "auto"
    imgsz: int = 1280
    confidence: float = 0.05
    nms_iou: float | None = None
    sample_fps: float = 5.0
    buffer_ms: float = 250.0
    output_dir: Path = ROOT / "runs/video"
    no_display: bool = False
    baseline: bool = False
    tracking: str = "off"
    track_max_lost_seconds: float = 3.0
    track_gap_max_ms: float = 300.0
    track_smooth_alpha: float = 0.65

    def validate(self):
        if self.mode not in ("full", "fast"):
            raise ValueError("mode phải là full hoặc fast.")
        if not isinstance(self.imgsz, int) or self.imgsz <= 0:
            raise ValueError("imgsz phải là số nguyên dương.")
        for name in ("confidence", "sample_fps", "buffer_ms"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} phải hữu hạn.")
        if not 0 < self.confidence <= 1 or self.sample_fps <= 0 or self.buffer_ms < 0:
            raise ValueError("confidence trong (0,1], sample-fps > 0 và buffer-ms >= 0.")
        if self.nms_iou is not None and (
            not math.isfinite(self.nms_iou) or not 0 <= self.nms_iou <= 1
        ):
            raise ValueError("nms-iou phải trong [0,1].")
        if self.baseline and self.mode != "fast":
            raise ValueError("--baseline chỉ dùng với --mode fast (không YOLO).")
        if self.tracking not in ("off", "bytetrack"):
            raise ValueError("--tracking phải là off hoặc bytetrack.")
        if self.tracking != "off" and (self.mode != "full" or self.baseline):
            raise ValueError("Tracking chỉ dùng với full, không dùng trong fast/baseline.")
        for name in ("track_max_lost_seconds", "track_gap_max_ms", "track_smooth_alpha"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} phải hữu hạn.")
        if (
            self.track_max_lost_seconds < 0
            or self.track_gap_max_ms < 0
            or not 0 < self.track_smooth_alpha <= 1
        ):
            raise ValueError("Thời gian tracking không âm; track-smooth-alpha trong (0,1].")


class LatestMailbox:
    """One pending task; taking it transfers ownership to the worker."""

    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.finished = False
        self.replaced = 0

    def offer(self, task):
        with self.condition:
            if self.finished:
                return
            if self.pending is not None:
                self.replaced += 1
            self.pending = task
            self.condition.notify_all()

    def take(self, stop):
        with self.condition:
            while self.pending is None and not self.finished and not stop.is_set():
                self.condition.wait(0.05)
            if stop.is_set():
                return None
            task, self.pending = self.pending, None
            return task

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()


class FrameBuffer:
    def __init__(self, max_frames=12, max_bytes=128 * 1024 * 1024):
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.frames = deque()
        self.condition = threading.Condition()
        self.bytes_used = 0
        self.peak_bytes = 0
        self.peak_frames = 0
        self.finished = False

    def put(self, frame, stop, playing=None):
        size = frame.image.nbytes
        if size > self.max_bytes:
            raise ValueError("Một frame vượt giới hạn bộ đệm 128 MiB.")
        with self.condition:
            while not stop.is_set():
                paused = playing is not None and not playing.is_set()
                if (
                    not paused
                    and len(self.frames) < self.max_frames
                    and self.bytes_used + size <= self.max_bytes
                ):
                    self.frames.append(frame)
                    self.bytes_used += size
                    self.peak_bytes = max(self.peak_bytes, self.bytes_used)
                    self.peak_frames = max(self.peak_frames, len(self.frames))
                    self.condition.notify_all()
                    return True
                self.condition.wait(0.02)
        return False

    def get(self, timeout=0.02):
        with self.condition:
            if not self.frames and not self.finished:
                self.condition.wait(timeout)
            if not self.frames:
                return None
            frame = self.frames.popleft()
            self.bytes_used -= frame.image.nbytes
            self.condition.notify_all()
            return frame

    def ready(self, count):
        with self.condition:
            return len(self.frames) >= count or self.finished

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()


class Results:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = OrderedDict()
        self.latest_shown = None

    def publish(self, result):
        with self.lock:
            self.rows[result["frame_index"]] = result
            while len(self.rows) > 64:
                self.rows.popitem(last=False)

    def at(self, frame_index):
        with self.lock:
            exact = self.rows.get(frame_index)
            eligible = [key for key in self.rows if key <= frame_index]
            for key in eligible:
                self.latest_shown = self.rows.pop(key)
            return exact, self.latest_shown


class TimeSampler:
    def __init__(self, fps: float):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("sample-fps phải hữu hạn và lớn hơn 0.")
        self.fps = fps
        self.next_tick = 0

    def select(self, timestamp: float) -> bool:
        if timestamp + 1e-8 < self.next_tick / self.fps:
            return False
        self.next_tick = math.floor((timestamp + 1e-8) * self.fps) + 1
        return True


class PlaybackClock:
    """Wall clock keeps original media speed between explicit pauses/stalls."""

    def __init__(self):
        self.origin = None
        self.suspended_at = None
        self.suspensions = set()

    def start(self, wall_time, media_time):
        self.origin = wall_time - media_time

    def deadline(self, media_time):
        return self.origin + media_time

    def suspend(self, now, reason="pause"):
        self.suspensions.add(reason)
        if self.suspended_at is None:
            self.suspended_at = now

    def resume(self, now, reason="pause"):
        self.suspensions.discard(reason)
        if self.suspended_at is not None and not self.suspensions:
            if self.origin is not None:
                self.origin += now - self.suspended_at
            self.suspended_at = None
