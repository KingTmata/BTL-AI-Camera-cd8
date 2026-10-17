"""metrics for the native video pipeline."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque

import numpy as np
import psutil


class Metrics:
    """Lifetime counters plus bounded stage distributions and resource samples."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.perf_counter()
        self.counts = {}
        self.stages = {}
        self.resources = deque(maxlen=3600)
        self.values = {}

    def count(self, key, n=1):
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + n

    def observe(self, key, milliseconds):
        with self.lock:
            stage = self.stages.setdefault(
                key, {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "recent": deque(maxlen=2000)}
            )
            stage["count"] += 1
            stage["total_ms"] += milliseconds
            stage["max_ms"] = max(stage["max_ms"], milliseconds)
            stage["recent"].append(milliseconds)

    def report(self):
        with self.lock:
            stages = {}
            for key, stage in self.stages.items():
                stages[key] = {k: v for k, v in stage.items() if k != "recent"}
                stages[key].update(
                    {f"p{p}_ms": float(np.percentile(stage["recent"], p)) for p in (50, 95)}
                )
                stages[key]["percentile_window_samples"] = len(stage["recent"])
            return {
                "elapsed_seconds": time.perf_counter() - self.started,
                "counts": dict(self.counts),
                "stages": stages,
                **self.values,
                "resource_samples": list(self.resources),
            }


class Progress:
    """Stage labels have elapsed cost, not invented loading percentages."""

    def __init__(self, metrics, emit):
        self.metrics, self.emit = metrics, emit
        self.lock = threading.Lock()
        self.stage = "Preparing"
        self.started = time.perf_counter()
        self.last_emit = 0.0
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.initial_rss = self.process.memory_info().rss
        self.peak_rss = self.initial_rss
        self.metrics.values["lifecycle"] = []
        self.notice = None

    def set(self, stage):
        with self.lock:
            self.stage, self.started = stage, time.perf_counter()
            self.metrics.values["lifecycle"].append(
                {"stage": stage, "cold_elapsed_ms": (self.started - self.metrics.started) * 1000}
            )

    def model_ready(self, load_ms, warmup_ms):
        rss = self.process.memory_info().rss
        self.metrics.values["model_ready_rss_bytes"] = rss
        with self.lock:
            self.notice = f"MODEL READY: load {load_ms:.0f}ms + warmup {warmup_ms:.0f}ms | RAM {rss / 1024**2:.1f} MiB"

    def tick(self, source=None, display=None):
        now = time.perf_counter()
        if now - self.last_emit < 1:
            return
        self.last_emit = now
        with self.lock:
            stage, elapsed = self.stage, now - self.started
            notice, self.notice = self.notice, None
        data = self.metrics.report()
        counts = data["counts"]
        processed = counts.get("frames_inferred", 0)
        text = f"{stage} | stage {elapsed:.1f}s | total {data['elapsed_seconds']:.1f}s"
        if source:
            if stage in (
                "Processing full video",
                "Detecting and tracking",
                "Rendering tracked video",
            ):
                if stage == "Rendering tracked video":
                    processed = counts.get("frames_encoded", 0)
                pct = min(100.0, 100 * processed / source.frame_count)
                eta = elapsed / processed * (source.frame_count - processed) if processed else None
                text += f" | {processed}/{source.frame_count} ({pct:.1f}%)"
                if eta is not None:
                    text += f" | ETA {eta:.1f}s"
            else:
                consumed = counts.get("frames_consumed", 0)
                text += f" | played {consumed}/{source.frame_count} | YOLO {processed}"
        process = self.process
        rss = process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        sample = {
            "elapsed_seconds": data["elapsed_seconds"],
            "rss_bytes": rss,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
        }
        try:
            torch = sys.modules.get("torch")
            loading = getattr(getattr(torch, "__spec__", None), "_initializing", False)
            cuda = getattr(torch, "cuda", None) if not loading else None
            if cuda is not None and cuda.is_initialized():
                device = self.metrics.values.get("resolved_device", "cpu")
                if device.startswith("cuda:"):
                    sample.update(
                        vram_allocated_bytes=cuda.memory_allocated(device),
                        vram_reserved_bytes=cuda.memory_reserved(device),
                    )
        except ImportError:
            pass
        with self.metrics.lock:
            self.metrics.resources.append(sample)
        text += f" | RAM {rss / 1024**2:.1f} MiB"
        if notice:
            text += f" | {notice}"
        if "vram_allocated_bytes" in sample:
            text += f" | VRAM {sample['vram_allocated_bytes'] / 1024**2:.1f} MiB"
        self.emit(text)
        if display and not counts.get("frames_consumed"):
            display.status(text)

    def preflight(self, count, total):
        now = time.perf_counter()
        if now - self.last_emit < 1:
            return
        self.last_emit = now
        fraction = (
            f"{count}/{total} ({min(100, count / total * 100):.1f}%)"
            if total
            else f"{count} frames scanned"
        )
        self.emit(f"Validating CFR | {fraction} | elapsed {now - self.started:.1f}s")
