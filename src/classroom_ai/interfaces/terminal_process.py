"""Small subprocess controller: text/status only, never imports YOLO/OpenCV."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from classroom_ai.paths import PROJECT_ROOT as ROOT


def progress_fraction(line):
    if not line.startswith(
        (
            "Processing full video",
            "Detecting and tracking",
            "Rendering tracked video",
            "Playing ",
            "Playback baseline",
            "Validating CFR",
        )
    ):
        return None
    match = re.search(r"(\d+)/(\d+)", line)
    if match and int(match[2]) > 0:
        return min(int(match[1]), int(match[2])), int(match[2])
    return None


class BackendJob:
    def __init__(self, output_root=None):
        self.output_root = Path(output_root) if output_root else ROOT / "runs/video_tui"
        self.process = None
        self.directory = self.stop_file = None
        self.log = None
        self.offset = 0
        self.partial = b""

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    @property
    def returncode(self):
        return self.process.poll() if self.process is not None else None

    @property
    def result_dir(self):
        if self.directory is not None:
            report = next(self.directory.glob("*/report.json"), None)
            return report.parent if report else self.directory
        return None

    def start(self, source, mode, options=None):
        if self.running:
            raise ValueError("Backend đang chạy. Hãy dừng hoặc đợi hoàn tất.")
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Chọn một file video có thật.")
        if mode not in ("fast", "full"):
            raise ValueError("Chế độ phải là fast hoặc full.")
        self.close()
        self.directory = self.output_root.resolve() / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:10]
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stop_file = self.directory / "stop.flag"
        self.offset, self.partial = 0, b""
        command = [
            sys.executable,
            "-u",
            "-m",
            "classroom_ai.video",
            "--source",
            str(source),
            "--mode",
            mode,
            "--stop-file",
            str(self.stop_file),
            "--output-dir",
            str(self.directory),
        ]
        # Preserve normal CLI options without accepting command strings or a shell.
        allowed = {
            "weights",
            "device",
            "imgsz",
            "confidence",
            "nms_iou",
            "sample_fps",
            "buffer_ms",
            "no_display",
            "baseline",
        }
        allowed |= {"tracking", "track_max_lost_seconds", "track_gap_max_ms", "track_smooth_alpha"}
        for key, value in (options or {}).items():
            if key not in allowed or value is None or value is False:
                continue
            command.append("--" + key.replace("_", "-"))
            if value is not True:
                command.append(str(value))
        self.log = (self.directory / "console.log").open("xb")
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"}
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self.log.close()
            self.log = None
            raise

    def read_updates(self):
        if self.directory is None:
            return []
        with (self.directory / "console.log").open("rb") as stream:
            stream.seek(self.offset)
            chunk = stream.read(65536)
            self.offset += len(chunk)
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()[-65536:]
        if not self.running and self.partial:
            lines.append(self.partial)
            self.partial = b""
        return [line.decode("utf-8", "replace").rstrip("\r") for line in lines]

    def request_stop(self):
        if self.running:
            self.stop_file.touch(exist_ok=True)

    def force_stop(self):
        if self.running:
            import psutil

            try:
                parent = psutil.Process(self.process.pid)
                # Windows venv launchers can have a Python child, plus FFmpeg.
                for child in reversed(parent.children(recursive=True)):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                self.process.kill()
                self.process.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass

    def close(self):
        if self.running:
            self.force_stop()
        if self.log is not None:
            self.log.close()
            self.log = None
