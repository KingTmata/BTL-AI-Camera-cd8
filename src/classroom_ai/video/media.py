"""Local CFR validation and incremental MP4 export using installed FFmpeg."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2

from classroom_ai.video.core import Frame

CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


def executable(name):
    path = shutil.which(name)
    if not path:
        raise ValueError(f"Cần {name} trong PATH. Cài FFmpeg gồm ffmpeg và ffprobe.")
    return path


def validate_timestamps(timestamps, fps, tolerance=None, on_progress=None):
    period = 1 / float(fps)
    tolerance = tolerance if tolerance is not None else max(0.0001, period * 0.01)
    first = None
    previous = None
    count = 0
    for count, timestamp in enumerate(timestamps, 1):
        if not math.isfinite(timestamp):
            raise ValueError("Timestamp video không hợp lệ.")
        if first is None:
            first = timestamp
        if previous is not None and timestamp <= previous:
            raise ValueError("Video VFR hoặc timestamp trùng/đảo thứ tự chưa được hỗ trợ.")
        if abs(timestamp - first - (count - 1) * period) > tolerance:
            raise ValueError(
                "Video VFR hoặc timestamp gián đoạn chưa được hỗ trợ; hãy cung cấp video CFR."
            )
        previous = timestamp
        if on_progress is not None:
            on_progress(count)
    if not count:
        raise ValueError("Không đọc được timestamp/frame video.")
    return count


@dataclass(frozen=True)
class VideoSource:
    path: Path
    fps: Fraction
    width: int
    height: int
    frame_count: int
    audio: bool
    start_time: float = 0.0
    audio_offset_seconds: float = 0.0

    @property
    def duration(self):
        return self.frame_count / float(self.fps)

    def metadata(self):
        return {
            "path": str(self.path),
            "fps_fraction": str(self.fps),
            "fps": float(self.fps),
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration,
            "has_audio": self.audio,
            "video_start_time": self.start_time,
            "audio_offset_seconds": self.audio_offset_seconds,
        }


def probe_source(path: Path, *, scan_timestamps=True, on_progress=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Không thấy file video local: {path}")
    ffprobe = executable("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        creationflags=CREATE_FLAGS,
    )
    if result.returncode:
        raise ValueError(f"Không mở được video: {result.stderr[-1500:]}")
    streams = json.loads(result.stdout).get("streams", [])
    stream = next(
        (
            s
            for s in streams
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if stream is None:
        raise ValueError("Nguồn không có video stream.")
    try:
        fps = Fraction(stream["avg_frame_rate"])
        nominal = Fraction(stream["r_frame_rate"])
        tick = float(Fraction(stream["time_base"]))
        width, height = int(stream["width"]), int(stream["height"])
        if (
            fps <= 0
            or nominal <= 0
            or width <= 0
            or height <= 0
            or abs(float(fps / nominal) - 1) > 0.001
        ):
            raise ValueError()
    except (KeyError, ValueError, ZeroDivisionError):
        raise ValueError("Video VFR hoặc FPS/kích thước không hợp lệ; cần nguồn CFR.") from None
    count = (
        int(stream.get("nb_frames", 0) or 0) if str(stream.get("nb_frames", "")).isdigit() else 0
    )
    if scan_timestamps:
        # Scan incrementally: strict CFR detection without retaining long-video timestamps.
        command = [
            ffprobe,
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ]
        errors = deque(maxlen=30)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_FLAGS,
        )

        def drain_errors():
            for line in process.stderr:
                errors.append(line)

        drainer = threading.Thread(target=drain_errors, name="ffprobe-errors")
        drainer.start()
        try:

            def timestamps():
                for line in process.stdout:
                    token = line.split(",")[0].strip()
                    if token:
                        yield float(token)

            def scanned_progress(number):
                if on_progress:
                    on_progress(number, count or None)

            period = 1 / float(fps)
            tolerance = max(0.00001, min(2 * tick, 0.1 * period), 0.01 * period)
            scanned = validate_timestamps(timestamps(), fps, tolerance, scanned_progress)
            code = process.wait(timeout=30)
            drainer.join()
            error = "".join(errors).strip()
            if code or error:
                raise ValueError(f"Video có lỗi giải mã: {error[-1500:]}")
            if count and count != scanned:
                raise ValueError(f"Số frame metadata {count} khác số frame đọc được {scanned}.")
            count = scanned
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            drainer.join()
            process.stdout.close()
            process.stderr.close()
    if count <= 0:
        raise ValueError("Không xác định được số frame video.")
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    start = float(stream.get("start_time", 0))
    offset = float(audio.get("start_time", start)) - start if audio else 0.0
    return VideoSource(path, fps, width, height, count, audio is not None, start, offset)


def frames(source, metrics, stop, playing=None, *, counter="frames_decoded", decode_stage="decode"):
    capture = cv2.VideoCapture(str(source.path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV không mở được video: {source.path}")
        # FFprobe dimensions refer to the encoded raster; do not silently autorotate it.
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        index = 0
        while not stop.is_set():
            if playing is not None and not playing.wait(0.05):
                continue
            started = time.perf_counter()
            ok, image = capture.read()
            metrics.observe(decode_stage, (time.perf_counter() - started) * 1000)
            if not ok:
                if index != source.frame_count:
                    raise ValueError(f"Video bị ngắt: đọc {index}/{source.frame_count} frame.")
                break
            if image.shape[:2] != (source.height, source.width):
                raise ValueError("Kích thước frame giải mã không khớp metadata.")
            metrics.count(counter)
            yield Frame(index, index / float(source.fps), image)
            index += 1
    finally:
        capture.release()


class VideoWriter:
    def __init__(self, source, run_dir):
        self.source = source
        self.temporary = run_dir / "annotated.partial.mp4"
        self.destination = run_dir / "annotated.mp4"
        self.count = 0
        ffmpeg = executable("ffmpeg")
        self.errors = (run_dir / "encoder.log").open("xb")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{source.width}x{source.height}",
            "-framerate",
            str(source.fps),
            "-i",
            "pipe:0",
        ]
        if source.audio:
            command += ["-protocol_whitelist", "file,pipe", "-i", str(source.path)]
        command += ["-map", "0:v:0"]
        if source.audio:
            trim = max(0.0, -source.audio_offset_seconds)
            delay_ms = max(0.0, source.audio_offset_seconds) * 1000
            audio_filter = f"asetpts=PTS-STARTPTS,atrim=start={trim},asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1"
            command += ["-map", "1:a:0", "-af", audio_filter, "-c:a", "aac"]
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(source.duration),
            "-movflags",
            "+faststart",
            str(self.temporary),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.errors,
                creationflags=CREATE_FLAGS,
            )
        except BaseException:
            self.errors.close()
            raise

    def write(self, image):
        try:
            self.process.stdin.write(image.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                "FFmpeg không ghi được video; xem encoder.log (codec/ổ đĩa)."
            ) from exc
        self.count += 1

    def finish(self):
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=120)
            if code:
                raise RuntimeError("FFmpeg xuất video thất bại; xem encoder.log.")
            output = probe_source(self.temporary)
            if (
                self.count != self.source.frame_count
                or output.frame_count != self.count
                or output.fps != self.source.fps
                or output.audio != self.source.audio
                or abs(output.duration - self.source.duration) > 1 / float(self.source.fps)
            ):
                raise RuntimeError(
                    "Video kết quả không đạt kiểm tra số frame/FPS/thời lượng/âm thanh."
                )
            if self.destination.exists():
                raise FileExistsError(self.destination)
            self.temporary.rename(self.destination)
            return self.destination
        finally:
            self.close()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        try:
            if not self.process.stdin.closed:
                self.process.stdin.close()
        except BrokenPipeError:
            # Preserve the original encoder/decode error during cleanup.
            pass
        finally:
            self.errors.close()
