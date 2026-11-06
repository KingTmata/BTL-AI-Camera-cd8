"""Local timestamp validation and incremental CFR/VFR MP4 export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import cv2

from classroom_ai.video.core import Frame

CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


class MediaCancelled(RuntimeError):
    """Cooperative media cancellation, caught by the owning worker."""


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise MediaCancelled("Đã hủy xử lý media.")


@contextmanager
def _process(command, *, cancel=None, **kwargs):
    """Own a child; a 50 ms watchdog also interrupts blocked pipe reads.

    Cancellation callbacks return bool and must be safe to read from this thread.
    This interrupts the child, rather than waiting for it to emit another line.
    """
    _check_cancel(cancel)
    with subprocess.Popen(command, **kwargs) as process:
        done = threading.Event()

        def watch():
            while not done.wait(0.05):
                if cancel() and process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    return

        watcher = threading.Thread(target=watch, name="media-cancel") if cancel else None
        if watcher:
            watcher.start()
        try:
            yield process
        except Exception:
            _check_cancel(cancel)
            raise
        else:
            _check_cancel(cancel)
        finally:
            done.set()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if watcher:
                watcher.join(timeout=1)


def _wait_process(process, *, cancel=None, timeout=120):
    deadline = time.monotonic() + timeout
    while True:
        _check_cancel(cancel)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(0.05, remaining))
        except subprocess.TimeoutExpired:
            continue


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
    time_base: str = "1/1"
    vfr: bool = False
    duration_seconds: float | None = None
    sample_aspect_ratio: str = "1:1"
    rotation_degrees: float = 0.0
    audio_duration_seconds: float | None = None

    @property
    def duration(self):
        return (
            self.duration_seconds
            if self.duration_seconds is not None
            else self.frame_count / float(self.fps)
        )

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
            "time_base": self.time_base,
            "fps_kind": "average" if self.vfr else "constant",
            "variable_frame_rate": self.vfr,
            "timing": "FFprobe best_effort_timestamp; relative to first decoded PTS",
        }


def probe_source(
    path: Path, *, scan_timestamps=True, on_progress=None, allow_vfr=False, cancel=None
):
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
    with _process(
        command,
        cancel=cancel,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_FLAGS,
    ) as process:
        stdout, stderr = process.communicate(timeout=30)
        _check_cancel(cancel)
        if process.returncode:
            raise ValueError(f"Không mở được video: {stderr[-1500:]}")
    streams = json.loads(stdout).get("streams", [])
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
            or (not allow_vfr and abs(float(fps / nominal) - 1) > 0.001)
        ):
            raise ValueError()
    except (KeyError, ValueError, ZeroDivisionError):
        raise ValueError("Video VFR hoặc FPS/kích thước không hợp lệ; cần nguồn CFR.") from None
    count = (
        int(stream.get("nb_frames", 0) or 0) if str(stream.get("nb_frames", "")).isdigit() else 0
    )
    vfr = abs(float(fps / nominal) - 1) > 0.001
    first_pts = last_pts = last_duration = None
    if allow_vfr and not scan_timestamps:
        raise ValueError("VFR cần quét PTS để kiểm tra thứ tự và thời lượng.")
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
            "frame=best_effort_timestamp,pkt_duration",
            "-of",
            "csv=p=0",
            str(path),
        ]
        errors = deque(maxlen=30)
        with _process(
            command,
            cancel=cancel,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_FLAGS,
        ) as process:

            def drain_errors():
                for line in process.stderr:
                    errors.append(line)

            drainer = threading.Thread(target=drain_errors, name="ffprobe-errors")
            drainer.start()
            try:

                def timestamps():
                    nonlocal first_pts, last_pts, last_duration, vfr
                    for line in process.stdout:
                        _check_cancel(cancel)
                        fields = line.strip().split(",")
                        token = fields[0].strip()
                        if token:
                            try:
                                pts = int(token)
                            except ValueError:
                                raise ValueError("Video thiếu PTS nguyên hợp lệ.") from None
                            if first_pts is None:
                                first_pts = pts
                            if last_pts is not None and pts <= last_pts:
                                raise ValueError("Video có timestamp trùng/đảo thứ tự.")
                            last_pts = pts
                            last_duration = (
                                int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else None
                            )
                            yield float(pts * Fraction(stream["time_base"]))

                def scanned_progress(number):
                    if on_progress:
                        on_progress(number, count or None)

                period = 1 / float(fps)
                tolerance = max(0.00001, min(2 * tick, 0.1 * period), 0.01 * period)
                if allow_vfr:
                    first = None
                    scanned = 0
                    for scanned, timestamp in enumerate(timestamps(), 1):
                        if first is None:
                            first = timestamp
                        if abs(timestamp - first - (scanned - 1) * period) > tolerance:
                            vfr = True
                        scanned_progress(scanned)
                    if not scanned:
                        raise ValueError("Không đọc được timestamp/frame video.")
                else:
                    scanned = validate_timestamps(timestamps(), fps, tolerance, scanned_progress)
                code = process.wait(timeout=30)
                _check_cancel(cancel)
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
    start = (
        float(first_pts * Fraction(stream["time_base"]))
        if first_pts is not None
        else float(stream.get("start_time", 0))
    )
    offset = float(audio.get("start_time", start)) - start if audio else 0.0
    audio_duration = None
    if audio:
        try:
            audio_duration = float(int(audio["duration_ts"]) * Fraction(audio["time_base"]))
        except (KeyError, ValueError, TypeError):
            audio_duration = float(audio.get("duration", 0) or 0)
        if not math.isfinite(audio_duration) or audio_duration <= 0:
            audio_duration = None
    try:
        duration = float(int(stream["duration_ts"]) * Fraction(stream["time_base"]))
    except (KeyError, ValueError, TypeError):
        duration = float(stream.get("duration", 0) or 0)
    if first_pts is not None and last_pts is not None:
        last_relative = float((last_pts - first_pts) * Fraction(stream["time_base"]))
        if duration <= last_relative:
            if not last_duration:
                raise ValueError("Không xác định được thời lượng frame cuối video.")
            duration = float((last_pts - first_pts + last_duration) * Fraction(stream["time_base"]))
    if not math.isfinite(duration) or duration <= 0:
        if vfr:
            raise ValueError("Không xác định được thời lượng video VFR.")
        duration = count / float(fps)
    return VideoSource(
        path,
        fps,
        width,
        height,
        count,
        audio is not None,
        start,
        offset,
        stream["time_base"],
        vfr,
        duration,
        stream.get("sample_aspect_ratio", "1:1"),
        float(
            next(
                (
                    side["rotation"]
                    for side in stream.get("side_data_list", [])
                    if "rotation" in side
                ),
                stream.get("tags", {}).get("rotate", 0),
            )
        ),
        audio_duration,
    )


def source_pts(source, *, cancel=None):
    """Stream original presentation timestamps; retain no long-video array."""
    command = [
        executable("ffprobe"),
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp",
        "-of",
        "csv=p=0",
        str(source.path),
    ]
    with _process(
        command,
        cancel=cancel,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=CREATE_FLAGS,
    ) as process:
        try:
            for line in process.stdout:
                _check_cancel(cancel)
                token = line.split(",")[0].strip()
                if token:
                    yield int(token)
            _check_cancel(cancel)
            if process.wait(timeout=30):
                raise ValueError("FFprobe không đọc được PTS nguồn.")
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()


def decoder_info(decoder):
    """Describe an explicit decoder; dependency errors never trigger installation."""
    if decoder == "opencv":
        return {"backend": "opencv", "version": cv2.__version__, "pts_backend": "ffprobe"}
    if decoder != "pyav":
        raise ValueError("decoder phải là opencv hoặc pyav.")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu PyAV (av); cài requirements/web.txt để dùng decoder pyav."
        ) from exc
    return {
        "backend": "pyav",
        "version": av.__version__,
        "pts_backend": "native VideoFrame.pts/time_base, one software decoder",
        "thread_type": "AUTO",
        "thread_count": min(4, max(1, os.cpu_count() or 1)),
    }


def _frames_pyav(source, metrics, stop, playing, *, counter, decode_stage, cancel):
    """Pixels and PTS from one native decode; never infer time from average FPS.

    Frame threading is bounded to four software threads. Native decode/color
    conversion checks cancellation when it returns. No rotation filter is used;
    conversion retains the encoded raster rather than display orientation.
    """
    info = decoder_info("pyav")
    import av

    _check_cancel(cancel)
    if stop.is_set():
        return
    with av.open(
        str(source.path), mode="r", options={"protocol_whitelist": "file,pipe"}, timeout=(5.0, 5.0)
    ) as container:
        _check_cancel(cancel)
        stream = next(
            (
                s
                for s in container.streams.video
                if not s.disposition & av.stream.Disposition.attached_pic
            ),
            None,
        )
        if stream is None:
            raise ValueError("PyAV không tìm thấy video stream.")
        stream.thread_type = info["thread_type"]
        stream.codec_context.thread_count = info["thread_count"]
        decoded = iter(container.decode(stream))
        first_pts, previous_pts, index = None, None, 0
        source_base = Fraction(source.time_base)
        while not stop.is_set():
            _check_cancel(cancel)
            if playing is not None and not playing.wait(0.05):
                continue
            started = time.perf_counter()
            try:
                native = next(decoded)
            except StopIteration:
                if index != source.frame_count:
                    raise ValueError(
                        f"Video bị ngắt: đọc {index}/{source.frame_count} frame."
                    ) from None
                break
            _check_cancel(cancel)
            if stop.is_set():
                break
            if index >= source.frame_count:
                raise ValueError("Số frame PyAV vượt số PTS nguồn đã kiểm tra.")
            if native.pts is None or native.time_base is None:
                raise ValueError("PyAV thiếu native PTS/time_base; không đoán timestamp từ FPS.")
            native_base = Fraction(native.time_base)
            if native_base <= 0 or source_base <= 0:
                raise ValueError("PTS time_base phải dương.")
            converted = Fraction(native.pts) * native_base / source_base
            if converted.denominator != 1:
                raise ValueError("Native PTS không biểu diễn nguyên trong source time_base.")
            pts = int(converted)
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError("Native PTS PyAV trùng hoặc đảo thứ tự.")
            if (native.height, native.width) != (source.height, source.width):
                raise ValueError("Kích thước frame PyAV không khớp metadata raster nguồn.")
            image = native.to_ndarray(format="bgr24")
            _check_cancel(cancel)
            if stop.is_set():
                break
            if image.shape != (source.height, source.width, 3):
                raise ValueError("Kích thước ảnh PyAV không khớp metadata raster nguồn.")
            metrics.observe(decode_stage, (time.perf_counter() - started) * 1000)
            metrics.count(counter)
            if first_pts is None:
                first_pts = pts
            previous_pts = pts
            yield Frame(index, float((pts - first_pts) * source_base), image, pts, source.time_base)
            index += 1


def frames(
    source,
    metrics,
    stop,
    playing=None,
    *,
    counter="frames_decoded",
    decode_stage="decode",
    cancel=None,
    decoder="opencv",
):
    _check_cancel(cancel)
    if decoder == "pyav":
        yield from _frames_pyav(
            source,
            metrics,
            stop,
            playing,
            counter=counter,
            decode_stage=decode_stage,
            cancel=cancel,
        )
        return
    if decoder != "opencv":
        raise ValueError("decoder phải là opencv hoặc pyav.")
    options = (
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
        if cancel
        else []
    )
    capture = cv2.VideoCapture(str(source.path), cv2.CAP_FFMPEG, options)
    timestamps = source_pts(source, cancel=cancel)
    first_pts = None
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV không mở được video: {source.path}")
        # FFprobe dimensions refer to the encoded raster; do not silently autorotate it.
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        index = 0
        while not stop.is_set():
            _check_cancel(cancel)
            if playing is not None and not playing.wait(0.05):
                continue
            started = time.perf_counter()
            ok, image = capture.read()
            _check_cancel(cancel)
            metrics.observe(decode_stage, (time.perf_counter() - started) * 1000)
            if not ok:
                if index != source.frame_count:
                    raise ValueError(f"Video bị ngắt: đọc {index}/{source.frame_count} frame.")
                break
            if image.shape[:2] != (source.height, source.width):
                raise ValueError("Kích thước frame giải mã không khớp metadata.")
            metrics.count(counter)
            try:
                pts = next(timestamps)
            except StopIteration:
                raise ValueError("Thiếu PTS so với frame giải mã.") from None
            if first_pts is None:
                first_pts = pts
            timestamp = float((pts - first_pts) * Fraction(source.time_base))
            yield Frame(index, timestamp, image, pts, source.time_base)
            index += 1
    finally:
        capture.release()
        timestamps.close()


class VideoWriter:
    def __init__(self, source, run_dir, *, expected_timeline=None):
        self.source = source
        self.expected_timeline = Path(expected_timeline) if expected_timeline is not None else None
        if self.expected_timeline is not None and not source.vfr:
            raise ValueError("Expected timeline cần đường encode native PTS (vfr=True).")
        self.temporary = run_dir / "annotated.partial.mp4"
        self.destination = run_dir / "annotated.mp4"
        self.count = 0
        self.process = None
        self.container = None
        self.errors = None
        self.pending = None
        self.packet_durations = {}
        self.first_pts = self.last_pts = None
        self.closed = False
        if source.vfr:
            try:
                import av
            except ImportError:
                raise ValueError(
                    "Xuất video VFR cần PyAV: cài av từ requirements/web.txt."
                ) from None
            self.av = av
            self.video_temporary = run_dir / "annotated.video.partial.mp4"
            if (
                self.video_temporary.exists()
                or self.temporary.exists()
                or self.destination.exists()
            ):
                raise FileExistsError("File kết quả đã tồn tại.")
            try:
                self.container = av.open(str(self.video_temporary), mode="w", format="mp4")
                self.stream = self.container.add_stream("libx264", rate=source.fps)
                self.stream.width = (source.width + 1) // 2 * 2
                self.stream.height = (source.height + 1) // 2 * 2
                self.stream.pix_fmt = "yuv420p"
                self.stream.time_base = Fraction(source.time_base)
                self.stream.codec_context.time_base = Fraction(source.time_base)
                self.stream.sample_aspect_ratio = Fraction(1, 1)
                self.stream.codec_context.sample_aspect_ratio = Fraction(1, 1)
                # MP4 stores durations in DTS order. Reordered B-frame packets
                # can shift a quantized final extent by one tick despite exact
                # PTS. Keep this exact-timestamp path in presentation order.
                self.stream.codec_context.max_b_frames = 0
                self.stream.options = {"preset": "veryfast", "crf": "20"}
            except BaseException:
                self.close()
                raise
            return
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
            command += ["-protocol_whitelist", "file,pipe", "-noautorotate", "-i", str(source.path)]
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

    def write(self, image, *, frame=None):
        if self.closed:
            raise RuntimeError("VideoWriter đã đóng.")
        if image.shape[:2] != (self.source.height, self.source.width):
            raise ValueError("Kích thước frame ghi không khớp nguồn.")
        if self.source.vfr:
            if frame is None or frame.source_pts is None or frame.time_base is None:
                raise ValueError("Xuất VFR cần frame chứa source PTS và time_base.")
            if Fraction(frame.time_base) != Fraction(self.source.time_base):
                raise ValueError("Time base frame không khớp nguồn.")
            if frame.frame_index != self.count:
                raise ValueError("Xuất VFR cần mọi frame đúng thứ tự.")
            pts = frame.source_pts
            if self.last_pts is not None and pts <= self.last_pts:
                raise ValueError("PTS frame ghi phải tăng nghiêm ngặt.")
            if self.first_pts is None:
                self.first_pts = pts
            if self.pending is not None:
                self._encode_pending(pts - self.last_pts)
            # Own one frame while waiting for its actual presentation duration.
            self.pending = image.copy()
            self.last_pts = pts
            self.count += 1
            return
        try:
            self.process.stdin.write(image.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                "FFmpeg không ghi được video; xem encoder.log (codec/ổ đĩa)."
            ) from exc
        self.count += 1

    def _encode_pending(self, duration):
        pts = self.last_pts - self.first_pts
        image = self.pending
        if image.shape[:2] != (self.stream.height, self.stream.width):
            image = cv2.copyMakeBorder(
                image,
                0,
                self.stream.height - image.shape[0],
                0,
                self.stream.width - image.shape[1],
                cv2.BORDER_CONSTANT,
            )
        video_frame = self.av.VideoFrame.from_ndarray(image, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = Fraction(self.source.time_base)
        self.packet_durations[pts] = duration
        for packet in self.stream.encode(video_frame):
            self._mux_packet(packet)
        self.pending = None

    def _mux_packet(self, packet):
        # Encoder output is in codec time base; MP4 may choose another track base.
        key = int(packet.pts * packet.time_base / Fraction(self.source.time_base))
        duration = self.packet_durations.pop(key)
        packet.duration = int(duration * Fraction(self.source.time_base) / packet.time_base)
        self.container.mux(packet)

    def _finish_vfr(self, *, cancel=None):
        _check_cancel(cancel)
        if self.count != self.source.frame_count or self.pending is None:
            raise RuntimeError("Xuất VFR thiếu frame nguồn.")
        last_relative = (self.last_pts - self.first_pts) * Fraction(self.source.time_base)
        final_duration = round(
            (Fraction(str(self.source.duration)) - last_relative) / Fraction(self.source.time_base)
        )
        if final_duration <= 0:
            raise RuntimeError("Thời lượng frame cuối VFR không hợp lệ.")
        self._encode_pending(final_duration)
        for packet in self.stream.encode():
            _check_cancel(cancel)
            self._mux_packet(packet)
        self.container.close()
        self.container = None
        command = [
            executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-protocol_whitelist",
            "file,pipe",
            "-noautorotate",
            "-i",
            str(self.video_temporary),
        ]
        if self.source.audio:
            command += [
                "-protocol_whitelist",
                "file,pipe",
                "-noautorotate",
                "-i",
                str(self.source.path),
            ]
        command += ["-map", "0:v:0", "-c:v", "copy"]
        if self.source.audio:
            trim = max(0.0, -self.source.audio_offset_seconds)
            delay_ms = max(0.0, self.source.audio_offset_seconds) * 1000
            audio_filter = f"asetpts=PTS-STARTPTS,atrim=start={trim},asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1"
            if self.expected_timeline is not None:
                audio_filter += (
                    f",apad=whole_dur={self.source.duration},atrim=duration={self.source.duration}"
                )
            command += [
                "-map",
                "1:a:0",
                "-af",
                audio_filter,
                "-c:a",
                "aac",
            ]
        command += ["-t", str(self.source.duration), "-movflags", "+faststart", str(self.temporary)]
        with (self.temporary.parent / "encoder.log").open("xb") as errors:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                creationflags=CREATE_FLAGS,
            )
            if _wait_process(self.process, cancel=cancel):
                raise RuntimeError("FFmpeg ghép âm thanh VFR thất bại; xem encoder.log.")
        self.video_temporary.unlink()

    def _validate_vfr_output(self, output, *, cancel=None):
        tolerance = max(Fraction(self.source.time_base), Fraction(output.time_base))
        left = (
            self._expected_pts(cancel=cancel)
            if self.expected_timeline is not None
            else source_pts(self.source, cancel=cancel)
        )
        right = source_pts(output, cancel=cancel)
        try:
            first_left = next(left)
            first_right = next(right)
            if first_right != 0:
                raise RuntimeError("PTS đầu kết quả VFR không bắt đầu tại 0.")
            for a, b in zip(left, right, strict=True):
                _check_cancel(cancel)
                original = (a - first_left) * Fraction(self.source.time_base)
                encoded = (b - first_right) * Fraction(output.time_base)
                if original != encoded:
                    raise RuntimeError("PTS kết quả VFR khác nguồn.")
        finally:
            left.close()
            right.close()
        if abs(output.duration - self.source.duration) > float(tolerance) * 2:
            raise RuntimeError("Thời lượng kết quả VFR khác nguồn.")

    def _expected_pts(self, *, cancel=None):
        """Stream the clip's durable mapping instead of re-reading full-source PTS."""
        cursor, count = 0, 0
        with self.expected_timeline.open(encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                _check_cancel(cancel)
                row = json.loads(line)
                pts, duration = row.get("output_pts"), row.get("duration_ticks")
                if (
                    row.get("output_frame_index") != count - 1
                    or isinstance(pts, bool)
                    or not isinstance(pts, int)
                    or pts != cursor
                    or isinstance(duration, bool)
                    or not isinstance(duration, int)
                    or duration <= 0
                    or Fraction(row["time_base"]) != Fraction(self.source.time_base)
                ):
                    raise ValueError("Timeline PTS/duration kết quả không hợp lệ.")
                cursor += duration
                yield pts
        expected_end = round(Fraction(str(self.source.duration)) / Fraction(self.source.time_base))
        if count != self.source.frame_count or cursor != expected_end:
            raise ValueError("Timeline thiếu frame hoặc thời lượng.")

    def finish(self, *, cancel=None):
        try:
            _check_cancel(cancel)
            if self.source.vfr:
                self._finish_vfr(cancel=cancel)
            else:
                self.process.stdin.close()
                code = _wait_process(self.process, cancel=cancel)
                if code:
                    raise RuntimeError("FFmpeg xuất video thất bại; xem encoder.log.")
            output = probe_source(
                self.temporary,
                allow_vfr=self.source.vfr,
                cancel=cancel,
                on_progress=lambda *_: _check_cancel(cancel),
            )
            if (
                self.count != self.source.frame_count
                or output.frame_count != self.count
                or (not self.source.vfr and output.fps != self.source.fps)
                or output.audio != self.source.audio
                or output.width != (self.source.width + 1) // 2 * 2
                or output.height != (self.source.height + 1) // 2 * 2
                or output.sample_aspect_ratio != "1:1"
                or output.rotation_degrees % 360 != 0
                or abs(output.duration - self.source.duration) > 1 / float(self.source.fps)
            ):
                raise RuntimeError(
                    "Video kết quả không đạt kiểm tra số frame/FPS/thời lượng/âm thanh."
                )
            if self.source.vfr:
                self._validate_vfr_output(output, cancel=cancel)
            _check_cancel(cancel)
            if self.destination.exists():
                raise FileExistsError(self.destination)
            self.temporary.rename(self.destination)
            return self.destination
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pending = None
        if self.container is not None:
            self.container.close()
            self.container = None
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        try:
            if (
                self.process is not None
                and self.process.stdin is not None
                and not self.process.stdin.closed
            ):
                self.process.stdin.close()
        except BrokenPipeError:
            # Preserve the original encoder/decode error during cleanup.
            pass
        finally:
            if self.errors is not None:
                self.errors.close()


def prepare_annotation_source(source, destination_dir, *, cancel=None, on_progress=None) -> Path:
    """Prepare native-PTS media for manual annotation, without model inference.

    Returns a NEW directory containing a verified preview and source-frame
    index, never predictions/GT. Source partition0 is an unsegmented container
    range, not a semantic scene annotation. Empty model arrays assert no model
    ran; they never assert absence of people/actions. All source pixels stay in
    the encoded raster coordinate system, matching the existing preview contract.

    Progress dict keys are phase, frames_written, frames_total, source_time_s.
    Unknown total/time are None. Every callback and cancellation boundary is
    before publication (including complete); successful return confirms commit.
    Native decode/encode checks cancellation when its call returns. FFmpeg and
    ffprobe reuse the cancellable process watchdogs in this module.
    """
    from classroom_ai.video.metrics import Metrics

    _check_cancel(cancel)
    source_path, destination = Path(source).resolve(), Path(destination_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not source_path.is_file():
        raise ValueError("Không thấy video nguồn local.")
    written, total, source_time = 0, None, None
    started = time.monotonic()

    def progress(phase):
        _check_cancel(cancel)
        if on_progress is not None:
            on_progress(
                dict(
                    phase=phase,
                    frames_written=written,
                    frames_total=total,
                    source_time_s=source_time,
                )
            )
        _check_cancel(cancel)

    def source_hash():
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                _check_cancel(cancel)
                digest.update(block)
        return digest.hexdigest()

    progress("hashing")
    digest = source_hash()
    progress("probing")
    original = probe_source(
        source_path, allow_vfr=True, cancel=cancel, on_progress=lambda *_: progress("probing")
    )
    total = original.frame_count
    if total <= 0 or original.width * original.height * 3 > 128 * 1024 * 1024:
        raise ValueError("Số frame không hợp lệ hoặc một frame vượt giới hạn 128 MiB.")
    base = Fraction(original.time_base)
    run_id = uuid.uuid4().hex
    identity = dict(
        schema_version=2,
        kind="annotation_source_preparation",
        processing_mode="annotation_prep",
        run_id=run_id,
        actor_namespace="run:" + run_id,
    )
    settings = dict(
        source=str(source_path), mode="annotation", analysis_task="annotation", tracking="off"
    )
    partition = dict(
        segment_id=0,
        start_s=0.0,
        end_s=original.duration,
        meaning="unsegmented_source_range_not_semantic_scene_gt",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".annotation-source-", dir=destination.parent
    ) as temporary:
        private = Path(temporary)
        writer, decoded = None, None
        first_pts, previous_pts = None, None
        try:
            writer = VideoWriter(replace(original, vfr=True), private)
            decoded = frames(original, Metrics(), threading.Event(), decoder="pyav", cancel=cancel)
            with (
                (private / "frames.jsonl").open("xb") as ledger,
                (private / "frames.idx").open("xb") as index,
                (private / "frames.times").open("xb") as timestamps,
            ):
                progress("decoding")
                for frame in decoded:
                    _check_cancel(cancel)
                    pts = frame.source_pts
                    if (
                        isinstance(frame.frame_index, bool)
                        or frame.frame_index != written
                        or written >= total
                        or isinstance(pts, bool)
                        or not isinstance(pts, int)
                        or frame.time_base is None
                        or Fraction(frame.time_base) != base
                        or previous_pts is not None
                        and pts <= previous_pts
                        or frame.image.shape != (original.height, original.width, 3)
                    ):
                        raise ValueError(
                            "Native frame index/PTS/time_base/raster nguồn không khớp."
                        )
                    first_pts = pts if first_pts is None else first_pts
                    relative = float((pts - first_pts) * base)
                    if (
                        isinstance(frame.timestamp_seconds, bool)
                        or not math.isfinite(frame.timestamp_seconds)
                        or abs(frame.timestamp_seconds - relative) > 1e-6
                        or not 0 <= relative < original.duration
                    ):
                        raise ValueError("Thời gian frame không khớp native PTS nguồn.")
                    row = dict(
                        schema_version=2,
                        kind="annotation_source_frame",
                        analysis_task="annotation",
                        observation="not_inferred",
                        frame_index=written,
                        source_pts=pts,
                        time_base=original.time_base,
                        timestamp_seconds=relative,
                        segment_id=0,
                        detections=[],
                        tracks=[],
                        display_tracks=[],
                    )
                    index.write(struct.pack("<Q", ledger.tell()))
                    timestamps.write(struct.pack("<d", relative))
                    ledger.write((json.dumps(row, allow_nan=False) + "\n").encode("utf-8"))
                    writer.write(frame.image, frame=frame)
                    written += 1
                    source_time, previous_pts = relative, pts
                    progress("decoding")
                if written != total:
                    raise ValueError("Số frame chuẩn bị không khớp nguồn đã probe.")
                for stream in (ledger, index, timestamps):
                    stream.flush()
                    os.fsync(stream.fileno())
            progress("finalizing")
            completed = writer.finish(cancel=cancel)
            completed.rename(private / "preview.mp4")
        finally:
            if decoded is not None:
                decoded.close()
            if writer is not None:
                writer.close()
        (private / "tracks.jsonl").touch(exist_ok=False)
        inference = dict(status="not_run", frames_inferred=0)
        notes = (
            "Manual annotation source preparation only; no model/weights/inference, actors, "
            "events or reviewed coverage created. Empty model arrays are not negative GT. "
            "Partition0 is only an unsegmented source range. Preview retains native relative "
            "PTS and encoded-raster pixels; rotation/SAR metadata is normalized like preview."
        )
        metadata = {
            **identity,
            "source": original.metadata(),
            "source_sha256": digest,
            "config": settings,
            "inference": inference,
            "partition": partition,
            "notes": notes,
        }
        manifest = {
            **metadata,
            "status": "completed",
            "coverage": dict(frames_inferred=0, source_frames=total, actions_reviewed=False),
        }
        report = {
            **identity,
            "status": "completed",
            "source_sha256": digest,
            "inference": inference,
            "counts": dict(frames_decoded=written, frames_encoded=written, frames_inferred=0),
            "source": original.metadata(),
            "partition": partition,
            "ground_truth": False,
            "preview": dict(
                native_pts_verified=True,
                rotation_degrees=0,
                sample_aspect_ratio="1:1",
                width=(original.width + 1) // 2 * 2,
                height=(original.height + 1) // 2 * 2,
                source_rotation_degrees=original.rotation_degrees,
                source_sample_aspect_ratio=original.sample_aspect_ratio,
            ),
            "elapsed_seconds": time.monotonic() - started,
            "notes": notes,
        }
        for name, value in (
            ("metadata.json", metadata),
            ("manifest.json", manifest),
            ("report.json", report),
        ):
            with (private / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        (private / "encoder.log").unlink(missing_ok=True)
        progress("verifying")
        progress("publishing")
        progress("complete")
        if source_hash() != digest:
            raise ValueError("Source changed / video nguồn thay đổi trong khi chuẩn bị.")
        _check_cancel(cancel)
        if destination.exists():
            raise FileExistsError(destination)
        private.rename(destination)
    return destination
