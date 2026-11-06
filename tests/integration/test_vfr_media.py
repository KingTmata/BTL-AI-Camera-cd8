"""Real FFmpeg/PyAV variable-timestamp round trips; no detector or network."""

import json
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from classroom_ai.paths import PROJECT_ROOT
from classroom_ai.video.media import VideoWriter, executable, frames, probe_source, source_pts


class Metrics:
    def observe(self, *args):
        pass

    def count(self, *args):
        pass


@pytest.fixture
def tmp_path():
    # Use the writable workspace like the other video integration tests. Windows
    # sandbox temp permits writes but can reject atomic result publication.
    scratch = PROJECT_ROOT / ".temp" / "vfr_media_tests"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        yield Path(directory)


@pytest.fixture
def vfr(tmp_path):
    path = tmp_path / "VFR có dấu.mp4"
    subprocess.run(
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=65x49:rate=30000/1001:duration=1.2",
            "-itsoffset",
            "0.12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.08",
            "-vf",
            "select=not(eq(mod(n\\,5)\\,2))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-bf",
            "3",
            "-pix_fmt",
            "yuv444p",
            "-c:a",
            "aac",
            "-video_track_timescale",
            "30000",
            "-output_ts_offset",
            "1.25",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path


def timestamps(source):
    points = list(source_pts(source))
    return [Fraction(p - points[0]) * Fraction(source.time_base) for p in points]


def test_vfr_is_explicit_and_actual_duration_is_not_count_over_average(vfr):
    with pytest.raises(ValueError, match="VFR"):
        probe_source(vfr)
    source = probe_source(vfr, allow_vfr=True)
    assert source.vfr
    assert source.start_time > 1
    assert 0.08 < source.audio_offset_seconds < 0.14
    assert len(set(b - a for a, b in zip(timestamps(source), timestamps(source)[1:]))) > 1
    assert source.duration > float(timestamps(source)[-1])
    assert source.metadata()["fps_kind"] == "average"
    assert source.metadata()["variable_frame_rate"] is True


def test_vfr_round_trip_preserves_b_frame_pts_audio_and_end_time(vfr, tmp_path):
    source = probe_source(vfr, allow_vfr=True)
    run = tmp_path / "run"
    run.mkdir()
    writer = VideoWriter(source, run)
    try:
        decoded = list(frames(source, Metrics(), threading.Event()))
        assert [f.source_pts for f in decoded] == list(source_pts(source))
        for frame in decoded:
            writer.write(frame.image, frame=frame)
        destination = writer.finish()
    finally:
        writer.close()
    output = probe_source(destination, allow_vfr=True)
    assert output.vfr
    assert output.audio
    assert output.frame_count == source.frame_count
    assert output.width == source.width + 1 and output.height == source.height + 1
    tick = max(Fraction(source.time_base), Fraction(output.time_base))
    assert all(abs(a - b) <= tick for a, b in zip(timestamps(source), timestamps(output)))
    assert abs(output.duration - source.duration) <= float(tick) * 2
    assert abs(output.start_time) <= float(tick)
    assert not writer.temporary.exists()
    data = json.loads(
        subprocess.run(
            [
                executable("ffprobe"),
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    assert float(audio["duration"]) >= source.duration - 0.05
    pcm = subprocess.run(
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(destination),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "48000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    samples = np.frombuffer(pcm, dtype="<f4")
    assert np.max(np.abs(samples[: int(0.06 * 48000)])) < 0.001
    assert np.sqrt(np.mean(samples[int(0.18 * 48000) : int(0.24 * 48000)] ** 2)) > 0.02


def test_vfr_missing_pts_or_incomplete_export_never_publishes(vfr, tmp_path):
    source = probe_source(vfr, allow_vfr=True)
    run = tmp_path / "partial"
    run.mkdir()
    writer = VideoWriter(source, run)
    frame = next(frames(source, Metrics(), threading.Event()))
    with pytest.raises(ValueError, match="PTS"):
        writer.write(frame.image)
    writer.write(frame.image, frame=frame)
    with pytest.raises(ValueError, match="thứ tự"):
        writer.write(frame.image, frame=frame)
    with pytest.raises((ValueError, RuntimeError)):
        writer.finish()
    writer.close()
    assert not writer.destination.exists()


def test_vfr_missing_pyav_has_actionable_error(vfr, tmp_path):
    source = probe_source(vfr, allow_vfr=True)
    with patch.dict(sys.modules, {"av": None}), pytest.raises(ValueError, match="PyAV"):
        VideoWriter(source, tmp_path)


def test_vfr_without_audio_and_cancel_cleanup(vfr, tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-copyts",
            "-i",
            str(vfr),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            str(silent),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    source = probe_source(silent, allow_vfr=True)
    assert source.vfr and not source.audio
    cancelled = tmp_path / "cancelled"
    cancelled.mkdir()
    writer = VideoWriter(source, cancelled)
    for frame in frames(source, Metrics(), threading.Event()):
        writer.write(frame.image, frame=frame)
        break
    writer.close()
    writer.close()
    assert writer.container is None
    assert not writer.destination.exists()
    complete = tmp_path / "complete"
    complete.mkdir()
    writer = VideoWriter(source, complete)
    try:
        for frame in frames(source, Metrics(), threading.Event()):
            writer.write(frame.image, frame=frame)
        output = probe_source(writer.finish(), allow_vfr=True)
        assert not output.audio
        assert timestamps(output) == timestamps(source)
    finally:
        writer.close()


@pytest.mark.parametrize("kind", ["offset_vfr", "audio_before_video", "rotated_sar", "coarse_cfr"])
def test_served_clean_preview_preserves_exact_frame_times_and_raster(kind, vfr, tmp_path):
    from bisect import bisect_right
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from classroom_ai.interfaces.web import create_app

    path = vfr
    if kind != "offset_vfr":
        path = tmp_path / f"{kind}.mp4"
        command = [executable("ffmpeg"), "-v", "error", "-nostdin", "-n"]
        if kind == "audio_before_video":
            command += ["-itsoffset", "0.3"]
        command += ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=0.8"]
        if kind == "audio_before_video":
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.1"]
            command += [
                "-vf",
                "select=not(eq(mod(n\\,3)\\,1))",
                "-fps_mode",
                "vfr",
                "-af",
                "volume=0:enable=lt(t\\,0.5)",
                "-c:a",
                "aac",
            ]
        elif kind == "rotated_sar":
            command += ["-vf", "setsar=2/1"]
        command += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "1000",
            str(path),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        if kind == "rotated_sar":
            rotated = tmp_path / "rotation.mp4"
            subprocess.run(
                [
                    executable("ffmpeg"),
                    "-v",
                    "error",
                    "-nostdin",
                    "-n",
                    "-display_rotation",
                    "90",
                    "-i",
                    str(path),
                    "-c",
                    "copy",
                    str(rotated),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            path = rotated
    source = probe_source(path, allow_vfr=True)
    if kind == "audio_before_video":
        assert source.audio_offset_seconds < -0.29
    if kind == "coarse_cfr":
        assert not source.vfr
    app = create_app(tmp_path, runner=lambda *a, **k: None)
    directory = tmp_path / "preview-run"
    directory.mkdir()
    job_id = "a" * 32
    job = {
        "id": job_id,
        "state": "complete",
        "home": tmp_path,
        "directory": directory,
        "stop": tmp_path / "stop.flag",
        "config": SimpleNamespace(source=path),
    }
    app.state.jobs.jobs[job_id] = job
    with TestClient(app, base_url="http://127.0.0.1") as client:
        app.state.jobs.preview(job)
        assert job["preview_contract"] == 1
        assert (
            json.loads((job["home"] / "job.json").read_text(encoding="utf-8"))["preview_contract"]
            == 1
        )
        destination = directory / "preview.mp4"
        assert client.get(f"/api/jobs/{job_id}").json()["preview_verified"] is True
        response = client.get(f"/api/jobs/{job_id}/artifacts/preview.mp4")
        assert response.status_code == 200
        assert response.content == destination.read_bytes()
        output = probe_source(destination, allow_vfr=True)
        actual_pts = [Fraction(p) * Fraction(output.time_base) for p in source_pts(output)]
        expected_pts = timestamps(source)
        assert actual_pts == expected_pts
        # This is the browser's floor lookup, with its sub-microsecond tolerance.
        assert [
            bisect_right([float(p) for p in expected_pts], float(t) + 1e-7) - 1 for t in actual_pts
        ] == list(range(source.frame_count))
        assert (output.width, output.height) == (
            (source.width + 1) // 2 * 2,
            (source.height + 1) // 2 * 2,
        )
        assert output.audio == source.audio
        assert abs(output.duration - source.duration) <= float(Fraction(output.time_base)) * 2
        data = json.loads(
            subprocess.run(
                [
                    executable("ffprobe"),
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        video = next(s for s in data["streams"] if s["codec_type"] == "video")
        assert video.get("sample_aspect_ratio") == "1:1"
        assert all(side.get("rotation", 0) == 0 for side in video.get("side_data_list", []))
        if kind == "rotated_sar":
            original_metadata = json.loads(
                subprocess.run(
                    [
                        executable("ffprobe"),
                        "-v",
                        "error",
                        "-show_streams",
                        "-of",
                        "json",
                        str(path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )["streams"][0]
            assert original_metadata["sample_aspect_ratio"] == "2:1"
            assert any(
                abs(side.get("rotation", 0)) == 90
                for side in original_metadata.get("side_data_list", [])
            )
        if source.audio:
            pcm = subprocess.run(
                [
                    executable("ffmpeg"),
                    "-v",
                    "error",
                    "-i",
                    str(destination),
                    "-map",
                    "0:a:0",
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
            samples = np.frombuffer(pcm, dtype="<f4")
            assert np.max(np.abs(samples[: int(0.04 * 48000)])) < 0.001
            assert np.sqrt(np.mean(samples[int(0.3 * 48000) : int(0.4 * 48000)] ** 2)) > 0.02
        # Raster decoding must keep the unrotated model-coordinate geometry.
        original = next(frames(source, Metrics(), threading.Event())).image
        clean = next(frames(output, Metrics(), threading.Event())).image[
            : source.height, : source.width
        ]
        # Compare luma: 4:4:4 -> browser-compatible 4:2:0 intentionally changes
        # high-frequency chroma in the tiny odd-sized fixture.
        import cv2

        left_luma = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(float)
        right_luma = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(float)
        assert np.mean(np.abs(left_luma - right_luma)) < 8


@pytest.mark.parametrize(
    "stage", ["metadata", "source_scan", "source_pts", "mux", "output_scan", "pts_validation"]
)
def test_cancellation_kills_owned_long_stage_before_eof(stage, vfr, tmp_path, monkeypatch):
    """A real child with a file barrier never reaches EOF unless released/killed."""
    from classroom_ai.video import media

    source = probe_source(vfr, allow_vfr=True)
    writer = None
    if stage in ("mux", "output_scan", "pts_validation"):
        run = tmp_path / "cancel-finish"
        run.mkdir()
        writer = VideoWriter(source, run)
        for frame in frames(source, Metrics(), threading.Event()):
            writer.write(frame.image, frame=frame)
    entered = tmp_path / "child-entered"
    release = tmp_path / "child-release"
    stop, exited = threading.Event(), threading.Event()
    original_popen = subprocess.Popen
    children, errors = [], []

    def intercept(command, *args, **kwargs):
        is_probe = "ffprobe" in Path(command[0]).name.lower()
        selected = (
            stage == "metadata"
            and is_probe
            and "-show_streams" in command
            or stage == "source_scan"
            and "frame=best_effort_timestamp,pkt_duration" in command
            and command[-1] == str(source.path)
            or stage in ("source_pts", "pts_validation")
            and "frame=best_effort_timestamp" in command
            and command[-1] == str(source.path)
            or stage == "mux"
            and "ffmpeg" in Path(command[0]).name.lower()
            or stage == "output_scan"
            and "frame=best_effort_timestamp,pkt_duration" in command
            and str(command[-1]).endswith("annotated.partial.mp4")
        )
        if not selected:
            return original_popen(command, *args, **kwargs)
        # Supply one valid point, then hold the pipe/process open behind a barrier.
        token = "0,1000" if stage in ("source_scan", "output_scan") else "0"
        script = "import sys,time;from pathlib import Path;Path(sys.argv[1]).write_text('entered');print(sys.argv[3],flush=True);\nwhile not Path(sys.argv[2]).exists():time.sleep(.01)"
        child = original_popen(
            [sys.executable, "-u", "-c", script, str(entered), str(release), token], *args, **kwargs
        )
        children.append(child)
        return child

    def work():
        try:
            if stage in ("metadata", "source_scan"):
                probe_source(vfr, allow_vfr=True, cancel=stop.is_set)
            elif stage == "source_pts":
                list(source_pts(source, cancel=stop.is_set))
            else:
                writer.finish(cancel=stop.is_set)
        except Exception as exc:
            errors.append(exc)
        finally:
            exited.set()

    monkeypatch.setattr(subprocess, "Popen", intercept)
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not entered.exists() and not exited.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert entered.exists(), f"did not reach {stage} barrier: {errors}"
        stop.set()
        assert exited.wait(1.5), f"cancellation waited for {stage} EOF"
        assert not release.exists()
        assert len(errors) == 1 and isinstance(errors[0], media.MediaCancelled), errors
        assert all(child.poll() is not None for child in children)
        if writer:
            assert not writer.destination.exists()
    finally:
        release.touch()
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        thread.join(3)
        if writer:
            writer.close()


@pytest.mark.parametrize("corrupt", [{"rotation_degrees": 90}, {"sample_aspect_ratio": "2:1"}])
def test_preview_geometry_failure_never_publishes_contract_marker(
    corrupt, vfr, tmp_path, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace

    from classroom_ai.interfaces.web import DemoJobs
    from classroom_ai.video import media

    original_probe = media.probe_source

    def probe(path, **kwargs):
        source = original_probe(path, **kwargs)
        return replace(source, **corrupt) if Path(path).name == "annotated.partial.mp4" else source

    monkeypatch.setattr(media, "probe_source", probe)
    jobs = DemoJobs(tmp_path, lambda *a, **k: None)
    directory = tmp_path / "bad-geometry"
    directory.mkdir()
    job = {
        "id": "a" * 32,
        "state": "rendering",
        "home": tmp_path,
        "directory": directory,
        "stop": tmp_path / "stop.flag",
        "config": SimpleNamespace(source=vfr),
    }
    try:
        with pytest.raises(RuntimeError, match="kiểm tra"):
            jobs.preview(job)
        assert "preview_contract" not in job
        assert not (directory / "preview.mp4").exists()
        assert not list(directory.glob(".preview-*"))
    finally:
        jobs.close()
