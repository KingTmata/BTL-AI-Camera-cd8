"""Real source-only preview/index preparation; no model, GT, server or GUI."""

import builtins
import hashlib
import json
import struct
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from classroom_ai.paths import PROJECT_ROOT
from classroom_ai.video import media


@pytest.fixture
def tmp_path():
    scratch = PROJECT_ROOT / ".temp" / "annotation_source_integration"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        yield Path(folder)


def make_video(folder, kind):
    source = folder / "original.mp4"
    command = [media.executable("ffmpeg"), "-v", "error", "-nostdin", "-n"]
    if kind == "early_audio":
        command += ["-itsoffset", "0.3"]
    command += ["-f", "lavfi", "-i", "testsrc=size=65x49:rate=10:duration=0.8"]
    if kind != "cfr":
        if kind == "rotated_vfr_audio":
            command += ["-itsoffset", "0.12"]
        command += [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.1",
            "-c:a",
            "aac",
        ]
        if kind == "early_audio":
            command += ["-af", r"volume=0:enable=lt(t\,0.5)"]
    vf = "setsar=2/1" if kind == "cfr" else r"settb=1/1000,setpts=PTS+30*gte(N\,2),setsar=2/1"
    command += [
        "-vf",
        vf,
        "-fps_mode",
        "vfr",
        "-enc_time_base",
        "1:1000",
        "-c:v",
        "libx264",
        "-bf",
        "2",
        "-pix_fmt",
        "yuv444p",
        "-video_track_timescale",
        "1000",
        "-output_ts_offset",
        "1.25",
        str(source),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    if kind == "rotated_vfr_audio":
        rotated = folder / "original_rotated.mp4"
        subprocess.run(
            [
                media.executable("ffmpeg"),
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-copyts",
                "-display_rotation",
                "90",
                "-i",
                str(source),
                "-c",
                "copy",
                str(rotated),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        source = rotated
    return source


@pytest.mark.parametrize("kind", ["cfr", "rotated_vfr_audio", "early_audio"])
def test_real_prep_matches_all_native_pts_raster_audio_and_never_loads_nn(
    tmp_path, monkeypatch, kind
):
    source = make_video(tmp_path, kind)
    original = media.probe_source(source, allow_vfr=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_points = list(media.source_pts(original))
    assert original.start_time > 1
    importer = builtins.__import__

    def no_nn(name, *args, **kwargs):
        if name.startswith(
            ("torch", "ultralytics", "classroom_ai.inference", "classroom_ai.video.pipeline")
        ):
            pytest.fail("Annotation prep must not load inference: " + name)
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_nn)
    output = media.prepare_annotation_source(source, tmp_path / "prepared")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    preview = media.probe_source(output / "preview.mp4", allow_vfr=True)
    preview_points = list(media.source_pts(preview))
    expected = [(p - source_points[0]) * Fraction(original.time_base) for p in source_points]
    actual = [p * Fraction(preview.time_base) for p in preview_points]
    assert expected == actual
    assert (preview.width, preview.height) == (66, 50)
    assert preview.sample_aspect_ratio == "1:1" and preview.rotation_degrees == 0
    assert preview.audio == original.audio
    assert report["counts"] == {
        "frames_decoded": len(expected),
        "frames_encoded": len(expected),
        "frames_inferred": 0,
    }
    assert report["preview"]["native_pts_verified"] is True
    assert manifest["coverage"]["actions_reviewed"] is False
    rows = [json.loads(row) for row in (output / "frames.jsonl").read_text().splitlines()]
    assert [r["source_pts"] for r in rows] == source_points
    assert all(
        r["observation"] == "not_inferred" and r["tracks"] == r["detections"] == [] for r in rows
    )
    times = struct.unpack("<" + "d" * len(expected), (output / "frames.times").read_bytes())
    assert list(times) == [float(t) for t in expected]
    assert (output / "tracks.jsonl").stat().st_size == 0
    assert not (output / "events.jsonl").exists() and not list(
        tmp_path.glob(".annotation-source-*")
    )
    if original.audio:
        pcm = subprocess.run(
            [
                media.executable("ffmpeg"),
                "-v",
                "error",
                "-i",
                str(output / "preview.mp4"),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        samples = np.frombuffer(pcm, dtype="<f4")
        assert np.max(np.abs(samples[: int(0.04 * 48000)])) < 0.005
        assert np.sqrt(np.mean(samples[int(0.3 * 48000) : int(0.4 * 48000)] ** 2)) > 0.02


def test_real_cancel_removes_partial_preview_and_preserves_source(tmp_path):
    source = make_video(tmp_path, "cfr")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    stopped = False

    def progress(value):
        nonlocal stopped
        if value["phase"] == "decoding" and value["frames_written"] == 2:
            stopped = True

    with pytest.raises(media.MediaCancelled):
        media.prepare_annotation_source(
            source, tmp_path / "prepared", cancel=lambda: stopped, on_progress=progress
        )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert not (tmp_path / "prepared").exists() and not list(tmp_path.glob(".annotation-source-*"))
