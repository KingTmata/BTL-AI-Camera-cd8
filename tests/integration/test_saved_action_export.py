"""Real native-PTS H264/AAC clips; deterministic saved Pose, never NN inference."""

import hashlib
import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from classroom_ai.paths import PROJECT_ROOT
from classroom_ai.video.media import executable, probe_source, source_pts
from classroom_ai.video.rendering import export_saved_actions


@pytest.fixture
def tmp_path():
    scratch = PROJECT_ROOT / ".temp" / "saved_action_export_integration"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as path:
        yield Path(path)


@pytest.fixture
def make_saved(tmp_path):
    def create(kind="silent", rotation=False):
        source_path = tmp_path / (kind + ".mp4")
        cmd = [executable("ffmpeg"), "-v", "error", "-nostdin", "-n"]
        if kind == "early":
            cmd += ["-itsoffset", "0.3"]
        cmd += ["-f", "lavfi", "-i", "testsrc=size=65x49:rate=10:duration=1.2"]
        if kind != "silent":
            offset = {"late": 0.4, "future": 2}.get(kind, 0)
            if offset:
                cmd += ["-itsoffset", str(offset)]
            duration = 0.15 if kind == "ended" else 0.2 if kind == "future" else 1.5
            cmd += [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=48000:duration={duration}",
                "-c:a",
                "aac",
            ]
            if kind == "early":
                cmd += ["-af", r"volume=0:enable=lt(t\,0.5)"]
        cmd += [
            "-vf",
            r"settb=1/1000,setpts=PTS+30*gte(N\,2),setsar=2/1",
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
            str(source_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        if rotation:
            rotated = tmp_path / (kind + "_rotated.mp4")
            subprocess.run(
                [
                    executable("ffmpeg"),
                    "-v",
                    "error",
                    "-nostdin",
                    "-n",
                    "-copyts",
                    "-display_rotation",
                    "90",
                    "-i",
                    str(source_path),
                    "-c",
                    "copy",
                    str(rotated),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            source_path = rotated
        source = probe_source(source_path, allow_vfr=True)
        points = list(source_pts(source))
        run = tmp_path / (kind + "_run")
        run.mkdir()
        rows = []
        for i, pts in enumerate(points):
            kp = [[30, 30, 1] for _ in range(17)]
            kp[0], kp[5], kp[6] = [30, 15, 1], [20, 35, 1], [40, 35, 1]
            kp[7:11] = [[20, 20, 1], [40, 40, 1], [20, 5, 1], [40, 45, 1]]
            rows.append(
                dict(
                    schema_version=2,
                    analysis_task="pose",
                    frame_index=i,
                    source_pts=pts,
                    time_base=source.time_base,
                    timestamp_seconds=float((pts - points[0]) * Fraction(source.time_base)),
                    segment_id=0,
                    tracks=[
                        dict(
                            track_id=7,
                            confirmed=True,
                            state="observed",
                            bbox_xyxy=[5, 1, 60, 47],
                            keypoints17=kp,
                        )
                    ],
                )
            )
        (run / "frames.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        manifest = dict(
            schema_version=2,
            status="completed",
            run_id="fixture",
            source=source.metadata(),
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            config=dict(mode="full", analysis_task="pose", tracking="bytetrack"),
        )
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run, source, rows

    return create


def test_nonzero_vfr_crop_native_pts_held_frame_rotation_padding_and_no_nn(
    make_saved, tmp_path, monkeypatch
):
    from classroom_ai.inference.pose import PoseInspector

    monkeypatch.setattr(PoseInspector, "__init__", lambda *a, **kw: pytest.fail("No NN allowed"))
    run, source, rows = make_saved(rotation=True)
    assert source.start_time > 1 and source.vfr and abs(source.rotation_degrees) == 90
    result = export_saved_actions(
        run, tmp_path / "export", start_s=0.71, end_s=0.94, action_rules="raise_hand"
    )
    report = json.loads((result / "report.json").read_text(encoding="utf-8"))
    timeline = [json.loads(row) for row in (result / "timeline.jsonl").read_text().splitlines()]
    output = probe_source(result / "video.mp4", allow_vfr=True)
    actual = [Fraction(pts) * Fraction(output.time_base) for pts in source_pts(output)]
    assert actual == [Fraction(row["output_pts"]) * Fraction(source.time_base) for row in timeline]
    assert timeline[0]["source_time_s"] < 0.71 and timeline[0]["output_pts"] == 0
    assert report["actual_interval_s"] == [0.71, 0.94]
    assert abs(output.duration - 0.23) <= 2 * float(Fraction(output.time_base))
    assert (output.width, output.height) == (66, 50) and not output.audio
    assert output.rotation_degrees == 0 and output.sample_aspect_ratio == "1:1"
    assert report["observation_counts"]["frames_seen"] > output.frame_count
    assert not list(tmp_path.glob(".action-export-*"))


@pytest.mark.parametrize("kind", ["late", "early", "ended", "future"])
def test_clip_audio_offsets_and_silent_no_overlap(make_saved, tmp_path, kind):
    run, source, rows = make_saved(kind)
    start, end = (
        (0.6, 1)
        if kind == "ended"
        else (0, 0.4)
        if kind == "future"
        else (0.1, 0.9)
        if kind == "early"
        else (0.2, 0.9)
    )
    result = export_saved_actions(run, tmp_path / "export", start_s=start, end_s=end)
    output = probe_source(result / "video.mp4", allow_vfr=True)
    assert output.audio and abs(output.duration - (end - start)) < 0.003
    report = json.loads((result / "report.json").read_text(encoding="utf-8"))
    pcm = subprocess.run(
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(result / "video.mp4"),
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
    assert len(samples) >= int((end - start - 0.03) * 48000)
    if kind in ("ended", "future"):
        assert np.max(np.abs(samples)) < 0.001
        assert report["audio"]["silence_only"] is True
        assert report["audio"]["stream_overlap_seconds"] == 0
    else:
        assert np.max(np.abs(samples[: int(0.04 * 48000)])) < 0.003
        assert np.sqrt(np.mean(samples[int(0.3 * 48000) : int(0.4 * 48000)] ** 2)) > 0.02


def test_real_export_cancel_during_render_removes_private_files(make_saved, tmp_path):
    from classroom_ai.video.media import MediaCancelled

    run, source, rows = make_saved()
    cancelled = False

    def progress(value):
        nonlocal cancelled
        if value["phase"] == "rendering" and value["frames_written"] == 2:
            cancelled = True

    with pytest.raises(MediaCancelled):
        export_saved_actions(
            run, tmp_path / "export", cancel=lambda: cancelled, on_progress=progress
        )
    assert not (tmp_path / "export").exists()
    assert not list(tmp_path.glob(".action-export-*"))
