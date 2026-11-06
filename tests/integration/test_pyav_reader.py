"""All-frame CFR/VFR native PTS parity with independent FFprobe timestamps."""

import subprocess
import threading
from fractions import Fraction

import pytest

from classroom_ai.video import media
from classroom_ai.video.metrics import Metrics


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
@pytest.mark.parametrize("vfr", [False, True])
@pytest.mark.parametrize("rotation", [0, 90])
def test_pyav_cfr_vfr_all_frame_pts_match_ffprobe(tmp_path, monkeypatch, codec, vfr, rotation):
    path = tmp_path / "source.mp4"
    command = [
        media.executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x48:rate=10",
    ]
    if vfr:
        command += ["-vf", r"settb=1/1000,setpts=PTS+30*gte(N\,2)"]
    command += ["-frames:v", "8", "-fps_mode", "vfr", "-enc_time_base", "1:1000", "-c:v", codec]
    if codec == "libx265":
        command += ["-x265-params", "pools=1:frame-threads=1:log-level=error"]
    command += ["-y", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    if rotation:
        rotated = tmp_path / "rotated.mp4"
        subprocess.run(
            [
                media.executable("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-display_rotation",
                str(rotation),
                "-i",
                str(path),
                "-c",
                "copy",
                "-y",
                str(rotated),
            ],
            check=True,
            capture_output=True,
        )
        path = rotated
    source = media.probe_source(path, allow_vfr=True)
    if rotation:
        assert abs(source.rotation_degrees) == rotation
    expected = list(media.source_pts(source))
    monkeypatch.setattr(
        media, "source_pts", lambda *args, **kwargs: pytest.fail("Second PTS decode is forbidden")
    )
    packets = list(media.frames(source, Metrics(), threading.Event(), decoder="pyav"))
    assert len(packets) == source.frame_count == len(expected) == 8
    assert [packet.source_pts for packet in packets] == expected
    assert [packet.frame_index for packet in packets] == list(range(8))
    assert all(packet.time_base == source.time_base for packet in packets)
    assert [packet.timestamp_seconds for packet in packets] == [
        float((pts - expected[0]) * Fraction(source.time_base)) for pts in expected
    ]
    assert all(packet.image.shape == (source.height, source.width, 3) for packet in packets)
