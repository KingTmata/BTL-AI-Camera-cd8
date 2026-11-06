"""Real FFmpeg/VFR/ByteTrack integration with a deterministic Pose predictor."""

import hashlib
import json
import subprocess
import threading
import time

import numpy as np
import pytest

from classroom_ai.video import online
from classroom_ai.video.media import executable, probe_source, source_pts


@pytest.mark.parametrize("action_rules", ["off", "raise_hand"])
def test_recorded_vfr_replay_preserves_pts_and_publishes_before_eof(
    tmp_path, monkeypatch, action_rules
):
    source = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=64x48:r=10",
            "-vf",
            r"settb=1/1000,setpts=PTS+30*gte(N\,2)",
            "-frames:v",
            "6",
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            "1:1000",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-y",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    metadata = probe_source(source, allow_vfr=True)
    original_pts = list(source_pts(metadata))
    assert len(set(np.diff(original_pts))) > 1
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"fake Pose checkpoint; no NN used")
    reader_done = threading.Event()
    reader = online.frames

    def frames(*args, **kwargs):
        try:
            yield from reader(*args, **kwargs)
        finally:
            reader_done.set()

    monkeypatch.setattr(online, "frames", frames)

    class Clock:
        now = 10.0

        def __call__(self):
            return self.now

        def wait(self, stop, delay):
            self.now += delay
            time.sleep(0.002)
            return stop.is_set()

    class Pose:
        def __init__(self, *args):
            pass

        def predict(self, frame, confidence, **kwargs):
            points = [[20, 20, 0.9] for _ in range(17)]
            points[0] = [25, 15, 0.9]
            points[5:11] = [
                [20, 30, 0.9],
                [30, 30, 0.9],
                [20, 20, 0.9],
                [30, 35, 0.9],
                [20, 5, 0.9],
                [30, 40, 0.9],
            ]
            return {
                "processing_ms": 0,
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.9,
                        "bbox_xyxy": [10, 5, 40, 45],
                        "keypoints17": points,
                        "keypoints13": [[20, 20, 0.9]] * 13,
                    }
                ],
            }

    updates, timing, events = [], [], []

    def update(row):
        updates.append(row)
        timing.append(reader_done.is_set())

    clock = Clock()
    config = online.ReplayConfig(
        source,
        weights,
        device="cpu",
        sample_fps=10,
        output_dir=tmp_path / "artifacts",
        action_rules=action_rules,
    )
    directory = online.run_replay(
        config,
        on_update=update,
        on_event=events.append,
        predictor_factory=Pose,
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert report["status"] == "completed", report
    assert updates and not all(timing)
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert [row["source_pts"] for row in updates] == [
        original_pts[row["source_frame_index"]] for row in updates
    ]
    assert [row["observation_index"] for row in updates] == list(range(len(updates)))
    assert any(track["confirmed"] for row in updates for track in row["tracks"])
    assert not (directory / "preview.mp4").exists()
    assert report["web_presentation_latency_ms"] is None
    ledger = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    assert ledger == events
    if action_rules == "raise_hand":
        assert len(events) == 1
        assert events[0]["event"]["origin"] == "rule" and events[0]["event"]["review"] == "draft"
        assert events[0]["event"]["start_s"] == updates[1]["timestamp_seconds"]
        assert events[0]["event"]["end_s"] == updates[-1]["timestamp_seconds"]
        assert events[0]["availability"]["source_pts"] == original_pts[-1]
        assert events[0]["availability"]["source_release_monotonic"] is None
        assert updates[0]["action_people"][0]["status"] == "unsupported"
        assert updates[-1]["action_people"][0]["status"] == "active"
        assert updates[-1]["action_people"][0]["label"] == "raise_hand"
        for row in updates:
            assert all(
                person["observed_through_s"] == row["timestamp_seconds"]
                for person in row["action_people"]
            )
    else:
        assert not events
        assert all(row["action_people"] == [] for row in updates)
