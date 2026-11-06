import io
import json
from pathlib import Path

import numpy as np
import pytest

from classroom_ai.inference.detector import class_mapping
from classroom_ai.video.core import Frame, VideoConfig


def test_person_only_profile_is_exact():
    assert class_mapping({0: "person"}) == {0: 0}
    for names in ({0: "student"}, {1: "person"}, {0: "person", 1: "other"}):
        with pytest.raises(ValueError):
            class_mapping(names)


def test_analysis_task_must_be_explicit_and_valid():
    assert VideoConfig(Path("x.mp4"), "full").analysis_task == "detect"
    with pytest.raises(ValueError, match="analysis_task"):
        VideoConfig(Path("x.mp4"), "full", analysis_task="action").validate()


def test_intel_gpu_is_explicit_pose_only():
    VideoConfig(Path("x.mp4"), "full", analysis_task="pose", device="intel:gpu").validate()
    with pytest.raises(ValueError, match="Pose|pose"):
        VideoConfig(Path("x.mp4"), "full", analysis_task="detect", device="intel:gpu").validate()


def test_frame_result_preserves_source_time_and_pose_fields(tmp_path):
    from classroom_ai.video.metrics import Metrics, Progress
    from classroom_ai.video.pipeline import Session

    class Guard:
        from contextlib import contextmanager

        @contextmanager
        def inference_slot(self):
            yield {}

    class Predictor:
        def predict(self, *args, **kwargs):
            return {
                "detections": [{"class_id": 0, "keypoints13": [None] * 13}],
                "processing_ms": 1,
                "runtime": {"backend": "openvino", "execution_devices": ["GPU.0"]},
            }

    config = VideoConfig(Path("x.mp4"), "full", analysis_task="pose")
    metrics = Metrics()
    session = Session(config, metrics, Progress(metrics, lambda _: None), tmp_path, Guard(), None)
    session.predictor = Predictor()
    session.warm_started = metrics.started
    stream = io.StringIO()
    frame = Frame(2, 0.2, np.zeros((10, 10, 3), dtype=np.uint8), source_pts=120, time_base="1/100")
    row = session.infer(frame, stream)
    assert row["schema_version"] == 2
    assert row["source_pts"] == 120 and row["time_base"] == "1/100"
    assert row["analysis_task"] == "pose"
    assert json.loads(stream.getvalue())["detections"][0]["keypoints13"] == [None] * 13
    assert metrics.values["runtime"]["execution_devices"] == ["GPU.0"]
