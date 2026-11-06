import hashlib
import json
import threading
import time
from fractions import Fraction
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from classroom_ai.video import online
from classroom_ai.video.core import Frame
from classroom_ai.video.media import VideoSource


class Clock:
    def __init__(self):
        self.now = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def wait(self, stop, delay):
        with self.lock:
            self.now += max(0, delay)
        time.sleep(0.001)
        return stop.is_set()


@pytest.fixture
def replay(tmp_path, monkeypatch):
    source_path, weights = tmp_path / "source.mp4", tmp_path / "pose.pt"
    source_path.write_bytes(b"source bytes")
    weights.write_bytes(b"weights bytes")
    pts = [0, 70, 230, 410, 600]
    source = VideoSource(
        source_path,
        Fraction(10),
        32,
        24,
        len(pts),
        False,
        time_base="1/1000",
        vfr=True,
        duration_seconds=0.7,
    )
    state = SimpleNamespace(released=[], predicted=[], warmups=0)

    def probe(path, **kwargs):
        assert kwargs["allow_vfr"] and kwargs["cancel"] is not None
        return source

    def read(*args, **kwargs):
        for i, value in enumerate(pts):
            yield Frame(i, value / 1000, np.full((24, 32, 3), i, np.uint8), value, "1/1000")

    class Predictor:
        def set_cancel(self, cancel):
            self.cancel = cancel

        def predict(self, frame, confidence, **kwargs):
            index = int(frame[0, 0, 0])
            if not state.warmups:
                state.warmups += 1
            else:
                state.predicted.append(index)
            return {"detections": [], "processing_ms": 1, "runtime": {"backend": "fake"}}

    monkeypatch.setattr(online, "probe_source", probe)
    monkeypatch.setattr(online, "frames", read)
    config = online.ReplayConfig(
        source_path, weights, device="cpu", tracking="off", output_dir=tmp_path / "output"
    )
    return config, Predictor, state


def test_pts_pacing_same_frame_jpeg_and_complete_provenance(replay):
    config, Predictor, _ = replay
    clock, updates, images = Clock(), [], []
    directory = online.run_replay(
        config,
        on_update=updates.append,
        on_frame=lambda jpeg, row: images.append((jpeg, row)),
        predictor_factory=lambda *args: Predictor(),
        clock=clock,
        wait=clock.wait,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed" and updates
    assert manifest["mode"] == "recorded_stream_replay_1x"
    assert manifest["source_sha256"] == hashlib.sha256(b"source bytes").hexdigest()
    assert manifest["weights_sha256"] == hashlib.sha256(b"weights bytes").hexdigest()
    origin = report["source_clock_origin_monotonic"]
    for row, (jpeg, image_row) in zip(updates, images):
        assert row["source_release_monotonic"] == pytest.approx(origin + row["timestamp_seconds"])
        assert row["infer_started_monotonic"] >= row["source_release_monotonic"]
        assert row["publish_monotonic"] >= row["infer_started_monotonic"]
        assert row["source_pts"] / 1000 == row["timestamp_seconds"]
        assert image_row["sequence"] == row["sequence"]
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image[0, 0, 0] == row["frame_index"]
    assert report["web_presentation_latency_ms"] is None
    assert report["first_update_warm_ms"] >= 0
    assert report["first_update_cold_ms"] >= report["first_update_warm_ms"]
    assert len((directory / "latencies.jsonl").read_text().splitlines()) == len(updates)
    assert all(row["action_status"] == "not_run" for row in updates)
    assert all(row["action_people"] == [] for row in updates)
    assert (directory / "actions.jsonl").read_text() == ""
    assert manifest["action_rules"]["mode"] == "off"


@pytest.fixture
def action_replay(replay, monkeypatch):
    from dataclasses import replace

    config, Predictor, _ = replay
    config.action_rules, config.tracking = "raise_hand", "bytetrack"
    probe = online.probe_source
    monkeypatch.setattr(
        online,
        "probe_source",
        lambda *a, **kw: replace(
            probe(*a, **kw), fps=Fraction(30), frame_count=49, duration_seconds=1.8
        ),
    )
    acknowledged = threading.Event()

    def read(*args, **kwargs):
        for i in range(9):
            acknowledged.clear()
            timestamp = Pose.times[i]
            yield Frame(
                i * 6,
                timestamp,
                np.full((24, 32, 3), i, np.uint8),
                round(timestamp * 1000),
                "1/1000",
            )
            if not kwargs["cancel"]():
                assert acknowledged.wait(2), "worker did not publish"

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, frame_index, timestamp, detections, **kwargs):
            return {
                "segment_id": 0,
                "tracks": [
                    dict(
                        detections[0],
                        track_id=7,
                        confirmed=Pose.confirmed,
                        detection_index=0 if Pose.matched else None,
                    )
                ],
            }

        def reset(self):
            pass

        def finish(self):
            pass

    class Pose(Predictor):
        all_raised = False
        confirmed = True
        matched = True
        times = [i * 0.2 for i in range(9)]

        def predict(self, image, *args, **kwargs):
            i = int(image[0, 0, 0])
            points = [[50, 70, 1] for _ in range(17)]
            points[0] = [50, 50, 1]
            points[5:11] = [
                [40, 100, 1],
                [60, 100, 1],
                [40, 60, 1],
                [60, 110, 1],
                [40, 20 if self.all_raised or 1 <= i <= 4 else 120, 1],
                [60, 120, 1],
            ]
            # Keep the synthetic Pose inside the 32x24 source raster while
            # preserving relative arm geometry and all temporal assertions.
            points = [[x / 6, y / 6, score] for x, y, score in points]
            return {
                "processing_ms": 1,
                "detections": [
                    {
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox_xyxy": [1, 1, 25, 22],
                        "keypoints17": points,
                        "display_keypoints17": [[0, 0, 0]] * 17,
                    }
                ],
            }

    monkeypatch.setattr(online, "frames", read)
    monkeypatch.setattr(online, "PersonTracker", Tracker)
    return config, Pose, acknowledged


def test_replay_raise_hand_closed_event_durable_and_same_source_frame(action_replay):
    config, Pose, acknowledged = action_replay
    clock, updates, events = Clock(), [], []

    def update(row):
        updates.append(row)
        acknowledged.set()

    directory = online.run_replay(
        config,
        on_update=update,
        on_event=events.append,
        predictor_factory=lambda *a: Pose(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert report["status"] == "completed", report
    ledger = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    assert ledger == events and len(events) == 1
    event = events[0]
    assert event["event"]["start_s"] == pytest.approx(0.2)
    assert event["event"]["end_s"] == pytest.approx(1)
    assert event["event"]["review"] == "draft" and event["event"]["origin"] == "rule"
    available = event["availability"]
    assert available["source_frame_index"] == 42 and available["source_pts"] == 1400
    assert available["observed_through_s"] == pytest.approx(1.4)
    assert event["observation_history_s"] == pytest.approx(0.8)
    assert event["decision_history_s"] == pytest.approx(1.2)
    assert available["source_to_available_ms"] >= 0
    assert [row["sequence"] for row in updates] == list(range(9))
    emitted_at = [row for row in updates if row["action_events"]]
    assert len(emitted_at) == 1 and emitted_at[0]["source_frame_index"] == 42
    assert emitted_at[0]["action_events"] == events
    assert all(row["tracks"][0]["state"] == "observed" for row in updates)
    assert [row["action_people"][0]["status"] for row in updates] == [
        "no_suggestion",
        "candidate",
        "candidate",
        "active",
        "active",
        "release_pending",
        "release_pending",
        "no_suggestion",
        "no_suggestion",
    ]
    assert all(
        row["action_people"][0]["observed_through_s"] == row["timestamp_seconds"] for row in updates
    )
    assert updates[-1]["action_people"][0]["label"] is None
    assert updates[3]["action_people"][0]["label"] == "raise_hand"
    assert report["action_summary"]["emitted_draft_events"] == 1
    assert manifest["action_rules"]["config"]["max_gap_s"] == 0.25


@pytest.mark.parametrize("cancel", [False, True])
def test_replay_terminal_event_has_no_synthetic_frame_or_release(action_replay, cancel):
    config, Pose, acknowledged = action_replay
    Pose.all_raised = True
    stop, clock, updates, events = threading.Event(), Clock(), [], []

    def update(row):
        updates.append(row)
        if cancel and len(updates) == 4:
            stop.set()
        acknowledged.set()

    directory = online.run_replay(
        config,
        on_update=update,
        on_event=events.append,
        stop=stop,
        predictor_factory=lambda *a: Pose(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == ("cancelled" if cancel else "completed"), report
    assert len(events) == 1
    end = updates[-1]
    assert events[0]["event"]["end_s"] == end["timestamp_seconds"]
    assert events[0]["availability"]["reason"] == ("cancelled" if cancel else "source_end")
    assert events[0]["availability"]["source_frame_index"] == end["source_frame_index"]
    assert events[0]["availability"]["source_release_monotonic"] is None
    assert events[0]["availability"]["source_to_available_ms"] is None
    assert not any(row["action_events"] for row in updates)
    assert report["action_summary"]["terminal_flush_events"] == 1


def test_replay_event_callback_failure_preserves_ledger_and_reports_failure(action_replay):
    config, Pose, acknowledged = action_replay
    Pose.all_raised = True
    clock = Clock()

    def fail(envelope):
        raise RuntimeError("event consumer failed")

    directory = online.run_replay(
        config,
        on_update=lambda row: acknowledged.set(),
        on_event=fail,
        predictor_factory=lambda *a: Pose(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed" and "event consumer failed" in report["error"]
    assert len((directory / "actions.jsonl").read_text().splitlines()) == 1


def test_replay_action_configuration_requires_observed_tracks(replay):
    config, _, _ = replay
    config.action_rules = "raise_hand"
    with pytest.raises(ValueError, match="bytetrack"):
        config.validate()
    config.action_rules = "four_actions"
    with pytest.raises(ValueError, match="action_rules"):
        config.validate()


@pytest.mark.parametrize("field", ["confirmed", "matched"])
def test_replay_rule_rejects_unconfirmed_or_unmatched_pose(action_replay, field):
    config, Pose, acknowledged = action_replay
    Pose.all_raised = True
    setattr(Pose, field, False)
    clock, updates = Clock(), []

    def update(row):
        updates.append(row)
        acknowledged.set()

    directory = online.run_replay(
        config, on_update=update, predictor_factory=lambda *a: Pose(), clock=clock, wait=clock.wait
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed", report
    assert report["action_summary"]["emitted_draft_events"] == 0
    assert all(row["tracks"][0]["state"] == "unknown" for row in updates)
    assert all(row["action_people"][0]["status"] == "unsupported" for row in updates)
    assert all(row["action_people"][0]["label"] is None for row in updates)


def test_replay_gap_does_not_count_unknown_time_as_observed_history(action_replay):
    config, Pose, acknowledged = action_replay
    Pose.all_raised = True
    Pose.times = [0, 0.2, 0.4, 1, 1.2, 1.4, 1.6, 1.8, 2]
    clock, events = Clock(), []
    directory = online.run_replay(
        config,
        on_update=lambda row: acknowledged.set(),
        on_event=events.append,
        predictor_factory=lambda *a: Pose(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed", report
    assert len(events) == 2
    first = events[0]
    assert first["event"]["end_s"] == 0.4
    assert first["availability"]["observed_through_s"] == 1
    assert first["observation_history_s"] == 0.4
    assert first["decision_history_s"] == 1
    assert first["availability"]["reason"] == "observation_gap"


def test_event_callback_cost_is_separate_from_frame_publish_latency(action_replay):
    config, Pose, acknowledged = action_replay
    Pose.all_raised = True
    clock = Clock()

    def event(envelope):
        clock.now += 0.5

    directory = online.run_replay(
        config,
        on_update=lambda row: acknowledged.set(),
        on_event=event,
        predictor_factory=lambda *a: Pose(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    rows = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    assert report["status"] == "completed", report
    assert report["action_summary"]["callback_ms"] == pytest.approx(500)
    assert all(row["timing"]["publication_callbacks_ms"] == 0 for row in rows)


def test_sparse_action_cadence_is_explicitly_unobservable_not_negative(action_replay):
    config, Pose, acknowledged = action_replay
    config.sample_fps = 2
    Pose.all_raised = True
    Pose.times = [i * 0.5 for i in range(9)]
    clock = Clock()
    directory = online.run_replay(
        config,
        on_update=lambda row: acknowledged.set(),
        predictor_factory=lambda *a: Pose(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed", report
    assert report["action_rules"]["cadence"]["nominally_sufficient"] is False
    assert report["action_rules"]["cadence"]["max_gap_s"] == 0.25
    assert "nominal_cadence_below_rule_max_gap" in report["action_rules"]["limitations"]
    assert report["action_summary"]["emitted_draft_events"] == 0
    assert report["action_rules"]["coverage"] == "not_reviewed"
    observed = report["action_summary"]["observation_counts"]
    assert observed["supported_actor_observations"] == 9
    assert observed["observation_gap_resets"] == 8


def test_basic_pose_replay_emits_three_current_hints_and_records_config(action_replay):
    config, Pose, acknowledged = action_replay
    config.action_rules = "basic_pose"
    clock, updates = Clock(), []

    def update(row):
        updates.append(row)
        acknowledged.set()

    directory = online.run_replay(
        config, on_update=update, predictor_factory=lambda *a: Pose(), clock=clock, wait=clock.wait
    )
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert report["status"] == "completed", report
    assert report["action_status"] == "basic_pose_rule_draft"
    assert manifest["action_rules"]["supported_labels"] == [
        "raise_hand",
        "stand_up",
        "move_to_board",
    ]
    assert manifest["action_rules"]["config"]["thresholds"]["recent_completed_s"] == 1
    assert updates
    for row in updates:
        hints = row["action_people"][0]["actions"]
        assert [hint["label"] for hint in hints] == ["raise_hand", "stand_up", "move_to_board"]
        assert hints[-1]["status"] == "not_configured"
        assert all(hint["observed_through_s"] == row["timestamp_seconds"] for hint in hints)


def test_replay_roi_requires_source_binding_and_rechecks_before_model(replay):
    config, _, _ = replay
    config.action_rules, config.tracking = "basic_pose", "bytetrack"
    config.board_roi = [0.7, 0.7, 0.95, 0.95]
    with pytest.raises(ValueError, match="board_roi_source_sha256"):
        config.validate()
    config.board_roi_source_sha256 = "f" * 64
    called = []
    directory = online.run_replay(
        config, on_update=lambda row: None, predictor_factory=lambda *a: called.append(a)
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed" and "board_roi" in report["error"]
    assert not called


def test_basic_pose_three_events_use_actual_source_pts_and_current_people(
    action_replay, monkeypatch
):
    from dataclasses import replace

    from tests.unit.test_pose_rules import frame as pose_frame

    config, Pose, acknowledged = action_replay
    config.action_rules, config.board_roi = "basic_pose", [0.7, 0.72, 0.95, 0.98]
    config.board_roi_source_sha256 = hashlib.sha256(b"source bytes").hexdigest()
    probe = online.probe_source
    monkeypatch.setattr(
        online,
        "probe_source",
        lambda *a, **kw: replace(probe(*a, **kw), frame_count=61, duration_seconds=2.2),
    )

    def frames(*args, **kwargs):
        for i in range(11):
            acknowledged.clear()
            yield Frame(i * 6, i * 0.2, np.full((24, 32, 3), i, np.uint8), i * 200, "1/1000")
            assert acknowledged.wait(2)

    class ThreePose(Pose):
        def predict(self, image, *args, **kwargs):
            i = int(image[0, 0, 0])
            x = 100 if i < 3 else 500 if i == 3 else 600 if i == 4 else 750
            raw = pose_frame(i * 0.2, seated=i < 3, x=x, raised=True)["tracks"][0]["keypoints17"]
            for point in raw:
                point[0] *= 32 / 1000
                point[1] *= 24 / 1000
            return {
                "processing_ms": 0,
                "detections": [
                    {
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox_xyxy": [0, 0, 32, 24],
                        "keypoints17": raw,
                    }
                ],
            }

    monkeypatch.setattr(online, "frames", frames)
    updates, events, clock = [], [], Clock()

    def update(row):
        updates.append(row)
        acknowledged.set()

    directory = online.run_replay(
        config,
        on_update=update,
        on_event=events.append,
        predictor_factory=lambda *a: ThreePose(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed", report
    assert [row["event"]["label"] for row in events] == ["stand_up", "move_to_board", "raise_hand"]
    assert events[0]["availability"]["source_frame_index"] == 30
    assert events[1]["availability"]["source_pts"] == 2000
    assert events[-1]["availability"]["source_release_monotonic"] is None
    assert [hint["status"] for hint in updates[5]["action_people"][0]["actions"]] == [
        "active",
        "completed",
        "candidate",
    ]
    assert report["action_summary"]["emitted_draft_events"] == 3
    assert report["action_summary"]["terminal_flush_events"] == 1


def test_slow_inference_replaces_pending_without_fifo_growth(replay, monkeypatch):
    config, Predictor, _ = replay
    clock = Clock()
    first_started, release = threading.Event(), threading.Event()

    class Slow(Predictor):
        def predict(self, frame, confidence, **kwargs):
            if first_started.is_set():
                return super().predict(frame, confidence, **kwargs)
            # First call is warmup; block first observed frame only.
            if getattr(self, "warm", False):
                first_started.set()
                release.wait(1)
            self.warm = True
            return super().predict(frame, confidence, **kwargs)

    original_wait = clock.wait

    def wait(stop, delay):
        result = original_wait(stop, delay)
        if clock() > 100.4:
            release.set()
        return result

    updates = []
    directory = online.run_replay(
        config,
        on_update=updates.append,
        predictor_factory=lambda *args: Slow(),
        clock=clock,
        wait=wait,
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["pending_capacity"] == 1
    assert report["counts"]["pending_replaced"] >= 1
    assert report["counts"]["observation_gaps"] >= 1
    assert len(updates) < report["counts"]["frames_selected"]
    assert [r["timestamp_seconds"] for r in updates] == sorted(
        r["timestamp_seconds"] for r in updates
    )


def test_cancelled_before_source_releases_writes_partial_report(replay):
    config, Predictor, state = replay
    stop = threading.Event()
    stop.set()
    directory = online.run_replay(
        config,
        on_update=lambda row: pytest.fail("No result after cancel"),
        stop=stop,
        predictor_factory=lambda *args: Predictor(),
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "cancelled" and not state.predicted
    assert report["source_clock_origin_monotonic"] is None


def test_predictor_cancel_is_attached_before_warmup(replay):
    config, Predictor, _ = replay
    stop = threading.Event()

    class Cancel(Predictor):
        def predict(self, *args, **kwargs):
            assert callable(self.cancel)
            stop.set()
            raise InterruptedError("cancel warmup")

    directory = online.run_replay(
        config, on_update=lambda row: None, stop=stop, predictor_factory=lambda *args: Cancel()
    )
    assert json.loads((directory / "report.json").read_text())["status"] == "cancelled"


def test_invalid_config_fails_before_any_model_load(replay):
    config, _, _ = replay
    config.sample_fps = 0
    with pytest.raises(ValueError):
        online.run_replay(
            config,
            on_update=lambda row: None,
            predictor_factory=lambda *args: pytest.fail("No model"),
        )


def test_predict_never_receives_future_pts(replay):
    config, Predictor, _ = replay
    clock = Clock()
    calls = []
    stamps = [0, 0.07, 0.23, 0.41, 0.6]
    config.sample_fps = 100

    class Causal(Predictor):
        warmed = False

        def predict(self, frame, *args, **kwargs):
            if self.warmed:
                index = int(frame[0, 0, 0])
                assert clock() >= 100 + stamps[index]
                calls.append(index)
            self.warmed = True
            return super().predict(frame, *args, **kwargs)

    directory = online.run_replay(
        config,
        on_update=lambda row: None,
        predictor_factory=lambda *args: Causal(),
        clock=clock,
        wait=clock.wait,
    )
    assert json.loads((directory / "report.json").read_text())["status"] == "completed"
    assert calls


def test_model_provenance_disagreement_is_failure_before_source_clock(replay):
    config, Predictor, _ = replay

    class WrongModel(Predictor):
        def predict(self, *args, **kwargs):
            value = super().predict(*args, **kwargs)
            value["runtime"]["source_sha256"] = "wrong weights hash"
            return value

    directory = online.run_replay(
        config,
        on_update=lambda row: pytest.fail("No stream on mismatch"),
        predictor_factory=lambda *args: WrongModel(),
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed" and report["source_clock_origin_monotonic"] is None


def test_source_replaced_during_stream_cannot_keep_old_provenance(replay):
    config, Predictor, _ = replay
    clock = Clock()

    def update(row):
        config.source.write_bytes(b"replaced source during replay")

    directory = online.run_replay(
        config,
        on_update=update,
        predictor_factory=lambda *args: Predictor(),
        clock=clock,
        wait=clock.wait,
    )
    assert json.loads((directory / "report.json").read_text())["status"] == "failed"


def test_auto_resolves_inside_engine_and_keeps_requested_device(replay, monkeypatch):
    from classroom_ai.inference import devices

    config, Predictor, _ = replay
    config.device = "auto"
    monkeypatch.setattr(devices, "cuda_devices", lambda: ({}, "CPU build"))
    received = []

    def factory(weights, device):
        received.append(device)
        return Predictor()

    clock = Clock()
    directory = online.run_replay(
        config, on_update=lambda row: None, predictor_factory=factory, clock=clock, wait=clock.wait
    )
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert received == ["cpu"]
    assert manifest["requested_device"] == "auto" and manifest["resolved_device"] == "cpu"
    assert report["requested_device"] == "auto" and report["resolved_device"] == "cpu"


def test_explicit_intel_failure_cannot_fallback_to_cpu(replay, monkeypatch):
    from classroom_ai.inference import devices

    config, _, _ = replay
    config.device = "intel:gpu"
    monkeypatch.setattr(devices, "intel_gpu_devices", lambda: ({}, "GPU missing"))
    called = []
    directory = online.run_replay(
        config, on_update=lambda row: None, predictor_factory=lambda *args: called.append(args)
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed" and "GPU missing" in report["error"] and not called


def test_oversized_frame_metadata_fails_before_model_allocation(replay, monkeypatch):
    from dataclasses import replace

    config, _, _ = replay
    probe = online.probe_source
    monkeypatch.setattr(
        online,
        "probe_source",
        lambda *args, **kwargs: replace(probe(*args, **kwargs), width=100_000, height=100_000),
    )
    called = []
    directory = online.run_replay(
        config, on_update=lambda row: None, predictor_factory=lambda *args: called.append(args)
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed" and "128 MiB" in report["error"] and not called


def test_oversampling_slow_source_still_confirms_adjacent_observations(replay, monkeypatch):
    from dataclasses import replace

    config, Predictor, _ = replay
    config.sample_fps, config.tracking = 30, "bytetrack"
    probe = online.probe_source
    monkeypatch.setattr(
        online,
        "probe_source",
        lambda *args, **kwargs: replace(
            probe(*args, **kwargs), fps=Fraction(10), frame_count=3, duration_seconds=0.3
        ),
    )

    def frames(*args, **kwargs):
        for i in range(3):
            yield Frame(i, i / 10, np.zeros((24, 32, 3), np.uint8), i * 100, "1/1000")

    monkeypatch.setattr(online, "frames", frames)

    class Person(Predictor):
        def predict(self, *args, **kwargs):
            output = super().predict(*args, **kwargs)
            output["detections"] = [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_xyxy": [2, 1, 25, 22],
                }
            ]
            return output

    updates = []
    clock = Clock()
    directory = online.run_replay(
        config,
        on_update=updates.append,
        predictor_factory=lambda *args: Person(),
        clock=clock,
        wait=clock.wait,
    )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed"
    assert any(track["confirmed"] for row in updates for track in row["tracks"])
    assert report["effective_nominal_sample_fps"] == 10
