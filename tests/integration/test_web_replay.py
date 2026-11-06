import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from classroom_ai.interfaces.web import create_app


class ReplayFixture:
    def __init__(self):
        self.ready = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.callbacks = None

    def __call__(self, config, *, on_frame, on_update, stop, predictor_factory, manager):
        directory = config.output_dir / ("replay_fixture_" + str(len(self.calls)))
        directory.mkdir(parents=True)
        self.calls.append((config, predictor_factory(config.weights, config.device), manager))
        self.callbacks = on_frame, on_update
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "source_sha256": "0" * 64,
                    "mode": "recorded_stream_replay_1x",
                    "source": {
                        "width": 8,
                        "height": 8,
                        "fps": 30,
                        "frame_count": 60,
                        "duration_seconds": 2,
                    },
                }
            )
        )
        (directory / "updates.jsonl").write_text("")
        (directory / "latencies.jsonl").write_text("")
        (directory / "actions.jsonl").write_text("")
        self.emit(0)
        self.ready.set()
        while not stop.wait(0.01) and not self.release.is_set():
            pass
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "status": "cancelled" if stop.is_set() else "completed",
                    "error": None,
                    "web_presentation_latency_ms": None,
                }
            )
        )
        return directory

    def emit(self, sequence, **changes):
        row = dict(
            schema_version=2,
            mode="recorded_stream_replay_1x",
            sequence=sequence,
            source_frame_index=sequence * 6,
            frame_index=sequence * 6,
            source_pts=sequence * 200,
            time_base="1/1000",
            timestamp_seconds=sequence / 5,
            width=8,
            height=8,
            detections=[],
            tracks=[],
            segment_id=0,
            timing={"source_to_publish_ms": 30},
        )
        row.update(changes)
        self.callbacks[0](b"\xff\xd8jpeg" + bytes([sequence % 256]) + b"\xff\xd9", row)
        self.callbacks[1](row)
        return row


def setup_replay(tmp_path, monkeypatch):
    (tmp_path / "weights").mkdir()
    weights = tmp_path / "weights" / "pose.pt"
    weights.write_bytes(b"fake")
    (tmp_path / "clip.mp4").write_bytes(b"fake")
    fixture = ReplayFixture()
    app = create_app(tmp_path, replay_runner=fixture)
    loaded = []

    class Inspector:
        def __init__(self, *args):
            loaded.append(self)

    monkeypatch.setattr("classroom_ai.inference.pose.PoseInspector", Inspector)
    return app, TestClient(app, base_url="http://127.0.0.1"), fixture, loaded


def submit_replay(client, **values):
    settings = client.get("/api/config").json()
    model = next(m for m in settings["models"] if m["task"] == "pose")
    return client.post(
        "/api/jobs",
        headers={"X-Demo-Token": settings["token"]},
        json={
            "source": "clip.mp4",
            "model_id": model["id"],
            "analysis_task": "pose",
            "processing_mode": "replay",
            "device": "cpu",
            **values,
        },
    )


def terminal(client, job_id):
    for _ in range(150):
        row = client.get(f"/api/jobs/{job_id}").json()
        if row["state"] in ("complete", "cancelled", "failed"):
            return row
        time.sleep(0.01)
    pytest.fail("replay worker did not terminate")


def test_replay_publishes_one_exact_pair_and_keeps_full_review_disabled(tmp_path, monkeypatch):
    app, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    with client:
        response = submit_replay(client)
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        assert fixture.ready.wait(2)
        prefix = f"/api/jobs/{job_id}"
        latest = client.get(prefix + "/latest").json()
        assert latest["sequence"] == 0 and latest["update"]["source_pts"] == 0
        image = client.get(prefix + "/live-frame?sequence=0")
        assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
        assert image.headers["x-replay-sequence"] == "0"
        fixture.emit(1)
        assert client.get(prefix + "/live-frame?sequence=0").status_code == 409
        assert client.get(prefix + "/latest").json()["update"]["source_frame_index"] == 6
        fixture.release.set()
        final = terminal(client, job_id)
        assert final["state"] == "complete" and final["processing_mode"] == "replay"
        assert final["preview_verified"] is False and final["preview_warning"] is None
        assert {"manifest.json", "updates.jsonl", "latencies.jsonl", "report.json"} <= set(
            final["artifacts"]
        )
        assert client.get(prefix + "/review-context").status_code == 409
        assert client.get(prefix + "/class-summary").status_code == 409
        assert not list(app.state.jobs.find(job_id)["directory"].glob("*.jpg"))


def test_replay_pair_rejects_duplicates_mismatches_and_oversized_callbacks(tmp_path, monkeypatch):
    app, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    with client:
        job_id = submit_replay(client).json()["id"]
        assert fixture.ready.wait(2)
        on_frame, on_update = fixture.callbacks
        with pytest.raises(ValueError):
            fixture.emit(0)
        with pytest.raises(ValueError):
            on_frame(b"x" * (4 * 1024 * 1024 + 1), {"sequence": 1})
        row = fixture.emit(1)
        row["detections"].append({"later_mutation": True})
        assert client.get(f"/api/jobs/{job_id}/latest").json()["update"]["detections"] == []
        on_frame(b"\xff\xd8ok\xff\xd9", {**row, "sequence": 2})
        with pytest.raises(ValueError):
            on_update({**row, "sequence": 3})
        assert client.get(f"/api/jobs/{job_id}/latest").json()["sequence"] == 1
        fixture.release.set()
        terminal(client, job_id)


def test_replay_cancel_releases_worker_and_new_job_reuses_model_without_old_pair(
    tmp_path, monkeypatch
):
    app, client, fixture, loaded = setup_replay(tmp_path, monkeypatch)
    with client:
        first = submit_replay(client).json()["id"]
        assert fixture.ready.wait(2)
        assert submit_replay(client).status_code == 409
        token = client.get("/api/config").json()["token"]
        client.post(f"/api/jobs/{first}/cancel", headers={"X-Demo-Token": token})
        assert terminal(client, first)["state"] == "cancelled"
        fixture.ready.clear()
        second = submit_replay(client).json()["id"]
        assert fixture.ready.wait(2)
        assert client.get(f"/api/jobs/{first}/latest").json()["sequence"] is None
        assert len(loaded) == 1
        assert fixture.calls[0][1] is fixture.calls[1][1]
        assert fixture.calls[0][2].root == tmp_path
        fixture.release.set()
        assert terminal(client, second)["state"] == "complete"


@pytest.mark.parametrize(
    "values",
    [
        {"sample_fps": 0},
        {"sample_fps": 11},
        {"processing_mode": "camera"},
        {"analysis_task": "detect"},
    ],
)
def test_replay_configuration_is_scoped_and_bounded(tmp_path, monkeypatch, values):
    _, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    fixture.release.set()
    with client:
        assert submit_replay(client, **values).status_code == 422


def test_replay_failure_is_terminal_and_restart_does_not_invent_a_cached_image(
    tmp_path, monkeypatch
):
    app, client, fixture, _ = setup_replay(tmp_path, monkeypatch)

    def failed(config, **kwargs):
        raise RuntimeError("fixture replay failure")

    app.state.jobs.replay_runner = failed
    with client:
        job_id = submit_replay(client).json()["id"]
        result = terminal(client, job_id)
        assert result["state"] == "failed"
        assert "fixture replay failure" in result["error"]
        assert client.get(f"/api/jobs/{job_id}/latest").json()["sequence"] is None
        app.state.jobs.replay_runner = fixture
        fixture.release.set()
        complete_id = submit_replay(client).json()["id"]
        assert terminal(client, complete_id)["state"] == "complete"
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as restarted:
        restored = restarted.get(f"/api/jobs/{complete_id}").json()
        assert restored["processing_mode"] == "replay" and restored["state"] == "complete"
        assert restarted.get(f"/api/jobs/{complete_id}/latest").json()["sequence"] is None
        assert restarted.get(f"/api/jobs/{complete_id}/live-frame?sequence=0").status_code == 409
        assert restarted.get(f"/api/jobs/{complete_id}/artifacts/updates.jsonl").status_code == 200


def test_replay_registry_and_total_cached_images_remain_bounded(tmp_path, monkeypatch):
    app, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    fixture.release.set()
    with client:
        for _ in range(22):
            job_id = submit_replay(client).json()["id"]
            assert terminal(client, job_id)["state"] == "complete"
        assert len(app.state.jobs.jobs) <= 20
        assert sum("_live_pair" in job for job in app.state.jobs.jobs.values()) == 1


def test_status_publication_retries_only_transient_replace_permission_errors(tmp_path, monkeypatch):
    from pathlib import Path

    from classroom_ai.interfaces import web

    target = tmp_path / "state.json"
    target.write_text('{"state":"loading"}')
    original = Path.replace
    attempts, waits = [], []

    def transient(path, destination):
        attempts.append(path)
        if len(attempts) == 1:
            assert json.loads(target.read_text())["state"] == "loading"
            raise PermissionError("simulated Windows sharing violation")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr(time, "sleep", waits.append)
    web._write_json(target, {"state": "complete"})
    assert json.loads(target.read_text())["state"] == "complete"
    assert len(attempts) == 2 and waits == [0.01]

    def persistent(*args):
        raise PermissionError("persistent denial")

    monkeypatch.setattr(Path, "replace", persistent)
    with pytest.raises(PermissionError, match="persistent denial"):
        web._write_json(target, {"state": "failed"})
    assert json.loads(target.read_text())["state"] == "complete"
    assert len(waits) == 5


def test_clock_has_monotonic_bounds_and_process_epoch_for_replay_clients(tmp_path, monkeypatch):
    app, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    fixture.release.set()
    with client:
        before = time.monotonic()
        response = client.get("/api/clock")
        after = time.monotonic()
        assert response.status_code == 200, response.text
        clock = response.json()
        assert before <= clock["monotonic_seconds"] <= after
        assert len(clock["clock_id"]) == 32
        assert client.get("/api/clock").json()["clock_id"] == clock["clock_id"]
        job_id = submit_replay(client).json()["id"]
        assert terminal(client, job_id)["state"] == "complete"
        assert client.get(f"/api/jobs/{job_id}/latest").json()["clock_id"] == clock["clock_id"]
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as restarted:
        assert restarted.get("/api/clock").json()["clock_id"] != clock["clock_id"]


def test_opt_in_action_rule_reaches_engine_and_draft_ledger_is_downloadable(tmp_path, monkeypatch):
    _, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    fixture.release.set()
    with client:
        response = submit_replay(client, tracking="bytetrack", action_rules="raise_hand")
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        result = terminal(client, job_id)
        assert fixture.calls[0][0].action_rules == "raise_hand"
        assert result["replay_settings"]["action_rules"] == "raise_hand"
        assert client.get(f"/api/jobs/{job_id}/artifacts/actions.jsonl").status_code == 200
        assert client.get(f"/api/jobs/{job_id}/gt-bundle").status_code == 409


@pytest.mark.parametrize(
    "values",
    [
        {"action_rules": "unknown"},
        {"action_rules": "raise_hand", "tracking": "off"},
        {"action_rules": "raise_hand", "tracking": "bytetrack", "processing_mode": "full"},
    ],
)
def test_rule_scope_cannot_silently_change_or_enable_full_gt_workflow(
    tmp_path, monkeypatch, values
):
    _, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    fixture.release.set()
    with client:
        assert submit_replay(client, **values).status_code == 422


def test_replay_rule_default_stays_off(tmp_path, monkeypatch):
    _, client, fixture, _ = setup_replay(tmp_path, monkeypatch)
    fixture.release.set()
    with client:
        job_id = submit_replay(client).json()["id"]
        result = terminal(client, job_id)
        assert fixture.calls[0][0].action_rules == "off"
        assert result["replay_settings"]["action_rules"] == "off"
