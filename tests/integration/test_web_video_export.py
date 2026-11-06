"""Export lifecycle tests use a controlled encoder; real media has its own tests."""

import json
import threading
import time

from fastapi.testclient import TestClient

from classroom_ai.interfaces.web import create_app
from classroom_ai.video.media import MediaCancelled
from tests.integration.test_web_workspace import (
    action_labels_job,
    prepare_action_job,
    submit,
)


class ControlledExport:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.fail = False
        self.on_return = None

    def __call__(self, directory, destination, **options):
        self.calls.append((directory, destination, options))
        self.started.set()
        options["on_progress"](
            {"phase": "encoding", "frames_written": 1, "frames_total": 2, "source_time_s": 0.1}
        )
        while not self.release.wait(0.01):
            if options["cancel"]():
                raise MediaCancelled("cancelled controlled export")
        if self.fail:
            raise ValueError("controlled encoder failure")
        destination.mkdir(parents=True)
        (destination / "video.mp4").write_bytes(b"verified-fixture-video")
        (destination / "report.json").write_text(json.dumps({"status": "completed"}))
        if self.on_return:
            self.on_return()
        return destination


def wait_export(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = client.get(f"/api/jobs/{job_id}").json()["video_export"]
        if value and value["state"] in {"complete", "failed", "cancelled"}:
            return value
        time.sleep(0.01)
    raise AssertionError("export did not finish")


def test_export_does_not_replace_analysis_and_publishes_unique_inline_video(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    app.state.jobs.export_runner = encoder
    with client:
        job_id, headers, _ = prepare_action_job(client)
        original = client.get(f"/api/jobs/{job_id}").json()
        assert original["video_export_eligible"] is True
        route = f"/api/jobs/{job_id}/video-export"
        response = client.post(route, headers=headers, json={"start_s": 0.1, "end_s": 1.5})
        assert response.status_code == 202, response.text
        first = response.json()
        assert encoder.started.wait(2)
        current = client.get(f"/api/jobs/{job_id}").json()
        assert current["state"] == "complete" and current["summary"] == original["summary"]
        assert current["video_export"]["state"] == "running"
        assert client.post(route, headers=headers, json={}).status_code == 409
        assert submit(client).status_code == 409
        encoder.release.set()
        complete = wait_export(client, job_id)
        assert complete["state"] == "complete" and complete["id"] == first["id"]
        video = client.get(complete["video_url"], headers={"Range": "bytes=2-5"})
        assert video.status_code == 206 and video.content == b"rifi"
        assert video.headers["content-type"] == "video/mp4"
        assert video.headers["content-disposition"].startswith("inline")
        assert client.get(complete["report_url"]).json()["status"] == "completed"
        response = client.post(route, headers=headers, json={})
        assert response.status_code == 202
        second = wait_export(client, job_id)
        assert second["id"] != first["id"]
        assert client.get(complete["video_url"]).status_code == 200
        assert client.get(second["video_url"]).status_code == 200
        assert client.get(route + "/" + second["id"] + "/frames.jsonl").status_code == 404


def test_cancel_export_preserves_complete_parent_and_original_artifacts(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    app.state.jobs.export_runner = encoder
    with client:
        job_id, headers, _ = prepare_action_job(client)
        job = app.state.jobs.find(job_id)
        before = (job["directory"] / "frames.jsonl").read_bytes()
        route = f"/api/jobs/{job_id}/video-export"
        assert client.post(route, headers=headers, json={}).status_code == 202
        assert encoder.started.wait(2)
        assert client.post(route + "/cancel", headers=headers).status_code == 200
        assert wait_export(client, job_id)["state"] == "cancelled"
        assert not job["stop"].exists()
        assert client.get(f"/api/jobs/{job_id}").json()["state"] == "complete"
        assert (job["directory"] / "frames.jsonl").read_bytes() == before


def test_failed_export_and_restart_recovery_are_separate_from_analysis(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    encoder.fail = True
    encoder.release.set()
    app.state.jobs.export_runner = encoder
    with client:
        job_id, headers, _ = prepare_action_job(client)
        route = f"/api/jobs/{job_id}/video-export"
        assert client.post(route, headers=headers, json={}).status_code == 202
        failed = wait_export(client, job_id)
        assert failed["state"] == "failed" and "controlled" in failed["error"]
        assert failed["video_url"] is None
        job = app.state.jobs.find(job_id)
        saved = json.loads((job["home"] / "job.json").read_text(encoding="utf-8"))
        saved["video_export"]["state"] = "running"
        (job["home"] / "job.json").write_text(json.dumps(saved), encoding="utf-8")
    restored = create_app(tmp_path)
    with TestClient(restored) as other:
        value = other.get(f"/api/jobs/{job_id}").json()
        assert value["state"] == "complete"
        assert value["video_export"]["state"] == "failed"
        assert "dừng" in value["video_export"]["error"]


def test_export_validates_interval_roi_binding_and_disables_non_pose(tmp_path):
    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, _ = prepare_action_job(client)
        route = f"/api/jobs/{job_id}/video-export"
        for body in (
            {"start_s": -1},
            {"start_s": 1, "end_s": 0.5},
            {"end_s": 100},
            {"end_s": 2.1000005},
            {"board_roi": [0.1, 0.1, 0.8, 0.8]},
            {"board_roi": [0.1, 0.1, 0.8, 0.8], "board_roi_source_sha256": "1" * 64},
        ):
            assert client.post(route, headers=headers, json=body).status_code == 422
        job = app.state.jobs.find(job_id)
        path = job["directory"] / "manifest.json"
        data = json.loads(path.read_text())
        data["config"]["analysis_task"] = "detect"
        path.write_text(json.dumps(data))
        public = client.get(f"/api/jobs/{job_id}").json()
        assert public["video_export_eligible"] is False
        assert client.post(route, headers=headers, json={}).status_code == 409


def test_new_pose_full_can_auto_save_without_second_inference(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    encoder.release.set()
    app.state.jobs.export_runner = encoder
    with client:
        response = submit(
            client,
            model_id="pose",
            analysis_task="pose",
            tracking="bytetrack",
            save_action_video=True,
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        result = wait_export(client, job_id)
        assert result["state"] == "complete"
        assert len(encoder.calls) == 1
        assert encoder.calls[0][2]["action_rules"] == "basic_pose"
        assert encoder.calls[0][2]["start_s"] == 0
        assert client.get(f"/api/jobs/{job_id}").json()["state"] == "complete"
        assert submit(client, save_action_video=True).status_code == 422


def test_shutdown_cancels_export_without_cancelling_analysis(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    app.state.jobs.export_runner = encoder
    with client:
        job_id, headers, _ = prepare_action_job(client)
        response = client.post(f"/api/jobs/{job_id}/video-export", headers=headers, json={})
        assert response.status_code == 202 and encoder.started.wait(2)
        job = app.state.jobs.find(job_id)
    persisted = json.loads((job["home"] / "job.json").read_text(encoding="utf-8"))
    assert persisted["state"] == "complete"
    assert persisted["video_export"]["state"] == "cancelled"
    assert not job["stop"].exists()


def test_cancel_accepted_after_render_cannot_publish_success(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    encoder.release.set()
    app.state.jobs.export_runner = encoder
    with client:
        job_id, headers, _ = prepare_action_job(client)
        job = app.state.jobs.find(job_id)
        encoder.on_return = lambda: app.state.jobs.cancel_video_export(job)
        route = f"/api/jobs/{job_id}/video-export"
        assert client.post(route, headers=headers, json={}).status_code == 202
        value = wait_export(client, job_id)
        assert value["state"] == "cancelled" and value["video_url"] is None
        assert client.get(route + "/" + value["id"] + "/video.mp4").status_code == 404


def test_rejected_export_remains_unpublished_when_later_export_succeeds(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    encoder.release.set()
    app.state.jobs.export_runner = encoder
    with client:
        job_id, headers, _ = prepare_action_job(client)
        job = app.state.jobs.find(job_id)
        raw = job["directory"] / "frames.jsonl"
        before = raw.read_bytes()
        encoder.on_return = lambda: raw.write_bytes(before + b"\n")
        route = f"/api/jobs/{job_id}/video-export"
        assert client.post(route, headers=headers, json={}).status_code == 202
        failed = wait_export(client, job_id)
        assert failed["state"] == "failed"
        raw.write_bytes(before)
        encoder.on_return = None
        assert client.post(route, headers=headers, json={}).status_code == 202
        assert wait_export(client, job_id)["state"] == "complete"
        assert client.get(route + "/" + failed["id"] + "/video.mp4").status_code == 404


def test_auto_export_is_reserved_before_dispatch_and_does_not_hold_registry_lock(tmp_path):
    app, client = action_labels_job(tmp_path)
    encoder = ControlledExport()
    encoder.release.set()
    app.state.jobs.export_runner = encoder
    entered, release = threading.Event(), threading.Event()
    original = app.state.jobs.dispatch_video_export

    def gated_dispatch(job, reservation):
        entered.set()
        assert release.wait(5)
        return original(job, reservation)

    app.state.jobs.dispatch_video_export = gated_dispatch
    with client:
        response = submit(
            client,
            model_id="pose",
            analysis_task="pose",
            tracking="bytetrack",
            save_action_video=True,
        )
        job_id = response.json()["id"]
        try:
            assert entered.wait(2)
            public = client.get(f"/api/jobs/{job_id}").json()
            assert public["state"] == "complete"
            assert public["video_export"]["state"] == "queued"
            assert submit(client).status_code == 409
        finally:
            release.set()
        assert wait_export(client, job_id)["state"] == "complete"
