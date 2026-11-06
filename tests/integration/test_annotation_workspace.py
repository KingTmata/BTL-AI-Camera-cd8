"""Annotation preparation is source-only; human labels stay explicit."""

import hashlib
import json
import struct
import time

import pytest
from fastapi.testclient import TestClient

from classroom_ai.interfaces.web import create_app


def source_app(tmp_path):
    (tmp_path / "clip.MOV").write_bytes(b"original-video")

    def forbidden(*args, **kwargs):
        raise AssertionError("annotation preparation called inference")

    app = create_app(tmp_path, runner=forbidden, replay_runner=forbidden)

    def prepare(source, destination, *, cancel, on_progress):
        destination.mkdir(parents=True)
        metadata = dict(
            path=str(source),
            width=100,
            height=80,
            fps=5,
            frame_count=2,
            duration_seconds=0.4,
            time_base="1/1000",
            video_start_time=1,
        )
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = dict(
            schema_version=2,
            kind="annotation_source_preparation",
            processing_mode="annotation_prep",
            status="completed",
            run_id="annotation-fixture",
            source=metadata,
            source_sha256=digest,
            config=dict(mode="annotation", analysis_task="annotation", tracking="off"),
            inference=dict(status="not_run"),
        )
        (destination / "manifest.json").write_text(json.dumps(manifest))
        (destination / "metadata.json").write_text(json.dumps({"source": metadata}))
        (destination / "report.json").write_text(
            json.dumps({"status": "completed", "counts": {"frames_inferred": 0}})
        )
        (destination / "preview.mp4").write_bytes(b"verified-preview")
        (destination / "tracks.jsonl").write_bytes(b"")
        with (
            (destination / "frames.jsonl").open("wb") as frames,
            (destination / "frames.idx").open("wb") as offsets,
            (destination / "frames.times").open("wb") as times,
        ):
            for index in range(2):
                offsets.write(struct.pack("<Q", frames.tell()))
                times.write(struct.pack("<d", index / 5))
                row = dict(
                    schema_version=2,
                    kind="annotation_source_frame",
                    analysis_task="annotation",
                    observation="not_inferred",
                    frame_index=index,
                    source_pts=1000 + 200 * index,
                    time_base="1/1000",
                    timestamp_seconds=index / 5,
                    segment_id=0,
                    tracks=[],
                    display_tracks=[],
                    detections=[],
                )
                frames.write((json.dumps(row) + "\n").encode())
        on_progress(dict(phase="complete", frames_written=2, frames_total=2, source_time_s=0.2))
        return destination

    app.state.jobs.annotation_runner = prepare
    return app


def start_source(client):
    headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
    response = client.post("/api/annotation-sources", headers=headers, json={"source": "clip.MOV"})
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + job_id).json()
        if job["state"] in {"complete", "failed", "cancelled"}:
            assert job["state"] == "complete", job
            return job_id, headers, job
        time.sleep(0.01)
    raise AssertionError("source preparation did not finish")


def review_payload():
    human = dict(segment_id=0, track_id="human:" + "a" * 32, start_s=0, end_s=0.4)
    return dict(
        events=[],
        coverage=[dict(human, label="raise_hand", review="reviewed")],
        human_actors=[human],
        actor_links=[],
        human_actor_references=[],
        review_metadata=dict(
            reviewer_id="group-a",
            rubric_version="fixture-v1",
            source_group="fixture",
            split="train",
            adjudication_state="reviewed",
        ),
    )


def reference():
    return dict(
        segment_id=0,
        track_id="human:" + "a" * 32,
        frame_index=0,
        source_pts=1000,
        time_base="1/1000",
        timestamp_seconds=0,
        bbox_xyxyn=[0.1, 0.2, 0.4, 0.8],
        note="Áo tối, hàng trước",
    )


def test_annotation_snapshot_precondition_prevents_stale_draft_overwrite(tmp_path):
    app = source_app(tmp_path)
    with TestClient(app) as client:
        job_id, headers, _ = start_source(client)
        prefix = "/api/jobs/" + job_id
        route = prefix + "/annotations?reviewer_id=group-a"
        payload = review_payload()
        payload["human_actor_references"] = [reference()]
        payload["expected_snapshot"] = None
        first = client.post(route, headers=headers, json=payload)
        assert first.status_code == 200, first.text
        snapshot = first.json()["snapshot"]
        directory = app.state.jobs.find(job_id)["directory"]
        before = set(directory.glob("review-*.json"))
        stale = client.post(route, headers=headers, json=payload)
        assert stale.status_code == 409, stale.text
        assert set(directory.glob("review-*.json")) == before
        payload["expected_snapshot"] = snapshot
        current = client.post(route, headers=headers, json=payload)
        assert current.status_code == 200, current.text
        assert current.json()["snapshot"] != snapshot
        assert (directory / snapshot).exists()


def test_source_only_job_supports_native_review_without_models(tmp_path):
    app = source_app(tmp_path)
    with TestClient(app) as client:
        job_id, headers, job = start_source(client)
        assert job["kind"] == "annotation_source_preparation"
        assert job["processing_mode"] == "annotation_prep" and job["inference_status"] == "not_run"
        assert job["preview_verified"] and not job["video_export_eligible"]
        prefix = "/api/jobs/" + job_id
        context = client.get(prefix + "/review-context?reviewer_id=group-a").json()
        assert context["model_track_segments"] == []
        assert (
            context["human_actors"] == [] and context["events"] == [] and context["coverage"] == []
        )
        assert (
            context["actor_references_required"] is True
            and context["source_kind"] == "annotation_source_preparation"
        )
        assert client.get(prefix + "/frame/1").json()["source_pts"] == 1200
        assert client.get(prefix + "/artifacts/frames.times").content == struct.pack("<dd", 0, 0.2)
        listing = client.get("/api/annotation-sources").json()["sources"]
        assert any(item["job_id"] == job_id and item["name"] == "clip.MOV" for item in listing)
        before = set(app.state.jobs.find(job_id)["directory"].iterdir())
        for route in ("/proposals", "/action-labels", "/video-export"):
            assert client.post(prefix + route, headers=headers, json={}).status_code == 409
        assert set(app.state.jobs.find(job_id)["directory"].iterdir()) == before


def test_source_review_requires_real_anchor_before_gt_and_branches_stay_separate(tmp_path):
    app = source_app(tmp_path)
    with TestClient(app) as client:
        job_id, headers, _ = start_source(client)
        prefix = "/api/jobs/" + job_id
        payload = review_payload()
        response = client.post(
            prefix + "/annotations?reviewer_id=group-a", headers=headers, json=payload
        )
        assert response.status_code == 200, response.text
        assert "missing_actor_reference" in response.json()["readiness_reasons"]
        assert client.get(prefix + "/gt-bundle?reviewer_id=group-a").status_code == 409
        payload["human_actor_references"] = [reference()]
        response = client.post(
            prefix + "/annotations?reviewer_id=group-a", headers=headers, json=payload
        )
        assert response.status_code == 200, response.text
        bundle = client.get(prefix + "/gt-bundle?reviewer_id=group-a")
        assert bundle.status_code == 200, bundle.text
        assert bundle.json()["human_actor_references"] == [reference()]
        other = client.get(prefix + "/review-context?reviewer_id=group-b").json()
        assert other["human_actor_references"] == [] and other["events"] == []


@pytest.mark.parametrize(
    "change",
    [{"frame_index": 1}, {"source_pts": 1001}, {"time_base": "1/600"}, {"timestamp_seconds": 0.1}],
)
def test_reference_must_match_actual_source_frame_before_any_snapshot(tmp_path, change):
    app = source_app(tmp_path)
    with TestClient(app) as client:
        job_id, headers, _ = start_source(client)
        prefix = "/api/jobs/" + job_id
        payload = review_payload()
        payload["human_actor_references"] = [reference() | change]
        response = client.post(
            prefix + "/annotations?reviewer_id=group-a", headers=headers, json=payload
        )
        assert response.status_code == 422, response.text
        assert not list(app.state.jobs.find(job_id)["directory"].glob("review-*.json"))


def test_rubric_serves_canonical_definitions_without_approval(tmp_path):
    app = source_app(tmp_path)
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "action_rubric.yaml").write_text(
        "version: pilot-v1\nstatus: draft\nlabels: []\nprinciples: []\nsource_reference: {}\n",
        encoding="utf-8",
    )
    with TestClient(app) as client:
        value = client.get("/api/action-rubric").json()
        assert value["version"] == "pilot-v1" and value["status"] == "draft"


@pytest.mark.parametrize("state", ["queued", "processing"])
def test_prepared_list_recovers_interrupted_jobs_before_individual_get(tmp_path, state):
    job_id = "c" * 32
    home = tmp_path / "runs" / "demo" / "jobs" / job_id
    home.mkdir(parents=True)
    (home / "job.json").write_text(
        json.dumps(
            dict(
                id=job_id,
                processing_mode="annotation_prep",
                kind="annotation_source_preparation",
                source_name="clip.MOV",
                state=state,
            )
        ),
        encoding="utf-8",
    )
    app = source_app(tmp_path)
    with TestClient(app) as client:
        sources = client.get("/api/annotation-sources").json()["sources"]
        assert len(sources) == 1
        assert sources[0]["job_id"] == job_id
        assert sources[0]["state"] == "failed"
        assert client.get("/api/jobs/" + job_id).json()["state"] == "failed"
