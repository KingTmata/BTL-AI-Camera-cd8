import hashlib
import json
import os
import struct
import threading
import time

import pytest
from fastapi.testclient import TestClient

from classroom_ai.interfaces.web import create_app


def action_labels_job(tmp_path):
    def runner(config, **kwargs):
        directory = completed_runner(config, **kwargs)
        source = {
            "width": 100,
            "height": 100,
            "fps": 10,
            "frame_count": 21,
            "duration_seconds": 2.1,
            "time_base": "1/1000",
        }
        (directory / "metadata.json").write_text(json.dumps({"source": source}))
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": "fixture-action-run",
                    "status": "completed",
                    "source_sha256": "0" * 64,
                    "source": source,
                    "config": {"analysis_task": "pose", "tracking": "bytetrack", "mode": "full"},
                }
            )
        )
        with (
            (directory / "frames.jsonl").open("wb") as frames,
            (directory / "frames.idx").open("wb") as offsets,
            (directory / "frames.times").open("wb") as times,
        ):
            for index in range(21):
                raised = 1 <= index <= 10
                points = [[50, 70, 1] for _ in range(17)]
                for joint, point in {
                    0: [50, 40, 1],
                    5: [40, 80, 1],
                    6: [60, 80, 1],
                    7: [40, 30 if raised else 85, 1],
                    8: [60, 85, 1],
                    9: [40, 10 if raised else 95, 1],
                    10: [60, 95, 1],
                }.items():
                    points[joint] = point
                row = {
                    "analysis_task": "pose",
                    "frame_index": index,
                    "source_pts": index * 100,
                    "time_base": "1/1000",
                    "timestamp_seconds": index / 10,
                    "segment_id": 0,
                    "tracks": [
                        {
                            "track_id": 1,
                            "state": "observed",
                            "confirmed": True,
                            "detection_index": 0,
                            "keypoints17": points,
                        }
                    ],
                    "display_tracks": [],
                }
                offsets.write(struct.pack("<Q", frames.tell()))
                times.write(struct.pack("<d", index / 10))
                frames.write((json.dumps(row) + "\n").encode())
        return directory

    app, client = setup_app(tmp_path, runner)
    app.state.jobs.models["pose"] = {
        "id": "pose",
        "name": "fixture-pose.pt",
        "task": "pose",
        "path": tmp_path / "weights" / "test.pt",
    }
    app.state.jobs.preview = lambda job: verified_preview(app, job)
    return app, client


def prepare_action_job(client):
    job_id = submit(client, model_id="pose", analysis_task="pose", tracking="bytetrack").json()[
        "id"
    ]
    assert wait_job(client, job_id)["state"] == "complete"
    headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
    return job_id, headers, f"/api/jobs/{job_id}/action-labels"


def test_explicit_tracker_thresholds_reach_full_runner_and_reject_bool(tmp_path):
    captured = []

    def runner(config, **kwargs):
        captured.append(config)
        return completed_runner(config, **kwargs)

    app, client = setup_app(tmp_path, runner)
    app.state.jobs.preview = lambda job: verified_preview(app, job)
    with client:
        rejected = submit(client, tracking="bytetrack", new_track_thresh=True)
        assert rejected.status_code == 422 and captured == []
        response = submit(
            client,
            tracking="bytetrack",
            track_high_thresh=0.1,
            track_low_thresh=0.05,
            new_track_thresh=0.15,
            recovery_confirm_observations=3,
        )
        assert response.status_code == 202, response.text
        finished = wait_job(client, response.json()["id"])
        assert finished["state"] == "complete", finished
        assert captured[0].track_high_thresh == 0.1
        assert captured[0].track_low_thresh == 0.05
        assert captured[0].new_track_thresh == 0.15
        assert captured[0].recovery_confirm_observations == 3


def test_pose_quality_config_creates_source_bound_generation_and_tracks_helper_code(tmp_path):
    app, client = action_labels_job(tmp_path)
    with client:
        _, headers, route = prepare_action_job(client)
        first = client.post(route, headers=headers, json={})
        assert first.status_code == 200, first.text
        changed = client.post(
            route, headers=headers, json={"pose_quality": {"temporal_guard": True}}
        )
        assert changed.status_code == 200, changed.text
        result = changed.json()
        assert result["generation"] != first.json()["generation"]
        assert result["rule_config"]["pose_quality"]["temporal_guard"] is True
        assert "classroom_ai/video/pose_quality.py" in result["rule_implementation_sources"]
        cached = client.post(
            route, headers=headers, json={"pose_quality": {"temporal_guard": True}}
        )
        assert cached.status_code == 200 and cached.json()["cache_hit"] is True
        assert cached.json()["generation"] == result["generation"]
        assert (
            client.post(
                route, headers=headers, json={"pose_quality": {"temporal_guard": "yes"}}
            ).status_code
            == 422
        )


def test_shoulder_cue_is_an_explicit_separate_cached_rule(tmp_path):
    _, client = action_labels_job(tmp_path)
    with client:
        _, headers, route = prepare_action_job(client)
        legacy = client.post(route, headers=headers, json={}).json()
        response = client.post(route, headers=headers, json={"hand_cue": "shoulder_any_arm_v1"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["hand_cue"] == "shoulder_any_arm_v1"
        assert result["rule_config"]["raise_hand"]["rule"] == "coco17_shoulder_any_arm_v1"
        assert result["generation"] != legacy["generation"]
        assert client.post(route, headers=headers, json={"hand_cue": "other"}).status_code == 422


def test_proposals_check_pose_against_manifest_raster_when_raw_rows_omit_bounds(tmp_path):
    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, _ = prepare_action_job(client)
        directory = app.state.jobs.find(job_id)["directory"]
        path = directory / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert all("width" not in row for row in rows)
        for row in rows:
            row["tracks"][0]["keypoints17"][9][0] = 201
        with path.open("wb") as frames, (directory / "frames.idx").open("wb") as offsets:
            for row in rows:
                offsets.write(struct.pack("<Q", frames.tell()))
                frames.write((json.dumps(row) + "\n").encode())
        before = path.read_bytes()
        response = client.post(f"/api/jobs/{job_id}/proposals", headers=headers, json={})
        assert response.status_code == 200, response.text
        assert response.json()["events"] == []
        assert path.read_bytes() == before


def test_full_action_labels_basic_mode_roi_changes_generation_and_pins_reads(tmp_path):
    app, client = action_labels_job(tmp_path)
    with client:
        _, headers, route = prepare_action_job(client)
        response = client.post(route, headers=headers, json={})
        assert response.status_code == 200, response.text
        first = response.json()
        assert first["action_rules"] == "basic_pose" and first["board_roi"] is None
        people = client.get(route + "?count=1&generation=" + first["generation"]).json()["frames"][
            0
        ]["action_people"]
        assert {action["label"] for action in people[0]["actions"]} == {
            "raise_hand",
            "stand_up",
            "move_to_board",
        }
        board = next(
            action for action in people[0]["actions"] if action["label"] == "move_to_board"
        )
        assert board["status"] == "not_configured"
        second = client.post(
            route,
            headers=headers,
            json={"board_roi": [0.1, 0.2, 0.8, 0.9], "board_roi_source_sha256": "0" * 64},
        ).json()
        assert second["generation"] != first["generation"]
        assert second["board_roi"] == [0.1, 0.2, 0.8, 0.9] and not second["cache_hit"]
        assert client.get(route + "?generation=" + first["generation"]).status_code == 409
        third = client.post(route, headers=headers, json={"action_rules": "raise_hand"}).json()
        assert third["generation"] != second["generation"] and third["action_rules"] == "raise_hand"


@pytest.mark.parametrize(
    "payload",
    [
        {"action_rules": "all_actions"},
        {"board_roi": [0, 0, 1]},
        {"board_roi": [0, 0, 2, 1]},
        {"board_roi": [1, 0, 0, 1]},
        {"board_roi": [False, 0, 1, 1]},
        {"unknown": True},
    ],
)
def test_full_action_labels_reject_invalid_mode_or_board_rect(tmp_path, payload):
    _, client = action_labels_job(tmp_path)
    with client:
        _, headers, route = prepare_action_job(client)
        assert client.post(route, headers=headers, json=payload).status_code == 422


def test_full_action_labels_prepare_once_causal_indexed_and_raw_unchanged(tmp_path):
    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, route = prepare_action_job(client)
        directory = app.state.jobs.find(job_id)["directory"]
        originals = {
            name: (directory / name).read_bytes()
            for name in ("frames.jsonl", "frames.idx", "frames.times", "manifest.json")
        }
        assert client.get(route).status_code == 409
        response = client.post(route, headers=headers, json={"action_rules": "raise_hand"})
        assert response.status_code == 200, response.text
        meta = response.json()
        assert meta["job_id"] == job_id and meta["frame_count"] == 21
        assert meta["actor_namespace"] == "run:fixture-action-run"
        assert meta["source_sha256"] == "0" * 64 and not meta["ground_truth"]
        assert meta["rule_config"]["calibrated"] is False and not meta["cache_hit"]
        assert len(meta["raw_frames_sha256"]) == 64
        first = client.get(route + "?start=0&count=6").json()
        assert first["generation"] == meta["generation"]
        assert (first["start"], first["count"]) == (0, 6)
        assert [row["frame_index"] for row in first["frames"]] == list(range(6))
        assert first["frames"][0]["action_people"][0]["status"] == "no_suggestion"
        assert first["frames"][1]["action_people"][0]["status"] == "candidate"
        assert first["frames"][5]["action_people"][0]["status"] == "active"
        assert first["frames"][5]["source_pts"] == 500
        assert first["frames"][5]["action_people"][0]["observed_through_s"] == 0.5
        last = client.get(route + "?start=11&count=32").json()
        assert last["count"] == 10
        assert last["frames"][0]["action_people"][0]["status"] == "release_pending"
        assert last["frames"][-1]["action_people"][0]["status"] == "no_suggestion"
        repeat = client.post(route, headers=headers, json={"action_rules": "raise_hand"}).json()
        assert repeat["cache_hit"] and repeat["generation"] == meta["generation"]
        assert originals == {name: (directory / name).read_bytes() for name in originals}


@pytest.mark.parametrize("fault", ["pts", "time_base", "index", "frame_count", "timestamp"])
def test_full_action_labels_reject_bad_native_ledger_without_publishing(tmp_path, fault):
    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, route = prepare_action_job(client)
        directory = app.state.jobs.find(job_id)["directory"]
        path = directory / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "pts":
            rows[1]["source_pts"] = 0
        elif fault == "time_base":
            rows[1]["time_base"] = "1/30"
        elif fault == "index":
            rows[1]["frame_index"] = 5
        elif fault == "timestamp":
            rows[1]["timestamp_seconds"] = 0.12
        else:
            rows.pop()
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        response = client.post(route, headers=headers, json={})
        assert response.status_code == 409, response.text
        assert client.get(route).status_code == 409
        assert not (directory / "action-labels.meta.json").exists()


def test_full_action_labels_stale_raw_and_tampered_cache_require_explicit_rebuild(tmp_path):
    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, route = prepare_action_job(client)
        directory = app.state.jobs.find(job_id)["directory"]
        meta = client.post(route, headers=headers, json={}).json()
        cache = directory / ".action-labels-cache" / meta["generation"] / "frames.jsonl"
        cache.write_bytes(cache.read_bytes().replace(b'"candidate"', b'"corrupted"', 1))
        before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        assert client.get(route).status_code == 409
        assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        repaired = client.post(route, headers=headers, json={}).json()
        assert repaired["generation"] != meta["generation"] and not repaired["cache_hit"]
        raw = directory / "frames.jsonl"
        raw.write_bytes(raw.read_bytes() + b"\n")
        assert client.get(route).status_code == 409


def test_full_action_labels_bounds_gate_and_midbuild_input_change(tmp_path, monkeypatch):
    import classroom_ai.interfaces.web as module

    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, route = prepare_action_job(client)
        job = app.state.jobs.find(job_id)
        for query in ("?start=-1", "?count=0", "?count=33"):
            assert client.get(route + query).status_code == 422
        original = module.RaiseHandState.update
        changed = False

        def update(state, row):
            nonlocal changed
            result = original(state, row)
            if not changed:
                changed = True
                with (job["directory"] / "frames.jsonl").open("ab") as stream:
                    stream.write(b"\n")
            return result

        monkeypatch.setattr(module.RaiseHandState, "update", update)
        assert client.post(route, headers=headers, json={}).status_code == 409
        assert not (job["directory"] / "action-labels.meta.json").exists()


def test_full_action_labels_require_pose_tracking_complete_verified(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        assert (
            client.post(f"/api/jobs/{job_id}/action-labels", headers=headers, json={}).status_code
            == 409
        )


def test_public_source_hash_and_full_board_roi_source_binding(tmp_path):
    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, route = prepare_action_job(client)
        assert client.get(f"/api/jobs/{job_id}").json()["source_sha256"] == "0" * 64
        for binding in (None, "1" * 64):
            response = client.post(
                route,
                headers=headers,
                json={"board_roi": [0, 0, 1, 1], "board_roi_source_sha256": binding},
            )
            assert response.status_code == 422, response.text
        job = app.state.jobs.find(job_id)
        manifest_path = job["directory"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_sha256"] = "a" * 64
        manifest_path.write_text(json.dumps(manifest))
        assert client.get(f"/api/jobs/{job_id}").json()["source_sha256"] == "a" * 64
        manifest_path.write_text("{broken")
        assert client.get(f"/api/jobs/{job_id}").json()["source_sha256"] is None


def test_actions_preview_artifact_inline_range_and_safe_attachment_default(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        directory = app.state.jobs.find(job_id)["directory"]
        (directory / "actions_preview.mp4").write_bytes(b"0123456789")
        prefix = f"/api/jobs/{job_id}/artifacts/"
        download = client.get(prefix + "actions_preview.mp4")
        assert download.status_code == 200 and download.content == b"0123456789"
        assert download.headers["content-disposition"].startswith("attachment;")
        inline = client.get(
            prefix + "actions_preview.mp4?inline=true", headers={"Range": "bytes=2-5"}
        )
        assert inline.status_code == 206 and inline.content == b"2345"
        assert inline.headers["content-type"] == "video/mp4"
        assert inline.headers["content-range"] == "bytes 2-5/10"
        assert inline.headers["content-disposition"].startswith("inline;")
        # inline is a video presentation option, never a way to execute JSON/HTML.
        ordinary = client.get(prefix + "manifest.json?inline=true")
        assert ordinary.status_code == 200
        assert ordinary.headers["content-disposition"].startswith("attachment;")
        assert client.get(prefix + "unlisted.mp4?inline=true").status_code == 404
        assert "actions_preview.mp4" in client.get(f"/api/jobs/{job_id}").json()["artifacts"]


def test_full_action_labels_digest_cache_reuses_stable_file_identity(tmp_path, monkeypatch):
    import classroom_ai.interfaces.web as module

    app, client = action_labels_job(tmp_path)
    digests = []
    original = module.hashlib.file_digest

    def digest(stream, *args, **kwargs):
        digests.append(str(stream.name))
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(module.hashlib, "file_digest", digest)
    with client:
        job_id, headers, route = prepare_action_job(client)
        assert (
            client.post(route, headers=headers, json={"action_rules": "raise_hand"}).status_code
            == 200
        )
        assert client.get(route + "?count=1").status_code == 200
        digests.clear()
        response = client.get(route + "?start=10&count=1")
        assert response.status_code == 200, response.text
        assert not digests, "unchanged windows must not hash complete raw/derived videos"
        assert "stable_stat" in response.json()["integrity_scope"]
        raw = app.state.jobs.find(job_id)["directory"] / "frames.jsonl"
        replacement = raw.with_suffix(".replacement")
        replacement.write_bytes(raw.read_bytes())
        old = raw.stat()
        os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
        replacement.replace(raw)
        assert client.get(route).status_code == 409
        assert str(raw) in digests


def test_full_action_labels_rule_implementation_change_requires_explicit_rebuild(
    tmp_path, monkeypatch
):
    import classroom_ai.interfaces.web as module

    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, route = prepare_action_job(client)
        first = client.post(route, headers=headers, json={"action_rules": "raise_hand"}).json()
        assert len(first["rule_implementation_sha256"]) == 64
        assert set(first["rule_implementation_sources"]) == {
            "classroom_ai/video/actions.py",
            "classroom_ai/video/pose_rules.py",
            "classroom_ai/video/pose_quality.py",
        }
        # Simulate the implementation constants in a newly deployed application;
        # changing files under an already imported process requires a restart.
        sources = dict(first["rule_implementation_sources"])
        sources["classroom_ai/video/actions.py"] = "f" * 64
        fingerprint = hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        monkeypatch.setattr(module, "ACTION_RULE_IMPLEMENTATION_SOURCES", sources)
        monkeypatch.setattr(module, "ACTION_RULE_IMPLEMENTATION_SHA256", fingerprint)
        directory = app.state.jobs.find(job_id)["directory"]
        before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        assert client.get(route).status_code == 409
        assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        second = client.post(route, headers=headers, json={"action_rules": "raise_hand"}).json()
        assert second["generation"] != first["generation"] and not second["cache_hit"]
        assert second["rule_implementation_sha256"] == fingerprint
        assert client.get(route).status_code == 200


def test_replay_board_roi_submit_hash_binding_and_config_propagation(tmp_path, monkeypatch):
    app, client = setup_app(tmp_path, completed_runner)
    app.state.jobs.models["pose"] = {
        "id": "pose",
        "name": "fixture-pose.pt",
        "task": "pose",
        "path": tmp_path / "weights" / "test.pt",
    }
    monkeypatch.setattr(app.state.jobs.executor, "submit", lambda *args: None)
    with client:
        body = dict(
            model_id="pose",
            analysis_task="pose",
            processing_mode="replay",
            tracking="bytetrack",
            action_rules="basic_pose",
            board_roi=[0.1, 0.2, 0.8, 0.9],
        )
        for binding in (None, "0" * 64):
            response = submit(client, **body, board_roi_source_sha256=binding)
            assert response.status_code == 422, response.text
        digest = hashlib.sha256(b"fake").hexdigest()
        response = submit(client, **body, board_roi_source_sha256=digest)
        assert response.status_code == 202, response.text
        data = response.json()
        assert data["source_sha256"] is None
        config = app.state.jobs.find(data["id"])["config"]
        assert config.action_rules == "basic_pose" and config.board_roi == [0.1, 0.2, 0.8, 0.9]
        assert config.board_roi_source_sha256 == digest
        assert data["replay_settings"]["board_roi_source_sha256"] == digest


def test_full_action_labels_per_job_lock_does_not_hold_global_registry(tmp_path, monkeypatch):
    import classroom_ai.interfaces.web as module

    app, client = action_labels_job(tmp_path)
    with client:
        job_id, _, _ = prepare_action_job(client)
        job = app.state.jobs.find(job_id)
        started, release = threading.Event(), threading.Event()
        original = module.RaiseHandState.update

        def update(state, row):
            if row["frame_index"] == 0:
                started.set()
                assert release.wait(3)
            return original(state, row)

        monkeypatch.setattr(module.RaiseHandState, "update", update)
        results, errors = [], []

        def prepare():
            try:
                results.append(
                    app.state.jobs.prepare_action_labels(
                        job, module.ActionLabelsInput(action_rules="raise_hand")
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        first, second = threading.Thread(target=prepare), threading.Thread(target=prepare)
        first.start()
        try:
            assert started.wait(3)
            second.start()
            acquired = app.state.jobs.lock.acquire(timeout=0.2)
            if acquired:
                app.state.jobs.lock.release()
            assert acquired
        finally:
            release.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
        assert not first.is_alive() and not second.is_alive() and not errors
        assert len(results) == 2 and results[0]["generation"] == results[1]["generation"]
        assert sorted(result["cache_hit"] for result in results) == [False, True]


@pytest.mark.parametrize("name", ["action-labels.meta.json", "manifest.json", "frames.jsonl"])
def test_full_action_labels_malformed_json_is_conflict_not_server_error(tmp_path, name):
    app, client = action_labels_job(tmp_path)
    with client:
        job_id, headers, route = prepare_action_job(client)
        assert (
            client.post(route, headers=headers, json={"action_rules": "raise_hand"}).status_code
            == 200
        )
        directory = app.state.jobs.find(job_id)["directory"]
        (directory / name).write_text("[]\n")
        assert client.get(route).status_code == 409
        if name != "action-labels.meta.json":
            assert (
                client.post(route, headers=headers, json={"action_rules": "raise_hand"}).status_code
                == 409
            )


def completed_job(tmp_path):
    app, client = setup_app(tmp_path, completed_runner)
    app.state.jobs.preview = lambda job: verified_preview(app, job)
    return app, client


def pilot_manifest_fixture(app, job_id):
    job = app.state.jobs.find(job_id)
    directory = job["directory"]
    job["source_meta"]["time_base"] = "1/1000"
    rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    with (
        (directory / "frames.jsonl").open("wb") as frames,
        (directory / "frames.idx").open("wb") as offsets,
    ):
        for row in rows:
            row.update(source_pts=round(row["timestamp_seconds"] * 1000), time_base="1/1000")
            offsets.write(struct.pack("<Q", frames.tell()))
            frames.write((json.dumps(row) + "\n").encode())
    path = app.state.jobs.directory / "pilot_fixture" / "pilot_manifest.json"
    path.parent.mkdir()
    value = {
        "schema_version": 2,
        "source_sha256": "0" * 64,
        "source": dict(job["source_meta"]),
        "approved_source_split": "train",
        "candidates": [
            {
                "candidate_id": "first",
                "source_first_frame": 0,
                "source_last_frame": 0,
                "source_start_pts": 0,
                "source_last_pts": 0,
                "start_s": 0,
                "end_s": 0.033,
                "frame_count": 1,
            },
            {
                "candidate_id": "second",
                "source_first_frame": 1,
                "source_last_frame": 2,
                "source_start_pts": 33,
                "source_last_pts": 67,
                "start_s": 0.033,
                "end_s": 0.1,
                "frame_count": 2,
            },
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_pilot_context_is_read_only_exact_pts_and_branch_scoped(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        path, value = pilot_manifest_fixture(app, job_id)
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        for branch in ("group-a", "group-b"):
            response = client.get(f"/api/jobs/{job_id}/pilot-context?reviewer_id={branch}")
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["job_id"] == job_id and data["review_branch"] == branch
            assert data["source_sha256"] == value["source_sha256"]
            assert data["pilot"]["approved_source_split"] == "train"
            assert data["pilot"]["windows"][1]["start_s"] == 0.033
            assert data["pilot"]["windows"][1]["source_start_pts"] == 33
            assert data["pilot"]["windows"][1]["time_base"] == "1/1000"
            assert not data["audit"]["declared_actor_scope_complete"]
            assert all(w["status"] == "unstarted" for w in data["audit"]["windows"])
            assert (
                client.get(f"/api/jobs/{job_id}/gt-bundle?reviewer_id={branch}").status_code == 409
            )
        assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        assert path.is_file()


def test_pilot_context_absence_is_optional_and_cannot_reveal_another_branch(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        empty = client.get(prefix + "/pilot-context?reviewer_id=group-a")
        assert empty.status_code == 200, empty.text
        assert empty.json()["pilot"] is None and empty.json()["audit"] is None
        pilot_manifest_fixture(app, job_id)
        context = client.get(prefix + "/review-context?reviewer_id=group-a").json()
        human = {**context["scene_segments"][0], "track_id": "human:" + "a" * 32}
        payload = {
            "events": [],
            "coverage": [dict(human, label="raise_hand", review="reviewed")],
            "human_actors": [human],
            "actor_links": [],
            "review_metadata": {
                "reviewer_id": "group-a",
                "rubric_version": "v1",
                "source_group": "source",
                "split": "train",
                "adjudication_state": "reviewed",
            },
        }
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        assert (
            client.post(
                prefix + "/annotations?reviewer_id=group-a", headers=headers, json=payload
            ).status_code
            == 200
        )
        assert client.get(prefix + "/gt-bundle?reviewer_id=group-a").status_code == 200
        first = client.get(prefix + "/pilot-context?reviewer_id=group-a").json()
        other = client.get(prefix + "/pilot-context?reviewer_id=group-b").json()
        assert first["audit"]["saved_snapshot_available"]
        assert all(w["status"] == "partial" for w in first["audit"]["windows"])
        assert not other["audit"]["saved_snapshot_available"]
        assert all(w["status"] == "unstarted" for w in other["audit"]["windows"])
        assert "human:" not in json.dumps(other)


@pytest.mark.parametrize("fault", ["timestamp", "pts", "duration", "frame", "duplicate"])
def test_pilot_context_rejects_unverified_or_ambiguous_source_windows(tmp_path, fault):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        path, value = pilot_manifest_fixture(app, job_id)
        if fault == "timestamp":
            value["candidates"][0]["start_s"] = 0.001
        elif fault == "pts":
            value["candidates"][0]["source_start_pts"] = 1
        elif fault == "duration":
            value["source"]["duration_seconds"] = 1
        elif fault == "frame":
            value["candidates"][0]["source_first_frame"] = True
        else:
            duplicate = path.parent.parent / "pilot_duplicate" / path.name
            duplicate.parent.mkdir()
            duplicate.write_text(json.dumps(value), encoding="utf-8")
        path.write_text(json.dumps(value), encoding="utf-8")
        response = client.get(f"/api/jobs/{job_id}/pilot-context?reviewer_id=group-a")
        assert response.status_code == 200, response.text
        assert response.json()["pilot"] is None and response.json()["audit"] is None
        assert response.json()["warnings"]


def test_web_pose_gpu_and_nms_options_reach_shared_pipeline(tmp_path):
    received = []

    def runner(config, **kwargs):
        received.append(config)
        return completed_runner(config, **kwargs)

    app, client = setup_app(tmp_path, runner)
    app.state.jobs.models["pose"] = {
        "id": "pose",
        "name": "fixture-pose.pt",
        "task": "pose",
        "path": tmp_path / "weights" / "test.pt",
    }
    app.state.jobs.preview = lambda job: verified_preview(app, job)
    with client:
        response = submit(
            client, model_id="pose", analysis_task="pose", device="intel:gpu", nms_iou=0.5
        )
        assert response.status_code == 202, response.text
        assert wait_job(client, response.json()["id"])["state"] == "complete"
        assert received[0].device == "intel:gpu"
        assert received[0].nms_iou == 0.5
        assert submit(client, device="intel:gpu").status_code == 422
        assert submit(client, nms_iou=1.1).status_code == 422


def test_model_cache_reloads_when_checkpoint_bytes_change_at_same_path(tmp_path, monkeypatch):
    from classroom_ai.inference import pose
    from classroom_ai.interfaces.web import DemoJobs

    created = []

    def inspector(weights, device):
        instance = (weights.read_bytes(), device, object())
        created.append(instance)
        return instance

    monkeypatch.setattr(pose, "PoseInspector", inspector)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"first")
    jobs = DemoJobs(tmp_path, lambda *args, **kwargs: None)
    try:
        first = jobs.predictor("pose", weights, "intel:gpu")
        assert jobs.predictor("pose", weights, "intel:gpu") is first
        weights.write_bytes(b"other")
        second = jobs.predictor("pose", weights, "intel:gpu")
        assert second is not first
        assert second[0] == b"other" and len(created) == 2
    finally:
        jobs.executor.shutdown(wait=True)


def test_independent_missed_actor_review_and_gt_export_preserve_raw_observations(tmp_path):
    app, client = completed_job(tmp_path)
    human = "human:" + "a" * 32
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        context = client.get(prefix + "/review-context").json()
        assert context["evaluation_ready"] is False
        assert client.get(prefix + "/gt-bundle").status_code == 409
        scene = context["scene_segments"][0]
        span = {**scene, "track_id": human}
        area = {**span, "label": "raise_hand", "review": "reviewed"}
        value = {
            "events": [],
            "coverage": [area],
            "human_actors": [span],
            "actor_links": [],
            "review_metadata": {
                "reviewer_id": "r1",
                "rubric_version": "v1",
                "source_group": "pilot-source",
                "split": "train",
                "adjudication_state": "reviewed",
            },
            "actor_namespace": context["actor_namespace"],
            "source_sha256": context["source_sha256"],
        }
        raw_path = app.state.jobs.find(job_id)["directory"] / "frames.jsonl"
        raw_before = raw_path.read_bytes()
        response = client.post(prefix + "/annotations", headers=headers, json=value)
        assert response.status_code == 200, response.text
        assert response.json()["evaluation_ready"] is True
        restored = client.get(prefix + "/review-context").json()
        assert restored["human_actors"] == [span]
        assert restored["model_track_segments"] != restored["track_segments"]
        bundle = client.get(prefix + "/gt-bundle")
        assert bundle.status_code == 200
        assert bundle.json()["human_actors"] == [span]
        assert bundle.json()["review_metadata"]["reviewer_id"] == "r1"
        assert bundle.json()["events"] == []  # Reviewed negative coverage is valid GT.
        assert client.get(prefix + "/readiness?reviewer_id=r1").json()["evaluation_ready"] is True
        assert client.get(prefix + "/gt-bundle?reviewer_id=r1").json()["human_actors"] == [span]
        assert client.get(prefix + "/readiness?reviewer_id=r2").json()["evaluation_ready"] is False
        assert client.get(prefix + "/gt-bundle?reviewer_id=r2").status_code == 409
        assert raw_path.read_bytes() == raw_before
        bad = {**value, "actor_namespace": "run:other"}
        assert client.post(prefix + "/annotations", headers=headers, json=bad).status_code == 422


def test_rule_prediction_export_retains_source_and_actor_namespace_without_gt_promotion(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        context = client.get(prefix + "/review-context").json()
        response = client.post(prefix + "/proposals", headers=headers, json={})
        assert response.status_code == 200
        assert response.json()["source_sha256"] == context["source_sha256"]
        assert response.json()["actor_namespace"] == context["actor_namespace"]
        assert response.json()["kind"] == "model_prediction"
        assert client.get(prefix + "/gt-bundle").status_code == 409


def test_frame_ranges_are_exact_bounded_and_verified(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        response = client.get(prefix + "/frames?start=1&count=32")
        assert response.status_code == 200
        bundle = response.json()
        assert bundle["start"] == 1 and bundle["count"] == 2
        assert bundle["frames"] == [client.get(prefix + f"/frame/{i}").json() for i in (1, 2)]
        for query in ("start=-1", "count=0", "count=33", "start=1.5", "start=..%5Csecret"):
            assert client.get(prefix + "/frames?" + query).status_code == 422
        assert client.get(prefix + "/frames?start=3").status_code == 404
        job = app.state.jobs.find(job_id)
        app.state.jobs.update(job, state="processing")
        assert client.get(prefix + "/frames").status_code == 409
        app.state.jobs.update(job, state="complete", preview_contract=None)
        assert client.get(prefix + "/frames").status_code == 409


def test_frame_ranges_limit_read_and_response_bytes(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        directory = app.state.jobs.find(job_id)["directory"]
        # Thirty-three rows establish the count cap independently from the byte cap.
        with (
            (directory / "frames.jsonl").open("wb") as rows,
            (directory / "frames.idx").open("wb") as offsets,
        ):
            for index in range(33):
                offsets.write(struct.pack("<Q", rows.tell()))
                rows.write((json.dumps({"frame_index": index}) + "\n").encode())
        route = f"/api/jobs/{job_id}/frames"
        response = client.get(route)
        assert response.status_code == 200 and response.json()["count"] == 32
        assert [r["frame_index"] for r in response.json()["frames"]] == list(range(32))
        row = json.dumps({"padding": "x" * (2 * 1024 * 1024)}).encode() + b"\n"
        (directory / "frames.jsonl").write_bytes(row + row)
        (directory / "frames.idx").write_bytes(struct.pack("<QQ", 0, len(row)))
        response = client.get(route + "?count=1")
        assert response.status_code == 200 and len(response.content) <= 4 * 1024 * 1024
        assert client.get(route + "?count=2").status_code == 413
        # An oversized first row must fail before reading/serializing the full row.
        (directory / "frames.jsonl").write_bytes(
            b'{"padding":"' + b"x" * (4 * 1024 * 1024) + b'"}\n'
        )
        (directory / "frames.idx").write_bytes(struct.pack("<Q", 0))
        response = client.get(route)
        assert response.status_code == 413
        assert len(response.content) < 1024


def verified_preview(app, job):
    # This fixture replaces the codec stage, whose actual validation and marker
    # publication are exercised separately by served real-codec fixtures.
    (job["directory"] / "preview.mp4").write_bytes(b"video")
    app.state.jobs.update(job, preview_contract=1)


def test_legacy_completed_job_requires_verified_preview_for_review(tmp_path):
    app, client = completed_job(tmp_path)
    app.state.jobs.preview = lambda job: None
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        job = app.state.jobs.find(job_id)
        (job["directory"] / "preview.mp4").write_bytes(b"legacy-unverified")
    restored = create_app(tmp_path, runner=completed_runner)
    with TestClient(restored, base_url="http://127.0.0.1") as client:
        prefix = f"/api/jobs/{job_id}"
        public = client.get(prefix).json()
        assert public.get("preview_verified") is False
        assert public.get("preview_warning")
        assert client.get(prefix + "/artifacts/preview.mp4").content == b"legacy-unverified"
        assert client.get(prefix + "/review-context").status_code == 409
        assert client.get(prefix + "/frame/0").status_code == 409
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        for route in ("annotations", "proposals"):
            response = client.post(
                prefix + "/" + route, headers=headers, json={"events": [], "coverage": []}
            )
            assert response.status_code == 409
            assert "xác minh" in response.json()["detail"]


def test_preview_cancel_during_probe_finishes_worker_before_scan_eof(tmp_path, monkeypatch):
    from classroom_ai.video import media

    entered, eof, exited = threading.Event(), threading.Event(), threading.Event()
    original_probe = media.probe_source

    def long_probe(*args, **kwargs):
        entered.set()
        try:
            while not eof.wait(0.01):
                cancel = kwargs.get("cancel")
                if cancel and cancel():
                    raise RuntimeError("cancelled probe")
            return original_probe(*args, **kwargs)
        finally:
            exited.set()

    app, client = setup_app(tmp_path, completed_runner)
    monkeypatch.setattr(media, "probe_source", long_probe)
    with client:
        job_id = submit(client).json()["id"]
        assert entered.wait(2)
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        client.post(f"/api/jobs/{job_id}/cancel", headers=headers)
        interrupted = exited.wait(0.7)
        # Always release the synthetic EOF even on the old implementation.
        eof.set()
        assert interrupted, "preview waited for scan EOF after cancellation"
        assert wait_job(client, job_id)["state"] == "cancelled"


def test_encoded_windows_review_traversal_and_noncanonical_names_are_rejected(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        directory = app.state.jobs.find(job_id)["directory"]
        (tmp_path / "secret.json").write_text('{"private":true}')
        name = "review-x" + "%5C.." * 7 + "%5Csecret.json"
        assert client.get(f"/api/jobs/{job_id}/artifacts/{name}").status_code == 404
        (directory / "review-arbitrary.json").write_text('{"invalid":true}')
        assert client.get(f"/api/jobs/{job_id}/artifacts/review-arbitrary.json").status_code == 404


def test_allowed_artifact_resolves_inside_job(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        directory = app.state.jobs.find(job_id)["directory"]
        outside = tmp_path / "private.json"
        outside.write_text('{"private":true}')
        (directory / "manifest.json").unlink()
        try:
            (directory / "manifest.json").symlink_to(outside)
        except OSError:
            pytest.skip("Windows symlink privilege unavailable")
        assert client.get(f"/api/jobs/{job_id}/artifacts/manifest.json").status_code == 404


def test_cancel_cannot_overwrite_interleaved_terminal_update(tmp_path, monkeypatch):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        jobs = app.state.jobs
        job = jobs.find(job_id)
        jobs.update(job, state="rendering")
        original_touch = type(job["stop"]).touch
        finished = threading.Event()
        threads = []

        def touch(path, *args, **kwargs):
            def finish():
                jobs.update(job, state="complete")
                finished.set()

            thread = threading.Thread(target=finish)
            threads.append(thread)
            thread.start()
            finished.wait(0.2)
            return original_touch(path, *args, **kwargs)

        monkeypatch.setattr(type(job["stop"]), "touch", touch)
        token = client.get("/api/config").json()["token"]
        response = client.post(f"/api/jobs/{job_id}/cancel", headers={"X-Demo-Token": token})
        assert response.status_code == 200
        for thread in threads:
            thread.join(2)
        assert job["state"] == "complete"


def test_hydrating_many_old_jobs_is_bounded_and_retains_active(tmp_path):
    from classroom_ai.interfaces.web import DemoJobs

    jobs = DemoJobs(tmp_path, lambda *a, **k: None)
    active = {"id": "active", "state": "processing", "stop": tmp_path / "active.stop"}
    jobs.jobs["active"] = active
    try:
        for index in range(35):
            job_id = f"{index:032x}"
            home = jobs.directory / "jobs" / job_id
            home.mkdir(parents=True)
            (home / "job.json").write_text(json.dumps({"id": job_id, "state": "complete"}))
            restored = jobs.find(job_id)
            restored["track_segments"] = [{"track_id": index}]
            assert restored["state"] == "complete"
            assert len(jobs.jobs) <= 20
            assert jobs.jobs["active"] is active
    finally:
        jobs.close()


def test_snapshot_failure_midwrite_does_not_publish_or_hide_previous(tmp_path, monkeypatch):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        token = client.get("/api/config").json()["token"]
        headers = {"X-Demo-Token": token}
        assert (
            client.post(
                prefix + "/annotations", headers=headers, json={"events": [], "coverage": []}
            ).status_code
            == 200
        )
        directory = app.state.jobs.find(job_id)["directory"]
        previous = list(directory.glob("review-*.json"))
        original_open = type(directory).open

        class BrokenWrite:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.handle.close()

            def write(self, data):
                self.handle.write(data[:10])
                raise OSError("simulated disk interruption")

        def broken_open(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            if path.name.startswith("review-") and "x" in mode:
                return BrokenWrite(handle)
            return handle

        with monkeypatch.context() as patcher:
            patcher.setattr(type(directory), "open", broken_open)
            with pytest.raises(OSError, match="disk interruption"):
                client.post(
                    prefix + "/annotations", headers=headers, json={"events": [], "coverage": []}
                )
        assert list(directory.glob("review-*.json")) == previous
        assert client.get(prefix + "/review-context").status_code == 200


def test_invalid_newest_review_recovers_previous_with_warning(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        token = client.get("/api/config").json()["token"]
        coverage = [
            {
                "segment_id": 0,
                "track_id": 1,
                "label": "standing",
                "start_s": 0,
                "end_s": 0.1,
                "review": "reviewed",
            }
        ]
        assert (
            client.post(
                prefix + "/annotations",
                headers={"X-Demo-Token": token},
                json={"events": [], "coverage": coverage},
            ).status_code
            == 200
        )
        directory = app.state.jobs.find(job_id)["directory"]
        corrupt = directory / ("review-" + "f" * 32 + ".json")
        corrupt.write_text('{"events":')
        future = time.time() + 10
        os.utime(corrupt, (future, future))
        response = client.get(prefix + "/review-context")
        assert response.status_code == 200
        assert response.json()["coverage"] == coverage
        assert response.json().get("warnings")


def test_review_uuid_collision_never_overwrites_committed_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from classroom_ai.interfaces import web

    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        monkeypatch.setattr(web.uuid, "uuid4", lambda: SimpleNamespace(hex="b" * 32))
        first = client.post(
            prefix + "/annotations", headers=headers, json={"events": [], "coverage": []}
        )
        assert first.status_code == 200
        directory = app.state.jobs.find(job_id)["directory"]
        path = directory / first.json()["snapshot"]
        original = path.read_bytes()
        coverage = [
            {
                "segment_id": 0,
                "track_id": 1,
                "label": "standing",
                "start_s": 0,
                "end_s": 0.1,
                "review": "reviewed",
            }
        ]
        with pytest.raises(FileExistsError):
            client.post(
                prefix + "/annotations", headers=headers, json={"events": [], "coverage": coverage}
            )
        assert path.read_bytes() == original
        assert not list(directory.glob("review-*.tmp"))


def test_api_overflow_coverage_rejected_before_snapshot_creation(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {
            "X-Demo-Token": client.get("/api/config").json()["token"],
            "Content-Type": "application/json",
        }
        response = client.post(
            prefix + "/annotations",
            headers=headers,
            content='{"events":[],"coverage":[{"segment_id":0,"track_id":1,"label":"standing","start_s":0,"end_s":0.1,"review":"reviewed","extra":1e999}]}',
        )
        assert response.status_code == 422
        assert not list(app.state.jobs.find(job_id)["directory"].glob("review-*"))


def test_reviewed_rule_snapshot_keeps_censored_onset_out_of_counts(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        coverage = [
            {
                "segment_id": 0,
                "track_id": 1,
                "label": "raise_hand",
                "start_s": 0,
                "end_s": 0.1,
                "review": "reviewed",
            }
        ]
        event = {
            "segment_id": 0,
            "track_id": 1,
            "label": "raise_hand",
            "start_s": 0.02,
            "end_s": 0.08,
            "review": "reviewed",
            "origin": "rule",
            "uncertain": False,
            "evidence": {"boundary_truncated": {"start": True, "end": False}},
        }
        response = client.post(
            prefix + "/annotations", headers=headers, json={"events": [event], "coverage": coverage}
        )
        assert response.status_code == 200
        bundle = client.get(prefix + "/artifacts/" + response.json()["snapshot"]).json()
        assert bundle["events"][0] == event
        total = bundle["summary"]["session"]["raise_hand"]
        assert total["count"] == 0
        assert total["observed_duration_s"] == pytest.approx(0.06)


def wait_job(client, job_id):
    for _ in range(100):
        row = client.get(f"/api/jobs/{job_id}").json()
        if row["state"] in ("complete", "failed", "cancelled"):
            return row
        time.sleep(0.02)
    pytest.fail("job failed to terminate")


def setup_app(tmp_path, runner):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "test.pt").write_bytes(b"fake")
    (tmp_path / "clip.mp4").write_bytes(b"fake")
    app = create_app(tmp_path, runner=runner)
    return app, TestClient(app, base_url="http://127.0.0.1")


def submit(client, **kwargs):
    settings = client.get("/api/config").json()
    model_id = settings["models"][0]["id"]
    return client.post(
        "/api/jobs",
        headers={"X-Demo-Token": settings["token"]},
        json={
            "source": "clip.mp4",
            "model_id": model_id,
            "analysis_task": "detect",
            "device": "cpu",
            "tracking": "off",
            **kwargs,
        },
    )


def test_local_paths_and_mutation_origin_guard(tmp_path):
    _, client = setup_app(tmp_path, lambda *a, **k: None)
    with client:
        assert client.get("/").status_code == 200
        assert client.post("/api/jobs", json={}).status_code == 403
        assert submit(client, source="../secret.mp4").status_code == 422
        assert submit(client, model_id="../secret.pt").status_code == 422
        token = client.get("/api/config").json()["token"]
        assert (
            client.post(
                "/api/jobs", headers={"Origin": "https://evil.test", "X-Demo-Token": token}, json={}
            ).status_code
            == 403
        )
        assert client.get("/api/config", headers={"Host": "evil.test"}).status_code == 400


def test_failure_and_cancel_are_terminal_and_single_worker(tmp_path):
    entered = threading.Event()

    def runner(config, *, stop_file, **kwargs):
        entered.set()
        for _ in range(200):
            if stop_file.exists():
                return None
            time.sleep(0.01)
        raise RuntimeError("test runner exceeded cancel deadline")

    _, client = setup_app(tmp_path, runner)
    with client:
        response = submit(client)
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert entered.wait(2)
        assert submit(client).status_code == 409
        token = client.get("/api/config").json()["token"]
        assert (
            client.post(f"/api/jobs/{job_id}/cancel", headers={"X-Demo-Token": token}).status_code
            == 200
        )
        assert wait_job(client, job_id)["state"] == "cancelled"


def test_failed_job_does_not_publish_complete(tmp_path):
    def runner(*args, **kwargs):
        raise RuntimeError("intentional broken inference")

    _, client = setup_app(tmp_path, runner)
    with client:
        job_id = submit(client).json()["id"]
        row = wait_job(client, job_id)
        assert row["state"] == "failed"
        assert "intentional broken inference" in row["error"]
        assert client.get(f"/api/jobs/{job_id}/artifacts/annotated.mp4").status_code == 404


def test_upload_is_bounded_and_private(tmp_path):
    app, client = setup_app(tmp_path, lambda *a, **k: None)
    app.state.max_upload_bytes = 5
    with client:
        token = client.get("/api/config").json()["token"]
        headers = {"X-Demo-Token": token, "Content-Type": "application/octet-stream"}
        assert (
            client.post(
                "/api/uploads?name=video.mp4", content=b"123456", headers=headers
            ).status_code
            == 413
        )
        result = client.post("/api/uploads?name=video.mp4", content=b"123", headers=headers)
        assert result.status_code == 201
        path = tmp_path / result.json()["source"]
        assert path.read_bytes() == b"123"
        assert client.get("/api/config").headers["cache-control"] == "no-store"


def completed_runner(config, **kwargs):
    directory = config.output_dir / "fixture-run"
    directory.mkdir(parents=True)
    metadata = {
        "source": {
            "width": 8,
            "height": 8,
            "fps": 30,
            "frame_count": 3,
            "duration_seconds": 0.1,
            "variable_frame_rate": False,
        }
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "report.json").write_text(json.dumps({"status": "completed", "error": None}))
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "source_sha256": "0" * 64})
    )
    with (
        (directory / "frames.jsonl").open("wb") as frames,
        (directory / "frames.idx").open("wb") as offsets,
        (directory / "frames.times").open("wb") as times,
    ):
        for index, timestamp in enumerate((0, 0.033, 0.067)):
            offsets.write(struct.pack("<Q", frames.tell()))
            times.write(struct.pack("<d", timestamp))
            row = {
                "frame_index": index,
                "timestamp_seconds": timestamp,
                "segment_id": 0,
                "detections": [],
                "display_tracks": [],
                "tracks": [{"track_id": 1, "confirmed": True, "state": "observed"}],
            }
            frames.write((json.dumps(row) + "\n").encode())
    return directory


def test_complete_replay_review_and_restart_preserve_source_contract(tmp_path):
    app, client = setup_app(tmp_path, completed_runner)
    app.state.jobs.preview = lambda job: verified_preview(app, job)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        assert client.get(f"/api/jobs/{job_id}/frame/2").json()["timestamp_seconds"] == 0.067
        assert client.get(f"/api/jobs/{job_id}/frame/-1").status_code == 404
        prefix = f"/api/jobs/{job_id}"
        assert client.get(prefix + "/artifacts/frames.times").content == struct.pack(
            "<ddd", 0, 0.033, 0.067
        )
        response = client.get(prefix + "/artifacts/preview.mp4", headers={"Range": "bytes=0-1"})
        assert response.status_code == 206 and response.content == b"vi"
        context = client.get(prefix + "/review-context").json()
        assert context["track_segments"][0]["end_s"] == 0.1
        coverage = [
            {
                "segment_id": 0,
                "track_id": 1,
                "start_s": 0,
                "end_s": 0.1,
                "label": label,
                "review": "reviewed",
            }
            for label in ("standing", "raise_hand")
        ]
        events = [
            {
                "segment_id": 0,
                "track_id": 1,
                "label": label,
                "start_s": 0.01,
                "end_s": 0.08,
                "origin": "human",
                "review": "reviewed",
                "uncertain": False,
                "evidence": {"boundary_truncated": {"start": False, "end": False}},
            }
            for label in ("standing", "raise_hand")
        ]
        token = client.get("/api/config").json()["token"]
        headers = {"X-Demo-Token": token}
        assert (
            client.post(
                prefix + "/annotations", headers=headers, json={"events": events, "coverage": []}
            ).status_code
            == 422
        )
        snapshots = []
        for _ in range(2):
            saved = client.post(
                prefix + "/annotations",
                headers=headers,
                json={"events": events, "coverage": coverage},
            )
            assert saved.status_code == 200, saved.text
            snapshots.append(saved.json()["snapshot"])
        assert snapshots[0] != snapshots[1]
        bundle = client.get(prefix + "/artifacts/" + snapshots[0]).json()
        assert bundle["source_sha256"] == "0" * 64 and bundle["coverage"] == coverage
    restored = create_app(tmp_path, runner=completed_runner)
    with TestClient(restored, base_url="http://127.0.0.1") as restarted:
        assert restarted.get(prefix).json()["state"] == "complete"
        assert restarted.get(prefix).json()["preview_verified"] is True
        assert restarted.get(prefix + "/review-context").json()["events"] == events


def test_variable_pts_and_missing_observation_split_review_extents(tmp_path):
    def runner(config, **kwargs):
        directory = completed_runner(config, **kwargs)
        path = directory / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["timestamp_seconds"] = 0
        rows[1]["timestamp_seconds"] = 0.02
        rows[1]["tracks"] = []
        rows[2]["timestamp_seconds"] = 0.09
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return directory

    app, client = setup_app(tmp_path, runner)
    app.state.jobs.preview = lambda job: verified_preview(app, job)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        extents = client.get(f"/api/jobs/{job_id}/review-context").json()["track_segments"]
        assert [(row["start_s"], row["end_s"]) for row in extents] == [(0, 0.02), (0.09, 0.1)]


def test_class_summary_is_read_only_and_distinguishes_unreviewed_from_zero(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        directory = app.state.jobs.find(job_id)["directory"]
        before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        response = client.get(prefix + "/class-summary?window_seconds=0.05")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["schema_version"] == 2
        assert payload["source_sha256"] == "0" * 64
        assert payload["review_snapshot"] is None
        assert len(payload["class_summary"]["windows"]) == 2
        assert payload["class_summary"]["session"]["raise_hand"]["count"] is None
        assert payload["class_summary"]["session"]["raise_hand"]["duration_s"] is None
        assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        coverage = [
            {
                "segment_id": 0,
                "track_id": 1,
                "label": "raise_hand",
                "start_s": 0,
                "end_s": 0.1,
                "review": "reviewed",
            }
        ]
        saved = client.post(
            prefix + "/annotations", headers=headers, json={"events": [], "coverage": coverage}
        )
        assert saved.status_code == 200, saved.text
        payload = client.get(prefix + "/class-summary").json()
        assert payload["review_snapshot"] == saved.json()["snapshot"]
        assert payload["class_summary"]["session"]["raise_hand"]["count"] == 0
        assert payload["class_summary"]["session"]["raise_hand"]["reviewed_coverage_actor_s"] == 0.1
        assert payload["class_summary"]["session"]["stand_up"]["count"] is None


def test_class_summary_latest_snapshot_reload_and_window_uncertainty(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        coverage = [
            {
                "segment_id": 0,
                "track_id": 1,
                "label": "raise_hand",
                "start_s": 0,
                "end_s": 0.1,
                "review": "reviewed",
            }
        ]
        events = [
            dict(
                segment_id=0,
                track_id=1,
                label="raise_hand",
                start_s=0.01,
                end_s=0.08,
                origin="human",
                review="reviewed",
                uncertain=False,
                evidence={"boundary_truncated": {"start": True, "end": False}},
            ),
            dict(
                segment_id=0,
                track_id=1,
                label="unknown",
                start_s=0.02,
                end_s=0.09,
                origin="human",
                review="draft",
                uncertain=True,
                evidence={},
            ),
        ]
        coverage.append(
            {
                "segment_id": 0,
                "track_id": 1,
                "label": "unknown",
                "start_s": 0,
                "end_s": 0.1,
                "review": "draft",
            }
        )
        saved = client.post(
            prefix + "/annotations", headers=headers, json={"events": events, "coverage": coverage}
        )
        assert saved.status_code == 200, saved.text
        response = client.get(prefix + "/class-summary?window_seconds=0.05")
        assert response.status_code == 200, response.text
        payload = response.json()
        totals = payload["class_summary"]
        assert totals["session"]["raise_hand"]["count"] == 0
        assert totals["session"]["raise_hand"]["duration_s"] == pytest.approx(0.07)
        assert totals["session"]["raise_hand"]["truncated_episode_count"] == 1
        assert totals["saved_intervals"] == {"unknown": 1, "uncertain": 1, "draft": 1}
        assert totals["windows"][0]["saved_intervals"] == totals["saved_intervals"]
        assert payload["events"] == events
        assert payload["review_snapshot"] == saved.json()["snapshot"]
    with TestClient(
        create_app(tmp_path, runner=completed_runner), base_url="http://127.0.0.1"
    ) as restored:
        after = restored.get(prefix + "/class-summary?window_seconds=0.05").json()
        assert after == payload


@pytest.mark.parametrize("window", ["0", "-1", "nan", "inf", "0.000001"])
def test_class_summary_rejects_invalid_or_unbounded_windows(tmp_path, window):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        response = client.get(f"/api/jobs/{job_id}/class-summary?window_seconds={window}")
        assert response.status_code == 422, response.text


def test_class_export_is_allowlisted_read_only_and_keeps_unreviewed_metrics_null(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        directory = app.state.jobs.find(job_id)["directory"]
        before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        response = client.get(prefix + "/class-export?window_seconds=0.05")
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["schema_version"] == 1
        assert value["kind"] == "class_observation_summary"
        assert value["class_session"]["raise_hand"]["count"] is None
        assert value["class_session"]["raise_hand"]["duration_s"] is None
        assert value["saved_data_only"] is True and not value["saved_snapshot_available"]
        assert value["source_ref"] == {"sha256": "0" * 64}
        assert len(value["windows"]) == 2
        assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before
        for query in ("0", "nan", "0.000001"):
            assert client.get(prefix + "/class-export?window_seconds=" + query).status_code == 422


def test_class_export_matches_saved_aggregate_without_identity_or_free_text(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}
        area = dict(
            segment_id=0, track_id=1, start_s=0, end_s=0.1, label="raise_hand", review="reviewed"
        )
        event = dict(
            segment_id=0,
            track_id=1,
            start_s=0.01,
            end_s=0.08,
            label="raise_hand",
            review="reviewed",
            origin="human",
            uncertain=False,
            evidence={"note": "PRIVATE_NOTE_SENTINEL"},
        )
        uncertain = {
            **event,
            "label": "unknown",
            "start_s": 0.03,
            "end_s": 0.06,
            "review": "draft",
            "uncertain": True,
        }
        unknown_area = {**area, "label": "unknown", "review": "draft"}
        saved = client.post(
            prefix + "/annotations",
            headers=headers,
            json={
                "events": [event, uncertain],
                "coverage": [area, unknown_area],
                "review_metadata": {"reviewer_id": "PRIVATE_REVIEWER_SENTINEL"},
            },
        )
        assert saved.status_code == 200, saved.text
        full = client.get(prefix + "/class-summary?window_seconds=0.05").json()
        response = client.get(prefix + "/class-export?window_seconds=0.05")
        assert response.status_code == 200, response.text
        exported = response.json()
        assert exported["class_session"] == full["class_summary"]["session"]
        assert exported["uncertainty_counts"] == full["class_summary"]["saved_intervals"]
        assert exported["uncertainty_counts"] == {"unknown": 1, "uncertain": 1, "draft": 1}
        assert exported["saved_snapshot_available"] is True
        assert exported["windows"][1]["labels"] == full["class_summary"]["windows"][1]["labels"]
        assert exported["class_session"]["raise_hand"]["duration_s"] == pytest.approx(0.07)
        assert exported["class_session"]["write_board"]["count"] is None
        assert exported["class_session"]["unknown"]["count"] is None
        forbidden = {
            "events",
            "track_id",
            "segment_id",
            "human_actors",
            "actor_links",
            "track_segments",
            "review_metadata",
            "reviewer_id",
            "notes",
            "evidence",
            "weights",
            "model",
            "config",
            "path",
            "artifacts",
            "detections",
            "keypoints13",
            "keypoints17",
            "image",
            "pixels",
        }

        def inspect(item):
            if isinstance(item, dict):
                assert not forbidden.intersection(item)
                for nested in item.values():
                    inspect(nested)
            elif isinstance(item, list):
                for nested in item:
                    inspect(nested)

        inspect(exported)
        assert "PRIVATE_NOTE_SENTINEL" not in response.text
        assert "PRIVATE_REVIEWER_SENTINEL" not in response.text
        assert str(tmp_path) not in response.text


def test_independent_reviewer_branches_restore_only_their_own_saved_annotations(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        prefix = f"/api/jobs/{job_id}"
        headers = {"X-Demo-Token": client.get("/api/config").json()["token"]}

        def save(reviewer, label):
            event = dict(
                segment_id=0,
                track_id=1,
                start_s=0.01,
                end_s=0.08,
                label=label,
                origin="human",
                review="reviewed",
                uncertain=False,
                evidence={},
            )
            area = dict(
                segment_id=0, track_id=1, start_s=0, end_s=0.1, label=label, review="reviewed"
            )
            value = {
                "events": [event],
                "coverage": [area],
                "review_metadata": {"reviewer_id": reviewer},
            }
            response = client.post(
                prefix + "/annotations?reviewer_id=" + reviewer, headers=headers, json=value
            )
            assert response.status_code == 200, response.text
            return response.json(), value

        saved_a, value_a = save("group-a", "raise_hand")
        branch_b = client.get(prefix + "/review-context?reviewer_id=group-b").json()
        assert branch_b["events"] == branch_b["coverage"] == branch_b["human_actors"] == []
        assert branch_b["review_metadata"] == {
            "reviewer_id": "group-b",
            "adjudication_state": "draft",
        }
        assert branch_b["review_snapshot"] is None and not branch_b["evaluation_ready"]
        assert branch_b["review_branch"] == "group-b"
        summary_b = client.get(prefix + "/class-summary?reviewer_id=group-b").json()
        assert summary_b["class_summary"]["session"]["raise_hand"]["count"] is None
        assert summary_b["review_snapshot"] is None
        assert not client.get(prefix + "/class-export?reviewer_id=group-b").json()[
            "saved_snapshot_available"
        ]
        saved_b, _ = save("group-b", "stand_up")
        branch_a = client.get(prefix + "/review-context?reviewer_id=group-a").json()
        assert branch_a["events"] == value_a["events"]
        assert branch_a["review_snapshot"] == saved_a["snapshot"]
        assert branch_a["actor_namespace"] == branch_b["actor_namespace"]
        assert branch_a["source_sha256"] == branch_b["source_sha256"]
        assert (
            client.get(prefix + "/review-context").json()["review_snapshot"] == saved_b["snapshot"]
        )
        wrong = {**value_a, "review_metadata": {"reviewer_id": "group-b"}}
        assert (
            client.post(
                prefix + "/annotations?reviewer_id=group-a", headers=headers, json=wrong
            ).status_code
            == 422
        )
        for invalid in ("..%2Fsecret", "a" * 41, "bad%20name"):
            assert client.get(prefix + "/review-context?reviewer_id=" + invalid).status_code == 422


def test_reviewer_branch_skips_nonobject_snapshot_without_crashing_or_rewriting_it(tmp_path):
    app, client = completed_job(tmp_path)
    with client:
        job_id = submit(client).json()["id"]
        assert wait_job(client, job_id)["state"] == "complete"
        directory = app.state.jobs.find(job_id)["directory"]
        corrupt = directory / ("review-" + "f" * 32 + ".json")
        corrupt.write_text("[]")
        response = client.get(f"/api/jobs/{job_id}/review-context?reviewer_id=group-b")
        assert response.status_code == 200
        assert response.json()["events"] == []
        assert response.json()["warnings"]
        assert corrupt.read_text() == "[]"
