"""No-NN export orchestration, temporal boundaries and failure publication."""

import hashlib
import json
import tempfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from classroom_ai.paths import PROJECT_ROOT
from classroom_ai.video import media, rendering
from classroom_ai.video.core import Frame


@pytest.fixture
def tmp_path():
    # Windows sandbox temp can reject rename; publication tests use workspace.
    scratch = PROJECT_ROOT / ".temp" / "saved_action_export_tests"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        yield Path(directory)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    run = tmp_path / "completed"
    run.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-source-no-NN")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    meta = media.VideoSource(
        source,
        Fraction(5),
        64,
        48,
        4,
        False,
        start_time=0.1,
        time_base="1/1000",
        vfr=True,
        duration_seconds=0.8,
    )
    manifest = dict(
        schema_version=2,
        status="completed",
        run_id="fixture",
        source=meta.metadata(),
        source_sha256=sha,
        config=dict(mode="full", analysis_task="pose", tracking="bytetrack"),
    )
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for i in range(4):
        points = [[30, 30, 1] for _ in range(17)]
        points[0], points[5], points[6] = [30, 15, 1], [20, 35, 1], [40, 35, 1]
        points[7:11] = [[20, 20, 1], [40, 40, 1], [20, 5, 1], [40, 45, 1]]
        rows.append(
            dict(
                schema_version=2,
                analysis_task="pose",
                frame_index=i,
                source_pts=100 + i * 200,
                time_base="1/1000",
                timestamp_seconds=i * 0.2,
                segment_id=0,
                tracks=[
                    dict(
                        track_id=7,
                        confirmed=True,
                        state="observed",
                        bbox_xyxy=[10, 1, 50, 47],
                        keypoints17=points,
                    )
                ],
            )
        )
    ledger = run / "frames.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    state = SimpleNamespace(writes=[], hints=[], writer_source=None)
    monkeypatch.setattr(media, "probe_source", lambda *a, **kw: meta)

    def frames(*args, **kwargs):
        for row in rows:
            yield Frame(
                row["frame_index"],
                row["timestamp_seconds"],
                np.zeros((48, 64, 3), np.uint8),
                row["source_pts"],
                "1/1000",
            )

    class Writer:
        def __init__(self, selected, directory, *, expected_timeline=None):
            state.writer_source = selected
            self.directory = directory
            self.timeline = expected_timeline

        def write(self, image, *, frame):
            state.writes.append(frame)

        def finish(self, *, cancel=None):
            assert Path(self.timeline).is_file()
            path = self.directory / "annotated.mp4"
            path.write_bytes(b"validated-fake-mp4")
            return path

        def close(self):
            pass

    def draw(image, tracks, people, **kwargs):
        state.hints.append(people)
        return image.copy()

    monkeypatch.setattr(media, "frames", frames)
    monkeypatch.setattr(media, "VideoWriter", Writer)
    monkeypatch.setattr(rendering, "draw_pose_action_demo", draw)
    return run, source, rows, ledger, sha, state


def test_export_visual_filter_is_shared_within_generation_and_raw_is_untouched(
    saved, tmp_path, monkeypatch
):
    run, source, rows, ledger, sha, state = saved
    before = ledger.read_bytes()
    supplied = []

    def draw(image, tracks, people, **kwargs):
        supplied.append(kwargs["display_stabilizer"])
        return image.copy()

    monkeypatch.setattr(rendering, "draw_pose_action_demo", draw)
    output = rendering.export_saved_actions(run, tmp_path / "stable", stabilize_display=True)
    assert supplied and supplied[0] is not None
    assert all(item is supplied[0] for item in supplied)
    report = json.loads((output / "report.json").read_text())
    assert report["display_stability"]["imputation"] is False
    assert report["display_stability"]["version"] == "causal_display_ema_v1"
    assert ledger.read_bytes() == before


def test_whiteboard_demo_requires_regions_and_source_binding(saved, tmp_path):
    run, source, rows, ledger, sha, state = saved
    with pytest.raises(ValueError, match="separate"):
        rendering.export_saved_actions(run, tmp_path / "missing-board", board_demo_proximity=True)
    with pytest.raises(ValueError, match="whiteboard ROI"):
        rendering.export_saved_actions(
            run,
            tmp_path / "foreign-board",
            whiteboard_roi=[0.1, 0.1, 0.4, 0.4],
            whiteboard_roi_source_sha256="f" * 64,
        )
    assert not (tmp_path / "foreign-board").exists()


@pytest.mark.parametrize("mode", ["raise_hand", "basic_pose"])
def test_export_forwards_opt_in_cue_and_quality_with_causal_prefix(saved, tmp_path, mode):
    run, source, rows, ledger, sha, state = saved
    output = rendering.export_saved_actions(
        run,
        tmp_path / "configured-export",
        start_s=0.45,
        end_s=0.75,
        action_rules=mode,
        hand_cue="shoulder_any_arm_v1",
        pose_quality={"temporal_guard": True, "max_joint_speed": 1},
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    cfg = report["rule_config"]
    hand = cfg["raise_hand"] if mode == "basic_pose" else cfg
    assert hand["rule"] == "coco17_shoulder_any_arm_v1"
    assert hand["hand_cue"] == "shoulder_any_arm_v1"
    assert hand["pose_quality"]["temporal_guard"] is True
    assert hand["pose_quality"]["max_joint_speed"] == 1
    person = state.hints[-1][0]
    record = person["actions"][0] if mode == "basic_pose" else person
    assert record["status"] == "active" and record["start_s"] == 0
    assert report["ground_truth"] is False and report["new_model_inference"] is False


def test_clip_includes_held_frame_with_causal_prefix_and_exact_clipped_duration(saved, tmp_path):
    run, source, rows, ledger, sha, state = saved
    phases = []
    output = rendering.export_saved_actions(
        run,
        tmp_path / "export",
        start_s=0.45,
        end_s=0.75,
        action_rules="raise_hand",
        on_progress=phases.append,
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    timeline = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    assert (output / "video.mp4").is_file()
    assert report["status"] == "completed" and report["new_model_inference"] is False
    assert report["source_sha256"] == sha
    assert report["requested_interval_s"] == [0.45, 0.75] and report["actual_interval_s"] == [
        0.45,
        0.75,
    ]
    assert [r["source_frame_index"] for r in timeline] == [2, 3]
    assert [r["source_pts"] for r in timeline] == [500, 700]
    assert [r["output_pts"] for r in timeline] == [0, 150]
    assert [r["duration_ticks"] for r in timeline] == [150, 150]
    assert [f.frame_index for f in state.writes] == [0, 1]
    assert state.writer_source.duration == pytest.approx(0.3)
    assert state.hints[0][0]["status"] == "active"  # prefix0/.2/.4, no reset at crop
    assert set(phases[0]) == {"phase", "frames_written", "frames_total", "source_time_s"}
    assert phases[-1]["phase"] == "complete"


@pytest.mark.parametrize("change", ["source_hash", "mode", "state", "count", "pts", "geometry"])
def test_invalid_completed_inputs_never_publish(saved, tmp_path, change):
    run, source, rows, ledger, sha, state = saved
    manifest = json.loads((run / "manifest.json").read_text())
    if change == "source_hash":
        manifest["source_sha256"] = "f" * 64
    elif change == "mode":
        manifest["config"]["mode"] = "fast"
    elif change == "state":
        manifest["status"] = "failed"
    elif change == "count":
        manifest["source"]["frame_count"] = 5
    elif change == "geometry":
        manifest["source"]["width"] = 65
    else:
        rows[1]["source_pts"] = 999
        ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "manifest.json").write_text(json.dumps(manifest))
    target = tmp_path / "export"
    with pytest.raises(ValueError):
        rendering.export_saved_actions(run, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".action-export-*"))


@pytest.mark.parametrize("when", ["before", "rendering", "finalizing"])
def test_cancel_removes_private_output_and_preserves_originals(saved, tmp_path, when):
    run, source, rows, ledger, sha, state = saved
    cancelled = when == "before"

    def progress(value):
        nonlocal cancelled
        if value["phase"] == when:
            cancelled = True

    with pytest.raises(media.MediaCancelled):
        rendering.export_saved_actions(
            run, tmp_path / "export", cancel=lambda: cancelled, on_progress=progress
        )
    assert not (tmp_path / "export").exists() and source.read_bytes() == b"original-source-no-NN"
    assert not list(tmp_path.glob(".action-export-*"))


def test_no_overwrite_and_roi_requires_matching_source(saved, tmp_path):
    run, source, rows, ledger, sha, state = saved
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        rendering.export_saved_actions(run, destination)
    assert (destination / "keep").read_text() == "unchanged"
    with pytest.raises(ValueError, match="board_roi_source_sha256"):
        rendering.export_saved_actions(run, tmp_path / "export", board_roi=[0.1, 0.1, 0.8, 0.8])


def test_mutation_during_export_and_progress_exception_do_not_publish(saved, tmp_path):
    run, source, rows, ledger, sha, state = saved

    def mutate(value):
        if value["phase"] == "rendering":
            ledger.write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed|thay đổi"):
        rendering.export_saved_actions(run, tmp_path / "export", on_progress=mutate)
    assert not (tmp_path / "export").exists()


def test_clip_inside_one_vfr_frame_not_empty_and_eof_uses_source_extent(saved, tmp_path):
    run, source, rows, ledger, sha, state = saved
    output = rendering.export_saved_actions(run, tmp_path / "one", start_s=0.21, end_s=0.29)
    timeline = [json.loads(s) for s in (output / "timeline.jsonl").read_text().splitlines()]
    assert len(timeline) == 1 and timeline[0]["source_frame_index"] == 1
    assert timeline[0]["output_pts"] == 0 and timeline[0]["duration_ticks"] == 80
    output = rendering.export_saved_actions(run, tmp_path / "eof", start_s=0.65)
    timeline = [json.loads(s) for s in (output / "timeline.jsonl").read_text().splitlines()]
    assert len(timeline) == 1 and timeline[0]["duration_ticks"] == 150


@pytest.mark.parametrize("phase", ["verifying", "publishing", "complete"])
def test_callback_failure_before_publication_does_not_leave_completed_directory(
    saved, tmp_path, phase
):
    run, *_ = saved

    def progress(value):
        if value["phase"] == phase:
            raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        rendering.export_saved_actions(run, tmp_path / "export", on_progress=progress)
    assert not (tmp_path / "export").exists()
    assert not list(tmp_path.glob(".action-export-*"))


def test_cancel_at_last_progress_boundary_does_not_publish(saved, tmp_path):
    run, *_ = saved
    stopped = False

    def progress(value):
        nonlocal stopped
        if value["phase"] == "complete":
            stopped = True

    with pytest.raises(media.MediaCancelled):
        rendering.export_saved_actions(
            run, tmp_path / "export", cancel=lambda: stopped, on_progress=progress
        )
    assert not (tmp_path / "export").exists()
