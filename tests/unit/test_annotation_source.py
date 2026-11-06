"""Annotation preparation is media indexing, never inference or negative GT."""

import hashlib
import json
import struct
import tempfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from classroom_ai.paths import PROJECT_ROOT
from classroom_ai.video import media
from classroom_ai.video.core import Frame


@pytest.fixture
def tmp_path():
    scratch = PROJECT_ROOT / ".temp" / "annotation_source_tests"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        yield Path(folder)


@pytest.fixture
def prep(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"
    source.write_bytes(b"original-user-video")
    meta = media.VideoSource(
        source,
        Fraction(10),
        32,
        24,
        3,
        False,
        start_time=1.25,
        time_base="1/1000",
        vfr=True,
        duration_seconds=0.4,
    )
    rows = [
        Frame(i, (pts - 1250) / 1000, np.zeros((24, 32, 3), np.uint8), pts, "1/1000")
        for i, pts in enumerate((1250, 1340, 1490))
    ]
    state = SimpleNamespace(rows=rows, written=[], closed=False)
    monkeypatch.setattr(media, "probe_source", lambda *a, **kw: meta)

    def frames(*args, **kwargs):
        assert kwargs["decoder"] == "pyav"
        yield from state.rows

    class Writer:
        def __init__(self, source_meta, directory):
            assert source_meta.vfr is True
            self.directory = directory

        def write(self, image, *, frame):
            state.written.append(frame)

        def finish(self, *, cancel=None):
            result = self.directory / "annotated.mp4"
            result.write_bytes(b"verified-native-preview")
            return result

        def close(self):
            state.closed = True

    monkeypatch.setattr(media, "frames", frames)
    monkeypatch.setattr(media, "VideoWriter", Writer)
    return source, meta, state


def test_truthful_no_model_source_contract_and_binary_index(prep, tmp_path):
    source, meta, state = prep
    updates = []
    output = media.prepare_annotation_source(
        source, tmp_path / "prepared", on_progress=updates.append
    )
    expected = {
        "preview.mp4",
        "metadata.json",
        "manifest.json",
        "report.json",
        "frames.jsonl",
        "frames.idx",
        "frames.times",
        "tracks.jsonl",
    }
    assert expected <= {p.name for p in output.iterdir()}
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert metadata["source"] == meta.metadata()
    assert manifest["kind"] == "annotation_source_preparation"
    assert manifest["processing_mode"] == "annotation_prep"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        manifest["config"]["mode"] == "annotation"
        and manifest["config"]["analysis_task"] == "annotation"
    )
    assert manifest["config"]["tracking"] == "off"
    assert manifest["actor_namespace"] == "run:" + manifest["run_id"]
    assert manifest["inference"]["status"] == "not_run"
    assert report["status"] == "completed"
    assert report["counts"] == {"frames_decoded": 3, "frames_encoded": 3, "frames_inferred": 0}
    assert manifest["coverage"]["frames_inferred"] == 0
    assert not {"weights", "weights_sha256", "runtime", "human_actors", "events", "coverage"} & set(
        metadata
    )
    assert not (output / "events.jsonl").exists()
    assert (output / "tracks.jsonl").read_bytes() == b""
    assert "not_semantic" in manifest["partition"]["meaning"]
    with (
        (output / "frames.jsonl").open("rb") as ledger,
        (output / "frames.idx").open("rb") as index,
    ):
        for i, frame in enumerate(state.rows):
            assert struct.unpack("<Q", index.read(8))[0] == ledger.tell()
            row = json.loads(ledger.readline())
            assert row["kind"] == "annotation_source_frame" and row["observation"] == "not_inferred"
            assert row["analysis_task"] == "annotation" and row["frame_index"] == i
            assert row["source_pts"] == frame.source_pts and row["time_base"] == frame.time_base
            assert row["timestamp_seconds"] == frame.timestamp_seconds
            assert row["segment_id"] == 0
            assert row["tracks"] == row["display_tracks"] == row["detections"] == []
        assert not ledger.read(1) and not index.read(1)
    assert list(struct.unpack("<3d", (output / "frames.times").read_bytes())) == [0, 0.09, 0.24]
    assert state.closed and len(state.written) == 3
    assert set(updates[0]) == {"phase", "frames_written", "frames_total", "source_time_s"}
    assert updates[-1]["phase"] == "complete"


@pytest.mark.parametrize("phase", ["hashing", "decoding", "finalizing", "complete"])
def test_cancel_before_publication_never_exposes_partial(prep, tmp_path, phase):
    source, meta, state = prep
    stopped = False

    def update(value):
        nonlocal stopped
        if value["phase"] == phase:
            stopped = True

    with pytest.raises(media.MediaCancelled):
        media.prepare_annotation_source(
            source, tmp_path / "prepared", cancel=lambda: stopped, on_progress=update
        )
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".annotation-source-*"))
    assert source.read_bytes() == b"original-user-video"


@pytest.mark.parametrize("change", ["index", "pts", "time", "geometry", "missing", "extra"])
def test_bad_native_frame_or_count_does_not_publish(prep, tmp_path, change):
    source, meta, state = prep
    original = state.rows[1]
    if change == "missing":
        state.rows.pop()
    elif change == "extra":
        state.rows.append(Frame(3, 0.35, np.zeros((24, 32, 3), np.uint8), 1600, "1/1000"))
    else:
        state.rows[1] = Frame(
            99 if change == "index" else original.frame_index,
            0.5 if change == "time" else original.timestamp_seconds,
            np.zeros((48, 64, 3), np.uint8) if change == "geometry" else original.image,
            1200 if change == "pts" else original.source_pts,
            original.time_base,
        )
    with pytest.raises(ValueError):
        media.prepare_annotation_source(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".annotation-source-*"))


def test_no_overwrite_and_source_change_after_final_callback_rejected(prep, tmp_path):
    source, meta, state = prep
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        media.prepare_annotation_source(source, target)
    assert (target / "keep").read_text() == "original"

    def mutate(value):
        if value["phase"] == "complete":
            source.write_bytes(b"replaced-video")

    with pytest.raises(ValueError, match="changed|thay đổi"):
        media.prepare_annotation_source(source, tmp_path / "prepared", on_progress=mutate)
    assert not (tmp_path / "prepared").exists()


def test_callback_error_cleans_private_output(prep, tmp_path):
    source, meta, state = prep

    def fail(value):
        if value["phase"] == "complete":
            raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        media.prepare_annotation_source(source, tmp_path / "prepared", on_progress=fail)
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".annotation-source-*"))
