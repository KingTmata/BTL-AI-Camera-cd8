"""Single-decoder PTS contracts; fake codec frames, no model inference."""

import sys
import threading
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from classroom_ai.video import media
from classroom_ai.video.metrics import Metrics


@pytest.fixture
def decoder(tmp_path, monkeypatch):
    source = media.VideoSource(
        tmp_path / "source.mp4", Fraction(5), 4, 2, 2, False, time_base="1/1000"
    )
    state = SimpleNamespace(closed=False, decoded=False)

    class VideoFrame:
        width, height = 4, 2

        def __init__(self, pts, tb=Fraction(1, 1000)):
            self.pts, self.time_base = pts, tb

        def to_ndarray(self, format):
            assert format == "bgr24"
            return np.zeros((self.height, self.width, 3), np.uint8)

    native_frames = [VideoFrame(1000), VideoFrame(1200)]
    stream = SimpleNamespace(
        disposition=0, codec_context=SimpleNamespace(thread_count=0), thread_type=None
    )

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed = True

        def decode(self, chosen):
            assert chosen is stream
            state.decoded = True
            yield from native_frames

    module = SimpleNamespace(
        __version__="16.1.0",
        open=lambda *args, **kwargs: Container(),
        stream=SimpleNamespace(Disposition=SimpleNamespace(attached_pic=1024)),
    )
    monkeypatch.setitem(sys.modules, "av", module)
    monkeypatch.setattr(
        media,
        "source_pts",
        lambda *args, **kwargs: pytest.fail("PyAV must not start a second PTS decoder"),
    )
    return source, native_frames, VideoFrame, stream, state


def test_pyav_uses_native_pts_and_closes_without_second_decoder(decoder):
    source, _, _, stream, state = decoder
    rows = list(media.frames(source, Metrics(), threading.Event(), decoder="pyav"))
    assert [row.source_pts for row in rows] == [1000, 1200]
    assert [row.timestamp_seconds for row in rows] == [0, 0.2]
    assert [row.frame_index for row in rows] == [0, 1]
    assert all(row.time_base == "1/1000" for row in rows)
    assert state.closed and state.decoded
    assert stream.thread_type == "AUTO" and 1 <= stream.codec_context.thread_count <= 4


def test_native_timebase_rescales_only_integral_pts(decoder):
    source, frames, Frame, _, _ = decoder
    frames[:] = [Frame(2000, Fraction(1, 2000)), Frame(2400, Fraction(1, 2000))]
    assert [
        r.source_pts for r in media.frames(source, Metrics(), threading.Event(), decoder="pyav")
    ] == [1000, 1200]
    frames[0].pts = 2001
    with pytest.raises(ValueError, match="time_base|PTS"):
        list(media.frames(source, Metrics(), threading.Event(), decoder="pyav"))


@pytest.mark.parametrize(
    "problem", ["missing_pts", "missing_tb", "duplicate", "backwards", "geometry", "short", "long"]
)
def test_pyav_fails_closed_on_invalid_native_stream(decoder, problem):
    source, frames, Frame, _, state = decoder
    if problem == "missing_pts":
        frames[0].pts = None
    if problem == "missing_tb":
        frames[0].time_base = None
    if problem == "duplicate":
        frames[1].pts = frames[0].pts
    if problem == "backwards":
        frames[1].pts = 900
    if problem == "geometry":
        frames[0].width = 5
    if problem == "short":
        frames.pop()
    if problem == "long":
        frames.append(Frame(1400))
    with pytest.raises(ValueError):
        list(media.frames(source, Metrics(), threading.Event(), decoder="pyav"))
    assert state.closed


def test_pyav_cancellation_after_native_decode_closes_container(decoder):
    source, frames, _, _, state = decoder
    stop = threading.Event()
    native = frames[0].to_ndarray

    def convert(**kwargs):
        image = native(**kwargs)
        stop.set()
        return image

    frames[0].to_ndarray = convert
    with pytest.raises(media.MediaCancelled):
        list(media.frames(source, Metrics(), stop, decoder="pyav", cancel=stop.is_set))
    assert state.closed


def test_missing_pyav_is_clear_optional_dependency_error(decoder, monkeypatch):
    source, *_ = decoder
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(RuntimeError, match="PyAV|av"):
        list(media.frames(source, Metrics(), threading.Event(), decoder="pyav"))


def test_unknown_decoder_is_rejected_without_opening_file(decoder):
    source, *_ = decoder
    with pytest.raises(ValueError, match="decoder"):
        list(media.frames(source, Metrics(), threading.Event(), decoder="guess"))
