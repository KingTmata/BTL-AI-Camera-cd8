"""Full person tracking: one model pass, then streamed rendering without inference."""

import json
import time
from itertools import zip_longest

from classroom_ai.inference.detector import draw_boxes
from classroom_ai.video.media import VideoWriter, frames
from classroom_ai.video.rendering import draw_tracks, render_tracks
from classroom_ai.video.tracking import PersonTracker, SceneCuts


def json_rows(stream):
    for line in stream:
        yield json.loads(line)


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def process_tracked(session, source):
    config, metrics = session.config, session.metrics
    started = time.perf_counter()
    tracker = PersonTracker(source.fps, config.confidence, config.track_max_lost_seconds, metrics)
    metrics.observe("tracking_setup", (time.perf_counter() - started) * 1000)
    metrics.values["tracking_settings"] = {
        **tracker.settings,
        "algorithm": "bytetrack",
        "class_id": 0,
        "confirm_consecutive_frames": 2,
        "max_lost_seconds": config.track_max_lost_seconds,
        "gap_max_ms": config.track_gap_max_ms,
        "smooth_alpha": config.track_smooth_alpha,
        "scene_cut": {
            "width": 160,
            "height": 90,
            "histogram_bins": [32, 32],
            "bhattacharyya_threshold": 0.6,
            "gray_mean_threshold": 0.25,
        },
        "colors": "HSV pseudo-random by session ID; seed 20261015; stable across frames",
        "interpolation": "Two confirmed same-ID same-scene anchors; no extrapolation",
        "gap_size_ratio": [0.5, 2],
        "gap_center_diagonal_ratio": 0.75,
    }
    cuts = SceneCuts()
    raw_path, det_path = (
        session.directory / "tracks_raw.jsonl",
        session.directory / "detections.jsonl",
    )
    session.progress.set("Detecting and tracking")
    try:
        with (
            det_path.open("x", encoding="utf-8") as detections,
            raw_path.open("x", encoding="utf-8") as raw,
        ):
            for packet in frames(source, metrics, session.stop):
                result = session.infer(packet, detections)
                started = time.perf_counter()
                if cuts.update(packet.image):
                    tracker.reset()
                metrics.observe("track_scene_check", (time.perf_counter() - started) * 1000)
                started = time.perf_counter()
                row = tracker.update(
                    packet.frame_index, packet.timestamp_seconds, result["detections"]
                )
                metrics.observe("tracking", (time.perf_counter() - started) * 1000)
                write_row(raw, row)
                metrics.count("frames_tracked")
    finally:
        tracker.finish()
    if session.stop.is_set():
        return
    session.progress.set("Rendering tracked video")
    writer = VideoWriter(source, session.directory)
    try:
        with (
            raw_path.open(encoding="utf-8") as raw,
            det_path.open(encoding="utf-8") as detections,
            (session.directory / "tracks.jsonl").open("x", encoding="utf-8") as rendered,
        ):
            tracks = render_tracks(
                json_rows(raw),
                source.width,
                source.height,
                config.track_smooth_alpha,
                config.track_gap_max_ms,
                metrics,
            )
            decoded = frames(
                source,
                metrics,
                session.stop,
                counter="frames_render_decoded",
                decode_stage="render_decode",
            )
            try:
                for packet, track, detection in zip_longest(decoded, tracks, json_rows(detections)):
                    if session.stop.is_set():
                        break
                    if packet is None or track is None or detection is None:
                        raise ValueError("Số frame video/detection/track không khớp khi dựng.")
                    if (
                        packet.frame_index != track["frame_index"]
                        or packet.frame_index != detection["frame_index"]
                        or abs(packet.timestamp_seconds - track["timestamp_seconds"]) > 1e-6
                        or abs(packet.timestamp_seconds - detection["timestamp_seconds"]) > 1e-6
                    ):
                        raise ValueError("Timeline video/detection/track không khớp.")
                    started = time.perf_counter()
                    image = draw_boxes(
                        packet.image, [r for r in detection["detections"] if r["class_id"] != 0]
                    )
                    image = draw_tracks(image, track["tracks"])
                    metrics.observe("draw", (time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    writer.write(image)
                    metrics.observe("encode_write", (time.perf_counter() - started) * 1000)
                    write_row(rendered, track)
                    metrics.count("frames_encoded")
            finally:
                decoded.close()
                tracks.close()
        if not session.stop.is_set():
            session.progress.set("Finalizing and validating MP4")
            started = time.perf_counter()
            session.output = writer.finish()
            metrics.observe("export_finalize_validate", (time.perf_counter() - started) * 1000)
    finally:
        writer.close()
