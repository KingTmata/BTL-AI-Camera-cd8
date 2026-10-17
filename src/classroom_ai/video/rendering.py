"""Streaming track display rows with bounded metadata lookahead; no video images retained."""

import colorsys
import math
import random
import time
from collections import deque

import cv2
import numpy as np

from classroom_ai.video.tracking import BoxSmoother, center_size, clip_box, corners


def gap_allowed(left, right, max_seconds):
    delta = right["timestamp_seconds"] - left["timestamp_seconds"]
    if delta <= 0 or delta > max_seconds + 1e-9 or left["segment_id"] != right["segment_id"]:
        return False
    a, b = center_size(left["bbox_xyxy"]), center_size(right["bbox_xyxy"])
    ratio = b[2:] / a[2:]
    return bool(
        np.all((ratio >= 0.5) & (ratio <= 2))
        and np.linalg.norm(b[:2] - a[:2])
        <= 0.75 * (np.linalg.norm(a[2:]) + np.linalg.norm(b[2:])) / 2
    )


def render_tracks(records, width, height, alpha, gap_ms, metrics):
    """Yield each frame once. Only future metadata up to gap_seconds is buffered."""
    records = iter(records)
    gap = gap_ms / 1000
    smoother = BoxSmoother(alpha, width, height)
    pending, history = deque(), {}
    previous_input = -1
    segment = None
    exhausted = False

    def prepare(row):
        nonlocal previous_input, segment
        index = row["frame_index"]
        if index != previous_input + 1:
            raise ValueError("tracks_raw.jsonl bị thiếu hoặc sai thứ tự frame.")
        previous_input = index
        if segment != row["segment_id"]:
            smoother.previous.clear()
            segment = row["segment_id"]
        started = time.perf_counter()
        tracks = []
        for track in row["tracks"]:
            if not track["confirmed"]:
                continue
            box = smoother.update(track["track_id"], index, track["bbox_xyxy"])
            if box is not None:
                tracks.append(
                    {
                        **track,
                        "raw_bbox_xyxy": track["bbox_xyxy"],
                        "bbox_xyxy": box,
                        "state": "observed",
                        "anchor_frames": [index],
                        "anchor_timestamps": [row["timestamp_seconds"]],
                    }
                )
        smoother.prune(index)
        metrics.observe("track_smoothing", (time.perf_counter() - started) * 1000)
        return {**row, "tracks": tracks}

    def read_one():
        nonlocal exhausted
        try:
            pending.append(prepare(next(records)))
        except StopIteration:
            exhausted = True

    read_one()
    while pending:
        current = pending[0]
        timestamp = current["timestamp_seconds"]
        # One boundary record can lie just beyond the lookahead window.
        while not exhausted and pending[-1]["timestamp_seconds"] <= timestamp + gap + 1e-9:
            read_one()
        metrics.values["track_metadata_peak_frames"] = max(
            metrics.values.get("track_metadata_peak_frames", 0), len(pending)
        )
        history = {
            key: value
            for key, value in history.items()
            if timestamp - value["timestamp_seconds"] <= gap + 1e-9
            and value["segment_id"] == current["segment_id"]
        }
        observed = {t["track_id"]: t for t in current["tracks"]}
        rows = list(observed.values())
        started = time.perf_counter()
        for key, left in history.items():
            if key in observed:
                continue
            right = next(
                (
                    dict(
                        t,
                        timestamp_seconds=f["timestamp_seconds"],
                        segment_id=f["segment_id"],
                        frame_index=f["frame_index"],
                    )
                    for f in list(pending)[1:]
                    if f["timestamp_seconds"] <= left["timestamp_seconds"] + gap + 1e-9
                    and f["segment_id"] == left["segment_id"]
                    for t in f["tracks"]
                    if t["track_id"] == key
                ),
                None,
            )
            if right is None:
                continue
            first_gap_frame = current["frame_index"] == left["frame_index"] + 1
            if not gap_allowed(left, right, gap):
                if first_gap_frame:
                    metrics.count("track_gaps_rejected")
                continue
            fraction = (timestamp - left["timestamp_seconds"]) / (
                right["timestamp_seconds"] - left["timestamp_seconds"]
            )
            box = clip_box(
                corners(
                    (1 - fraction) * center_size(left["bbox_xyxy"])
                    + fraction * center_size(right["bbox_xyxy"])
                ),
                width,
                height,
            )
            rows.append(
                dict(
                    track_id=key,
                    class_id=0,
                    class_name="person",
                    bbox_xyxy=box,
                    confidence=None,
                    state="interpolated",
                    detection_index=None,
                    anchor_frames=[left["frame_index"], right["frame_index"]],
                    anchor_timestamps=[left["timestamp_seconds"], right["timestamp_seconds"]],
                )
            )
            metrics.count("track_boxes_interpolated")
            if first_gap_frame:
                metrics.count("track_gaps_interpolated")
                metrics.observe(
                    "track_gap_duration",
                    (right["timestamp_seconds"] - left["timestamp_seconds"]) * 1000,
                )
        if any(r["state"] == "interpolated" for r in rows):
            metrics.count("track_frames_interpolated")
        metrics.observe("track_interpolation", (time.perf_counter() - started) * 1000)
        for key, track in observed.items():
            history[key] = dict(
                track,
                frame_index=current["frame_index"],
                timestamp_seconds=timestamp,
                segment_id=current["segment_id"],
            )
        metrics.values["track_render_peak_history"] = max(
            metrics.values.get("track_render_peak_history", 0), len(history)
        )
        yield {**current, "tracks": rows}
        pending.popleft()
        if not pending and not exhausted:
            read_one()


def track_color(track_id):
    """Pseudo-random per ID, constant across frames and reproducible across runs."""
    rng = random.Random(20261015 + int(track_id))
    rgb = colorsys.hsv_to_rgb(rng.random(), 0.55 + 0.35 * rng.random(), 0.8 + 0.2 * rng.random())
    return tuple(round(c * 255) for c in reversed(rgb))


def draw_tracks(image, tracks):
    """Person labels are session IDs, never the per-frame detection enumeration."""
    canvas = image.copy()
    thickness = max(1, round(max(image.shape[:2]) / 500))
    for track in tracks:
        box = clip_box(track["bbox_xyxy"], image.shape[1], image.shape[0])
        if box is None:
            continue
        x1, y1, x2, y2 = map(round, box)
        interpolated = track["state"] == "interpolated"
        color = track_color(track["track_id"])
        if interpolated:
            for start, end in (
                ((x1, y1), (x2, y1)),
                ((x2, y1), (x2, y2)),
                ((x2, y2), (x1, y2)),
                ((x1, y2), (x1, y1)),
            ):
                length = math.dist(start, end)
                for offset in range(0, math.ceil(length), 12):
                    a, b = offset / max(1, length), min(offset + 7, length) / max(1, length)

                    def point(f):
                        return tuple(round(s + (e - s) * f) for s, e in zip(start, end))

                    cv2.line(canvas, point(a), point(b), color, thickness)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"P{track['track_id']}" + (" | noi suy" if interpolated else "")
        cv2.putText(
            canvas,
            label,
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(1, thickness),
            cv2.LINE_AA,
        )
    return canvas
