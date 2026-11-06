"""Streaming track display rows with bounded metadata lookahead; no video images retained."""

import colorsys
import hashlib
import json
import math
import os
import random
import tempfile
import threading
import time
from collections import deque
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

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


def render_tracks(records, width, height, alpha, gap_ms, metrics, *, pose_tau_ms=80):
    """Render display geometry while retaining raw points; pose_tau_ms=0 disables EMA."""
    if not math.isfinite(pose_tau_ms) or pose_tau_ms < 0:
        raise ValueError("pose_tau_ms must be finite and nonnegative")
    records = iter(records)
    gap = gap_ms / 1000
    pose_tau = pose_tau_ms / 1000
    smoother = BoxSmoother(alpha, width, height)
    pending, history = deque(), {}
    pose_previous = {}
    previous_input = -1
    segment = None
    exhausted = False

    def prepare(row):
        nonlocal previous_input, segment, pose_previous
        index = row["frame_index"]
        if index != previous_input + 1:
            raise ValueError("tracks_raw.jsonl bị thiếu hoặc sai thứ tự frame.")
        previous_input = index
        if segment != row["segment_id"]:
            smoother.previous.clear()
            pose_previous.clear()
            segment = row["segment_id"]
        started = time.perf_counter()
        tracks = []
        pose_current = {}
        for track in row["tracks"]:
            if not track["confirmed"]:
                continue
            box = smoother.update(track["track_id"], index, track["bbox_xyxy"])
            if box is not None:
                displayed = {
                    **track,
                    "raw_bbox_xyxy": track["bbox_xyxy"],
                    "bbox_xyxy": box,
                    "state": "observed",
                    "anchor_frames": [index],
                    "anchor_timestamps": [row["timestamp_seconds"]],
                }
                points = track.get("keypoints13")
                if points is not None:
                    previous = pose_previous.get(track["track_id"])
                    dt = row["timestamp_seconds"] - previous[0] if previous else 0
                    # Pose resets are independent of optional bbox gap interpolation.
                    can_smooth = pose_tau > 0 and previous is not None and 0 < dt <= 0.3
                    weight = -math.expm1(-dt / pose_tau) if can_smooth else 1
                    shown, smoothed = [], False
                    for joint, point in enumerate(points):
                        if point is None:
                            shown.append(None)
                            continue
                        old = previous[1][joint] if can_smooth else None
                        if old is None:
                            shown.append(list(point))
                        else:
                            shown.append(
                                [
                                    old[0] + weight * (point[0] - old[0]),
                                    old[1] + weight * (point[1] - old[1]),
                                    point[2],
                                ]
                            )
                            smoothed = True
                    displayed["display_keypoints13"] = shown
                    displayed["display_pose"] = {
                        "algorithm": "timestamp_ema",
                        "tau_ms": pose_tau_ms,
                        "smoothed": smoothed,
                    }
                    pose_current[track["track_id"]] = (row["timestamp_seconds"], shown)
                tracks.append(displayed)
        # Missing tracks/joints never retain an estimate across an observation gap.
        pose_previous = pose_current
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


def _caption_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _caption_provenance(action):
    return (
        action.get("origin") == "rule"
        and action.get("review") == "draft"
        and action.get("calibrated") is False
    )


def _caption_quality(action):
    quality = action.get("pose_quality")
    return (
        isinstance(quality, dict)
        and quality.get("gate_version") == "causal_pose_quality_v1"
        and isinstance(quality.get("temporal_guard"), bool)
        and quality.get("status") == "supported"
    )


def _caption_relation(person, actions):
    relations, t = person.get("context_relations"), person.get("observed_through_s")
    if not isinstance(relations, list) or len(relations) != 1 or not _caption_number(t):
        return False
    relation = relations[0]
    if not isinstance(relation, dict) or not _caption_provenance(relation):
        return False
    if (
        relation.get("relation") != "raise_hand_then_move_to_board"
        or relation.get("rule") != "raise_hand_then_move_to_board_v1"
        or relation.get("status") != "observed_sequence"
        or relation.get("continuity") != "same_scene_track_continuous_supported_pose"
    ):
        return False
    keys = (
        "raise_start_s",
        "raise_confirmed_at_s",
        "raise_observed_through_s",
        "board_start_s",
        "board_confirmed_at_s",
        "max_separation_s",
    )
    if not all(_caption_number(relation.get(k)) for k in keys):
        return False
    start, confirmed, through, board_start, board_confirmed, window = (relation[k] for k in keys)
    if not (
        start <= confirmed <= through <= t
        and confirmed <= board_start <= board_confirmed <= t
        and 0 < window <= 60
        and t - through <= window
        and t - board_confirmed < 1
    ):
        return False
    hand = [a for a in actions if a.get("label") == "raise_hand"]
    board = [a for a in actions if a.get("label") == "move_to_board"]
    return (
        len(hand) == len(board) == 1
        and all(
            _caption_provenance(a) and _caption_quality(a) and a.get("observed_through_s") == t
            for a in hand + board
        )
        and hand[0].get("rule") in ("coco17_wrist_above_nose_v1", "coco17_shoulder_any_arm_v1")
        and board[0].get("rule") == "coco17_floor_roi_arrival_v1"
        and board[0].get("status") == "completed"
        and board[0].get("start_s") == board_start
        and board[0].get("confirmed_at_s") == board_confirmed
    )


def action_caption_text(person):
    """Vietnamese current-state captions for a saved demo; no event-log lookup."""
    if person is None:
        return "Chưa phân loại"
    captions = {
        ("raise_hand", "active"): "Giơ tay",
        ("raise_hand", "candidate"): "Có thể giơ tay?",
        ("raise_hand", "release_pending"): "Đang hạ tay?",
        ("stand_up", "candidate"): "Đang đứng lên?",
        ("stand_up", "completed"): "Vừa đứng lên",
        ("move_to_board", "candidate"): "Đang đến vùng bảng?",
        ("move_to_board", "completed"): "Đến vùng bảng",
        ("write_board", "active"): "Viết bảng · quy ước gần bảng",
    }
    actions = person.get("actions", [person])
    actions = [a for a in actions if isinstance(a, dict)] if isinstance(actions, list) else []
    selected = [
        (
            "Đi lên bảng"
            if a.get("rule") == "user_board_proximity_demo_v1" and a.get("label") == "move_to_board"
            else captions[(a.get("label"), a.get("status"))]
        )
        for a in actions
        if (a.get("label"), a.get("status")) in captions
    ]
    for action in actions:
        if (
            action.get("label", "raise_hand") == "raise_hand"
            and action.get("rule") in ("coco17_wrist_above_nose_v1", "coco17_shoulder_any_arm_v1")
            and _caption_provenance(action)
            and action.get("pose_observation_scope") == "uncalibrated_raw_pose_geometry"
        ):
            if (
                action.get("pose_observation") == "not_raised"
                and action.get("pose_observation_reason") == "both_wrists_lowered"
                and _caption_quality(action)
                and action.get("status") != "unsupported"
            ):
                selected.append("Tay đang hạ · Pose nháp")
            elif action.get("pose_observation") == "unknown":
                selected.append("Chưa rõ tư thế tay")
    if _caption_relation(person, actions):
        selected.append("Liên hệ: Đã giơ tay → đến vùng bảng")
    if selected:
        return " + ".join(selected) + " · nháp"
    if any(a.get("status") == "unsupported" for a in actions):
        return "Chưa phân loại · thiếu khớp"
    return "Chưa có gợi ý · nháp"


def draw_pose_action_demo(
    image,
    tracks,
    people,
    *,
    segment_id,
    source_time,
    font_path,
    board_roi=None,
    source_title="",
    display_stabilizer=None,
    whiteboard_roi=None,
    whiteboard_offscreen=False,
):
    """Burn current raw Pose/ID hints into a copy; never reuse an old actor label.

    Only current confirmed observed tracks are drawn. People must match source
    time, scene and ID. A caller supplies a Unicode font for Vietnamese text.
    This is an explicit saved-demo renderer, not a replacement for live display.
    """
    from PIL import Image, ImageDraw, ImageFont

    from classroom_ai.inference.pose import draw_pose

    observed = [t for t in tracks if t.get("state") == "observed" and t.get("confirmed") is True]
    raw = [
        {k: v for k, v in t.items() if k not in ("display_keypoints13", "display_pose")}
        for t in observed
    ]
    if display_stabilizer is not None:
        raw = display_stabilizer.update(raw, segment=segment_id, t=source_time)
    canvas = draw_pose(draw_tracks(image, raw), raw)
    height, width = image.shape[:2]
    if board_roi is not None:
        a, b, c, d = board_roi
        cv2.rectangle(
            canvas,
            (round(a * width), round(b * height)),
            (round(c * width), round(d * height)),
            (70, 220, 180),
            2,
        )
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(str(font_path), max(14, round(width / 72)))
    title_font = ImageFont.truetype(str(font_path), max(17, round(width / 65)))
    if whiteboard_offscreen:
        draw.text(
            (20, height - 80),
            "HƯỚNG LÊN BẢNG: MÉP DƯỚI / PHÍA CAMERA (NGOÀI KHUNG)",
            font=font,
            fill=(70, 230, 255),
        )
    if whiteboard_roi is not None:
        a, b, c, d = whiteboard_roi
        draw.rectangle(
            (a * width, b * height, c * width, d * height), outline=(70, 230, 255), width=3
        )
        draw.text(
            (a * width, max(80, b * height - 30)), "BẢNG TRẮNG", font=font, fill=(70, 230, 255)
        )
    draw.rectangle((0, 0, width, round(height * 0.075)), fill=(12, 20, 30))
    title = (
        str(source_title) + " · " if source_title else ""
    ) + "POSE + ID + HÀNH VI CƠ BẢN · NHÁP"
    draw.text((14, 5), title, font=title_font, fill=(240, 245, 255))
    draw.text(
        (14, round(height * 0.04)),
        f"Nguồn {source_time:.3f}s · Dùng kết quả đã lưu · Không phải nhãn đã duyệt",
        font=font,
        fill=(200, 210, 225),
    )
    current = {
        (p.get("segment_id"), p.get("track_id")): p
        for p in people
        if isinstance(p.get("observed_through_s"), (int, float))
        and abs(p["observed_through_s"] - source_time) <= 1e-9
    }
    occupied = []
    for track in raw:
        box = clip_box(track["bbox_xyxy"], width, height)
        if box is None:
            continue
        x1, y1, _, _ = box
        text = f"#{track['track_id']}  " + action_caption_text(
            None
            if track.get("identity_uncertain") or track.get("uncertain")
            else current.get((segment_id, track["track_id"]))
        )
        bounds = draw.textbbox((0, 0), text, font=font)
        tw, th = bounds[2] - bounds[0] + 12, bounds[3] - bounds[1] + 12
        x = max(0, min(round(x1), width - tw))
        y = max(round(height * 0.08), round(y1) - th - 7)
        for old in occupied:
            if x < old[2] and x + tw > old[0] and y < old[3] and y + th > old[1]:
                y = old[3] + 3
        y = min(y, height - th - 35)
        draw.rectangle((x, y, x + tw, y + th), fill=(15, 21, 31))
        draw.text((x + 6, y + 4 - bounds[1]), text, font=font, fill=(255, 235, 160))
        occupied.append((x, y, x + tw, y + th))
    draw.rectangle((0, height - 34, width, height), fill=(12, 20, 30))
    draw.text(
        (14, height - 32),
        "Thiếu khớp / chưa có gợi ý không có nghĩa là không có hành vi.",
        font=font,
        fill=(210, 220, 235),
    )
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _export_hash(path, cancel):
    from classroom_ai.video.media import _check_cancel

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _check_cancel(cancel)
            digest.update(block)
    return digest.hexdigest()


def _export_rows(path, source, cancel):
    """Validate the complete raw ledger without retaining its frame history."""
    from classroom_ai.video.actions import _actor, _number
    from classroom_ai.video.media import _check_cancel

    first, previous = None, None
    base = Fraction(source.time_base)
    with path.open("rb") as stream:
        for index in range(source.frame_count):
            _check_cancel(cancel)
            encoded = stream.readline(4 * 1024 * 1024 + 1)
            if not encoded or len(encoded) > 4 * 1024 * 1024:
                raise ValueError("Raw ledger missing frame or exceeds 4 MiB/frame")
            try:
                row = json.loads(encoded)
                json.dumps(row, allow_nan=False)
            except (ValueError, TypeError) as exc:
                raise ValueError("Raw ledger changed or invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError("Raw frame must be an object")
            pts, timestamp = row.get("source_pts"), row.get("timestamp_seconds")
            if (
                row.get("analysis_task") != "pose"
                or isinstance(row.get("frame_index"), bool)
                or row.get("frame_index") != index
                or isinstance(pts, bool)
                or not isinstance(pts, int)
                or Fraction(row.get("time_base", "0")) != base
                or previous is not None
                and pts <= previous
                or not isinstance(row.get("tracks"), list)
            ):
                raise ValueError("Raw frame index/PTS/Pose schema mismatch")
            first = pts if first is None else first
            relative = (pts - first) * base
            if (
                relative < 0
                or relative >= Fraction(str(source.duration))
                or abs(_number(timestamp, "timestamp_seconds") - float(relative)) > 1e-6
            ):
                raise ValueError("Raw timestamp does not match source PTS")
            _actor(dict(segment_id=row.get("segment_id"), track_id=0))
            for track in row["tracks"]:
                if not isinstance(track, dict):
                    raise ValueError("Raw track must be an object")
                _actor(dict(segment_id=row["segment_id"], track_id=track.get("track_id")))
            previous = pts
            yield row
        if stream.read(1):
            raise ValueError("Raw ledger has extra frames")


def _export_font(font_path):
    from PIL import ImageFont

    candidates = (
        [font_path]
        if font_path is not None
        else [
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            ImageFont.truetype(str(candidate), 16)
            return str(candidate)
        except OSError:
            continue
    raise ValueError("Cần font Unicode Segoe UI/DejaVu Sans đã cài; không tự tải font.")


def export_saved_actions(
    completed_run_dir,
    destination_dir,
    *,
    source=None,
    start_s=0,
    end_s=None,
    action_rules="basic_pose",
    stabilize_display=True,
    whiteboard_roi=None,
    whiteboard_roi_source_sha256=None,
    board_demo_proximity=False,
    whiteboard_offscreen=False,
    pose_quality=None,
    hand_cue="nose_both_arms_v1",
    board_roi=None,
    board_roi_source_sha256=None,
    source_title="",
    font_path=None,
    cancel=None,
    on_progress=None,
) -> Path:
    """Export a NEW directory from completed raw Pose/IDs, with no model inference.

    Calls on_progress(dict) with exactly phase, frames_written, frames_total,
    source_time_s. Totals/time are None until known. The complete progress
    callback is the final pre-commit boundary; only successful return confirms
    atomic publication. Exceptions/cancellation
    publish no directory. Callbacks must return promptly; callback errors abort.

    Requested [start,end) bounds round to nearest source-clock ticks. Frames
    whose presentation spans overlap the actual interval are retained, including
    the held frame containing start. First/last display durations are clipped.
    Rules consume raw frames from source0 through the last retained observation.
    Only timestamp lookahead determines durations, never future Pose. Original
    raster coordinates are kept; rotation/SAR metadata is stripped like preview.
    Native decode/encode cancellation is cooperative when each call returns.
    """
    from classroom_ai.video.actions import RaiseHandState, _number
    from classroom_ai.video.board_demo import BoardDemoState
    from classroom_ai.video.core import Frame
    from classroom_ai.video.display_stability import PoseDisplayStabilizer
    from classroom_ai.video.media import VideoWriter, _check_cancel, frames, probe_source
    from classroom_ai.video.metrics import Metrics
    from classroom_ai.video.pose_rules import BasicPoseActionState, validate_board_roi

    _check_cancel(cancel)
    run, destination = Path(completed_run_dir).resolve(), Path(destination_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if action_rules not in ("off", "raise_hand", "basic_pose"):
        raise ValueError("Unsupported action_rules")
    board_roi = validate_board_roi(board_roi)
    whiteboard_roi = validate_board_roi(whiteboard_roi)
    if board_demo_proximity and (
        board_roi is None or whiteboard_roi is None and not whiteboard_offscreen
    ):
        raise ValueError("Board demo needs separate floor and whiteboard regions")
    manifest_path, raw_path = run / "manifest.json", run / "frames.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = manifest.get("config", {})
    if (
        manifest.get("status") != "completed"
        or settings.get("mode") != "full"
        or settings.get("analysis_task") != "pose"
        or settings.get("tracking") != "bytetrack"
    ):
        raise ValueError("Export requires a completed Full Pose + ByteTrack run")
    source_path = Path(source if source is not None else manifest["source"]["path"]).resolve()
    inputs = {"manifest.json": manifest_path, "frames.jsonl": raw_path, "source": source_path}
    written, total, source_time = 0, None, None

    def progress(phase):
        _check_cancel(cancel)
        if on_progress is not None:
            on_progress(
                dict(
                    phase=phase,
                    frames_written=written,
                    frames_total=total,
                    source_time_s=source_time,
                )
            )
        _check_cancel(cancel)

    progress("hashing")
    hashes = {name: _export_hash(path, cancel) for name, path in inputs.items()}
    if hashes["source"] != manifest.get("source_sha256"):
        raise ValueError("Source SHA256 mismatch")
    if board_roi is not None and board_roi_source_sha256 != hashes["source"]:
        raise ValueError("board_roi_source_sha256 must match the selected source")
    if whiteboard_roi is not None and whiteboard_roi_source_sha256 != hashes["source"]:
        raise ValueError("whiteboard ROI must match selected source SHA256")
    progress("validating")
    original = probe_source(source_path, allow_vfr=True, cancel=cancel)
    saved = manifest["source"]
    if (
        any(saved.get(k) != getattr(original, k) for k in ("width", "height", "frame_count"))
        or Fraction(saved["time_base"]) != Fraction(original.time_base)
        or abs(saved["duration_seconds"] - original.duration)
        > float(Fraction(original.time_base)) * 2
    ):
        raise ValueError("Source geometry/count/clock mismatch with completed run")
    if original.width * original.height * 3 > 128 * 1024 * 1024:
        raise ValueError("Một frame vượt giới hạn 128 MiB")
    requested_start = _number(start_s, "start_s")
    requested_end = original.duration if end_s is None else _number(end_s, "end_s")
    if not 0 <= requested_start < requested_end <= original.duration + 1e-9:
        raise ValueError("Expected 0 <= start_s < end_s <= source duration")
    base = Fraction(original.time_base)
    extent = round(Fraction(str(original.duration)) / base)
    left = max(0, round(Fraction(str(requested_start)) / base))
    right = min(extent, round(Fraction(str(requested_end)) / base))
    if right <= left:
        raise ValueError("Export interval is empty after source-tick quantization")
    actual_start, actual_end = float(left * base), float(right * base)
    font = _export_font(font_path)
    state = (
        BasicPoseActionState(board_roi, pose_quality=pose_quality, hand_cue=hand_cue)
        if action_rules == "basic_pose"
        else RaiseHandState(pose_quality=pose_quality, hand_cue=hand_cue)
        if action_rules == "raise_hand"
        else None
    )
    rule_config = state.config if state is not None else None
    display_stabilizer = PoseDisplayStabilizer() if stabilize_display else None
    board_demo = (
        BoardDemoState(board_roi, offscreen=whiteboard_offscreen) if board_demo_proximity else None
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".action-export-", dir=destination.parent) as temporary:
        private = Path(temporary)
        timeline_path = private / "timeline.jsonl"
        first_pts, previous = None, None
        total = 0

        def record(row, next_tick, stream):
            nonlocal total
            relative = row["source_pts"] - first_pts
            begin, end = max(relative, left), min(next_tick, right)
            if begin < end:
                item = dict(
                    output_frame_index=total,
                    source_frame_index=row["frame_index"],
                    source_pts=row["source_pts"],
                    time_base=original.time_base,
                    source_time_s=row["timestamp_seconds"],
                    output_pts=begin - left,
                    duration_ticks=end - begin,
                )
                stream.write(json.dumps(item, allow_nan=False) + "\n")
                total += 1

        with timeline_path.open("x", encoding="utf-8") as timeline:
            for row in _export_rows(raw_path, original, cancel):
                first_pts = row["source_pts"] if first_pts is None else first_pts
                if previous is not None:
                    record(previous, row["source_pts"] - first_pts, timeline)
                previous = row
            if previous is not None:
                record(previous, extent, timeline)
            timeline.flush()
            os.fsync(timeline.fileno())
        if not total:
            raise ValueError("No source presentation overlaps the interval")
        selected = replace(
            original,
            vfr=True,
            frame_count=total,
            duration_seconds=float((right - left) * base),
            audio_offset_seconds=original.audio_offset_seconds - actual_start,
        )
        progress("rendering")
        writer, decoded = None, None
        try:
            writer = VideoWriter(selected, private, expected_timeline=timeline_path)
            decoded = frames(original, Metrics(), threading.Event(), cancel=cancel, decoder="pyav")
            with timeline_path.open(encoding="utf-8") as timeline:
                expected = json.loads(next(timeline))
                rows = _export_rows(raw_path, original, cancel)
                try:
                    for packet in decoded:
                        _check_cancel(cancel)
                        row = next(rows, None)
                        if (
                            row is None
                            or packet.frame_index != row["frame_index"]
                            or packet.source_pts != row["source_pts"]
                            or Fraction(packet.time_base) != base
                            or abs(packet.timestamp_seconds - row["timestamp_seconds"]) > 1e-6
                            or packet.image.shape[:2] != (original.height, original.width)
                        ):
                            raise ValueError(
                                "Decoded source index/nativePTS/raster does not match raw ledger"
                            )
                        source_time = row["timestamp_seconds"]
                        if state is not None:
                            state.update(
                                {**row, "width": original.width, "height": original.height}
                            )
                        demo_people = state.people if state else []
                        if board_demo is not None:
                            hints = board_demo.update(
                                row["tracks"],
                                segment=row["segment_id"],
                                t=source_time,
                                width=original.width,
                                height=original.height,
                            )
                            indexed = {p["track_id"]: p for p in demo_people}
                            for tid, actions in hints.items():
                                person = indexed.setdefault(
                                    tid,
                                    dict(
                                        track_id=tid,
                                        segment_id=row["segment_id"],
                                        observed_through_s=source_time,
                                        actions=[],
                                    ),
                                )
                                person["actions"] = [
                                    a for a in person["actions"] if a["label"] != "move_to_board"
                                ] + actions
                            demo_people = list(indexed.values())
                        if row["frame_index"] == expected["source_frame_index"]:
                            image = draw_pose_action_demo(
                                packet.image,
                                row["tracks"],
                                demo_people,
                                segment_id=row["segment_id"],
                                source_time=source_time,
                                font_path=font,
                                board_roi=board_roi,
                                source_title=source_title,
                                display_stabilizer=display_stabilizer,
                                whiteboard_roi=whiteboard_roi,
                                whiteboard_offscreen=whiteboard_offscreen,
                            )
                            encoded_frame = Frame(
                                written,
                                float(expected["output_pts"] * base),
                                image,
                                expected["output_pts"],
                                original.time_base,
                            )
                            writer.write(image, frame=encoded_frame)
                            written += 1
                            progress("rendering")
                            value = next(timeline, None)
                            if value is None:
                                break
                            expected = json.loads(value)
                finally:
                    rows.close()
            if written != total:
                raise ValueError("Decoded source ended before selected frames")
            progress("finalizing")
            completed = writer.finish(cancel=cancel)
            completed.rename(private / "video.mp4")
        finally:
            if decoded is not None:
                decoded.close()
            if writer is not None:
                writer.close()
        progress("verifying")
        if {name: _export_hash(path, cancel) for name, path in inputs.items()} != hashes:
            raise ValueError("Export inputs changed / đầu vào thay đổi")
        if state is not None:
            state.finish()
        audio_end = (
            original.audio_offset_seconds + original.audio_duration_seconds
            if original.audio_duration_seconds is not None
            else None
        )
        overlap = (
            max(0.0, min(actual_end, audio_end) - max(actual_start, original.audio_offset_seconds))
            if audio_end is not None
            else 0.0
            if original.audio_offset_seconds >= actual_end
            else None
        )
        report = dict(
            schema_version=1,
            status="completed",
            kind="saved_action_video",
            source_sha256=hashes["source"],
            raw_frames_sha256=hashes["frames.jsonl"],
            input_sha256=hashes,
            actor_namespace="run:" + str(manifest.get("run_id", "")),
            new_model_inference=False,
            ground_truth=False,
            requested_interval_s=[requested_start, requested_end],
            actual_interval_s=[actual_start, actual_end],
            interval_policy="Nearest source tick, presentation overlap; held first frame and clipped first/last durations",
            frame_count=written,
            time_base=original.time_base,
            duration_seconds=selected.duration,
            action_rules=action_rules,
            rule_config=rule_config,
            board_demo_proximity=board_demo_proximity,
            whiteboard_offscreen=whiteboard_offscreen,
            board_demo_scope="Near board aliases writing; approaching aliases going to board; not visual writing GT"
            if board_demo_proximity
            else None,
            whiteboard_roi=whiteboard_roi,
            whiteboard_roi_source_sha256=whiteboard_roi_source_sha256,
            display_stability=PoseDisplayStabilizer.config if stabilize_display else None,
            board_roi_source_sha256=board_roi_source_sha256 if board_roi else None,
            source_raster=dict(
                width=original.width,
                height=original.height,
                rotation_degrees=original.rotation_degrees,
                sample_aspect_ratio=original.sample_aspect_ratio,
            ),
            output_raster=dict(
                width=(original.width + 1) // 2 * 2,
                height=(original.height + 1) // 2 * 2,
                rotation_degrees=0,
                sample_aspect_ratio="1:1",
            ),
            audio=dict(
                present=original.audio,
                source_offset_seconds=original.audio_offset_seconds,
                clip_offset_seconds=selected.audio_offset_seconds,
                stream_duration_seconds=original.audio_duration_seconds,
                stream_overlap_seconds=overlap if original.audio else None,
                silence_only=overlap == 0 if original.audio and overlap is not None else None,
                timing_scope="Stream extent, not a semantic audibility classification",
                policy="Source content normalized/trimmed/delayed; silence to clip duration if source has audio, none otherwise",
            ),
            output_pts_validated=True,
            video_sha256=_export_hash(private / "video.mp4", cancel),
            observation_counts=state.observation_counts if state is not None else {},
            note="Causal raw Pose/ID rule hints, uncalibrated drafts; zero hints is not absence. Encoded raster contract matches preview; no future Pose or display smoothing.",
        )
        with (private / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        progress("publishing")
        progress("complete")
        if {name: _export_hash(path, cancel) for name, path in inputs.items()} != hashes:
            raise ValueError("Export inputs changed / đầu vào thay đổi")
        _check_cancel(cancel)
        if destination.exists():
            raise FileExistsError(destination)
        private.rename(destination)
    return destination
