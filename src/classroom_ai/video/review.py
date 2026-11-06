"""Human actor spans and explicit review provenance, separate from model observations."""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction

from classroom_ai.video.actions import (
    ACTION_LABELS,
    _actor,
    _covered,
    _duration,
    _events,
    _extents,
    _number,
    _reviewed_areas,
    validate_events,
)

MAX_HUMAN_SPANS = 1000
MAX_ACTOR_LINKS = 2000
MAX_ACTOR_REFERENCES = 4000
MAX_ACTOR_REFERENCE_NOTE = 240
MAX_PILOT_WINDOWS = 100
MAX_PILOT_GAPS = 1024
HUMAN_ID = re.compile(r"human:[a-f0-9]{32}")
_SPAN_KEYS = {"segment_id", "track_id", "start_s", "end_s"}
_LINK_KEYS = {"segment_id", "model_track_id", "human_track_id", "start_s", "end_s", "review"}
_METADATA_KEYS = {"reviewer_id", "rubric_version", "source_group", "split", "adjudication_state"}
_REFERENCE_KEYS = {
    "segment_id",
    "track_id",
    "frame_index",
    "source_pts",
    "time_base",
    "timestamp_seconds",
    "bbox_xyxyn",
    "note",
}


def _identity(value):
    if isinstance(value, bool) or not (
        isinstance(value, int) and value >= 0 or isinstance(value, str) and value.strip()
    ):
        raise ValueError("actor link/segment identity must be a nonnegative int or nonempty string")


def _interval(row, duration):
    a, b = row.get("start_s"), row.get("end_s")
    if (
        any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (a, b)
        )
        or not 0 <= a < b <= duration
    ):
        raise ValueError("actor/link span must satisfy 0 <= start_s < end_s <= duration_s")
    return a, b


def _inside(row, spans, key="track_id"):
    end = row["start_s"]
    for span in sorted(spans, key=lambda item: item["start_s"]):
        if span["segment_id"] != row["segment_id"] or (key and span["track_id"] != row[key]):
            continue
        if span["start_s"] > end:
            return False
        end = max(end, span["end_s"])
        if end >= row["end_s"]:
            return True
    return False


def _actor_references(value, humans, duration):
    """Check visual anchors, never infer continuous identity or actual source PTS.

    Native PTS can have a nonzero origin. The API must resolve each anchor against
    the source frame ledger; this helper checks only structure and local spans.
    """
    refs = value.get("human_actor_references", [])
    if not isinstance(refs, list) or len(refs) > MAX_ACTOR_REFERENCES:
        raise ValueError(f"human actor reference list limit is {MAX_ACTOR_REFERENCES}")
    by_actor = {}
    for index, span in enumerate(humans):
        by_actor.setdefault((span["segment_id"], span["track_id"]), []).append((index, span))
    seen, anchored = set(), set()
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != _REFERENCE_KEYS:
            raise ValueError(
                "human actor reference requires exactly its source, box and note fields"
            )
        _identity(ref["segment_id"])
        _identity(ref["track_id"])
        for field in ("frame_index", "source_pts"):
            if isinstance(ref[field], bool) or not isinstance(ref[field], int) or ref[field] < 0:
                raise ValueError(
                    "actor reference frame index/source PTS must be nonnegative integers"
                )
        base = ref["time_base"]
        if (
            not isinstance(base, str)
            or len(base) > 80
            or re.fullmatch(r"[0-9]+/[0-9]+", base) is None
        ):
            raise ValueError("actor reference time_base requires a positive rational string")
        try:
            if Fraction(base) <= 0:
                raise ValueError("nonpositive time base")
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(
                "actor reference time_base requires a positive rational string"
            ) from exc
        timestamp = _number(ref["timestamp_seconds"], "actor reference timestamp_seconds")
        if not 0 <= timestamp < duration:
            raise ValueError("actor reference timestamp exceeds source bounds")
        matches = [
            index
            for index, span in by_actor.get((ref["segment_id"], ref["track_id"]), [])
            if span["start_s"] <= timestamp < span["end_s"]
        ]
        if not matches:
            raise ValueError("actor reference must lie in an independently declared human span")
        key = (ref["segment_id"], ref["track_id"], ref["frame_index"])
        if key in seen:
            raise ValueError("duplicate human actor reference anchor")
        seen.add(key)
        anchored.update(matches)
        box = ref["bbox_xyxyn"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("actor reference bbox_xyxyn requires four normalized coordinates")
        x1, y1, x2, y2 = [_number(v, "actor reference bbox_xyxyn") for v in box]
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError("actor reference bbox_xyxyn must have positive area within 0..1")
        if not isinstance(ref["note"], str) or len(ref["note"]) > MAX_ACTOR_REFERENCE_NOTE:
            raise ValueError(
                f"actor reference note must be text of at most {MAX_ACTOR_REFERENCE_NOTE} characters"
            )
    return refs, len(humans) - len(anchored)


def validate_review(
    payload,
    *,
    duration_s,
    model_segments,
    scene_segments,
    source_sha256,
    actor_namespace,
    require_actor_references=False,
):
    """Validate a sidecar snapshot; never create people, links or human approvals."""
    if not isinstance(payload, dict):
        raise ValueError("review must be an object")
    if not isinstance(require_actor_references, bool):
        raise ValueError("require_actor_references must be a boolean")
    duration_s = _duration(duration_s)
    if payload.get("source_sha256", source_sha256) != source_sha256:
        raise ValueError("review source hash mismatch")
    if payload.get("actor_namespace", actor_namespace) != actor_namespace:
        raise ValueError("review actor namespace mismatch")
    # Detached finite JSON protects immutable snapshots from aliases and NaN.
    value = json.loads(json.dumps(payload, allow_nan=False))
    humans, links = value.get("human_actors", []), value.get("actor_links", [])
    if not isinstance(humans, list) or len(humans) > MAX_HUMAN_SPANS:
        raise ValueError("human actor span limit is 1000")
    if not isinstance(links, list) or len(links) > MAX_ACTOR_LINKS:
        raise ValueError("actor link limit is 2000")
    scenes = {}
    for row in humans:
        if not isinstance(row, dict) or set(row) != _SPAN_KEYS:
            raise ValueError("human actor requires segment_id, track_id, start_s, end_s")
        tid = row["track_id"]
        _identity(row["segment_id"])
        if not isinstance(tid, str) or HUMAN_ID.fullmatch(tid) is None:
            raise ValueError("human actor track_id must be human:<32 lowercase hex>")
        _interval(row, duration_s)
        if not _inside(row, scene_segments, key=None):
            raise ValueError("human actor span must fit one source scene")
        if tid in scenes and scenes[tid] != row["segment_id"]:
            raise ValueError("human actor cannot cross source scenes")
        scenes[tid] = row["segment_id"]
    for i, row in enumerate(humans):
        if any(
            old["track_id"] == row["track_id"]
            and old["start_s"] < row["end_s"]
            and row["start_s"] < old["end_s"]
            for old in humans[:i]
        ):
            raise ValueError("overlapping spans for the same human actor")
    references, missing_reference_spans = _actor_references(value, humans, duration_s)
    for i, link in enumerate(links):
        if not isinstance(link, dict) or set(link) != _LINK_KEYS or link["review"] != "reviewed":
            raise ValueError("actor link requires explicit reviewed mapping and exact fields")
        _identity(link["segment_id"])
        _identity(link["model_track_id"])
        _interval(link, duration_s)
        if not _inside(link, model_segments, "model_track_id") or not _inside(
            link, humans, "human_track_id"
        ):
            raise ValueError(
                "actor link must fit known model and human actor spans in the same scene"
            )
        for old in links[:i]:
            if (
                old["segment_id"] == link["segment_id"]
                and old["start_s"] < link["end_s"]
                and link["start_s"] < old["end_s"]
            ):
                if old["model_track_id"] == link["model_track_id"]:
                    raise ValueError("overlapping/conflicting actor links")
    metadata = value.get("review_metadata", {})
    if not isinstance(metadata, dict) or set(metadata) - _METADATA_KEYS:
        raise ValueError("unsupported review_metadata fields")
    for name, item in metadata.items():
        if not isinstance(item, str) or len(item) > 160:
            raise ValueError("review metadata values must be strings of at most 160 characters")
        metadata[name] = item.strip()
    if metadata.get("split", "") not in ("", "train", "val", "test", "unassigned"):
        raise ValueError("unsupported declared split")
    if metadata.get("adjudication_state", "") not in ("", "draft", "reviewed", "adjudicated"):
        raise ValueError("unsupported adjudication_state")
    segments = [*model_segments, *humans]
    coverage = value.get("coverage", [])
    events = validate_events(
        value.get("events", []), duration_s=duration_s, track_segments=segments, coverage=coverage
    )
    if metadata.get("reviewer_id") == "codex-draft" and (
        metadata.get("adjudication_state") in ("reviewed", "adjudicated")
        or any(row["review"] == "reviewed" for row in [*events, *coverage])
    ):
        raise ValueError("codex-draft cannot claim reviewed annotations or human adjudication")
    reasons = []
    for field in ("reviewer_id", "rubric_version", "source_group"):
        if not metadata.get(field):
            reasons.append("missing_" + field)
    if metadata.get("adjudication_state") not in ("reviewed", "adjudicated"):
        reasons.append("human_review_not_confirmed")
    if not humans:
        reasons.append("missing_independent_human_actors")
    if not coverage:
        reasons.append("missing_reviewed_label_coverage")
    if any(row["review"] != "reviewed" for row in [*events, *coverage]):
        reasons.append("draft_annotations")
    if any(not _inside(row, humans) for row in [*events, *coverage]):
        reasons.append("annotations_reference_model_actors")
    if require_actor_references and missing_reference_spans:
        reasons.append("missing_actor_reference")
    return dict(
        events=events,
        coverage=coverage,
        human_actors=humans,
        human_actor_references=references,
        actor_reference_count=len(references),
        actor_reference_missing_span_count=missing_reference_spans,
        actor_references_required=require_actor_references,
        actor_links=links,
        review_metadata=metadata,
        evaluation_ready=not reasons,
        readiness_reasons=reasons,
        actor_namespace=actor_namespace,
        source_sha256=source_sha256,
        track_segments=segments,
    )


def _merged_spans(spans):
    merged = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _missing_spans(declared, covered):
    for start, end in declared:
        cursor = start
        for left, right in covered:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                yield cursor, min(left, end)
            cursor = max(cursor, min(right, end))
        if cursor < end:
            yield cursor, end


def audit_pilot_review(review, windows, *, duration_s):
    """Audit saved, validated human spans; coverage is not an exhaustive person inventory.

    This deliberately does not change validate_review's partial-GT export gate.
    Missing human actors give an unknown denominator, never a completed window.
    Gaps are bounded for presentation; all missing seconds still enter totals.
    """
    duration = _duration(duration_s)
    if not isinstance(windows, list) or not 1 <= len(windows) <= MAX_PILOT_WINDOWS:
        raise ValueError("pilot window limit is 1..100")
    seen = set()
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("pilot window must be an object")
        name = window.get("window_id")
        if not isinstance(name, str) or not name.strip() or len(name) > 160 or name in seen:
            raise ValueError("pilot window requires a unique nonempty window_id")
        seen.add(name)
        try:
            _interval(window, duration)
        except ValueError as exc:
            raise ValueError("pilot window interval is invalid") from exc
    humans = review.get("human_actors", [])
    if not isinstance(humans, list) or len(humans) > MAX_HUMAN_SPANS:
        raise ValueError("human actor span limit is 1000")
    actors = {}
    for span in humans:
        _interval(span, duration)
        tid = span.get("track_id")
        if not isinstance(tid, str) or HUMAN_ID.fullmatch(tid) is None:
            raise ValueError("pilot requires independent human actors")
        actors.setdefault(_actor(span), []).append((span["start_s"], span["end_s"]))
    actors = {key: _merged_spans(spans) for key, spans in actors.items()}
    rows = _events(review.get("events", []), duration)
    masks = [
        r for r in rows if r["review"] != "reviewed" or r["uncertain"] or r["label"] == "unknown"
    ]
    original_areas = _extents(review.get("coverage", []), duration, coverage=True)
    areas = _reviewed_areas(original_areas, masks)
    by_label = {}
    for area in areas:
        if area["label"] in ACTION_LABELS and _covered(area, humans):
            key = (_actor(area), area["label"])
            by_label.setdefault(key, []).append((area["start_s"], area["end_s"]))
    by_label = {key: _merged_spans(spans) for key, spans in by_label.items()}
    human_events = [r for r in rows if _covered(r, humans)]
    result, retained = [], 0
    for window in windows:
        start, end = window["start_s"], window["end_s"]
        selected = {
            key: _merged_spans((max(a, start), min(b, end)) for a, b in spans)
            for key, spans in actors.items()
        }
        selected = {key: spans for key, spans in selected.items() if spans}
        denominator = sum(b - a for spans in selected.values() for a, b in spans)
        labels = []
        active = [r for r in human_events if r["start_s"] < end and r["end_s"] > start]
        for label in ACTION_LABELS:
            gaps, missing, gap_count = [], 0.0, 0
            for key, spans in selected.items():
                for a, b in _missing_spans(spans, by_label.get((key, label), [])):
                    missing += b - a
                    gap_count += 1
                    if retained < MAX_PILOT_GAPS:
                        gaps.append(dict(segment_id=key[0], track_id=key[1], start_s=a, end_s=b))
                        retained += 1
            relevant = [r for r in active if r["label"] == label]
            labels.append(
                dict(
                    label=label,
                    reviewed_actor_seconds=max(0.0, denominator - missing),
                    declared_actor_seconds=denominator,
                    coverage_fraction=min(1.0, max(0.0, (denominator - missing) / denominator))
                    if denominator
                    else None,
                    unreviewed_actor_seconds=missing if denominator else None,
                    draft_intervals=sum(r["review"] != "reviewed" for r in relevant)
                    + sum(
                        r["review"] != "reviewed"
                        and r["label"] == label
                        and r["start_s"] < end
                        and r["end_s"] > start
                        and _covered(r, humans)
                        for r in original_areas
                    ),
                    uncertain_intervals=sum(r["uncertain"] for r in relevant),
                    reviewed_positive_intervals=sum(
                        r["review"] == "reviewed"
                        and not r["uncertain"]
                        and _covered(r, areas, label=True, reviewed=True)
                        for r in relevant
                    ),
                    coverage_gaps=gaps,
                    gap_count=gap_count,
                    gaps_truncated=gap_count > len(gaps),
                )
            )
        complete = bool(denominator) and all(r["unreviewed_actor_seconds"] <= 1e-9 for r in labels)
        started = any(r["reviewed_actor_seconds"] > 0 for r in labels)
        result.append(
            dict(
                window_id=window["window_id"],
                status="reviewed_declared_scope"
                if complete
                else "partial"
                if started
                else "unstarted",
                declared_actor_count=len(selected),
                labels=labels,
            )
        )
    return dict(
        scope="declared_human_actors",
        saved_snapshot_available=bool(review.get("review_snapshot")),
        source_inventory_confirmed=False,
        declared_actor_scope_complete=all(r["status"] == "reviewed_declared_scope" for r in result),
        windows=result,
    )
