"""Build a source-bound assistant draft for the existing annotations POST.

This module does not read files, call a model, send HTTP, or publish annotations.
The caller supplies the current ``codex-draft`` context and a resolver that reads
native frame rows from THAT context's source ledger. The returned payload is a
complete replacement snapshot, not an implicit merge. The API must compare its
``expected_snapshot`` again at publication to close the build/POST race.

Required draft keys are source_sha256, actor_namespace, human_actors,
human_actor_references, and events. Optional coverage denotes only working areas;
when absent, each event gets its own working interval. Optional review_metadata
can provide rubric/source-group/split. Any actor_links are discarded. Human UUIDs
and frame boxes must be deliberately authored from the source; this helper never
converts a model ID into a person or infers continuous identity from a box.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction

from .review import validate_review

MAX_IMPORT_BYTES = 2 * 1024 * 1024
MAX_IMPORT_EVENTS = 2000
_WRITABLE = (
    "source_sha256",
    "actor_namespace",
    "events",
    "coverage",
    "human_actors",
    "human_actor_references",
    "actor_links",
    "review_metadata",
)
_REQUIRED = {"source_sha256", "actor_namespace", "human_actors", "human_actor_references", "events"}


def _without_human_review(value):
    if isinstance(value, dict):
        return {
            key: _without_human_review(item) for key, item in value.items() if key != "human_review"
        }
    if isinstance(value, list):
        return [_without_human_review(item) for item in value]
    return value


def _native_reference(reference, row):
    try:
        index, pts, base, timestamp = (
            row["frame_index"],
            row["source_pts"],
            row["time_base"],
            row["timestamp_seconds"],
        )
        segment = row["segment_id"] if row["segment_id"] is not None else "source"
        valid = (
            type(index) is int
            and index == reference["frame_index"]
            and type(pts) is int
            and pts == reference["source_pts"]
            and isinstance(base, str)
            and Fraction(base) > 0
            and Fraction(base) == Fraction(reference["time_base"])
            and not isinstance(timestamp, bool)
            and isinstance(timestamp, (int, float))
            and math.isfinite(timestamp)
            and abs(timestamp - reference["timestamp_seconds"]) <= 1e-9
            and not isinstance(segment, bool)
            and segment == reference["segment_id"]
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid native source frame reference") from exc
    if not valid:
        raise ValueError("reference does not match native frame/PTS/time base/time/scene")
    # Canonical source values also match the viewer's exact time_base comparison.
    reference["time_base"] = base
    reference["timestamp_seconds"] = timestamp


def prepare_assistant_draft(draft, *, current_context, frame_resolver, expected_snapshot):
    """Return detached writable annotations fields plus expected_snapshot, or fail.

    ``frame_resolver(frame_index)`` must return the target ledger row, including
    frame_index/source_pts/time_base/timestamp_seconds/segment_id. It is called
    once per distinct referenced frame. No model observations are accepted as
    actor extents, even if a source context also contains model track segments.
    """
    if (
        not isinstance(current_context, dict)
        or current_context.get("review_branch") != "codex-draft"
    ):
        raise ValueError("assistant imports require the codex-draft branch")
    if (
        "review_snapshot" not in current_context
        or expected_snapshot != current_context["review_snapshot"]
        or expected_snapshot is not None
        and not isinstance(expected_snapshot, str)
    ):
        raise ValueError("current review snapshot changed or snapshot precondition is missing")
    if not callable(frame_resolver):
        raise ValueError("native source frame resolver is required")
    if not isinstance(draft, dict) or not _REQUIRED <= draft.keys():
        raise ValueError(
            "draft requires source identity, independent humans, references and events"
        )
    if set(draft) - set(_WRITABLE):
        raise ValueError("unsupported draft fields; supplementary provenance belongs in evidence")
    source_hash, namespace = (
        current_context.get("source_sha256"),
        current_context.get("actor_namespace"),
    )
    if (
        not isinstance(source_hash, str)
        or re.fullmatch(r"[a-f0-9]{64}", source_hash) is None
        or draft["source_sha256"] != source_hash
    ):
        raise ValueError("assistant draft source hash mismatch")
    if not isinstance(namespace, str) or not namespace or draft["actor_namespace"] != namespace:
        raise ValueError("assistant draft actor namespace mismatch")
    try:
        encoded = json.dumps(draft, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_IMPORT_BYTES:
            raise ValueError("assistant draft exceeds the 2 MiB import limit")
        value = json.loads(encoded)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("assistant draft requires finite JSON") from exc
    events = value["events"]
    if not isinstance(events, list) or len(events) > MAX_IMPORT_EVENTS:
        raise ValueError("assistant draft event limit is 2000")
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("evidence"), dict):
            raise ValueError("assistant event requires an evidence object")
        if not {"segment_id", "track_id", "label", "start_s", "end_s"} <= event.keys():
            raise ValueError("assistant event requires actor, label and temporal bounds")
        event.update(origin="assistant", review="draft")
        evidence = _without_human_review(event["evidence"])
        provenance = evidence.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("assistant provenance must be an object")
        if provenance.get("source_sha256", source_hash) != source_hash:
            raise ValueError("assistant provenance source hash mismatch")
        if provenance.get("actor_namespace", namespace) != namespace:
            raise ValueError("assistant provenance target actor namespace mismatch")
        hint = provenance.get("model_hint")
        if hint is not None and (
            not isinstance(hint, dict) or hint.get("source_sha256") != source_hash
        ):
            raise ValueError("model hint must explicitly bind the same source hash")
        provenance.update(
            producer="assistant",
            import_review_state="draft",
            source_sha256=source_hash,
            actor_namespace=namespace,
        )
        event["evidence"] = evidence
    if "coverage" not in value:
        value["coverage"] = [
            {key: event[key] for key in ("segment_id", "track_id", "label", "start_s", "end_s")}
            | {"review": "draft"}
            for event in events
        ]
    if not isinstance(value["coverage"], list):
        raise ValueError("draft coverage must be a list")
    for area in value["coverage"]:
        if not isinstance(area, dict):
            raise ValueError("draft coverage must contain objects")
        area["review"] = "draft"
    existing_metadata = current_context.get("review_metadata", {})
    metadata = value.get("review_metadata", {})
    if not isinstance(metadata, dict) or not isinstance(existing_metadata, dict):
        raise ValueError("review metadata must be an object")
    value["review_metadata"] = {
        **existing_metadata,
        **metadata,
        "reviewer_id": "codex-draft",
        "adjudication_state": "draft",
    }
    value["actor_links"] = []
    try:
        args = dict(
            duration_s=current_context["duration_s"],
            model_segments=[],
            scene_segments=current_context["scene_segments"],
            source_sha256=source_hash,
            actor_namespace=namespace,
            require_actor_references=True,
        )
    except KeyError as exc:
        raise ValueError("current review source context is incomplete") from exc
    checked = validate_review(value, **args)
    if checked["actor_reference_missing_span_count"]:
        raise ValueError("each imported human span needs an explicit native frame reference")
    frames = {}
    for reference in checked["human_actor_references"]:
        index = reference["frame_index"]
        if index not in frames:
            try:
                frames[index] = frame_resolver(index)
            except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError("cannot resolve native source frame reference") from exc
        _native_reference(reference, frames[index])
    checked = validate_review(checked, **args)
    return {key: checked[key] for key in _WRITABLE} | {"expected_snapshot": expected_snapshot}
