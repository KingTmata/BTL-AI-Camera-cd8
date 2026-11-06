"""Reproducible action evaluation from independent human review bundles.

Fixed-threshold, actor-aware event F1 is distinct from score-ranked AP/mAP.
The runner never creates GT, tunes a protocol or changes a dataset split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

from classroom_ai.video.actions import (
    ACTION_LABELS,
    _actor,
    _boundary_flags,
    _coverage_seconds,
    _covered,
    _events,
    _masks,
    _number,
    _reviewed_areas,
    _union_duration,
    evaluate_event_f1,
)
from classroom_ai.video.review import validate_review


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _protocol(value, metadata):
    protocol = _json_copy(value)
    if protocol.get("schema_version") != 1 or not protocol.get("protocol_id"):
        raise ValueError("protocol requires schema_version=1 and protocol_id")
    if protocol.get("phase") not in ("pilot", "dev", "test"):
        raise ValueError("protocol phase must be pilot, dev or test")
    if protocol.get("status") not in ("draft", "locked"):
        raise ValueError("protocol status must be draft or locked")
    if protocol.get("rubric_version") != metadata["rubric_version"]:
        raise ValueError("protocol and GT rubric version mismatch")
    threshold = _number(protocol.get("iou_threshold"), "iou_threshold")
    if not 0 < threshold <= 1:
        raise ValueError("iou_threshold must be in (0,1]")
    if "boundary_tolerance_s" not in protocol:
        raise ValueError("protocol must explicitly set boundary_tolerance_s (null disables it)")
    if protocol["boundary_tolerance_s"] is not None:
        if _number(protocol["boundary_tolerance_s"], "boundary_tolerance_s") < 0:
            raise ValueError("boundary_tolerance_s must be nonnegative")
    groups = []
    for name in ("development_source_groups", "test_source_groups"):
        values = protocol.get(name)
        if not isinstance(values, list) or any(
            not isinstance(s, str) or not s.strip() for s in values
        ):
            raise ValueError(f"{name} requires explicit source groups")
        groups.append(set(values))
    if groups[0] & groups[1]:
        raise ValueError("source group overlap leaks development sources into test")
    if protocol["phase"] == "test":
        if protocol["status"] != "locked":
            raise ValueError("test requires a locked protocol")
        if metadata.get("split") != "test" or metadata["source_group"] not in groups[1]:
            raise ValueError("GT source is not declared in the locked test split")
    elif metadata.get("split") == "test" or metadata["source_group"] in groups[1]:
        raise ValueError("held-out test source cannot be used for pilot/development")
    elif protocol["phase"] == "dev" and metadata["source_group"] not in groups[0]:
        raise ValueError("GT source is not declared in development source groups")
    return protocol


def _mapped_predictions(prediction, reference, review, actor_map):
    same_namespace = prediction["actor_namespace"] == reference["actor_namespace"]
    links = review["actor_links"]
    if actor_map is not None:
        mapping = _json_copy(actor_map)
        if (
            mapping.get("source_sha256") != reference["source_sha256"]
            or mapping.get("from_namespace") != prediction["actor_namespace"]
            or mapping.get("to_namespace") != reference["actor_namespace"]
            or not isinstance(mapping.get("reviewer_id"), str)
            or not mapping["reviewer_id"].strip()
        ):
            raise ValueError("actor mapping requires matching source/namespaces and reviewer")
        links = mapping.get("links")
        if not isinstance(links, list) or len(links) > 2000:
            raise ValueError("actor mapping links must be a bounded list")
    elif not same_namespace:
        raise ValueError("actor namespace mismatch requires an explicit reviewed actor mapping")
    duration = reference["duration_s"]
    humans = review["human_actors"]
    for index, link in enumerate(links):
        if set(link) != {
            "segment_id",
            "model_track_id",
            "human_track_id",
            "start_s",
            "end_s",
            "review",
        }:
            raise ValueError("actor mapping link has unsupported fields")
        _actor(dict(segment_id=link["segment_id"], track_id=link["model_track_id"]))
        start, end = (
            _number(link["start_s"], "mapping start"),
            _number(link["end_s"], "mapping end"),
        )
        human_span = dict(link, track_id=link["human_track_id"])
        if (
            link["review"] != "reviewed"
            or not 0 <= start < end <= duration
            or not _covered(human_span, humans)
        ):
            raise ValueError("actor mapping must fit an explicitly reviewed human actor span")
        for previous in links[:index]:
            if (
                previous["segment_id"] == link["segment_id"]
                and previous["model_track_id"] == link["model_track_id"]
                and previous["start_s"] < end
                and start < previous["end_s"]
            ):
                raise ValueError("conflicting overlapping actor mapping for one model actor")
    mapped, mapped_count = [], 0
    for row in prediction["events"]:
        event = _events([row], duration)[0]
        if same_namespace and _covered(event, humans):
            mapped.append(event)
            continue
        candidates = set()
        for human in humans:
            spans = [
                dict(link, track_id=link["model_track_id"])
                for link in links
                if link["human_track_id"] == human["track_id"]
            ]
            if _covered(event, spans):
                candidates.add(human["track_id"])
        if len(candidates) != 1:
            raise ValueError(
                "unmapped or ambiguous predicted actor; human mapping review is required"
            )
        event["track_id"] = candidates.pop()
        mapped.append(event)
        mapped_count += 1
    return mapped, dict(
        mapped_predictions=mapped_count,
        total_predictions=len(mapped),
        external_mapping=actor_map is not None,
    )


def _errors(values):
    values = [value for value in values if value is not None]
    return dict(
        count=len(values),
        mae=statistics.mean(abs(v) for v in values) if values else None,
        bias=statistics.mean(values) if values else None,
        max_abs=max(map(abs, values)) if values else None,
    )


def _durations(rows):
    actors = {}
    for row in rows:
        actors.setdefault(_actor(row), []).append((row["start_s"], row["end_s"]))
    return sum(_union_duration(intervals, 0, math.inf) for intervals in actors.values())


def review_snapshot_sha256(bundle):
    """Digest finite canonical JSON, distinct from a CLI input file's byte digest."""
    encoded = json.dumps(
        bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_actor_references(bundle):
    required = bundle.get("actor_references_required", False)
    if not isinstance(required, bool):
        raise ValueError("actor_references_required must be a boolean")
    return bundle.get("source_kind") == "annotation_source_preparation" or required


def compare_review_bundles(review_a, review_b, *, actor_map, protocol):
    """Descriptive human-human agreement, never accuracy or adjudicated GT.

    Identity pairs bind canonical snapshot digests, even when run namespaces or
    human UUIDs coincide. Episodes are never clipped to manufacture new onsets.
    """
    bundles = {"a": _json_copy(review_a), "b": _json_copy(review_b)}
    reviews, identities = {}, {}
    for side, bundle in bundles.items():
        if (
            bundle.get("schema_version") != 2
            or not isinstance(bundle.get("source_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", bundle["source_sha256"]) is None
            or not isinstance(bundle.get("actor_namespace"), str)
            or not bundle["actor_namespace"].strip()
        ):
            raise ValueError("comparison requires schema2 reviews with source hash and namespace")
        review = validate_review(
            bundle,
            duration_s=bundle["duration_s"],
            model_segments=bundle.get("model_track_segments", []),
            scene_segments=bundle.get("scene_segments", []),
            source_sha256=bundle["source_sha256"],
            actor_namespace=bundle["actor_namespace"],
            require_actor_references=_require_actor_references(bundle),
        )
        if not review["evaluation_ready"]:
            raise ValueError(
                "comparison requires reviewed provenance: " + ", ".join(review["readiness_reasons"])
            )
        reviews[side] = review
        identities[side] = dict(
            snapshot_sha256=review_snapshot_sha256(bundle),
            reviewer_id=review["review_metadata"]["reviewer_id"],
            actor_namespace=bundle["actor_namespace"],
        )
    a, b = bundles["a"], bundles["b"]
    if identities["a"]["reviewer_id"] == identities["b"]["reviewer_id"]:
        raise ValueError("comparison requires distinct reviewer IDs")
    for field in ("source_sha256", "duration_s", "scene_segments"):
        if a.get(field) != b.get(field):
            raise ValueError("review comparison mismatch: " + field)
    for field in ("rubric_version", "source_group", "split"):
        if reviews["a"]["review_metadata"].get(field) != reviews["b"]["review_metadata"].get(field):
            raise ValueError("review metadata mismatch: " + field)
    chosen = _protocol(protocol, reviews["a"]["review_metadata"])
    _protocol(protocol, reviews["b"]["review_metadata"])
    if chosen["boundary_tolerance_s"] is not None:
        raise ValueError(
            "human agreement requires null boundary_tolerance_s for symmetric matching"
        )
    mapping = _json_copy(actor_map)
    if (
        set(mapping)
        != {
            "schema_version",
            "kind",
            "source_sha256",
            "mapping_reviewer_id",
            "review",
            "a",
            "b",
            "pairs",
        }
        or mapping["schema_version"] != 1
        or mapping["kind"] != "human_actor_pairs"
        or mapping["source_sha256"] != a["source_sha256"]
        or mapping["review"] != "reviewed"
        or not isinstance(mapping["mapping_reviewer_id"], str)
        or not 0 < len(mapping["mapping_reviewer_id"].strip()) <= 160
        or any(mapping[side] != identities[side] for side in ("a", "b"))
        or not isinstance(mapping["pairs"], list)
        or len(mapping["pairs"]) > 2000
    ):
        raise ValueError("actor pairs require reviewed, snapshot-bound source/reviewer provenance")
    pairs = mapping["pairs"]
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or set(pair) != {
            "segment_id",
            "a_track_id",
            "b_track_id",
            "start_s",
            "end_s",
        }:
            raise ValueError("unsupported human actor pair fields")
        start, end = (_number(pair[key], key) for key in ("start_s", "end_s"))
        if not 0 <= start < end <= a["duration_s"]:
            raise ValueError("actor pair interval outside source")
        for side in ("a", "b"):
            span = dict(pair, track_id=pair[side + "_track_id"])
            if not _covered(span, reviews[side]["human_actors"]):
                raise ValueError("actor pair must fit both known human actor spans")
            if any(
                old["segment_id"] == pair["segment_id"]
                and old[side + "_track_id"] == pair[side + "_track_id"]
                and old["start_s"] < end
                and start < old["end_s"]
                for old in pairs[:index]
            ):
                raise ValueError("overlapping human actor pairs")
    # Merge adjacent identical correspondences; their arbitrary UI boundaries
    # must not turn a continuous episode into a discarded or split episode.
    merged = []
    for pair in sorted(pairs, key=lambda row: row["start_s"]):
        previous = next(
            (
                old
                for old in merged
                if all(old[k] == pair[k] for k in ("segment_id", "a_track_id", "b_track_id"))
                and old["end_s"] == pair["start_s"]
            ),
            None,
        )
        if previous is None:
            merged.append(dict(pair))
        else:
            previous["end_s"] = pair["end_s"]
    areas = {"a": [], "b": []}
    masks = []
    segments = []
    for pair_id, pair in enumerate(merged):
        segments.append(
            dict(
                segment_id=pair["segment_id"],
                track_id=pair_id,
                start_s=pair["start_s"],
                end_s=pair["end_s"],
            )
        )
        for side in ("a", "b"):
            for row in reviews[side]["coverage"]:
                if (
                    _actor(row) != (pair["segment_id"], pair[side + "_track_id"])
                    or row["review"] != "reviewed"
                ):
                    continue
                start, end = max(row["start_s"], pair["start_s"]), min(row["end_s"], pair["end_s"])
                if start < end:
                    areas[side].append(dict(row, track_id=pair_id, start_s=start, end_s=end))
            for row in reviews[side]["events"]:
                if (
                    row["review"] == "reviewed"
                    and not row["uncertain"]
                    and row["label"] != "unknown"
                ):
                    continue
                if _actor(row) == (pair["segment_id"], pair[side + "_track_id"]):
                    start, end = (
                        max(row["start_s"], pair["start_s"]),
                        min(row["end_s"], pair["end_s"]),
                    )
                    if start < end:
                        masks.append(dict(row, track_id=pair_id, start_s=start, end_s=end))
    shared = []
    for left in areas["a"]:
        for right in areas["b"]:
            if _actor(left) == _actor(right) and left["label"] == right["label"]:
                start, end = (
                    max(left["start_s"], right["start_s"]),
                    min(left["end_s"], right["end_s"]),
                )
                if start < end:
                    shared.append(dict(left, start_s=start, end_s=end))
    shared = _reviewed_areas(shared, masks)
    eligible, indices, audit = {}, {}, {}
    for side in ("a", "b"):
        review = reviews[side]
        eligible[side], indices[side], excluded = [], [], []
        for index, row in enumerate(review["events"]):
            if row["label"] not in ACTION_LABELS:
                continue
            candidates = [
                i
                for i, pair in enumerate(merged)
                if _covered(row, [dict(pair, track_id=pair[side + "_track_id"])])
            ]
            reason = None
            if row["uncertain"] or row["review"] != "reviewed":
                reason = "uncertain_or_draft"
            elif not candidates:
                reason = "unmapped_or_crosses_pair_boundary"
            else:
                normalized = dict(row, track_id=candidates[0])
                if any(_masks(normalized, mask) for mask in masks):
                    reason = "masked_by_either_reviewer"
                elif not _covered(normalized, shared, label=True, reviewed=True):
                    reason = "outside_shared_coverage"
                else:
                    eligible[side].append(normalized)
                    indices[side].append(index)
            if reason:
                excluded.append(dict(event_index=index, label=row["label"], reason=reason))
        own_masks = [
            row
            for row in review["events"]
            if row["uncertain"] or row["review"] != "reviewed" or row["label"] == "unknown"
        ]
        own_areas = _reviewed_areas(review["coverage"], own_masks)
        actors = list(dict.fromkeys(_actor(row) for row in review["human_actors"]))
        unmapped = [
            actor
            for actor in actors
            if not any(actor == (p["segment_id"], p[side + "_track_id"]) for p in merged)
        ]
        human_seconds = _durations(review["human_actors"])
        paired_seconds = sum(p["end_s"] - p["start_s"] for p in merged)
        audit[side] = dict(
            unmapped_actor_count=len(unmapped),
            unmapped_actors=[dict(segment_id=seg, track_id=tid) for seg, tid in unmapped],
            human_actor_s=human_seconds,
            paired_actor_s=paired_seconds,
            unpaired_actor_s=max(0, human_seconds - paired_seconds),
            unshared_coverage_actor_s={
                label: max(
                    0,
                    _coverage_seconds(own_areas, label, 0, a["duration_s"])
                    - _coverage_seconds(shared, label, 0, a["duration_s"]),
                )
                for label in ACTION_LABELS
            },
            excluded_event_count=len(excluded),
            excluded_events=excluded,
        )
    matching = evaluate_event_f1(
        eligible["a"],
        eligible["b"],
        duration_s=a["duration_s"],
        track_segments=segments,
        coverage=shared,
        iou_threshold=chosen["iou_threshold"],
        boundary_tolerance_s=None,
        include_matches=True,
    )
    matches = []
    for pair in matching["matches"]:
        ai, bi = indices["a"][pair["prediction_index"]], indices["b"][pair["reference_index"]]
        left, right = reviews["a"]["events"][ai], reviews["b"]["events"][bi]
        errors = {}
        for edge in ("start", "end"):
            attested = all(
                row["evidence"].get("boundary_truncated", {}).get(edge) is False
                for row in (left, right)
            )
            errors[edge + "_difference_s"] = (
                right[edge + "_s"] - left[edge + "_s"] if attested else None
            )
        matches.append(
            dict(
                a_event_index=ai,
                b_event_index=bi,
                label=left["label"],
                temporal_iou=pair["temporal_iou"],
                **errors,
            )
        )
    per_label = {}
    for label in ACTION_LABELS:
        count_a = sum(row["label"] == label for row in eligible["a"])
        count_b = sum(row["label"] == label for row in eligible["b"])
        paired = [row for row in matches if row["label"] == label]
        per_label[label] = dict(
            matched=len(paired),
            a_only=count_a - len(paired),
            b_only=count_b - len(paired),
            a_episodes=count_a,
            b_episodes=count_b,
            agreement_f1=2 * len(paired) / (count_a + count_b) if count_a + count_b else None,
            shared_coverage_actor_s=_coverage_seconds(shared, label, 0, a["duration_s"]),
            start_difference_s=_errors(row["start_difference_s"] for row in paired),
            end_difference_s=_errors(row["end_difference_s"] for row in paired),
        )
    return dict(
        schema_version=1,
        kind="inter_review_agreement",
        status="descriptive_agreement",
        source_sha256=a["source_sha256"],
        snapshots=identities,
        protocol=chosen,
        actor_map_sha256=review_snapshot_sha256(mapping),
        snapshot_digest_policy="sha256_utf8_json_sort_keys_compact_ensure_ascii_false_finite",
        per_label=per_label,
        audit=audit,
        matches=matches,
        matching_policy=matching["matching_policy"],
        caveat="Descriptive agreement, not accuracy or adjudicated GT. Distinct pseudonyms do not prove independent annotation. "
        "Only explicit actor pairs and jointly reviewed label coverage are compared; either reviewer's uncertainty masks apply. "
        "Unmapped actors/unshared coverage are unresolved, not negative examples. Whole episodes crossing coverage or mapping gaps are excluded, never clipped. "
        "Matching uses symmetric observed-interval tIoU without endpoint tolerance; endpoint differences B minus A require both explicit uncensored attestations. "
        "Maximum-cardinality ties use input order; endpoint summaries describe that pairing. No automatic approval.",
    )


def evaluate_bundle(prediction, reference, *, protocol, actor_map=None):
    """Evaluate only reviewed human coverage; fail closed on source/identity gaps."""
    prediction, reference = _json_copy(prediction), _json_copy(reference)
    for bundle in (prediction, reference):
        if bundle.get("schema_version") != 2:
            raise ValueError("evaluation requires bundle schema_version=2")
        if (
            not isinstance(bundle.get("source_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", bundle["source_sha256"]) is None
        ):
            raise ValueError("bundle requires a source SHA256")
        if (
            not isinstance(bundle.get("actor_namespace"), str)
            or not bundle["actor_namespace"].strip()
        ):
            raise ValueError("bundle requires an explicit actor namespace")
    if prediction["source_sha256"] != reference["source_sha256"]:
        raise ValueError("prediction and GT source hashes differ")
    review = validate_review(
        reference,
        duration_s=reference["duration_s"],
        model_segments=reference.get("model_track_segments", []),
        scene_segments=reference.get("scene_segments", []),
        source_sha256=reference["source_sha256"],
        actor_namespace=reference["actor_namespace"],
        require_actor_references=_require_actor_references(reference),
    )
    if not review["evaluation_ready"]:
        raise ValueError(
            "GT review is not evaluation-ready: " + ", ".join(review["readiness_reasons"])
        )
    chosen = _protocol(protocol, review["review_metadata"])
    predicted, mapping = _mapped_predictions(prediction, reference, review, actor_map)
    # Matching uses action-only rows internally, while audit indices must still
    # address the caller's original bundle (which may also contain postures).
    action_indices = [i for i, row in enumerate(predicted) if row["label"] in ACTION_LABELS]
    predicted = [predicted[i] for i in action_indices]
    matching = evaluate_event_f1(
        predicted,
        review["events"],
        duration_s=reference["duration_s"],
        track_segments=review["human_actors"],
        coverage=review["coverage"],
        iou_threshold=chosen["iou_threshold"],
        boundary_tolerance_s=chosen["boundary_tolerance_s"],
        include_matches=True,
    )
    good_preds = [predicted[i] for i in matching["eligible_prediction_indices"]]
    good_refs = [review["events"][i] for i in matching["eligible_reference_indices"]]
    per_label = {}
    for label in ACTION_LABELS:
        preds = [row for row in good_preds if row["label"] == label]
        refs = [row for row in good_refs if row["label"] == label]
        pairs = [
            pair
            for pair in matching["matches"]
            if review["events"][pair["reference_index"]]["label"] == label
        ]
        tp, fp, fn = len(pairs), len(preds) - len(pairs), len(refs) - len(pairs)
        covered = matching["reviewed_coverage_actor_s"][label]
        pred_count = sum(not _boundary_flags(row, review["coverage"])[0] for row in preds)
        ref_count = sum(not _boundary_flags(row, review["coverage"])[0] for row in refs)
        per_label[label] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
            reference_episodes=len(refs),
            prediction_episodes=len(preds),
            ignored_predictions=sum(row["label"] == label for row in predicted) - len(preds),
            ignored_references=sum(row["label"] == label for row in review["events"]) - len(refs),
            reviewed_coverage_actor_s=covered,
            predicted_onsets=pred_count if covered else None,
            reference_onsets=ref_count if covered else None,
            count_error=pred_count - ref_count if covered else None,
            duration_error_s=_durations(preds) - _durations(refs) if covered else None,
            start_error_s=_errors(pair["start_error_s"] for pair in pairs),
            end_error_s=_errors(pair["end_error_s"] for pair in pairs),
        )
    tp, fp, fn = (sum(row[key] for row in per_label.values()) for key in ("tp", "fp", "fn"))
    return dict(
        schema_version=1,
        kind="action_evaluation",
        status="locked_test_metrics" if chosen["phase"] == "test" else chosen["phase"] + "_metrics",
        source_sha256=reference["source_sha256"],
        actor_namespace=reference["actor_namespace"],
        review_metadata=review["review_metadata"],
        protocol=chosen,
        mapping=mapping,
        per_label=per_label,
        micro=dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        ),
        missing_positive_labels=[
            label for label, row in per_label.items() if not row["reference_episodes"]
        ],
        matching_policy=matching["matching_policy"],
        matches=[
            dict(pair, prediction_index=action_indices[pair["prediction_index"]])
            for pair in matching["matches"]
        ],
        caveat="Actor-scoped reviewed coverage only. Unreviewed/uncertain spans are excluded, not negatives. "
        "Duration errors use unions of observed actor-seconds; censored endpoints have no boundary error. "
        "Pilot/dev metrics are not independent test results. No automatic student interaction scores.",
    )


def evaluate_corpus(items, *, protocol):
    """One adjudicated review per source; never count sibling snapshots as new videos."""
    if not isinstance(items, list) or not items:
        raise ValueError("cannot evaluate an empty corpus")
    if len(items) > 10000:
        raise ValueError("corpus exceeds 10000 source bundles")
    results, sources, group_splits = [], set(), {}
    for item in items:
        result = evaluate_bundle(**item, protocol=protocol)
        if result["source_sha256"] in sources:
            raise ValueError("duplicate source hash: use one GT snapshot per source")
        sources.add(result["source_sha256"])
        metadata = result["review_metadata"]
        group, split = metadata["source_group"], metadata.get("split", "unassigned")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("one source group cannot span different dataset splits")
        group_splits[group] = split
        results.append(result)
    totals = {}
    for label in ACTION_LABELS:
        rows = [result["per_label"][label] for result in results]
        tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        totals[label] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
            reference_episodes=sum(row["reference_episodes"] for row in rows),
            prediction_episodes=sum(row["prediction_episodes"] for row in rows),
            reviewed_coverage_actor_s=sum(row["reviewed_coverage_actor_s"] for row in rows),
            ignored_predictions=sum(row["ignored_predictions"] for row in rows),
            ignored_references=sum(row["ignored_references"] for row in rows),
            count_error_sum=sum(row["count_error"] or 0 for row in rows)
            if any(row["count_error"] is not None for row in rows)
            else None,
            count_abs_error_sum=sum(abs(row["count_error"] or 0) for row in rows)
            if any(row["count_error"] is not None for row in rows)
            else None,
            duration_error_sum_s=sum(row["duration_error_s"] or 0 for row in rows)
            if any(row["duration_error_s"] is not None for row in rows)
            else None,
            duration_abs_error_sum_s=sum(abs(row["duration_error_s"] or 0) for row in rows)
            if any(row["duration_error_s"] is not None for row in rows)
            else None,
        )
        for endpoint in ("start_error_s", "end_error_s"):
            count = sum(row[endpoint]["count"] for row in rows)
            totals[label][endpoint] = dict(
                count=count,
                mae=sum((row[endpoint]["mae"] or 0) * row[endpoint]["count"] for row in rows)
                / count
                if count
                else None,
                bias=sum((row[endpoint]["bias"] or 0) * row[endpoint]["count"] for row in rows)
                / count
                if count
                else None,
                max_abs=max(
                    (row[endpoint]["max_abs"] for row in rows if row[endpoint]["count"]),
                    default=None,
                ),
            )
    tp, fp, fn = (sum(row[key] for row in totals.values()) for key in ("tp", "fp", "fn"))
    supported = [row["f1"] for row in totals.values() if row["reference_episodes"]]
    return dict(
        schema_version=1,
        kind="action_corpus_evaluation",
        status=results[0]["status"],
        protocol=_json_copy(protocol),
        source_count=len(sources),
        independent_source_groups=len(group_splits),
        per_label=totals,
        micro=dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        ),
        macro_f1_positive_labels=statistics.mean(supported) if supported else None,
        missing_positive_labels=[
            label for label, row in totals.items() if not row["reference_episodes"]
        ],
        sources=results,
        caveat="One snapshot per source hash; source independence follows declared source groups. "
        "Macro F1 includes only labels with positive reviewed support. "
        "Missing classes and actor-scoped unknown coverage remain unvalidated.",
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare-reviews":
        return _comparison_main(argv[1:])
    parser = argparse.ArgumentParser(description="Evaluate actions against independent human GT")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prediction", type=Path)
    inputs.add_argument("--corpus", type=Path)
    parser.add_argument("--reference", type=Path)
    for name in ("protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--actor-map", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = {"protocol": args.protocol}
        if args.corpus:
            if args.reference or args.actor_map:
                raise ValueError(
                    "corpus items must declare their own reference and actor_map paths"
                )
            paths["corpus"] = args.corpus
        else:
            if not args.reference:
                raise ValueError("single-source evaluation requires --reference")
            paths.update(prediction=args.prediction, reference=args.reference)
            if args.actor_map:
                paths["actor_map"] = args.actor_map
        raw = {name: path.read_bytes() for name, path in paths.items()}
        data = {name: json.loads(value) for name, value in raw.items()}
        if args.corpus:
            corpus = data["corpus"]
            if corpus.get("schema_version") != 1 or not isinstance(corpus.get("items"), list):
                raise ValueError("corpus manifest requires schema_version=1 and items")
            bundles, hashes = [], []
            for item in corpus["items"]:
                if set(item) - {"prediction", "reference", "actor_map"} or not {
                    "prediction",
                    "reference",
                } <= set(item):
                    raise ValueError(
                        "corpus item requires prediction/reference and optional actor_map paths"
                    )
                contents = {
                    name: (args.corpus.parent / path).read_bytes() for name, path in item.items()
                }
                bundles.append({name: json.loads(value) for name, value in contents.items()})
                hashes.append(
                    {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
                )
            report = evaluate_corpus(bundles, protocol=data["protocol"])
            report["source_input_sha256"] = hashes
        else:
            report = evaluate_bundle(data.pop("prediction"), data.pop("reference"), **data)
        report["input_sha256"] = {
            name: hashlib.sha256(value).hexdigest() for name, value in raw.items()
        }
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(f"{report['status']}: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _comparison_main(argv):
    parser = argparse.ArgumentParser(
        description="Describe agreement between immutable human reviews"
    )
    for name in ("review-a", "review-b", "actor-map", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = {
            name: getattr(args, name).read_bytes()
            for name in ("review_a", "review_b", "actor_map", "protocol")
        }
        data = {name: json.loads(value) for name, value in raw.items()}
        report = compare_review_bundles(data.pop("review_a"), data.pop("review_b"), **data)
        report["input_sha256"] = {
            name: hashlib.sha256(value).hexdigest() for name, value in raw.items()
        }
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(f"{report['status']}: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
