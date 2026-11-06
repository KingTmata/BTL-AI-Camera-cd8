"""Synthetic import contracts; these fixtures are not real classroom annotations."""

import copy
import importlib

import pytest

from classroom_ai.video.actions import aggregate_events, validate_events
from classroom_ai.video.review import validate_review

SHA = "a" * 64
NS = "run:prepared"
HUMAN = "human:" + "b" * 32
WRITABLE = {
    "source_sha256",
    "actor_namespace",
    "events",
    "coverage",
    "human_actors",
    "human_actor_references",
    "actor_links",
    "review_metadata",
    "expected_snapshot",
}


def context():
    return dict(
        source_sha256=SHA,
        actor_namespace=NS,
        review_branch="codex-draft",
        review_snapshot=None,
        duration_s=5,
        model_track_segments=[],
        scene_segments=[dict(segment_id=0, start_s=0, end_s=5)],
        review_metadata={"rubric_version": "pilot-v1", "source_group": "same-camera"},
    )


def frame(index):
    return dict(
        frame_index=index,
        source_pts=index * 100,
        time_base="1/1000",
        timestamp_seconds=index / 10,
        segment_id=0,
    )


def draft():
    return dict(
        source_sha256=SHA,
        actor_namespace=NS,
        human_actors=[dict(segment_id=0, track_id=HUMAN, start_s=0, end_s=5)],
        human_actor_references=[
            dict(
                segment_id=0,
                track_id=HUMAN,
                frame_index=10,
                source_pts=1000,
                time_base="1/1000",
                timestamp_seconds=1,
                bbox_xyxyn=[0.1, 0.2, 0.4, 0.8],
                note="Draft visual anchor, not confirmed identity",
            )
        ],
        events=[
            dict(
                segment_id=0,
                track_id=HUMAN,
                label="raise_hand",
                start_s=1,
                end_s=2,
                origin="human",
                review="reviewed",
                uncertain=False,
                evidence={
                    "note": "synthetic visual sequence",
                    "human_review": True,
                    "provenance": {"method": "visual_temporal", "human_review": True},
                },
            )
        ],
        review_metadata={"reviewer_id": "someone", "adjudication_state": "reviewed"},
    )


def prepare(value=None, *, ctx=None, resolver=frame, expected_snapshot=None):
    return importlib.import_module("classroom_ai.video.draft_annotations").prepare_assistant_draft(
        draft() if value is None else value,
        current_context=context() if ctx is None else ctx,
        frame_resolver=resolver,
        expected_snapshot=expected_snapshot,
    )


def validate(value):
    ctx = context()
    return validate_review(
        value,
        duration_s=5,
        model_segments=[],
        scene_segments=ctx["scene_segments"],
        source_sha256=SHA,
        actor_namespace=NS,
        require_actor_references=True,
    )


def test_assistant_origin_is_canonical_and_survives_a_later_real_human_review():
    value = prepare()
    value["events"][0]["review"] = "reviewed"
    value["coverage"] = [
        dict(
            segment_id=0, track_id=HUMAN, label="raise_hand", start_s=0, end_s=5, review="reviewed"
        )
    ]
    value["review_metadata"].update(
        reviewer_id="actual-reviewer",
        adjudication_state="reviewed",
        rubric_version="pilot-v1",
        source_group="camera",
    )
    result = validate(value)
    assert result["evaluation_ready"]
    assert result["events"][0]["origin"] == "assistant"
    assert "human_review" not in result["events"][0]["evidence"]
    summary = aggregate_events(result["events"], duration_s=5, coverage=result["coverage"])
    assert summary["session"]["raise_hand"]["count"] == 1


@pytest.mark.parametrize("reviewed", ["event", "coverage", "adjudication"])
def test_reserved_assistant_branch_rejects_claims_of_completed_human_review(reviewed):
    value = draft()
    value["events"][0].update(origin="rule", review="draft")
    value["coverage"] = [
        dict(segment_id=0, track_id=HUMAN, label="raise_hand", start_s=0, end_s=5, review="draft")
    ]
    value["review_metadata"].update(reviewer_id="codex-draft", adjudication_state="draft")
    if reviewed == "event":
        value["events"][0]["review"] = "reviewed"
        value["coverage"][0]["review"] = "reviewed"
    elif reviewed == "coverage":
        value["coverage"][0]["review"] = "reviewed"
    else:
        value["review_metadata"]["adjudication_state"] = "adjudicated"
    with pytest.raises(ValueError, match="codex-draft"):
        validate(value)


def test_prepare_forces_only_writable_draft_payload_and_does_not_mutate_input():
    value, ctx = draft(), context()
    value["actor_links"] = [{"review": "reviewed", "model_track_id": 7}]
    before, before_context = copy.deepcopy(value), copy.deepcopy(ctx)
    result = prepare(value, ctx=ctx)
    assert set(result) == WRITABLE
    assert value == before and ctx == before_context
    event = result["events"][0]
    assert event["origin"] == "assistant" and event["review"] == "draft"
    assert event["uncertain"] is False  # Unreviewed does not imply uncertain content.
    assert "human_review" not in event["evidence"]
    assert "human_review" not in event["evidence"]["provenance"]
    assert event["evidence"]["provenance"] == dict(
        method="visual_temporal",
        producer="assistant",
        import_review_state="draft",
        source_sha256=SHA,
        actor_namespace=NS,
    )
    assert result["actor_links"] == [] and result["expected_snapshot"] is None
    assert result["coverage"] == [
        dict(segment_id=0, track_id=HUMAN, label="raise_hand", start_s=1, end_s=2, review="draft")
    ]
    assert result["review_metadata"] == dict(
        reviewer_id="codex-draft",
        adjudication_state="draft",
        rubric_version="pilot-v1",
        source_group="same-camera",
    )
    result["human_actor_references"][0]["bbox_xyxyn"][0] = 0.3
    assert value == before


def test_prepared_payload_is_valid_but_cannot_produce_gt_or_reviewed_class_counts():
    result = validate(prepare())
    assert not result["evaluation_ready"]
    assert "draft_annotations" in result["readiness_reasons"]
    assert "human_review_not_confirmed" in result["readiness_reasons"]
    summary = aggregate_events(result["events"], duration_s=5, coverage=result["coverage"])
    assert summary["session"]["raise_hand"]["count"] is None
    assert summary["session"]["raise_hand"]["duration_s"] is None


@pytest.mark.parametrize(
    "field,value", [("source_sha256", "f" * 64), ("actor_namespace", "run:old")]
)
def test_foreign_source_or_namespace_is_rejected(field, value):
    value_draft = draft()
    value_draft[field] = value
    with pytest.raises(ValueError, match="source|namespace"):
        prepare(value_draft)


def test_branch_and_stale_snapshot_are_rejected_before_frame_resolution():
    ctx = context()
    ctx["review_snapshot"] = "review-" + "c" * 32 + ".json"
    with pytest.raises(ValueError, match="snapshot"):
        prepare(ctx=ctx, resolver=lambda _index: pytest.fail("stale import resolved frames"))
    assert (
        prepare(ctx=ctx, expected_snapshot=ctx["review_snapshot"])["expected_snapshot"]
        == ctx["review_snapshot"]
    )
    ctx["review_branch"] = "group-a"
    with pytest.raises(ValueError, match="codex-draft"):
        prepare(ctx=ctx, expected_snapshot=ctx["review_snapshot"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_index", 11),
        ("source_pts", 999),
        ("time_base", "1/600"),
        ("timestamp_seconds", 1.01),
        ("segment_id", 2),
    ],
)
def test_native_reference_identity_must_match_target_ledger(field, value):
    def resolve(index):
        return frame(index) | {field: value}

    with pytest.raises(ValueError, match="reference|frame|PTS|time|scene"):
        prepare(resolver=resolve)


def test_equivalent_time_base_is_normalized_to_native_ledger_identity():
    value = draft()
    value["human_actor_references"][0]["time_base"] = "2/2000"
    assert prepare(value)["human_actor_references"][0]["time_base"] == "1/1000"


@pytest.mark.parametrize("mutation", ["model_actor", "model_event", "missing_reference", "bad_box"])
def test_import_never_fabricates_human_mapping_or_missing_anchors(mutation):
    value = draft()
    if mutation == "model_actor":
        value["human_actors"][0]["track_id"] = 7
    elif mutation == "model_event":
        value["events"][0]["track_id"] = 7
    elif mutation == "missing_reference":
        value["human_actor_references"] = []
    else:
        value["human_actor_references"][0]["bbox_xyxyn"] = [0.5, 0.2, 0.1, 0.8]
    with pytest.raises(ValueError):
        prepare(value)


def test_explicit_working_coverage_is_demoted_but_must_contain_events():
    value = draft()
    value["coverage"] = [
        dict(
            segment_id=0, track_id=HUMAN, label="raise_hand", start_s=0, end_s=5, review="reviewed"
        )
    ]
    assert prepare(value)["coverage"][0]["review"] == "draft"
    value["coverage"] = []
    with pytest.raises(ValueError, match="coverage"):
        prepare(value)


@pytest.mark.parametrize("mutation", ["nan", "extra_event", "extra_metadata", "foreign_provenance"])
def test_malformed_or_conflicting_provenance_cannot_be_silently_imported(mutation):
    value = draft()
    if mutation == "nan":
        value["events"][0]["start_s"] = float("nan")
    elif mutation == "extra_event":
        value["events"][0]["score"] = 0.9
    elif mutation == "extra_metadata":
        value["review_metadata"]["claimed_human_review"] = "true"
    else:
        value["events"][0]["evidence"]["provenance"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        prepare(value)


def test_assistant_origin_stays_explicit_in_lower_level_event_contract():
    value = draft()["events"][0] | {"origin": "assistant", "review": "draft"}
    result = validate_events(
        [value],
        duration_s=5,
        track_segments=draft()["human_actors"],
        coverage=[
            dict(
                segment_id=0, track_id=HUMAN, label="raise_hand", start_s=1, end_s=2, review="draft"
            )
        ],
    )
    assert result[0]["origin"] == "assistant"


@pytest.mark.parametrize(
    "mutation", ["missing_label", "foreign_namespace", "foreign_model_hint", "boolean_scene"]
)
def test_incomplete_event_and_conflicting_identity_details_are_rejected(mutation):
    value = draft()
    resolver = frame
    if mutation == "missing_label":
        del value["events"][0]["label"]
    elif mutation == "foreign_namespace":
        value["events"][0]["evidence"]["provenance"]["actor_namespace"] = "run:another"
    elif mutation == "foreign_model_hint":
        value["events"][0]["evidence"]["provenance"]["model_hint"] = {"source_sha256": "f" * 64}
    else:

        def resolver(index):
            return frame(index) | {"segment_id": False}

    with pytest.raises(ValueError):
        prepare(value, resolver=resolver)


def test_resolver_is_cached_per_frame_without_mutating_shared_ledger_rows():
    value = draft()
    other = "human:" + "c" * 32
    value["human_actors"].append(value["human_actors"][0] | {"track_id": other})
    value["human_actor_references"].append(value["human_actor_references"][0] | {"track_id": other})
    rows, calls = {10: frame(10)}, []
    before = copy.deepcopy(rows)

    def resolve(index):
        calls.append(index)
        return rows[index]

    assert len(prepare(value, resolver=resolve)["human_actor_references"]) == 2
    assert calls == [10] and rows == before


@pytest.mark.parametrize("large", ["bytes", "events"])
def test_import_limits_bound_payload_and_interval_validation_work(large):
    value = draft()
    if large == "bytes":
        value["events"][0]["evidence"]["note"] = "x" * (2 * 1024 * 1024)
    else:
        value["events"] *= 2001
    with pytest.raises(ValueError, match="limit"):
        prepare(value)
