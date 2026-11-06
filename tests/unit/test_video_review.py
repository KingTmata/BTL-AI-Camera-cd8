"""Independent human actors and review provenance, without model or Label Studio writes."""

import copy
import importlib
import json

import pytest

HUMAN = "human:" + "a" * 32


def payload():
    return {
        "human_actors": [{"segment_id": 0, "track_id": HUMAN, "start_s": 0, "end_s": 5}],
        "actor_links": [],
        "events": [],
        "coverage": [
            {
                "segment_id": 0,
                "track_id": HUMAN,
                "label": "raise_hand",
                "start_s": 0,
                "end_s": 5,
                "review": "reviewed",
            }
        ],
        "review_metadata": {
            "reviewer_id": "reviewer-1",
            "rubric_version": "pilot-v1",
            "source_group": "source-1",
            "split": "train",
            "adjudication_state": "reviewed",
        },
    }


def validate(value, **kwargs):
    return importlib.import_module("classroom_ai.video.review").validate_review(
        value,
        duration_s=5,
        source_sha256="a" * 64,
        actor_namespace="run:fixture",
        model_segments=kwargs.get("model_segments", []),
        scene_segments=kwargs.get("scene_segments", [{"segment_id": 0, "start_s": 0, "end_s": 5}]),
        require_actor_references=kwargs.get("require_actor_references", False),
    )


def actor_reference(**changes):
    return (
        dict(
            segment_id=0,
            track_id=HUMAN,
            frame_index=30,
            source_pts=600,
            time_base="1/600",
            timestamp_seconds=1,
            bbox_xyxyn=[0.1, 0.2, 0.4, 0.8],
            note="Hàng bàn trước, cạnh lối đi",
        )
        | changes
    )


def test_actor_references_optional_legacy_and_required_readiness_only():
    value = payload()
    legacy = validate(value)
    assert legacy["evaluation_ready"] and legacy["human_actor_references"] == []
    assert legacy["actor_reference_count"] == 0 and not legacy["actor_references_required"]
    required = validate(value, require_actor_references=True)
    assert required["readiness_reasons"] == ["missing_actor_reference"]
    assert required["actor_reference_missing_span_count"] == 1
    assert required["actor_references_required"] and not required["evaluation_ready"]
    value["review_metadata"]["adjudication_state"] = "draft"
    draft = validate(value, require_actor_references=True)
    assert "missing_actor_reference" in draft["readiness_reasons"]
    assert draft["review_metadata"]["adjudication_state"] == "draft"


def test_actor_reference_roundtrip_is_detached_and_does_not_change_human_span_schema():
    value = payload()
    value["human_actor_references"] = [actor_reference()]
    original = copy.deepcopy(value)
    result = validate(value, require_actor_references=True)
    assert result["human_actor_references"] == value["human_actor_references"]
    assert (
        result["actor_reference_count"] == 1 and result["actor_reference_missing_span_count"] == 0
    )
    assert result["evaluation_ready"] and value == original
    assert set(result["human_actors"][0]) == {"segment_id", "track_id", "start_s", "end_s"}
    assert (
        validate(result, require_actor_references=True)["human_actor_references"]
        == original["human_actor_references"]
    )
    result["human_actor_references"][0]["bbox_xyxyn"][0] = 0.3
    assert value == original


def test_actor_reference_gate_requires_one_anchor_in_each_half_open_span():
    value = payload()
    value["human_actors"] = [
        dict(value["human_actors"][0], end_s=2),
        dict(value["human_actors"][0], start_s=2),
    ]
    value["human_actor_references"] = [actor_reference()]
    result = validate(value, require_actor_references=True)
    assert not result["evaluation_ready"] and result["actor_reference_missing_span_count"] == 1
    edge = actor_reference()
    edge.update(frame_index=60, source_pts=1200, timestamp_seconds=2)
    value["human_actor_references"].append(edge)
    result = validate(value, require_actor_references=True)
    assert result["evaluation_ready"] and result["actor_reference_count"] == 2
    value["human_actor_references"] = [edge]
    assert validate(value, require_actor_references=True)["actor_reference_missing_span_count"] == 1


@pytest.mark.parametrize(
    "field,item",
    [
        ("track_id", 1),
        ("track_id", "human:" + "b" * 32),
        ("segment_id", 1),
        ("segment_id", False),
        ("frame_index", True),
        ("frame_index", -1),
        ("frame_index", 1.5),
        ("source_pts", True),
        ("source_pts", -1),
        ("source_pts", 2.5),
        ("time_base", "0/600"),
        ("time_base", "1/0"),
        ("time_base", "-1/600"),
        ("time_base", "bad"),
        ("time_base", 1),
        ("timestamp_seconds", True),
        ("timestamp_seconds", -1),
        ("timestamp_seconds", 5),
        ("timestamp_seconds", float("nan")),
        ("bbox_xyxyn", [0, 0, 1]),
        ("bbox_xyxyn", [False, 0, 1, 1]),
        ("bbox_xyxyn", [0, 0, 1.1, 1]),
        ("bbox_xyxyn", [0.5, 0, 0.5, 1]),
        ("bbox_xyxyn", [0, 0.8, 1, 0.2]),
        ("bbox_xyxyn", [0, 0, float("inf"), 1]),
        ("note", None),
        ("note", "x" * 241),
    ],
)
def test_invalid_actor_reference_is_rejected_even_when_draft(field, item):
    value = payload()
    ref = actor_reference()
    ref[field] = item
    value["human_actor_references"] = [ref]
    value["review_metadata"]["adjudication_state"] = "draft"
    with pytest.raises(ValueError):
        validate(value)


def test_actor_reference_exact_keys_duplicate_anchor_and_orphan_after_span_edit():
    value = payload()
    ref = actor_reference()
    value["human_actor_references"] = [ref, copy.deepcopy(ref)]
    with pytest.raises(ValueError, match="duplicate"):
        validate(value)
    value["human_actor_references"] = [dict(ref, extra=True)]
    with pytest.raises(ValueError, match="reference"):
        validate(value)
    value["human_actor_references"] = [ref]
    value["human_actors"][0]["start_s"] = 2
    value["coverage"][0]["start_s"] = 2
    with pytest.raises(ValueError, match="span|reference"):
        validate(value)


def test_multiple_actors_can_have_references_on_same_frame_without_merging_identity():
    value = payload()
    other = "human:" + "b" * 32
    value["human_actors"].append(dict(value["human_actors"][0], track_id=other))
    value["human_actor_references"] = [actor_reference(), dict(actor_reference(), track_id=other)]
    result = validate(value, require_actor_references=True)
    assert result["evaluation_ready"] and result["actor_reference_count"] == 2
    # Native PTS may start away from zero. Only the API source-ledger resolver
    # can establish the relation between native PTS and relative timestamps.
    value["human_actor_references"][0]["source_pts"] = 90600
    assert validate(value, require_actor_references=True)["evaluation_ready"]


def test_actor_reference_list_and_count_are_bounded_before_entry_validation():
    module = importlib.import_module("classroom_ai.video.review")
    value = payload()
    value["human_actor_references"] = [{}] * (module.MAX_ACTOR_REFERENCES + 1)
    with pytest.raises(ValueError, match="limit"):
        validate(value)
    value["human_actor_references"] = None
    with pytest.raises(ValueError, match="list|limit"):
        validate(value)


def test_actor_reference_review_branches_roundtrip_without_aliasing_or_auto_review():
    a, b = payload(), payload()
    a["human_actor_references"] = [actor_reference(note="Bên trái lối đi")]
    b["human_actor_references"] = [
        actor_reference(note="Bên phải lối đi", bbox_xyxyn=[0.6, 0.1, 0.9, 0.9])
    ]
    b["review_metadata"].update(reviewer_id="reviewer-2", adjudication_state="draft")
    saved_a = json.loads(json.dumps(validate(a, require_actor_references=True)))
    saved_b = json.loads(json.dumps(validate(b, require_actor_references=True)))
    restored_a, restored_b = (
        validate(saved_a, require_actor_references=True),
        validate(saved_b, require_actor_references=True),
    )
    assert restored_a["review_metadata"]["reviewer_id"] == "reviewer-1"
    assert restored_b["review_metadata"]["reviewer_id"] == "reviewer-2"
    assert restored_a["human_actor_references"] == a["human_actor_references"]
    assert restored_b["human_actor_references"] == b["human_actor_references"]
    assert restored_a["evaluation_ready"] and not restored_b["evaluation_ready"]
    restored_b["human_actor_references"][0]["note"] = "Đã sửa bản nháp"
    assert restored_a["human_actor_references"][0]["note"] == "Bên trái lối đi"


def test_actor_reference_readiness_counts_are_recomputed_and_notes_can_be_empty():
    value = payload()
    value.update(
        actor_reference_count=999, actor_reference_missing_span_count=0, evaluation_ready=True
    )
    result = validate(value, require_actor_references=True)
    assert result["actor_reference_count"] == 0
    assert not result["evaluation_ready"] and result["actor_reference_missing_span_count"] == 1
    value["human_actor_references"] = [actor_reference(note="")]
    assert validate(value, require_actor_references=True)["evaluation_ready"]
    with pytest.raises(ValueError, match="boolean"):
        validate(value, require_actor_references="yes")


def test_missed_person_is_reviewable_without_any_model_actor():
    value = payload()
    original = copy.deepcopy(value)
    result = validate(value)
    assert result["evaluation_ready"] is True
    assert result["human_actors"] == value["human_actors"]
    assert result["actor_namespace"] == "run:fixture"
    assert value == original


def test_legacy_model_annotations_remain_readable_but_are_not_gt():
    model = [{"segment_id": 0, "track_id": 1, "start_s": 0, "end_s": 5}]
    result = validate({"events": [], "coverage": []}, model_segments=model)
    assert result["track_segments"] == model
    assert result["review_metadata"] == {}
    assert result["evaluation_ready"] is False
    assert result["readiness_reasons"]


@pytest.mark.parametrize(
    "field", ["reviewer_id", "rubric_version", "source_group", "adjudication_state"]
)
def test_metadata_is_not_invented_and_missing_fields_block_gt(field):
    value = payload()
    del value["review_metadata"][field]
    result = validate(value)
    assert field not in result["review_metadata"]
    assert not result["evaluation_ready"]


@pytest.mark.parametrize(
    "field,value", [("source_sha256", "b" * 64), ("actor_namespace", "run:other")]
)
def test_foreign_source_or_run_namespace_cannot_enter_review(field, value):
    row = payload()
    row[field] = value
    with pytest.raises(ValueError, match="source|namespace"):
        validate(row)


def test_reviewed_links_can_join_two_model_fragments_only_in_their_observed_intervals():
    model = [
        {"segment_id": 0, "track_id": i, "start_s": s, "end_s": e}
        for i, s, e in [(1, 0, 2), (2, 3, 5)]
    ]
    value = payload()
    value["actor_links"] = [
        {
            "segment_id": 0,
            "model_track_id": i,
            "human_track_id": HUMAN,
            "start_s": s,
            "end_s": e,
            "review": "reviewed",
        }
        for i, s, e in [(1, 0, 2), (2, 3, 5)]
    ]
    result = validate(value, model_segments=model)
    assert result["actor_links"] == value["actor_links"]
    assert result["human_actors"][0]["start_s"] == 0
    assert result["human_actors"][0]["end_s"] == 5
    bad = copy.deepcopy(value)
    bad["actor_links"][0]["end_s"] = 3
    with pytest.raises(ValueError, match="link"):
        validate(bad, model_segments=model)


@pytest.mark.parametrize(
    "change", ["unknown_model", "unknown_human", "cross_scene", "draft", "overlap"]
)
def test_invalid_or_conflicting_identity_links_are_rejected(change):
    model = [{"segment_id": 0, "track_id": 1, "start_s": 0, "end_s": 5}]
    value = payload()
    link = {
        "segment_id": 0,
        "model_track_id": 1,
        "human_track_id": HUMAN,
        "start_s": 0,
        "end_s": 5,
        "review": "reviewed",
    }
    value["actor_links"] = [link]
    if change == "unknown_model":
        link["model_track_id"] = 9
    elif change == "unknown_human":
        link["human_track_id"] = "human:" + "b" * 32
    elif change == "cross_scene":
        link["segment_id"] = 1
    elif change == "draft":
        link["review"] = "draft"
    else:
        value["actor_links"].append(copy.deepcopy(link))
    with pytest.raises(ValueError, match="link"):
        validate(value, model_segments=model)


def test_human_spans_must_stay_within_real_scene_and_not_overlap_same_actor():
    value = payload()
    value["human_actors"].append(copy.deepcopy(value["human_actors"][0]))
    with pytest.raises(ValueError, match="actor|span"):
        validate(value)
    value = payload()
    with pytest.raises(ValueError, match="scene|span"):
        validate(value, scene_segments=[{"segment_id": 0, "start_s": 0, "end_s": 2}])


def test_draft_or_model_actor_coverage_is_never_gt_ready():
    value = payload()
    value["coverage"][0]["review"] = "draft"
    assert not validate(value)["evaluation_ready"]
    value = payload()
    value["coverage"][0]["track_id"] = 1
    assert not validate(
        value, model_segments=[{"segment_id": 0, "track_id": 1, "start_s": 0, "end_s": 5}]
    )["evaluation_ready"]


def test_actor_and_link_counts_are_bounded_before_detailed_validation():
    value = payload()
    value["human_actors"] *= 1001
    with pytest.raises(ValueError, match="limit|1000"):
        validate(value)
    value = payload()
    value["actor_links"] = [{}] * 2001
    with pytest.raises(ValueError, match="limit|2000"):
        validate(value)


def test_explicit_duplicate_model_tracks_can_map_to_one_human_for_false_positive_evaluation():
    value = payload()
    model = [{"segment_id": 0, "track_id": tid, "start_s": 0, "end_s": 5} for tid in (1, 2)]
    value["actor_links"] = [
        {
            "segment_id": 0,
            "model_track_id": tid,
            "human_track_id": HUMAN,
            "start_s": 0,
            "end_s": 5,
            "review": "reviewed",
        }
        for tid in (1, 2)
    ]
    assert len(validate(value, model_segments=model)["actor_links"]) == 2


def test_boolean_scene_or_model_id_cannot_alias_numeric_actor_identity():
    value = payload()
    value["human_actors"][0]["segment_id"] = False
    value["coverage"] = []
    with pytest.raises(ValueError, match="identity|segment"):
        validate(value)
    value = payload()
    value["actor_links"] = [
        {
            "segment_id": 0,
            "model_track_id": True,
            "human_track_id": HUMAN,
            "start_s": 0,
            "end_s": 5,
            "review": "reviewed",
        }
    ]
    with pytest.raises(ValueError, match="identity|model"):
        validate(value, model_segments=[{"segment_id": 0, "track_id": 1, "start_s": 0, "end_s": 5}])


def pilot_audit(value, windows=None):
    review = validate(value)
    return importlib.import_module("classroom_ai.video.review").audit_pilot_review(
        review,
        windows or [{"window_id": "first", "start_s": 0, "end_s": 5}],
        duration_s=5,
    )


def label_audit(report, label="raise_hand", window=0):
    return next(row for row in report["windows"][window]["labels"] if row["label"] == label)


def test_pilot_partial_gt_is_valid_but_other_action_labels_remain_unreviewed():
    value = payload()
    assert validate(value)["evaluation_ready"] is True
    report = pilot_audit(value)
    assert report["scope"] == "declared_human_actors"
    assert not report["source_inventory_confirmed"]
    assert not report["declared_actor_scope_complete"]
    assert report["windows"][0]["status"] == "partial"
    assert label_audit(report)["coverage_fraction"] == 1
    other = label_audit(report, "write_board")
    assert other["coverage_fraction"] == 0
    assert other["unreviewed_actor_seconds"] == 5
    assert other["coverage_gaps"] == [value["human_actors"][0]]


def test_empty_pilot_has_unknown_denominator_and_never_becomes_complete():
    report = pilot_audit({})
    assert not report["declared_actor_scope_complete"]
    assert report["windows"][0]["status"] == "unstarted"
    assert report["windows"][0]["declared_actor_count"] == 0
    for row in report["windows"][0]["labels"]:
        assert row["declared_actor_seconds"] == row["reviewed_actor_seconds"] == 0
        assert row["coverage_fraction"] is None
        assert row["unreviewed_actor_seconds"] is None


def test_pilot_unions_coverage_and_clips_human_spans_to_each_window_without_mutation():
    value = payload()
    area = value["coverage"][0]
    value["coverage"] = [dict(area, end_s=3), dict(area, start_s=2, end_s=4)]
    before = copy.deepcopy(value)
    report = pilot_audit(
        value,
        [
            {"window_id": "a", "start_s": 0, "end_s": 2},
            {"window_id": "b", "start_s": 2, "end_s": 5},
        ],
    )
    assert label_audit(report, window=0)["reviewed_actor_seconds"] == 2
    later = label_audit(report, window=1)
    assert later["declared_actor_seconds"] == 3
    assert later["reviewed_actor_seconds"] == 2
    assert later["unreviewed_actor_seconds"] == 1
    assert later["coverage_gaps"] == [dict(value["human_actors"][0], start_s=4)]
    assert value == before


def test_pilot_masks_uncertain_action_but_unknown_posture_does_not_erase_action():
    value = payload()
    base = {
        **value["human_actors"][0],
        "start_s": 2,
        "end_s": 3,
        "origin": "human",
        "review": "reviewed",
        "evidence": {},
    }
    value["events"] = [dict(base, label="raise_hand", uncertain=True)]
    report = pilot_audit(value)
    row = label_audit(report)
    assert row["reviewed_actor_seconds"] == 4
    assert row["uncertain_intervals"] == 1
    assert row["reviewed_positive_intervals"] == 0
    assert row["coverage_gaps"] == [{**value["human_actors"][0], "start_s": 2, "end_s": 3}]
    value["events"] = [dict(base, label="unknown", uncertain=False)]
    value["coverage"].append(dict(value["human_actors"][0], label="unknown", review="reviewed"))
    assert label_audit(pilot_audit(value))["reviewed_actor_seconds"] == 5


def test_pilot_denominator_is_declared_actor_seconds_and_requires_each_action():
    value = payload()
    second = dict(value["human_actors"][0], track_id="human:" + "b" * 32)
    value["human_actors"].append(second)
    row = label_audit(pilot_audit(value))
    assert row["declared_actor_seconds"] == 10 and row["reviewed_actor_seconds"] == 5
    assert row["coverage_fraction"] == 0.5
    assert row["coverage_gaps"] == [second]
    value["coverage"] = [
        dict(span, label=label, review="reviewed")
        for span in value["human_actors"]
        for label in ("raise_hand", "stand_up", "move_to_board", "write_board")
    ]
    report = pilot_audit(value)
    assert report["declared_actor_scope_complete"]
    assert report["windows"][0]["status"] == "reviewed_declared_scope"
    assert report["source_inventory_confirmed"] is False
    assert "pilot_complete" not in report


@pytest.mark.parametrize(
    "windows",
    [
        [{"window_id": "x", "start_s": False, "end_s": 5}],
        [{"window_id": "x", "start_s": 0, "end_s": float("nan")}],
        [{"window_id": "x", "start_s": 0, "end_s": 6}],
        [
            {"window_id": "x", "start_s": 0, "end_s": 2},
            {"window_id": "x", "start_s": 2, "end_s": 5},
        ],
        [{"window_id": str(i), "start_s": 0, "end_s": 1} for i in range(101)],
    ],
)
def test_pilot_rejects_invalid_or_unbounded_window_definitions(windows):
    with pytest.raises(ValueError, match="window|interval|limit"):
        pilot_audit(payload(), windows)


def test_pilot_reports_draft_coverage_and_bounds_gaps_without_losing_missing_totals(monkeypatch):
    module = importlib.import_module("classroom_ai.video.review")
    monkeypatch.setattr(module, "MAX_PILOT_GAPS", 1)
    value = payload()
    value["coverage"][0]["review"] = "draft"
    report = pilot_audit(value)
    row = label_audit(report)
    assert row["draft_intervals"] == 1
    assert row["reviewed_actor_seconds"] == 0
    assert row["unreviewed_actor_seconds"] == 5
    assert sum(len(row["coverage_gaps"]) for row in report["windows"][0]["labels"]) == 1
    last = label_audit(report, "write_board")
    assert last["gap_count"] == 1 and last["gaps_truncated"]
    assert last["unreviewed_actor_seconds"] == 5
