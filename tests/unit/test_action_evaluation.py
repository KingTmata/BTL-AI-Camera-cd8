"""Synthetic arithmetic and provenance fixtures, never classroom GT."""

import copy
import importlib
import json

import pytest

HUMAN = "human:" + "a" * 32
SHA = "b" * 64


def fixture():
    actor = dict(segment_id=0, track_id=HUMAN, start_s=0, end_s=10)
    event = dict(
        segment_id=0,
        track_id=HUMAN,
        label="raise_hand",
        start_s=1,
        end_s=3,
        origin="human",
        review="reviewed",
        uncertain=False,
        evidence={"boundary_truncated": {"start": False, "end": False}},
    )
    reference = dict(
        schema_version=2,
        source_sha256=SHA,
        actor_namespace="run:fixture",
        duration_s=10,
        human_actors=[actor],
        scene_segments=[dict(segment_id=0, start_s=0, end_s=10)],
        model_track_segments=[],
        actor_links=[],
        events=[event],
        coverage=[dict(actor, label="raise_hand", review="reviewed")],
        review_metadata=dict(
            reviewer_id="reviewer-A",
            rubric_version="pilot-v1",
            source_group="source-A",
            split="train",
            adjudication_state="reviewed",
        ),
    )
    prediction = dict(schema_version=2, source_sha256=SHA, actor_namespace="run:fixture", events=[])
    protocol = dict(
        schema_version=1,
        protocol_id="synthetic-pilot",
        phase="pilot",
        status="draft",
        rubric_version="pilot-v1",
        iou_threshold=0.5,
        boundary_tolerance_s=None,
        development_source_groups=["source-A"],
        test_source_groups=[],
    )
    return prediction, reference, protocol


def api():
    return importlib.import_module("classroom_ai.video.action_evaluation")


def anchored(bundle):
    bundle["human_actor_references"] = [
        dict(
            segment_id=0,
            track_id=HUMAN,
            frame_index=30,
            source_pts=600,
            time_base="1/600",
            timestamp_seconds=1,
            bbox_xyxyn=[0.1, 0.1, 0.4, 0.8],
            note="Cạnh lối đi",
        )
    ]


@pytest.mark.parametrize(
    "provenance",
    [
        {"source_kind": "annotation_source_preparation"},
        {"actor_references_required": True},
        {"source_kind": "annotation_source_preparation", "actor_references_required": False},
    ],
)
def test_evaluator_preparation_reference_gate_cannot_be_bypassed(provenance):
    pred, ref, protocol = fixture()
    ref.update(provenance)
    ref["evaluation_ready"] = True
    with pytest.raises(ValueError, match="missing_actor_reference"):
        api().evaluate_bundle(pred, ref, protocol=protocol)
    anchored(ref)
    before = copy.deepcopy(ref)
    report = api().evaluate_bundle(pred, ref, protocol=protocol)
    assert report["micro"]["fn"] == 1
    assert ref == before and ref["human_actor_references"]


@pytest.mark.parametrize("flag", ["false", 0, 1, None])
def test_evaluator_requires_boolean_reference_gate_flag(flag):
    pred, ref, protocol = fixture()
    ref["actor_references_required"] = flag
    with pytest.raises(ValueError, match="boolean"):
        api().evaluate_bundle(pred, ref, protocol=protocol)


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize(
    "provenance",
    [
        {"source_kind": "annotation_source_preparation"},
        {"actor_references_required": True},
    ],
)
def test_comparison_revalidates_preparation_anchors_for_either_reviewer(side, provenance):
    a, b, mapping, protocol = comparison_fixture()
    bundle = a if side == "a" else b
    bundle.update(provenance)
    bind_comparison(a, b, mapping)
    with pytest.raises(ValueError, match="missing_actor_reference"):
        api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    anchored(bundle)
    bind_comparison(a, b, mapping)
    original = copy.deepcopy((a, b, mapping))
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["per_label"]["raise_hand"]["matched"] == 1
    assert report["snapshots"][side]["snapshot_sha256"] == api().review_snapshot_sha256(bundle)
    assert (a, b, mapping) == original


@pytest.mark.parametrize("side", ["a", "b"])
def test_comparison_rejects_string_reference_gate_flag_on_either_side(side):
    a, b, mapping, protocol = comparison_fixture()
    (a if side == "a" else b)["actor_references_required"] = "false"
    bind_comparison(a, b, mapping)
    with pytest.raises(ValueError, match="boolean"):
        api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)


def comparison_fixture():
    _, a, protocol = fixture()
    b = copy.deepcopy(a)
    b["review_metadata"]["reviewer_id"] = "reviewer-B"
    pair_map = dict(
        schema_version=1,
        kind="human_actor_pairs",
        source_sha256=SHA,
        mapping_reviewer_id="mapper",
        review="reviewed",
        pairs=[dict(segment_id=0, a_track_id=HUMAN, b_track_id=HUMAN, start_s=0, end_s=10)],
    )
    for side, bundle in (("a", a), ("b", b)):
        pair_map[side] = dict(
            snapshot_sha256=api().review_snapshot_sha256(bundle),
            reviewer_id=bundle["review_metadata"]["reviewer_id"],
            actor_namespace=bundle["actor_namespace"],
        )
    return a, b, pair_map, protocol


def bind_comparison(a, b, mapping):
    for side, bundle in (("a", a), ("b", b)):
        mapping[side]["snapshot_sha256"] = api().review_snapshot_sha256(bundle)
        mapping[side]["reviewer_id"] = bundle["review_metadata"]["reviewer_id"]
        mapping[side]["actor_namespace"] = bundle["actor_namespace"]


def test_review_comparison_same_uuid_requires_explicit_snapshot_bound_pair():
    a, b, mapping, protocol = comparison_fixture()
    original = copy.deepcopy((a, b, mapping, protocol))
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    row = report["per_label"]["raise_hand"]
    assert (row["matched"], row["a_only"], row["b_only"], row["agreement_f1"]) == (1, 0, 0, 1)
    assert row["shared_coverage_actor_s"] == 10
    assert report["kind"] == "inter_review_agreement"
    assert report["status"] == "descriptive_agreement"
    assert "accuracy" not in report and "evaluation_ready" not in report
    assert (a, b, mapping, protocol) == original
    mapping["pairs"] = []
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["per_label"]["raise_hand"]["agreement_f1"] is None
    assert report["audit"]["a"]["unmapped_actor_count"] == 1


@pytest.mark.parametrize(
    "change", ["snapshot", "review", "reviewer", "source", "rubric", "split", "group", "tolerance"]
)
def test_comparison_rejects_stale_or_unreviewed_provenance(change):
    a, b, mapping, protocol = comparison_fixture()
    if change == "snapshot":
        b["events"][0]["end_s"] = 4
    elif change == "review":
        mapping["review"] = "draft"
    elif change == "reviewer":
        b["review_metadata"]["reviewer_id"] = a["review_metadata"]["reviewer_id"]
    elif change == "source":
        b["source_sha256"] = "c" * 64
    elif change == "rubric":
        b["review_metadata"]["rubric_version"] = "different"
    elif change == "split":
        b["review_metadata"]["split"] = "val"
    elif change == "group":
        b["review_metadata"]["source_group"] = "other"
    else:
        protocol["boundary_tolerance_s"] = 1
    with pytest.raises(ValueError):
        api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)


def test_comparison_joint_coverage_excludes_whole_episode_not_clips():
    a, b, mapping, protocol = comparison_fixture()
    a["events"][0].update(start_s=1, end_s=5)
    b["events"] = []
    b["coverage"][0].update(start_s=2, end_s=8)
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["per_label"]["raise_hand"]["shared_coverage_actor_s"] == 6
    assert report["per_label"]["raise_hand"]["agreement_f1"] is None
    assert report["audit"]["a"]["excluded_events"][0]["reason"] == "outside_shared_coverage"
    assert report["audit"]["a"]["unshared_coverage_actor_s"]["raise_hand"] == 4


def test_comparison_both_masks_symmetric_but_unknown_posture_keeps_action():
    a, b, mapping, protocol = comparison_fixture()
    uncertain = dict(a["events"][0], uncertain=True)
    a["events"] = [uncertain]
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["per_label"]["raise_hand"]["shared_coverage_actor_s"] == 8
    assert report["per_label"]["raise_hand"]["agreement_f1"] is None
    assert report["audit"]["b"]["excluded_events"][0]["reason"] == "masked_by_either_reviewer"
    a["events"] = [
        dict(a["events"][0], uncertain=False),
        dict(uncertain, label="unknown", uncertain=False),
    ]
    a["coverage"].append(dict(a["coverage"][0], label="unknown"))
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["per_label"]["raise_hand"]["agreement_f1"] == 1


def test_comparison_merge_pair_boundaries_and_symmetric_censoring():
    a, b, mapping, protocol = comparison_fixture()
    mapping["pairs"] = [dict(mapping["pairs"][0], end_s=2), dict(mapping["pairs"][0], start_s=2)]
    b["events"][0].update(start_s=1.2, end_s=3.2)
    b["events"][0]["evidence"]["boundary_truncated"]["end"] = True
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    row = report["per_label"]["raise_hand"]
    assert row["matched"] == 1 and row["start_difference_s"]["bias"] == pytest.approx(0.2)
    assert row["end_difference_s"]["count"] == 0
    assert report["matches"][0]["end_difference_s"] is None
    reverse = copy.deepcopy(mapping)
    reverse["a"], reverse["b"] = mapping["b"], mapping["a"]
    reverse["pairs"] = [
        dict(p, a_track_id=p["b_track_id"], b_track_id=p["a_track_id"]) for p in mapping["pairs"]
    ]
    other = api().compare_review_bundles(b, a, actor_map=reverse, protocol=protocol)
    assert other["per_label"]["raise_hand"]["agreement_f1"] == row["agreement_f1"]
    assert other["per_label"]["raise_hand"]["start_difference_s"]["bias"] == pytest.approx(-0.2)


def test_comparison_rejects_overlap_and_preserves_missing_actor_coverage():
    a, b, mapping, protocol = comparison_fixture()
    other = "human:" + "d" * 32
    b["human_actors"].append(dict(b["human_actors"][0], track_id=other))
    b["coverage"].append(dict(b["coverage"][0], track_id=other))
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["audit"]["b"]["unmapped_actor_count"] == 1
    assert report["audit"]["b"]["unshared_coverage_actor_s"]["raise_hand"] == 10
    mapping["pairs"].append(dict(mapping["pairs"][0], b_track_id=other))
    with pytest.raises(ValueError, match="overlap"):
        api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)


def test_compare_reviews_cli_hashes_and_exclusive_output(tmp_path):
    a, b, mapping, protocol = comparison_fixture()
    data = {"review-a": a, "review-b": b, "actor-map": mapping, "protocol": protocol}
    args = ["compare-reviews"]
    for name, value in data.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value, indent=4), encoding="utf-8")
        args.extend(["--" + name, str(path)])
    output = tmp_path / "agreement.json"
    args.extend(["--output", str(output)])
    assert api().main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["input_sha256"]) == {"review_a", "review_b", "actor_map", "protocol"}
    assert report["input_sha256"]["review_a"] != report["snapshots"]["a"]["snapshot_sha256"]
    raw = output.read_bytes()
    assert api().main(args) == 2 and output.read_bytes() == raw


def test_comparison_rejects_boolean_scene_alias_in_pair():
    a, b, mapping, protocol = comparison_fixture()
    mapping["pairs"][0]["segment_id"] = False
    with pytest.raises(ValueError, match="identity|segment"):
        api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)


def test_comparison_actual_identity_change_excludes_crossing_episode():
    a, b, mapping, protocol = comparison_fixture()
    other = "human:" + "e" * 32
    b["human_actors"].append(dict(b["human_actors"][0], track_id=other))
    mapping["pairs"] = [
        dict(mapping["pairs"][0], end_s=2),
        dict(mapping["pairs"][0], start_s=2, b_track_id=other),
    ]
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert (
        report["audit"]["a"]["excluded_events"][0]["reason"] == "unmapped_or_crosses_pair_boundary"
    )
    assert report["per_label"]["raise_hand"]["agreement_f1"] is None


def test_comparison_unmatched_counts_and_all_negative_coverage_remain_descriptive():
    a, b, mapping, protocol = comparison_fixture()
    b["events"].append(dict(b["events"][0], start_s=5, end_s=6))
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    row = report["per_label"]["raise_hand"]
    assert (row["matched"], row["a_only"], row["b_only"]) == (1, 0, 1)
    assert row["agreement_f1"] == pytest.approx(2 / 3)
    a["events"], b["events"] = [], []
    bind_comparison(a, b, mapping)
    row = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)["per_label"][
        "raise_hand"
    ]
    assert row["shared_coverage_actor_s"] == 10
    assert row["agreement_f1"] is None and row["a_episodes"] == row["b_episodes"] == 0


def test_comparison_distinct_namespaces_and_ids_map_without_ai_links():
    a, b, mapping, protocol = comparison_fixture()
    other = "human:" + "f" * 32
    b["actor_namespace"] = "run:other"
    for name in ("human_actors", "coverage", "events"):
        for row in b[name]:
            row["track_id"] = other
    mapping["pairs"][0]["b_track_id"] = other
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["per_label"]["raise_hand"]["matched"] == 1
    b["review_metadata"]["adjudication_state"] = "draft"
    bind_comparison(a, b, mapping)
    with pytest.raises(ValueError, match="provenance"):
        api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)


@pytest.mark.parametrize("change", ["count", "actor", "span", "extra", "missing_identity"])
def test_comparison_pair_map_bounds_and_exact_known_identity(change):
    a, b, mapping, protocol = comparison_fixture()
    if change == "count":
        mapping["pairs"] *= 2001
    elif change == "actor":
        mapping["pairs"][0]["b_track_id"] = "human:" + "f" * 32
    elif change == "span":
        mapping["pairs"][0]["end_s"] = 11
    elif change == "extra":
        mapping["pairs"][0]["automatic"] = True
    else:
        del mapping["b"]["actor_namespace"]
    with pytest.raises(ValueError):
        api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)


def test_canonical_snapshot_hash_is_order_independent_finite_utf8():
    value = {"ghi_chú": "đã duyệt", "a": {"b": 1, "a": 2}}
    reverse = {"a": {"a": 2, "b": 1}, "ghi_chú": "đã duyệt"}
    assert api().review_snapshot_sha256(value) == api().review_snapshot_sha256(reverse)
    with pytest.raises(ValueError):
        api().review_snapshot_sha256({"not_finite": float("nan")})


def test_comparison_coverage_hole_and_missing_boundary_attestation_are_visible():
    a, b, mapping, protocol = comparison_fixture()
    b["events"] = []
    b["coverage"] = [dict(b["coverage"][0], end_s=2), dict(b["coverage"][0], start_s=2.5)]
    bind_comparison(a, b, mapping)
    report = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)
    assert report["per_label"]["raise_hand"]["shared_coverage_actor_s"] == 9.5
    assert report["audit"]["a"]["excluded_events"][0]["reason"] == "outside_shared_coverage"
    b["events"] = copy.deepcopy(a["events"])
    b["coverage"] = copy.deepcopy(a["coverage"])
    a["events"][0]["evidence"] = {}
    bind_comparison(a, b, mapping)
    row = api().compare_review_bundles(a, b, actor_map=mapping, protocol=protocol)["per_label"][
        "raise_hand"
    ]
    assert row["matched"] == 1
    assert row["start_difference_s"]["count"] == row["end_difference_s"]["count"] == 0


def test_missed_independent_human_is_fn_even_without_any_model_actor():
    pred, ref, protocol = fixture()
    report = api().evaluate_bundle(pred, ref, protocol=protocol)
    assert report["micro"]["fn"] == 1 and report["micro"]["tp"] == 0
    assert report["per_label"]["raise_hand"]["recall"] == 0
    assert report["per_label"]["stand_up"]["f1"] is None
    assert "stand_up" in report["missing_positive_labels"]
    assert report["status"] == "pilot_metrics"


def test_duplicate_prediction_is_fp_but_duration_uses_union_and_gt_is_unchanged():
    pred, ref, protocol = fixture()
    pred["events"] = [copy.deepcopy(ref["events"][0])] * 2
    before = copy.deepcopy((pred, ref, protocol))
    report = api().evaluate_bundle(pred, ref, protocol=protocol)
    row = report["per_label"]["raise_hand"]
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 0)
    assert row["count_error"] == 1 and row["duration_error_s"] == 0
    assert row["start_error_s"]["mae"] == 0
    assert (pred, ref, protocol) == before


@pytest.mark.parametrize(
    "field,value", [("source_sha256", "c" * 64), ("actor_namespace", "run:other")]
)
def test_source_or_unmapped_namespace_mismatch_is_rejected(field, value):
    pred, ref, protocol = fixture()
    pred[field] = value
    with pytest.raises(ValueError, match="source|namespace"):
        api().evaluate_bundle(pred, ref, protocol=protocol)


def test_review_ready_flag_cannot_replace_missing_human_provenance():
    pred, ref, protocol = fixture()
    ref["review_metadata"] = {}
    ref["evaluation_ready"] = True
    with pytest.raises(ValueError, match="review|GT"):
        api().evaluate_bundle(pred, ref, protocol=protocol)


def test_locked_test_rejects_development_source_and_unlocked_protocol():
    pred, ref, protocol = fixture()
    ref["review_metadata"]["split"] = "test"
    protocol.update(phase="test", status="locked", test_source_groups=["source-A"])
    with pytest.raises(ValueError, match="source|leak|overlap"):
        api().evaluate_bundle(pred, ref, protocol=protocol)
    protocol["development_source_groups"] = []
    protocol["status"] = "draft"
    with pytest.raises(ValueError, match="locked"):
        api().evaluate_bundle(pred, ref, protocol=protocol)


def test_pilot_cannot_use_declared_heldout_group_even_if_gt_split_text_says_train():
    pred, ref, protocol = fixture()
    protocol["development_source_groups"] = []
    protocol["test_source_groups"] = ["source-A"]
    with pytest.raises(ValueError, match="held.out|test source"):
        api().evaluate_bundle(pred, ref, protocol=protocol)


def test_explicit_mapping_reconciles_duplicate_ai_ids_without_silent_identity_merge():
    pred, ref, protocol = fixture()
    pred["actor_namespace"] = "run:new-model"
    pred["events"] = [dict(ref["events"][0], track_id=track) for track in (2, 3)]
    mapping = dict(
        source_sha256=SHA,
        from_namespace="run:new-model",
        to_namespace="run:fixture",
        reviewer_id="mapping-reviewer",
        links=[
            dict(
                segment_id=0,
                model_track_id=track,
                human_track_id=HUMAN,
                start_s=0,
                end_s=10,
                review="reviewed",
            )
            for track in (2, 3)
        ],
    )
    report = api().evaluate_bundle(pred, ref, protocol=protocol, actor_map=mapping)
    assert report["micro"]["tp"] == 1 and report["micro"]["fp"] == 1
    assert report["mapping"]["mapped_predictions"] == 2
    del mapping["links"][1]
    with pytest.raises(ValueError, match="mapping|unmapped"):
        api().evaluate_bundle(pred, ref, protocol=protocol, actor_map=mapping)


def test_cli_writes_exclusive_report_with_input_hashes(tmp_path):
    pred, ref, protocol = fixture()
    for name, data in (("prediction", pred), ("reference", ref), ("protocol", protocol)):
        (tmp_path / (name + ".json")).write_text(json.dumps(data), encoding="utf-8")
    args = [
        "--prediction",
        str(tmp_path / "prediction.json"),
        "--reference",
        str(tmp_path / "reference.json"),
        "--protocol",
        str(tmp_path / "protocol.json"),
        "--output",
        str(tmp_path / "report.json"),
    ]
    assert api().main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert set(report["input_sha256"]) == {"prediction", "reference", "protocol"}
    original = (tmp_path / "report.json").read_bytes()
    assert api().main(args) != 0
    assert (tmp_path / "report.json").read_bytes() == original


def test_corpus_rejects_duplicate_source_and_groups_spanning_different_splits():
    pred, ref, protocol = fixture()
    item = dict(prediction=pred, reference=ref)
    with pytest.raises(ValueError, match="duplicate source"):
        api().evaluate_corpus([item, item], protocol=protocol)
    other = copy.deepcopy(item)
    other["prediction"]["source_sha256"] = other["reference"]["source_sha256"] = "c" * 64
    other["reference"]["review_metadata"]["split"] = "val"
    with pytest.raises(ValueError, match="source group.*split"):
        api().evaluate_corpus([item, other], protocol=protocol)


def test_corpus_sums_event_metrics_without_inventing_missing_positive_classes():
    pred, ref, protocol = fixture()
    first = dict(prediction=pred, reference=ref)
    second = copy.deepcopy(first)
    second["prediction"]["source_sha256"] = second["reference"]["source_sha256"] = "c" * 64
    second["reference"]["review_metadata"]["source_group"] = "source-B"
    second["prediction"]["events"] = copy.deepcopy(second["reference"]["events"])
    report = api().evaluate_corpus([first, second], protocol=protocol)
    assert report["micro"]["tp"] == 1 and report["micro"]["fn"] == 1
    assert report["micro"]["f1"] == pytest.approx(2 / 3)
    assert report["source_count"] == report["independent_source_groups"] == 2
    assert "write_board" in report["missing_positive_labels"]
    with pytest.raises(ValueError, match="empty"):
        api().evaluate_corpus([], protocol=protocol)


def test_corpus_cli_resolves_relative_inputs_and_records_each_source_hash(tmp_path):
    pred, ref, protocol = fixture()
    for name, value in (("prediction", pred), ("reference", ref), ("protocol", protocol)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "corpus.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                items=[dict(prediction="prediction.json", reference="reference.json")],
            )
        ),
        encoding="utf-8",
    )
    assert (
        api().main(
            [
                "--corpus",
                str(tmp_path / "corpus.json"),
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["kind"] == "action_corpus_evaluation" and report["source_count"] == 1
    assert set(report["input_sha256"]) == {"corpus", "protocol"}
    assert set(report["source_input_sha256"][0]) == {"prediction", "reference"}


@pytest.mark.parametrize("external_mapping", [False, True])
def test_matches_reference_original_prediction_indices_after_posture_filter(external_mapping):
    pred, ref, protocol = fixture()
    positive = copy.deepcopy(ref["events"][0])
    pred["events"] = [dict(positive, label="standing", start_s=0, end_s=1), positive]
    mapping = None
    if external_mapping:
        pred["actor_namespace"] = "run:new-model"
        for row in pred["events"]:
            row["track_id"] = 2
        mapping = dict(
            source_sha256=SHA,
            from_namespace="run:new-model",
            to_namespace="run:fixture",
            reviewer_id="mapping-reviewer",
            links=[
                dict(
                    segment_id=0,
                    model_track_id=2,
                    human_track_id=HUMAN,
                    start_s=0,
                    end_s=10,
                    review="reviewed",
                )
            ],
        )
    report = api().evaluate_bundle(pred, ref, protocol=protocol, actor_map=mapping)
    pair = report["matches"][0]
    assert pair["prediction_index"] == 1
    assert pred["events"][pair["prediction_index"]]["label"] == "raise_hand"
    assert pair["reference_index"] == 0
    assert report["per_label"]["raise_hand"]["tp"] == 1
    assert report["per_label"]["raise_hand"]["prediction_episodes"] == 1
    if external_mapping:
        assert report["mapping"]["mapped_predictions"] == 2
