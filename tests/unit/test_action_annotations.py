"""Hand-built contract fixtures, not measured classroom ground truth."""

import importlib
import json

import pytest


def api():
    return importlib.import_module("classroom_ai.video.actions")


def event(label="raise_hand", start=1, end=3, **changes):
    return dict(
        segment_id="cut-1",
        track_id=7,
        label=label,
        start_s=start,
        end_s=end,
        origin="human",
        review="reviewed",
        uncertain=False,
        evidence={},
        **changes,
    )


def context():
    return dict(
        duration_s=10,
        track_segments=[dict(segment_id="cut-1", track_id=7, start_s=0, end_s=10)],
        coverage=[
            dict(
                segment_id="cut-1", track_id=7, label=label, start_s=0, end_s=10, review="reviewed"
            )
            for label in (
                "sitting",
                "standing",
                "transition",
                "unknown",
                "raise_hand",
                "stand_up",
                "move_to_board",
                "write_board",
            )
        ],
    )


@pytest.mark.parametrize("extra", ["note", "nonfinite"])
def test_coverage_rejects_extra_properties_in_validation_and_aggregation(extra):
    args = context()
    args["coverage"][0][extra] = float("inf") if extra == "nonfinite" else "unexpected"
    with pytest.raises(ValueError):
        api().validate_events([], **args)
    with pytest.raises(ValueError):
        api().aggregate_events([], duration_s=10, coverage=args["coverage"])


def test_standing_and_raise_hand_are_independent_and_input_is_copied():
    rows = [event("standing"), event()]
    result = api().validate_events(rows, **context())
    assert [e["label"] for e in result] == ["standing", "raise_hand"]
    result[0]["evidence"]["changed"] = True
    assert rows[0]["evidence"] == {}


def test_contradictory_posture_overlap_rejected_but_adjoining_allowed():
    with pytest.raises(ValueError, match="posture"):
        api().validate_events([event("standing"), event("sitting", 2, 4)], **context())
    assert len(api().validate_events([event("standing"), event("sitting", 3, 4)], **context())) == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"start_s": float("nan")},
        {"end_s": float("inf")},
        {"start_s": -1},
        {"end_s": 11},
        {"end_s": 1},
        {"label": "engaged"},
        {"origin": "model"},
        {"review": "approved"},
        {"uncertain": "false"},
        {"track_id": True},
        {"evidence": {"bad": float("nan")}},
        {"evidence": []},
    ],
)
def test_invalid_event_fields_rejected(updates):
    row = event()
    row.update(updates)
    with pytest.raises(ValueError):
        api().validate_events([row], **context())


def test_missing_actor_coverage_and_segment_cut_are_not_joined():
    args = context()
    args["track_segments"][0]["end_s"] = 2
    with pytest.raises(ValueError, match="segment"):
        api().validate_events([event()], **args)
    args = context()
    row = event()
    row["segment_id"] = "cut-2"
    with pytest.raises(ValueError, match="segment"):
        api().validate_events([row], **args)
    args = context()
    args["coverage"] = []
    assert api().validate_events([], **args) == []
    with pytest.raises(ValueError, match="coverage"):
        api().validate_events([event()], **args)


def test_coverage_gaps_and_draft_coverage_cannot_certify_reviewed_event():
    args = context()
    args["coverage"] = [
        dict(
            segment_id="cut-1",
            track_id=7,
            label="raise_hand",
            start_s=0,
            end_s=2,
            review="reviewed",
        ),
        dict(
            segment_id="cut-1",
            track_id=7,
            label="raise_hand",
            start_s=2.5,
            end_s=10,
            review="reviewed",
        ),
    ]
    with pytest.raises(ValueError, match="coverage"):
        api().validate_events([event()], **args)
    args["coverage"][1]["start_s"] = 2
    assert len(api().validate_events([event()], **args)) == 1
    for row in args["coverage"]:
        row["review"] = "draft"
    with pytest.raises(ValueError, match="coverage"):
        api().validate_events([event()], **args)


def test_save_is_exclusive_and_rejects_draft_without_touching_source(tmp_path):
    dest = tmp_path / "annotations.json"
    assert api().save_reviewed_events([event()], dest) == dest
    original = dest.read_bytes()
    assert json.loads(original)["events"][0]["label"] == "raise_hand"
    with pytest.raises(FileExistsError):
        api().save_reviewed_events([event("standing")], dest)
    assert dest.read_bytes() == original
    draft = event()
    draft["review"] = "draft"
    with pytest.raises(ValueError, match="reviewed"):
        api().save_reviewed_events([draft], tmp_path / "draft.json")
    assert not (tmp_path / "draft.json").exists()


def pose_frame(t, raised=True, segment="cut-1", confirmed=True, confidence=1):
    # COCO17: nose, shoulders, elbows, wrists. Both sides visible.
    points = [[50, 70, confidence] for _ in range(17)]
    points[0] = [50, 50, confidence]
    points[5], points[6] = [40, 100, confidence], [60, 100, confidence]
    points[7], points[8] = [40, 60, confidence], [60, 110, confidence]
    points[9], points[10] = [40, 20 if raised else 120, confidence], [60, 120, confidence]
    return dict(
        frame_index=int(t * 100),
        timestamp_seconds=t,
        segment_id=segment,
        tracks=[dict(track_id=7, state="observed", confirmed=confirmed, keypoints17=points)],
    )


def test_pose_observation_separates_supported_lowering_from_unknown_and_rule_status():
    state = api().RaiseHandState()
    state.update(pose_frame(0, False))
    person = state.people[0]
    assert person["status"] == "no_suggestion"
    assert person["pose_observation"] == "not_raised"
    assert person["pose_observation_scope"] == "uncalibrated_raw_pose_geometry"
    assert person["pose_quality"]["status"] == "supported"
    ambiguous = pose_frame(0.2)
    ambiguous["tracks"][0]["keypoints17"][9][1] = 90
    state.update(ambiguous)
    assert state.people[0]["status"] == "no_suggestion"
    assert state.people[0]["pose_observation"] == "unknown"
    low = pose_frame(0.4, False)
    low["tracks"][0]["keypoints17"][10][2] = 0.1
    state.update(low)
    assert state.people[0]["pose_observation"] == "unknown"
    assert state.people[0]["pose_quality"]["reason"] == "low_confidence"


@pytest.mark.parametrize(
    "prior", ["first", "after_lowered", "after_candidate", "after_active", "during_release"]
)
def test_ambiguous_geometry_never_establishes_known_absence_or_complete_boundaries(prior):
    state = api().RaiseHandState()
    t = 0
    if prior == "after_lowered":
        state.update(pose_frame(t, False))
        t += 0.2
    elif prior == "after_candidate":
        state.update(pose_frame(t, False))
        state.update(pose_frame(0.2))
        t = 0.4
    elif prior in ("after_active", "during_release"):
        for t in (0, 0.2, 0.4):
            state.update(pose_frame(t))
        t = 0.6
        if prior == "during_release":
            state.update(pose_frame(t, False))
            t = 0.8
    ambiguous = pose_frame(t)
    ambiguous["tracks"][0]["keypoints17"][9][1] = 90
    closed = state.update(ambiguous)
    if prior in ("after_active", "during_release"):
        assert len(closed) == 1
        assert closed[0]["evidence"]["boundary_truncated"]["end"] is True
    for offset in (0.2, 0.4, 0.6):
        state.update(pose_frame(round(t + offset, 6)))
    final = state.finish()
    assert len(final) == 1
    assert final[0]["evidence"]["boundary_truncated"]["start"] is True


@pytest.mark.parametrize("value", [-1, 201, float("nan"), float("inf"), True])
def test_raise_pose_invalid_raster_or_numeric_geometry_never_becomes_positive_or_negative(value):
    state = api().RaiseHandState()
    row = pose_frame(0)
    row.update(width=200, height=200)
    row["tracks"][0]["keypoints17"][9][0] = value
    assert not state.update(row)
    assert state.people[0]["status"] == "unsupported"
    assert state.people[0]["pose_observation"] == "unknown"
    assert state.finish() == []


def test_temporal_quality_outlier_censors_event_rebaselines_and_preserves_raw():
    import copy

    state = api().RaiseHandState(pose_quality={"temporal_guard": True, "max_joint_speed": 1})
    for t in (0, 0.2, 0.4):
        row = pose_frame(t)
        row["tracks"][0]["bbox_xyxy"] = [0, 0, 100, 200]
        state.update(row)
    outlier = pose_frame(0.5)
    outlier["tracks"][0]["bbox_xyxy"] = [0, 0, 100, 200]
    outlier["tracks"][0]["keypoints17"][9][0] += 60
    original = copy.deepcopy(outlier)
    closed = state.update(outlier)
    assert len(closed) == 1 and closed[0]["end_s"] == 0.4
    assert closed[0]["evidence"]["boundary_truncated"]["end"] is True
    assert state.people[0]["pose_quality"]["reason"] == "temporal_outlier"
    assert state.people[0]["pose_observation"] == "unknown"
    assert outlier == original
    recovered = copy.deepcopy(outlier)
    recovered.update(frame_index=60, timestamp_seconds=0.6)
    state.update(recovered)
    assert state.people[0]["status"] == "candidate"
    assert state.people[0]["start_s"] == 0.6
    assert state.people[0]["pose_observation"] == "raised"
    assert state.config["pose_quality"]["temporal_guard"] is True


@pytest.mark.parametrize("confusable", ["touch_head", "wrist_only", "hidden_other_arm"])
def test_confusable_pose_is_not_a_supported_negative_or_confirmed_raise(confusable):
    state = api().RaiseHandState()
    for t in (0, 0.2, 0.4):
        row = pose_frame(t)
        points = row["tracks"][0]["keypoints17"]
        if confusable == "touch_head":
            points[9] = [50, 50, 1]
        elif confusable == "wrist_only":
            points[7][1] = 110
        else:
            points[10][2] = 0.1
        assert state.update(row) == []
        assert state.people[0]["pose_observation"] == "unknown"
        assert state.people[0]["status"] in ("no_suggestion", "unsupported")
    assert state.finish() == []


def test_quality_snapshots_are_detached_and_prefix_causal():
    state = api().RaiseHandState(pose_quality={"temporal_guard": True})
    prefix = [pose_frame(t) for t in (0, 0.2, 0.4)]
    for row in prefix:
        state.update(row)
    snapshot = state.people
    snapshot[0]["pose_quality"]["status"] = "mutated"
    assert state.people[0]["pose_quality"]["status"] == "supported"
    future = pose_frame(0.6)
    future["tracks"][0]["keypoints17"][9][0] = float("nan")
    observed = state.people
    state.update(future)
    fresh = api().RaiseHandState(pose_quality={"temporal_guard": True})
    for row in prefix:
        fresh.update(row)
    assert observed == fresh.people
    config = fresh.config
    config["pose_quality"]["temporal_guard"] = False
    assert fresh.config["pose_quality"]["temporal_guard"] is True


@pytest.mark.parametrize("reset", ["cut", "gap", "missing", "duplicate", "identity"])
def test_temporal_quality_baseline_resets_with_observation_and_identity_context(reset):
    state = api().RaiseHandState(pose_quality={"temporal_guard": True})
    first = pose_frame(0)
    first["tracks"][0]["bbox_xyxy"] = [0, 0, 100, 200]
    state.update(first)
    middle = pose_frame(0.1)
    middle["tracks"][0]["bbox_xyxy"] = [0, 0, 100, 200]
    if reset == "missing":
        middle["tracks"] = []
    elif reset == "duplicate":
        middle["tracks"] *= 2
    elif reset == "identity":
        middle["tracks"][0]["identity_uncertain"] = True
    state.update(middle)
    current = pose_frame(0.6 if reset == "gap" else 0.2)
    if reset == "cut":
        current["segment_id"] = "other"
    current["tracks"][0]["bbox_xyxy"] = [0, 0, 400, 800]
    state.update(current)
    assert state.people[0]["pose_quality"]["status"] == "supported"
    assert state.people[0]["status"] == "candidate"


def test_raise_hand_requires_dwell_and_emits_only_draft_observable_action():
    frames = [pose_frame(t) for t in (0, 0.2, 0.4, 0.6)]
    rows = api().propose_raise_hand_events(frames)
    assert len(rows) == 1
    assert rows[0]["label"] == "raise_hand"
    assert rows[0]["start_s"] == 0
    assert rows[0]["end_s"] == 0.6
    assert rows[0]["origin"] == "rule" and rows[0]["review"] == "draft"
    assert rows[0]["evidence"]["calibrated"] is False
    assert api().propose_raise_hand_events([pose_frame(0), pose_frame(0.2)]) == []


def alternate_frame(t, *, right_raised=True, other_missing=True):
    row = pose_frame(t, False)
    points = row["tracks"][0]["keypoints17"]
    points[0][2] = 0.02
    if other_missing:
        points[7][2] = points[9][2] = 0.1
    points[8][1] = 60 if right_raised else 110
    points[10][1] = 20 if right_raised else 120
    return row


def test_explicit_alternate_cue_supports_one_arm_without_nose_or_hidden_other_arm():
    rows = [alternate_frame(t) for t in (0, 0.2, 0.4, 0.6)]
    legacy = api().RaiseHandState()
    state = api().RaiseHandState(hand_cue="shoulder_any_arm_v1")
    for row in rows:
        legacy.update(row)
        state.update(row)
    assert legacy.people[0]["status"] == "unsupported"
    person = state.people[0]
    assert person["status"] == "active" and person["pose_observation"] == "raised"
    assert person["pose_quality"]["supported_arms"] == ["right"]
    assert person["rule"] == "coco17_shoulder_any_arm_v1"
    assert state.config["min_dwell_s"] == 0.5
    assert state.config["enter_margin"] == 0.45 and state.config["exit_margin"] == 0.2
    event = state.finish()[0]
    assert event["evidence"]["rule"] == person["rule"]
    assert event["origin"] == "rule" and event["review"] == "draft"


def test_alternate_cue_negative_requires_both_arms_and_ambiguous_geometry_is_unknown():
    state = api().RaiseHandState(hand_cue="shoulder_any_arm_v1")
    state.update(alternate_frame(0, right_raised=False))
    assert state.people[0]["pose_observation"] == "unknown"
    state.update(alternate_frame(0.2, right_raised=False, other_missing=False))
    assert state.people[0]["pose_observation"] == "not_raised"
    ambiguous = alternate_frame(0.4, right_raised=False, other_missing=False)
    ambiguous["tracks"][0]["keypoints17"][10][1] = 95
    state.update(ambiguous)
    assert state.people[0]["pose_observation"] == "unknown"
    assert state.finish() == []


def test_alternate_per_arm_temporal_gate_does_not_contaminate_supported_other_arm():
    state = api().RaiseHandState(
        hand_cue="shoulder_any_arm_v1", pose_quality={"temporal_guard": True, "max_joint_speed": 1}
    )
    for t in (0, 0.2, 0.4, 0.6):
        row = alternate_frame(t, other_missing=False)
        row["tracks"][0]["bbox_xyxy"] = [0, 0, 100, 200]
        if t >= 0.4:
            row["tracks"][0]["keypoints17"][9][0] += 150
        state.update(row)
    assert state.people[0]["status"] == "active"
    assert state.people[0]["pose_observation"] == "raised"
    assert state.finish()[0]["evidence"]["rule"] == "coco17_shoulder_any_arm_v1"
    with pytest.raises(ValueError):
        api().RaiseHandState(hand_cue="invented")


@pytest.mark.parametrize("kind", ["missing", "confidence", "identity", "cut", "gap"])
def test_unknown_gap_or_identity_resets_raise_hand_dwell(kind):
    frames = [pose_frame(0), pose_frame(0.2), pose_frame(0.4), pose_frame(0.6)]
    if kind == "missing":
        frames[1]["tracks"] = []
    elif kind == "confidence":
        frames[1]["tracks"][0]["keypoints17"][9][2] = 0.1
    elif kind == "identity":
        frames[1]["tracks"][0]["confirmed"] = False
    elif kind == "cut":
        for frame in frames[1:]:
            frame["segment_id"] = "cut-2"
    else:
        frames = [pose_frame(0), pose_frame(0.4), pose_frame(0.6)]
    # .4 is not enough after the cut: the last new-cut span is .4,
    # so shorten it to .3 to distinguish a new run from accidental stitching.
    if kind == "cut":
        frames[-1]["timestamp_seconds"] = 0.5
    assert api().propose_raise_hand_events(frames) == []


def test_raise_hand_release_hysteresis_does_not_split_short_lowering():
    frames = [pose_frame(t) for t in (0, 0.2, 0.4)]
    frames += [
        pose_frame(0.6, False),
        pose_frame(0.8),
        pose_frame(1),
        pose_frame(1.2, False),
        pose_frame(1.4, False),
        pose_frame(1.6, False),
    ]
    rows = api().propose_raise_hand_events(frames)
    assert len(rows) == 1
    assert rows[0]["end_s"] == 1.2


def test_incremental_raise_hand_emits_closed_only_and_matches_offline():
    frames = [pose_frame(0, False)] + [pose_frame(t) for t in (0.2, 0.4, 0.6)]
    frames += [pose_frame(t, False) for t in (0.8, 1, 1.2)]
    state = api().RaiseHandState()
    emitted = []
    for frame in frames[:-1]:
        assert state.update(frame) == []
    assert state.counts == {"active_actor_count": 1, "pending_actor_count": 0}
    emitted.extend(state.update(frames[-1]))
    assert len(emitted) == 1 and emitted[0]["end_s"] == 0.8
    assert emitted[0]["evidence"]["boundary_truncated"] == {"start": False, "end": False}
    assert state.finish(1.4) == []
    assert emitted == api().propose_raise_hand_events(frames)
    assert state.finish(1.4) == []
    with pytest.raises(ValueError, match="finished"):
        state.update(pose_frame(1.6))


@pytest.mark.parametrize("reason", ["source_end", "cancelled", "failed"])
def test_incremental_finish_censors_at_last_observation_without_extrapolating(reason):
    state = api().RaiseHandState()
    for t in (0, 0.2, 0.4):
        assert state.update(pose_frame(t)) == []
    row = state.finish(100, reason=reason)[0]
    assert row["end_s"] == 0.4
    assert row["evidence"]["boundary_truncated"] == {"start": True, "end": True}
    assert row["evidence"]["boundary_reason"]["end"] == reason
    assert state.counts == {"active_actor_count": 0, "pending_actor_count": 0}


def test_incremental_prefix_gap_and_bounded_actor_memory():
    state = api().RaiseHandState()
    for t in (0, 0.2, 0.4):
        state.update(pose_frame(t))
    rows = state.update(pose_frame(1))
    assert len(rows) == 1 and rows[0]["end_s"] == 0.4
    assert rows[0]["evidence"]["boundary_reason"]["end"] == "observation_gap"
    # New local actors do not leave an ever-growing known-absent history.
    for i in range(500):
        frame = pose_frame(1.2 + i * 0.2, False)
        frame["tracks"][0]["track_id"] = i
        state.update(frame)
    assert len(state._known_absent) == 1 and not state._states
    assert state.config["rule"] == "coco17_wrist_above_nose_v1"
    assert state.config["max_gap_s"] == 0.25


def test_incremental_raw_pose_only_and_skipped_source_indices_are_not_missing_samples():
    state = api().RaiseHandState()
    for i, t in enumerate((0, 0.2, 0.4)):
        frame = pose_frame(t)
        frame["frame_index"] = i * 120
        frame["tracks"][0]["display_keypoints17"] = [[0, 0, 0]] * 17
        state.update(frame)
    assert len(state.finish()) == 1


@pytest.mark.parametrize(
    "termination,reason",
    [
        ("scene", "scene_cut"),
        ("unknown", "unknown_pose_or_identity"),
        ("missing", "missing_or_duplicate_track"),
        ("duplicate", "missing_or_duplicate_track"),
    ],
)
def test_incremental_causal_termination_never_uses_unknown_endpoint(termination, reason):
    state = api().RaiseHandState()
    for t in (0, 0.2, 0.4):
        assert state.update(pose_frame(t)) == []
    frame = pose_frame(0.6)
    if termination == "scene":
        frame["segment_id"] = "other-scene"
    elif termination == "unknown":
        frame["tracks"][0]["identity_uncertain"] = True
    elif termination == "missing":
        frame["tracks"] = []
    else:
        frame["tracks"] *= 2
    row = state.update(frame)[0]
    assert row["segment_id"] == "cut-1" and row["end_s"] == 0.4
    assert row["evidence"]["boundary_reason"]["end"] == reason
    assert row["evidence"]["boundary_truncated"]["end"] is True
    assert state.finish() == []


def test_incremental_configuration_isolated_and_matches_legacy_defaults():
    state, other = api().RaiseHandState(), api().RaiseHandState(min_dwell_s=0.2)
    config = state.config
    config["confidence"] = 0.01
    assert state.config["confidence"] == 0.5
    for t in (0, 0.2):
        state.update(pose_frame(t))
        other.update(pose_frame(t))
    assert state.finish() == []
    assert len(other.finish()) == 1


def test_incremental_people_follow_current_pose_not_closed_event_history():
    state = api().RaiseHandState()
    assert state.people == []
    for t, raised, status in (
        (0, True, "candidate"),
        (0.2, True, "candidate"),
        (0.4, True, "active"),
        (0.6, False, "release_pending"),
        (0.8, False, "release_pending"),
        (1, False, "no_suggestion"),
    ):
        state.update(pose_frame(t, raised))
        person = state.people[0]
        assert person["status"] == status
        assert person["observed_through_s"] == t
        assert person["segment_id"] == "cut-1" and person["track_id"] == 7
        assert person["origin"] == "rule" and person["review"] == "draft"
        assert person["calibrated"] is False
        assert person["rule"] == "coco17_wrist_above_nose_v1"
        assert person["label"] == (None if status == "no_suggestion" else "raise_hand")
        assert ("start_s" in person) == (status != "no_suggestion")
    # A closed event must not label the person in subsequent source frames.
    state.update(pose_frame(1.2, False))
    assert state.people[0]["status"] == "no_suggestion"
    state.finish()
    assert state.people == []


@pytest.mark.parametrize("change", ["confidence", "unconfirmed", "identity", "duplicate"])
def test_incremental_people_unsupported_is_current_uncertainty(change):
    state = api().RaiseHandState()
    for t in (0, 0.2, 0.4):
        state.update(pose_frame(t))
    frame = pose_frame(0.6)
    if change == "confidence":
        frame["tracks"][0]["keypoints17"][9][2] = 0.1
    elif change == "unconfirmed":
        frame["tracks"][0]["confirmed"] = False
    elif change == "identity":
        frame["tracks"][0]["identity_uncertain"] = True
    else:
        frame["tracks"] *= 2
    state.update(frame)
    assert len(state.people) == 1
    assert state.people[0]["status"] == "unsupported" and state.people[0]["label"] is None
    assert state.people[0]["reason"] == (
        "duplicate_track" if change == "duplicate" else "unknown_pose_or_identity"
    )
    assert "start_s" not in state.people[0]


def test_incremental_people_snapshot_detached_and_no_stale_missing_gap_or_scene_actor():
    state = api().RaiseHandState()
    for t in (0, 0.2, 0.4):
        state.update(pose_frame(t))
    original = state.people
    original[0]["status"] = "changed"
    assert state.people[0]["status"] == "active"
    frame = pose_frame(0.6)
    frame["tracks"] = []
    state.update(frame)
    assert state.people == []
    state.update(pose_frame(0.8))
    assert state.people[0]["status"] == "candidate" and state.people[0]["start_s"] == 0.8
    state.update(pose_frame(1.2))
    assert state.people[0]["status"] == "candidate" and state.people[0]["start_s"] == 1.2
    state.update(pose_frame(1.4, segment="new-scene"))
    assert state.people[0]["segment_id"] == "new-scene"
    assert state.people[0]["start_s"] == 1.4
    assert original[0]["observed_through_s"] == 0.4


def test_aggregation_counts_onsets_clips_durations_and_never_scores_people():
    rows = [event(start=1, end=7), event(start=8, end=10)]
    summary = api().aggregate_events(
        rows, duration_s=10, window_s=5, coverage=context()["coverage"]
    )
    assert summary["session"]["raise_hand"]["count"] == 2
    assert summary["session"]["raise_hand"]["duration_s"] == 8
    assert summary["windows"][0]["labels"]["raise_hand"]["count"] == 1
    assert summary["windows"][0]["labels"]["raise_hand"]["duration_s"] == 4
    assert summary["windows"][1]["labels"]["raise_hand"]["count"] == 1
    assert summary["windows"][1]["labels"]["raise_hand"]["duration_s"] == 4
    assert "actors" not in summary and "score" not in summary


def test_aggregation_empty_or_draft_is_unreviewed_not_negative():
    summary = api().aggregate_events([], duration_s=10, coverage=[])
    assert summary["session"]["raise_hand"]["count"] is None
    row = event()
    row["review"] = "draft"
    summary = api().aggregate_events([row], duration_s=10, coverage=[])
    assert summary["session"]["raise_hand"]["count"] is None
    assert summary["excluded_events"] == 1


def test_numeric_segment_ids_and_unknown_posture_are_separate_from_actions():
    args = context()
    for row in args["track_segments"] + args["coverage"]:
        row["segment_id"] = 0
    unknown, raised = event("unknown"), event()
    unknown["segment_id"] = raised["segment_id"] = 0
    assert len(api().validate_events([unknown, raised], **args)) == 2
    summary = api().aggregate_events([unknown, raised], duration_s=10, coverage=args["coverage"])
    assert summary["session"]["unknown"]["duration_s"] == 2
    assert summary["session"]["raise_hand"]["count"] == 1
    result = api().evaluate_event_f1([raised], [unknown, raised], **args)
    assert result["tp"] == 1 and result["ignored_references"] == 1


def test_fully_uncertain_label_coverage_has_no_negative_aggregate_count():
    row = event(start=0, end=10)
    row["uncertain"] = True
    summary = api().aggregate_events([row], duration_s=10, coverage=context()["coverage"])
    assert summary["session"]["raise_hand"]["count"] is None
    assert summary["session"]["raise_hand"]["reviewed_coverage_actor_s"] == 0


def test_pose_rejects_nonmonotonic_frames_and_invalid_hysteresis():
    with pytest.raises(ValueError, match="increasing"):
        api().propose_raise_hand_events([pose_frame(0.2), pose_frame(0)])
    with pytest.raises(ValueError, match="exit_margin"):
        api().propose_raise_hand_events([], enter_margin=0.1, exit_margin=0.2)


def test_active_rule_ends_at_missing_observation_and_does_not_bridge_it():
    frames = [pose_frame(t) for t in (0, 0.2, 0.4)]
    frames += [pose_frame(0.6), pose_frame(0.8), pose_frame(1)]
    frames[3]["tracks"] = []
    rows = api().propose_raise_hand_events(frames)
    assert len(rows) == 1 and rows[0]["end_s"] == 0.4


def test_pose_enter_exit_thresholds_have_a_real_hysteresis_band():
    frames = [pose_frame(t) for t in (0, 0.2, 0.4, 0.6, 0.8, 1)]
    # Shoulder width 20: enter wrist_y < 48, keep wrist_y < 49.
    # Once active, remaining at 48.5 must not trigger release.
    for frame in frames[3:]:
        frame["tracks"][0]["keypoints17"][9][1] = 48.5
    rows = api().propose_raise_hand_events(frames)
    assert len(rows) == 1 and rows[0]["end_s"] == 1
    for frame in frames:
        frame["tracks"][0]["keypoints17"][9][1] = 48.5
    assert api().propose_raise_hand_events(frames) == []


def test_evaluation_one_to_one_with_explicit_reviewed_negative_coverage():
    prediction = [event(start=1, end=3), event(start=1.1, end=3.1)]
    result = api().evaluate_event_f1(prediction, [event()], **context())
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["f1"] == pytest.approx(2 / 3)


def test_evaluation_unreviewed_and_uncertain_are_excluded_not_false_positives():
    args = context()
    args["coverage"] = []
    result = api().evaluate_event_f1([event()], [], **args)
    assert result["f1"] is None
    assert result["fp"] == 0 and result["ignored_predictions"] == 1
    unknown = event()
    unknown["uncertain"] = True
    result = api().evaluate_event_f1([event()], [unknown], **context())
    assert result["f1"] is None and result["fp"] == 0


def test_evaluation_honors_temporal_iou_and_reports_partial_review_coverage():
    args = context()
    args["coverage"] = [
        dict(
            segment_id="cut-1",
            track_id=7,
            label="raise_hand",
            start_s=0,
            end_s=5,
            review="reviewed",
        )
    ]
    # [1,3) vs [2,4): intersection 1, union 3, below IoU .5.
    result = api().evaluate_event_f1(
        [event(start=2, end=4), event(start=7, end=8)], [event()], **args
    )
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert result["f1"] == 0 and result["ignored_predictions"] == 1
    assert result["reviewed_coverage_actor_s"]["raise_hand"] == 5


def test_uncertain_posture_does_not_certify_other_postures_as_negative():
    row = event("sitting", start=0, end=10)
    row["uncertain"] = True
    summary = api().aggregate_events([row], duration_s=10, coverage=context()["coverage"])
    assert summary["session"]["standing"]["count"] is None
    assert summary["session"]["transition"]["count"] is None


def test_evaluation_uses_maximum_one_to_one_assignment_and_never_joins_cuts():
    # IoU threshold .2: p0 matches both refs, p1 only r0. Greedy p0-r0 loses TP.
    preds = [event(start=0, end=6), event(start=0, end=2)]
    refs = [event(start=0, end=3), event(start=3, end=9)]
    result = api().evaluate_event_f1(preds, refs, iou_threshold=0.2, **context())
    assert result["tp"] == 2 and result["f1"] == 1
    args = context()
    args["track_segments"].append(dict(segment_id="cut-2", track_id=7, start_s=0, end_s=10))
    args["coverage"].append(
        dict(
            segment_id="cut-2",
            track_id=7,
            label="raise_hand",
            start_s=0,
            end_s=10,
            review="reviewed",
        )
    )
    pred = event()
    pred["segment_id"] = "cut-2"
    result = api().evaluate_event_f1([pred], [event()], **args)
    assert result["tp"] == 0 and result["fp"] == 1 and result["fn"] == 1


@pytest.mark.parametrize(
    "boundary",
    [
        True,
        {},
        {"start": True},
        {"start": True, "end": "false"},
        {"start": False, "end": False, "other": False},
    ],
)
def test_boundary_truncation_evidence_requires_two_boolean_endpoints(boundary):
    row = event()
    row["evidence"] = {"boundary_truncated": boundary}
    with pytest.raises(ValueError, match="boundary_truncated"):
        api().validate_events([row], **context())


def test_clip_start_episode_without_evidence_is_not_a_new_onset():
    summary = api().aggregate_events(
        [event(start=0, end=7)], duration_s=10, coverage=context()["coverage"], window_s=5
    )
    session = summary["session"]["raise_hand"]
    assert session["count"] == 0
    assert session["truncated_episode_count"] == 1
    assert session["observed_duration_s"] == 7
    assert session["duration_s"] == 7
    assert summary["windows"][1]["labels"]["raise_hand"]["truncated_episode_count"] == 1


def test_review_boundary_without_prior_coverage_is_conservatively_left_truncated():
    args = context()
    args["coverage"] = [
        dict(
            segment_id="cut-1",
            track_id=7,
            label="raise_hand",
            start_s=2,
            end_s=7,
            review="reviewed",
        )
    ]
    summary = api().aggregate_events(
        [event(start=2, end=7)], duration_s=10, coverage=args["coverage"]
    )
    assert summary["session"]["raise_hand"]["count"] == 0
    assert summary["session"]["raise_hand"]["truncated_episode_count"] == 1


def test_explicit_boundary_evidence_preserves_observed_onsets_and_censored_durations():
    row = event(start=0, end=10)
    row["evidence"] = {"boundary_truncated": {"start": False, "end": True}}
    summary = api().aggregate_events([row], duration_s=10, coverage=context()["coverage"])
    assert summary["session"]["raise_hand"]["count"] == 1
    assert summary["session"]["raise_hand"]["truncated_episode_count"] == 1
    assert summary["session"]["raise_hand"]["observed_duration_s"] == 10
    assert api().validate_events([row], **context())[0]["evidence"] == row["evidence"]


def test_rule_flags_both_clip_boundaries_without_claiming_unknown_action_content():
    rows = api().propose_raise_hand_events([pose_frame(t) for t in (0, 0.2, 0.4, 0.6)])
    assert rows[0]["evidence"]["boundary_truncated"] == {"start": True, "end": True}
    assert rows[0]["evidence"]["boundary_reason"]["end"] == "source_end"
    # Action content is observed; only its temporal boundary is censored.
    assert rows[0]["uncertain"] is False and rows[0]["review"] == "draft"


def test_rule_observed_absence_then_release_proves_nontruncated_boundaries():
    frames = [pose_frame(0, False)] + [pose_frame(t) for t in (0.2, 0.4, 0.6)]
    frames += [pose_frame(t, False) for t in (0.8, 1, 1.2)]
    rows = api().propose_raise_hand_events(frames)
    assert rows[0]["evidence"]["boundary_truncated"] == {"start": False, "end": False}
    assert rows[0]["start_s"] == 0.2 and rows[0]["end_s"] == 0.8


@pytest.mark.parametrize("boundary", ["missing", "confidence", "cut", "gap"])
def test_rule_censors_right_boundary_and_resets_onset_knowledge_after_unknown(boundary):
    frames = [pose_frame(t) for t in (0, 0.2, 0.4)]
    if boundary == "missing":
        gap = pose_frame(0.6)
        gap["tracks"] = []
        frames += [gap] + [pose_frame(t) for t in (0.8, 1, 1.2)]
    elif boundary == "confidence":
        frames += [pose_frame(0.6, confidence=0.1)]
        frames += [pose_frame(t) for t in (0.8, 1, 1.2)]
    elif boundary == "cut":
        frames += [pose_frame(t, segment="cut-2") for t in (0.6, 0.8, 1)]
    else:
        frames += [pose_frame(t) for t in (0.8, 1, 1.2)]
    rows = api().propose_raise_hand_events(frames)
    assert len(rows) == 2
    assert all(
        row["evidence"]["boundary_truncated"] == {"start": True, "end": True} for row in rows
    )


def test_event_match_report_preserves_original_indices_with_duplicate_predictions():
    ignored = event(start=0, end=1)
    ignored["uncertain"] = True
    prediction = event(start=1.2, end=3.2)
    reference = event()
    result = api().evaluate_event_f1(
        [ignored, prediction, prediction], [reference], **context(), include_matches=True
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["matches"][0]["prediction_index"] in (1, 2)
    assert result["matches"][0]["reference_index"] == 0
    assert result["matches"][0]["start_error_s"] == pytest.approx(0.2)
    assert result["matches"][0]["end_error_s"] == pytest.approx(0.2)


def test_boundary_tolerance_is_explicit_and_does_not_grade_censored_endpoint():
    reference = event(start=0, end=4)
    reference["evidence"] = {"boundary_truncated": {"start": True, "end": False}}
    prediction = event(start=1, end=4.1)
    result = api().evaluate_event_f1(
        [prediction], [reference], **context(), boundary_tolerance_s=0.2, include_matches=True
    )
    assert result["tp"] == 1
    assert result["matches"][0]["start_error_s"] is None
    assert result["matches"][0]["end_error_s"] == pytest.approx(0.1)
    result = api().evaluate_event_f1(
        [event(start=1, end=4.3)], [reference], **context(), boundary_tolerance_s=0.2
    )
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


@pytest.mark.parametrize("tolerance", [-1, True, float("nan"), float("inf")])
def test_invalid_boundary_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="boundary_tolerance_s"):
        api().evaluate_event_f1([], [], **context(), boundary_tolerance_s=tolerance)


def test_long_augmenting_path_does_not_depend_on_python_recursion_depth():
    import sys

    count = 350
    args = context()
    args["duration_s"] = count + 1
    for span in [*args["track_segments"], *args["coverage"]]:
        span["end_s"] = count + 1
    references = [event(start=i, end=i + 1) for i in range(count)]
    predictions = [event(start=0, end=2), event(start=0, end=2)]
    predictions += [event(start=i - 1, end=i + 1) for i in range(2, count)]
    original = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(300)
        result = api().evaluate_event_f1(predictions, references, **args)
    finally:
        sys.setrecursionlimit(original)
    assert result["tp"] == count and result["fp"] == result["fn"] == 0
