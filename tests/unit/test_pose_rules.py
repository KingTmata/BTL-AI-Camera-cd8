"""Synthetic causal proxies, not empirical GT or calibrated classroom thresholds."""

import copy
import importlib
import math

import pytest

from classroom_ai.video.actions import propose_raise_hand_events


def api():
    return importlib.import_module("classroom_ai.video.pose_rules")


def frame(t, *, seated=False, x=300, raised=False, segment=0, rotate=0):
    points = [[x, 500, 1] for _ in range(17)]
    points[0] = [x, 50, 1]
    for shoulder, elbow, wrist, hip, knee, ankle, side in (
        (5, 7, 9, 11, 13, 15, -20),
        (6, 8, 10, 12, 14, 16, 20),
    ):
        sx = x + side
        points[shoulder] = [sx, 300 if seated else 100, 1]
        points[elbow] = [sx, 60 if raised else 500, 1]
        points[wrist] = [sx, 20 if raised else 600, 1]
        points[hip] = [sx, 600 if seated else 400, 1]
        points[knee] = [sx + (300 if seated else 0), 600 if seated else 650, 1]
        points[ankle] = [sx + (300 if seated else 0), 900, 1]
    if rotate:
        for point in points:
            px, py = point[0] - 500, point[1] - 500
            point[:2] = [
                500 + math.cos(rotate) * px - math.sin(rotate) * py,
                500 + math.sin(rotate) * px + math.cos(rotate) * py,
            ]
    return dict(
        frame_index=round(t * 100),
        timestamp_seconds=t,
        segment_id=segment,
        width=1000,
        height=1000,
        tracks=[dict(track_id=7, confirmed=True, state="observed", keypoints17=points)],
    )


def action(state, label):
    return next(row for row in state.people[0]["actions"] if row["label"] == label)


def test_quality_channels_do_not_erase_independently_supported_actions():
    state = api().BasicPoseActionState()
    events = []
    for i in range(6):
        row = frame(i * 0.2, seated=i < 3)
        row["tracks"][0]["keypoints17"][9][2] = 0.1
        events += state.update(row)
    assert [event["label"] for event in events] == ["stand_up"]
    assert action(state, "raise_hand")["pose_observation"] == "unknown"
    assert action(state, "stand_up")["pose_quality"]["status"] == "supported"
    other = api().BasicPoseActionState()
    for t in (0, 0.2, 0.4):
        row = frame(t, raised=True)
        row["tracks"][0]["keypoints17"][15][2] = 0.1
        other.update(row)
    assert action(other, "raise_hand")["status"] == "active"
    assert action(other, "stand_up")["status"] == "unsupported"
    assert other.people[0]["context_relations"] == []


def relation_frames():
    return [
        frame(i * 0.2, x=x, raised=i < 3)
        for i, x in enumerate((300, 300, 300, 400, 550, 750, 750, 750, 750, 750, 750))
    ]


def test_raise_then_board_is_draft_context_separate_from_events_and_current_pose():
    rows = relation_frames()
    original = copy.deepcopy(rows)
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    emitted = []
    for row in rows:
        emitted += state.update(row)
    assert sorted(event["label"] for event in emitted) == ["move_to_board", "raise_hand"]
    assert action(state, "raise_hand")["pose_observation"] == "not_raised"
    relation = state.people[0]["context_relations"][0]
    assert relation["relation"] == "raise_hand_then_move_to_board"
    assert relation["status"] == "observed_sequence"
    assert relation["raise_confirmed_at_s"] == pytest.approx(0.4)
    assert relation["board_start_s"] == pytest.approx(0.6)
    assert relation["board_confirmed_at_s"] == pytest.approx(2)
    assert relation["origin"] == "rule" and relation["review"] == "draft"
    assert relation["calibrated"] is False and "label" not in relation
    assert rows == original
    state.people[0]["context_relations"][0]["status"] = "bad"
    assert state.people[0]["context_relations"][0]["status"] == "observed_sequence"
    state.finish()
    assert state.people == []


@pytest.mark.parametrize(
    "change",
    ["hand_quality", "body_quality", "identity", "duplicate", "missing", "cut", "gap", "id_switch"],
)
def test_raise_board_context_never_crosses_missing_or_uncertain_evidence(change):
    rows = relation_frames()
    row = rows[4]
    if change == "hand_quality":
        row["tracks"][0]["keypoints17"][9][2] = 0.1
    elif change == "body_quality":
        row["tracks"][0]["keypoints17"][15][2] = 0.1
    elif change == "identity":
        row["tracks"][0]["identity_uncertain"] = True
    elif change == "duplicate":
        row["tracks"] *= 2
    elif change == "missing":
        row["tracks"] = []
    else:
        for item in rows[4:]:
            if change == "cut":
                item["segment_id"] = 1
            elif change == "gap":
                item["timestamp_seconds"] += 0.4
                item["frame_index"] += 40
            else:
                item["tracks"][0]["track_id"] = 8
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    for row in rows:
        state.update(row)
        assert all(not person["context_relations"] for person in state.people)


def test_raise_board_relation_expires_and_cannot_use_a_later_raise_as_prior_cause():
    roi = [0.7, 0.72, 0.95, 0.98]
    state = api().BasicPoseActionState(roi, relation_window_s=0.5)
    for row in relation_frames():
        state.update(row)
    assert state.people[0]["context_relations"] == []
    later = api().BasicPoseActionState(roi)
    for i, row in enumerate(relation_frames()):
        replacement = frame(
            row["timestamp_seconds"], x=row["tracks"][0]["keypoints17"][0][0], raised=i >= 4
        )
        later.update(replacement)
    assert action(later, "move_to_board")["status"] == "completed"
    assert later.people[0]["context_relations"] == []
    for value in (0, 61, True, float("nan")):
        with pytest.raises(ValueError):
            api().BasicPoseActionState(relation_window_s=value)


def test_recovered_body_cannot_reuse_raise_episode_from_before_quality_gap():
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    rows = [frame(i * 0.2, x=300 if i < 7 else 750, raised=True) for i in range(14)]
    rows[3]["tracks"][0]["keypoints17"][15][2] = 0.1
    for row in rows:
        state.update(row)
        assert all(not person["context_relations"] for person in state.people)
    assert action(state, "raise_hand")["status"] == "active"
    assert action(state, "move_to_board")["status"] == "completed"


@pytest.mark.parametrize("rotate", [0, 0.5, -0.4])
def test_stand_requires_observed_seated_baseline_then_rise_and_dwell(rotate):
    state = api().BasicPoseActionState()
    for t in (0, 0.2, 0.4):
        assert state.update(frame(t, seated=True, rotate=rotate)) == []
    for t in (0.6, 0.8):
        assert state.update(frame(t, rotate=rotate)) == []
        assert action(state, "stand_up")["status"] == "candidate"
    events = state.update(frame(1, rotate=rotate))
    assert len(events) == 1 and events[0]["label"] == "stand_up"
    assert events[0]["start_s"] == 0.6 and events[0]["end_s"] == 1
    assert events[0]["origin"] == "rule" and events[0]["review"] == "draft"
    assert events[0]["evidence"]["calibrated"] is False
    assert action(state, "stand_up")["status"] == "completed"
    assert action(state, "stand_up")["confirmed_at_s"] == 1
    assert state.counts["recent_completed_actor_count"] == 1
    for t in (1.2, 1.4, 1.6, 1.8):
        assert state.update(frame(t, rotate=rotate)) == []
    assert action(state, "stand_up")["status"] == "completed"
    state.update(frame(2, rotate=rotate))
    assert action(state, "stand_up")["status"] == "no_suggestion"
    assert state.finish() == []


def test_standing_at_start_and_brief_rise_never_complete_stand_up():
    state = api().BasicPoseActionState()
    for t in (0, 0.2, 0.4, 0.6, 0.8, 1):
        assert state.update(frame(t)) == []
    assert action(state, "stand_up")["status"] == "no_suggestion"
    for t in (1.2, 1.4, 1.6):
        state.update(frame(t, seated=True))
    assert state.update(frame(1.8)) == []
    assert state.update(frame(2, seated=True)) == []
    assert state.finish() == []
    assert state.observation_counts["incomplete_stand_candidates"] == 1


def test_upright_angles_without_projected_hip_rise_is_not_standing_transition():
    state = api().BasicPoseActionState()
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True))
    for t in (0.6, 0.8, 1, 1.2):
        row = frame(t)
        for point in row["tracks"][0]["keypoints17"]:
            point[1] += 200
        row["height"] = 1200
        assert state.update(row) == []
    assert action(state, "stand_up")["status"] != "completed"


def test_board_arrival_requires_outside_baseline_progress_and_stable_inside():
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    events = []
    for i, x in enumerate((300, 300, 300, 400, 550, 750, 750, 750, 750, 750, 750)):
        emitted = state.update(frame(i * 0.2, x=x))
        events += emitted
        if 3 <= i < 10:
            assert action(state, "move_to_board")["status"] == "candidate"
            assert not emitted
    assert len(events) == 1 and events[0]["label"] == "move_to_board"
    assert events[0]["start_s"] == pytest.approx(0.6) and events[0]["end_s"] == 2
    assert action(state, "move_to_board")["status"] == "completed"
    assert not state.update(frame(2.2, x=750))
    assert action(state, "move_to_board")["status"] == "completed"
    state.update(frame(2.4, x=600))
    assert action(state, "move_to_board")["status"] != "completed"
    assert state.finish() == []


@pytest.mark.parametrize(
    "xs",
    [
        [750] * 15,  # already in ROI
        [650, 655, 660, 650, 655, 660, 650, 655, 660, 650, 655, 660, 650, 655, 660],
        [300, 300, 300, 400, 550, 750, 750, 990, 990, 990, 990, 990, 990, 990, 990],
    ],
)
def test_board_hard_negatives_start_inside_jitter_and_brief_passage(xs):
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    events = [event for i, x in enumerate(xs) for event in state.update(frame(i * 0.2, x=x))]
    assert not events
    assert state.finish() == []


def test_no_board_roi_is_not_configured_not_action_absence():
    state = api().BasicPoseActionState()
    state.update(frame(0))
    assert action(state, "move_to_board")["status"] == "not_configured"
    assert action(state, "move_to_board")["reason"] == "missing_floor_roi"


@pytest.mark.parametrize("change", ["missing", "unknown", "identity", "gap", "scene", "duplicate"])
def test_temporal_baselines_reset_without_stitching_future_or_missing_pose(change):
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True))
    row = frame(0.6)
    if change == "missing":
        row["tracks"] = []
    elif change == "unknown":
        row["tracks"][0]["keypoints17"][15][2] = 0.1
    elif change == "identity":
        row["tracks"][0]["identity_uncertain"] = True
    elif change == "gap":
        row = frame(1)
    elif change == "scene":
        row["segment_id"] = 1
    else:
        row["tracks"] *= 2
    assert state.update(row) == []
    segment, base = row["segment_id"], row["timestamp_seconds"]
    for i in (1, 2, 3):
        assert state.update(frame(base + i * 0.2, segment=segment)) == []
    assert action(state, "stand_up")["status"] != "completed"


def test_raise_hand_parity_independence_and_detached_bounded_snapshots():
    rows = [frame(t, raised=True) for t in (0, 0.2, 0.4, 0.6)]
    state = api().BasicPoseActionState()
    events = [event for row in rows for event in state.update(copy.deepcopy(row))]
    people = state.people
    assert action(state, "raise_hand")["status"] == "active"
    people[0]["actions"][0]["status"] = "mutated"
    assert action(state, "raise_hand")["status"] == "active"
    events += state.finish(0.8)
    assert events == propose_raise_hand_events(rows)
    assert state.people == []
    assert state.finish() == []
    with pytest.raises(ValueError, match="finished"):
        state.update(frame(1))


@pytest.mark.parametrize(
    "roi",
    [
        [0.9, 0.5, 0.2, 0.9],
        [0, 0, 1],
        [0, 0, float("nan"), 1],
        [False, 0, 1, 1],
        [-0.1, 0, 1, 1],
        [0.5, 0.5, 0.51, 0.51],
    ],
)
def test_configuration_rejects_invalid_or_collapsed_roi(roi):
    with pytest.raises(ValueError, match="board_roi"):
        api().BasicPoseActionState(roi)


def test_config_provenance_defaults_and_no_alias():
    roi = [0.7, 0.72, 0.95, 0.98]
    state = api().BasicPoseActionState(roi)
    roi[0] = 0
    config = state.config
    assert config["board_roi"][0] == 0.7
    assert config["thresholds"]["recent_completed_s"] == 1
    assert config["thresholds"]["max_gap_s"] == 0.25
    config["thresholds"]["confidence"] = 0.01
    assert state.config["thresholds"]["confidence"] == 0.5
    with pytest.raises(ValueError):
        api().BasicPoseActionState(config={"max_gap_s": -1})


def test_all_three_hints_coexist_without_stand_reemission_during_standing():
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    events = []
    for t in (0, 0.2, 0.4):
        events += state.update(frame(t, seated=True, x=100, raised=True))
    events += state.update(frame(0.6, x=500, raised=True))
    events += state.update(frame(0.8, x=600, raised=True))
    events += state.update(frame(1, x=750, raised=True))
    assert [
        action(state, label)["status"] for label in ("raise_hand", "stand_up", "move_to_board")
    ] == ["active", "completed", "candidate"]
    for i in range(6, 31):
        events += state.update(frame(i * 0.2, x=750, raised=True))
    events += state.finish()
    assert [event["label"] for event in events].count("stand_up") == 1
    assert [event["label"] for event in events].count("move_to_board") == 1
    assert [event["label"] for event in events].count("raise_hand") == 1


def test_stand_candidate_timeout_even_when_hip_returns_to_baseline_height():
    state = api().BasicPoseActionState(config={"max_sequence_s": 0.5})
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True))
    state.update(frame(0.6))
    for t in (0.8, 1, 1.2):
        row = frame(t)
        row["height"] = 1200
        for point in row["tracks"][0]["keypoints17"]:
            point[1] += 200
        assert state.update(row) == []
    assert action(state, "stand_up")["status"] == "no_suggestion"
    assert state.observation_counts["incomplete_stand_candidates"] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e300, -1, 1001])
def test_extreme_or_outside_raster_lowerbody_is_unknown_not_imputed(bad):
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True))
    row = frame(0.6, x=750)
    row["tracks"][0]["keypoints17"][15][0] = bad
    state.update(row)
    assert action(state, "stand_up")["status"] == "unsupported"
    assert action(state, "move_to_board")["status"] == "unsupported"
    assert state.finish() == []


def test_completed_hints_clear_with_unreliable_current_pose_and_memory_is_current_only():
    state = api().BasicPoseActionState()
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True))
    for t in (0.6, 0.8, 1):
        state.update(frame(t))
    assert action(state, "stand_up")["status"] == "completed"
    row = frame(1.2)
    row["tracks"][0]["keypoints17"][13][2] = 0.1
    state.update(row)
    assert action(state, "stand_up")["status"] == "unsupported"
    state.update(frame(1.4))
    assert action(state, "stand_up")["status"] == "no_suggestion"
    for i in range(300):
        row = frame(1.6 + i * 0.2)
        row["tracks"][0]["track_id"] = i
        state.update(row)
    assert len(state._actors) == len(state.people) == 1


def test_old_seated_baseline_cannot_label_later_motion_of_already_standing_person():
    state = api().BasicPoseActionState()
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True))
    events = []
    for t in (0.6, 0.8, 1, 1.2):
        row = frame(t)
        row["height"] = 1200
        for point in row["tracks"][0]["keypoints17"]:
            point[1] += 200  # upright, but no rise from original seated hip
        events += state.update(row)
    for t in (1.4, 1.6, 1.8, 2):
        events += state.update(frame(t))
    assert not events
    assert action(state, "stand_up")["status"] == "no_suggestion"


def test_stand_timeout_starts_at_first_nonseated_not_later_qualifying_hip_rise():
    state = api().BasicPoseActionState(config={"max_sequence_s": 0.5})
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True))
    for t in (0.6, 0.8):
        row = frame(t)
        row["height"] = 1200
        for point in row["tracks"][0]["keypoints17"]:
            point[1] += 200
        state.update(row)
    events = []
    for t in (1, 1.2, 1.4, 1.6):
        events += state.update(frame(t))
    assert not events


def test_seated_foot_motion_into_board_roi_is_not_board_arrival():
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    events = []
    for i, x in enumerate((100, 100, 100, 200, 350, 450, 450, 450, 450, 450, 450, 450)):
        events += state.update(frame(i * 0.2, seated=True, x=x))
    assert not events
    assert action(state, "move_to_board")["status"] != "completed"
    assert state.finish() == []
    assert state.observation_counts["incomplete_board_candidates"] == 1


@pytest.mark.parametrize("reason", ["source_end", "cancelled", "failed"])
def test_finish_drops_unconfirmed_transitions_without_synthetic_completed_events(reason):
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    for t in (0, 0.2, 0.4):
        state.update(frame(t, seated=True, x=100))
    state.update(frame(0.6, x=500))
    assert action(state, "stand_up")["status"] == "candidate"
    assert action(state, "move_to_board")["status"] == "candidate"
    assert state.finish(100, reason=reason) == []
    assert state.people == []
    assert state.observation_counts["incomplete_stand_candidates"] == 1
    assert state.observation_counts["incomplete_board_candidates"] == 1


def test_board_approach_timeout_and_recent_completion_unknown_reset():
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98], config={"max_sequence_s": 0.5})
    for t in (0, 0.2, 0.4):
        state.update(frame(t, x=300))
    for t in (0.6, 0.8, 1, 1.2):
        assert state.update(frame(t, x=500)) == []
    assert state.observation_counts["incomplete_board_candidates"] == 1
    assert action(state, "move_to_board")["status"] == "no_suggestion"
    state = api().BasicPoseActionState([0.7, 0.72, 0.95, 0.98])
    for i, x in enumerate((300, 300, 300, 400, 550, 750, 750, 750, 750, 750, 750)):
        state.update(frame(i * 0.2, x=x))
    assert action(state, "move_to_board")["status"] == "completed"
    unknown = frame(2.2, x=750)
    unknown["tracks"][0]["keypoints17"][16][2] = 0.1
    state.update(unknown)
    assert action(state, "move_to_board")["status"] == "unsupported"
    state.update(frame(2.4, x=750))
    assert action(state, "move_to_board")["status"] == "no_suggestion"
