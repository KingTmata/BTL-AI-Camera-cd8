"""Synthetic quality/causality contracts; thresholds are not classroom calibration."""

import copy
import importlib

import pytest


def api():
    return importlib.import_module("classroom_ai.video.pose_quality")


def track(x=0):
    points = [[50 + x, 40, 1] for _ in range(17)]
    for i, point in {
        5: [40 + x, 80, 1],
        6: [60 + x, 80, 1],
        7: [35 + x, 100, 1],
        8: [65 + x, 100, 1],
        9: [30 + x, 120, 1],
        10: [70 + x, 120, 1],
        11: [40 + x, 140, 1],
        12: [60 + x, 140, 1],
    }.items():
        points[i] = point
    return dict(
        state="observed", confirmed=True, keypoints17=points, bbox_xyxy=[x, 0, x + 100, 200]
    )


UPPER = (0, 5, 6, 7, 8, 9, 10)


@pytest.mark.parametrize("bad", [-1, 401, float("nan"), float("inf"), True])
def test_invalid_required_geometry_is_unsupported_even_with_temporal_guard_off(bad):
    gate = api().PoseQualityGate()
    row = track()
    row["keypoints17"][9][0] = bad
    result = gate.check((0, 1), row, UPPER, t=0, confidence=0.5, bounds=(400, 400))
    assert result["status"] == "unsupported"
    assert result["gate_version"] == "causal_pose_quality_v1"
    assert result["temporal_guard"] is False


def test_temporal_guard_uses_elapsed_pts_robust_scale_and_rebaselines_after_outlier():
    gate = api().PoseQualityGate({"temporal_guard": True, "max_joint_speed": 1})
    a, b = track(), track()
    b["keypoints17"][9][0] += 50
    original = copy.deepcopy(b)
    args = dict(confidence=0.5, bounds=(400, 400))
    assert gate.check((0, 1), a, UPPER, t=0, **args)["status"] == "supported"
    assert gate.check((0, 1), b, UPPER, t=0.1, **args)["reason"] == "temporal_outlier"
    assert gate.check((0, 1), b, UPPER, t=0.2, **args)["status"] == "supported"
    slower = api().PoseQualityGate({"temporal_guard": True, "max_joint_speed": 1})
    slower.check((0, 1), a, UPPER, t=0, **args)
    assert slower.check((0, 1), b, UPPER, t=0.3, **args)["status"] == "supported"
    assert b == original


def test_common_translation_and_nearly_overlapping_shoulders_do_not_make_false_jump():
    gate = api().PoseQualityGate({"temporal_guard": True, "max_joint_speed": 0.1})
    a, b = track(), track(100)
    for row in (a, b):
        row["keypoints17"][6][0] = row["keypoints17"][5][0] + 0.01
    args = dict(confidence=0.5, bounds=(400, 400))
    gate.check((0, 1), a, UPPER, t=0, **args)
    assert gate.check((0, 1), b, UPPER, t=0.1, **args)["status"] == "supported"


def test_quality_is_action_specific_and_retains_only_current_actor_state():
    gate = api().PoseQualityGate({"temporal_guard": True})
    row = track()
    row["keypoints17"][15] = [float("nan"), 0, 0]
    args = dict(confidence=0.5, bounds=(400, 400))
    assert gate.check((0, 1), row, UPPER, t=0, **args)["status"] == "supported"
    gate.retain(set())
    row["keypoints17"][9][0] += 100
    assert gate.check((0, 1), row, UPPER, t=0.01, **args)["status"] == "supported"
    assert gate.state_count == 1
    gate.clear()
    assert gate.state_count == 0


@pytest.mark.parametrize(
    "config",
    [{"temporal_guard": "false"}, {"max_joint_speed": 0}, {"max_scale_ratio": 1}, {"extra": 1}],
)
def test_quality_config_is_strict_and_detached(config):
    with pytest.raises(ValueError):
        api().PoseQualityGate(config)
