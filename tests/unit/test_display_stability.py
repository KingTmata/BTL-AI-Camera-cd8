import copy
import inspect

from classroom_ai.video.display_stability import PoseDisplayStabilizer


def test_all_new_video_exports_default_to_stable_display_with_explicit_opt_out():
    from classroom_ai.interfaces.web import VideoExportInput
    from classroom_ai.video.rendering import export_saved_actions

    assert VideoExportInput().stabilize_display is True
    assert VideoExportInput(stabilize_display=False).stabilize_display is False
    assert inspect.signature(export_saved_actions).parameters["stabilize_display"].default is True


def track(x=0, uncertain=False):
    return dict(
        track_id=1,
        state="observed",
        confirmed=True,
        identity_uncertain=uncertain,
        bbox_xyxy=[x, 0, x + 100, 200],
        keypoints13=[[x + 30, 40, 0.9]] * 13,
        keypoints17=[[x + 30, 40, 0.9]] * 17,
    )


def test_visual_filter_attenuates_jitter_without_changing_raw_or_scores():
    state = PoseDisplayStabilizer()
    first = track()
    second = track(10)
    before = copy.deepcopy(second)
    state.update([first], segment=0, t=0)
    result = state.update([second], segment=0, t=1 / 30)[0]
    assert 0 < result["bbox_xyxy"][0] < 10
    assert 30 < result["display_keypoints13"][0][0] < 40
    assert result["keypoints17"] == second["keypoints17"]
    assert second == before


def test_missing_identity_uncertainty_scene_gap_and_jump_reset_visual_history():
    for reset in ("missing", "identity", "scene", "gap", "jump"):
        state = PoseDisplayStabilizer()
        state.update([track()], segment=0, t=0)
        if reset == "missing":
            state.update([], segment=0, t=0.01)
        row = track(500 if reset == "jump" else 10, reset == "identity")
        result = state.update(
            [row], segment=1 if reset == "scene" else 0, t=0.5 if reset == "gap" else 0.03
        )[0]
        assert result["bbox_xyxy"] == row["bbox_xyxy"], reset


def test_filter_is_causal_and_does_not_fill_missing_joints():
    state = PoseDisplayStabilizer()
    state.update([track()], segment=0, t=0)
    row = track(10)
    row["keypoints13"][0] = None
    result = state.update([row], segment=0, t=0.03)[0]
    assert result["display_keypoints13"][0] is None
    prefix = copy.deepcopy(result)
    state.update([track(20)], segment=0, t=0.06)
    assert result == prefix
