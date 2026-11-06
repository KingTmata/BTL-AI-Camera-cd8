import copy

import numpy as np
import pytest

from classroom_ai.video import rendering


def caption_person():
    quality = dict(
        status="supported",
        reason="valid_geometry",
        gate_version="causal_pose_quality_v1",
        temporal_guard=False,
    )
    common = dict(
        origin="rule", review="draft", calibrated=False, observed_through_s=2, pose_quality=quality
    )
    return dict(
        segment_id=0,
        track_id=7,
        observed_through_s=2,
        actions=[
            dict(
                common,
                label="raise_hand",
                status="no_suggestion",
                rule="coco17_wrist_above_nose_v1",
                pose_observation="not_raised",
                pose_observation_reason="both_wrists_lowered",
                pose_observation_scope="uncalibrated_raw_pose_geometry",
            ),
            dict(
                common,
                label="move_to_board",
                status="completed",
                rule="coco17_floor_roi_arrival_v1",
                start_s=0.6,
                confirmed_at_s=2,
            ),
        ],
        context_relations=[
            dict(
                relation="raise_hand_then_move_to_board",
                status="observed_sequence",
                rule="raise_hand_then_move_to_board_v1",
                origin="rule",
                review="draft",
                calibrated=False,
                raise_start_s=0,
                raise_confirmed_at_s=0.4,
                raise_observed_through_s=0.4,
                board_start_s=0.6,
                board_confirmed_at_s=2,
                max_separation_s=60,
                continuity="same_scene_track_continuous_supported_pose",
            )
        ],
    )


def test_explicit_supported_lowering_and_context_are_separate_draft_captions():
    person = caption_person()
    text = rendering.action_caption_text(person)
    assert "Tay đang hạ" in text and "Pose nháp" in text
    assert "Đã giơ tay → đến vùng bảng" in text and "Liên hệ:" in text
    assert "Đến vùng bảng" in text
    person["actions"][0]["pose_observation"] = "unknown"
    person["actions"][0]["pose_quality"] = dict(
        person["actions"][0]["pose_quality"], status="unsupported"
    )
    assert "Tay đang hạ" not in rendering.action_caption_text(person)
    assert "Chưa rõ tư thế tay" in rendering.action_caption_text(person)


@pytest.mark.parametrize(
    "field,value",
    [
        ("origin", "human"),
        ("review", "reviewed"),
        ("calibrated", True),
        ("pose_observation_scope", "guess"),
        ("rule", "invented"),
    ],
)
def test_negative_caption_requires_explicit_rule_geometry_provenance(field, value):
    person = caption_person()
    person["actions"][0][field] = value
    assert "Tay đang hạ" not in rendering.action_caption_text(person)


@pytest.mark.parametrize(
    "field,value",
    [
        ("relation", "invented"),
        ("origin", "human"),
        ("review", "reviewed"),
        ("calibrated", True),
        ("continuity", "same_number"),
        ("raise_start_s", True),
        ("raise_confirmed_at_s", 1),
        ("board_confirmed_at_s", 3),
        ("raise_observed_through_s", -1),
        ("max_separation_s", 61),
    ],
)
def test_context_caption_rejects_unknown_provenance_or_impossible_chronology(field, value):
    person = caption_person()
    person["context_relations"][0][field] = value
    assert "Đã giơ tay → đến vùng bảng" not in rendering.action_caption_text(person)


def test_caption_labels_are_current_rule_statuses_not_any_constant_label():
    person = {
        "actions": [
            {"label": "raise_hand", "status": "active"},
            {"label": "stand_up", "status": "completed"},
            {"label": "move_to_board", "status": "candidate"},
        ]
    }
    text = rendering.action_caption_text(person)
    assert "Giơ tay" in text and "Vừa đứng lên" in text and "Đang đến vùng bảng?" in text
    assert "nháp" in text
    for action in person["actions"]:
        action["status"] = "unsupported"
    assert "thiếu khớp" in rendering.action_caption_text(person)
    assert "Giơ tay" not in rendering.action_caption_text(person)


def test_render_caption_requires_same_person_scene_source_time_and_observation(monkeypatch):
    from PIL import ImageDraw, ImageFont

    default_font = ImageFont.load_default()
    monkeypatch.setattr(ImageFont, "truetype", lambda *a, **kw: default_font)
    texts = []
    original_text = ImageDraw.ImageDraw.text

    def text(self, xy, value, *args, **kwargs):
        texts.append(value)
        return original_text(self, xy, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    image = np.zeros((300, 500, 3), np.uint8)
    tracks = [
        {"track_id": 7, "bbox_xyxy": [50, 100, 250, 290], "state": "observed", "confirmed": True}
    ]
    person = {
        "segment_id": 0,
        "track_id": 7,
        "observed_through_s": 1,
        "actions": [{"label": "raise_hand", "status": "active"}],
    }
    captured = []
    original = rendering.action_caption_text

    def caption(row):
        captured.append(copy.deepcopy(row))
        return original(row)

    monkeypatch.setattr(rendering, "action_caption_text", caption)
    result = rendering.draw_pose_action_demo(
        image, tracks, [person], segment_id=0, source_time=1, font_path="unused"
    )
    assert captured == [person] and np.any(result) and not np.any(image)
    assert "THÀNH" not in texts[0]
    captured.clear()
    rendering.draw_pose_action_demo(
        image, tracks, [person], segment_id=1, source_time=1, font_path="unused"
    )
    assert captured == [None]
    captured.clear()
    rendering.draw_pose_action_demo(
        image, tracks, [person], segment_id=0, source_time=2, font_path="unused"
    )
    assert captured == [None]
    tracks[0]["confirmed"] = False
    captured.clear()
    rendering.draw_pose_action_demo(
        image, tracks, [person], segment_id=0, source_time=1, font_path="unused"
    )
    assert not captured
