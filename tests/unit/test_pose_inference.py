"""Pose contracts exercised without loading model weights or running inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from classroom_ai.inference import pose


def points(score=0.8):
    return [[float(i + 10), float(i + 20), score] for i in range(17)]


def result(poses=None, boxes=None, scores=None, classes=None):
    poses = [points()] if poses is None else poses
    count = len(poses)
    boxes = [[0, 0, 60, 60] for _ in poses] if boxes is None else boxes
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            conf=np.asarray(scores if scores is not None else [0.9] * count),
            cls=np.asarray(classes if classes is not None else [0] * count),
        ),
        keypoints=SimpleNamespace(data=np.asarray(poses, dtype=float).reshape(count, 17, 3)),
    )


@pytest.fixture
def engine(tmp_path, monkeypatch):
    import ultralytics

    weights = tmp_path / "pose.pt"
    weights.touch()
    native = SimpleNamespace(yaml={"kpt_shape": [17, 3]})
    model = SimpleNamespace(task="pose", names={0: "person"}, model=native)
    model.predict = lambda *args, **kwargs: [result()]
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: model)
    return pose.PoseInspector(weights), model


def test_head_uses_median_of_confident_face_and_keeps_overlapping_body_points():
    raw = points()
    raw[:5] = [[10, 20, 0.5], [90, 80, 0.9], [30, 60, 0.7], [999, 999, 0.49], [1, 1, 0.1]]
    raw[5] = raw[6] = [22, 33, 0.05]
    output = pose.coco17_to_pose13(raw)
    assert output[0] == pytest.approx([30, 60, 0.7])
    assert output[1:3] == [[22, 33, 0.05], [22, 33, 0.05]]
    assert output[-1] == raw[16]


def test_head_falls_back_to_all_five_face_points_and_masks_below_threshold():
    raw = points(0.049)
    raw[:5] = [[10, 50, 0.1], [20, 40, 0.2], [30, 30, 0.3], [40, 20, 0.4], [50, 10, 0.05]]
    output = pose.coco17_to_pose13(raw)
    assert output[0] == pytest.approx([30, 30, 0.21])
    assert output[1:] == [None] * 12
    assert pose.coco17_to_pose13(points(0.01)) == [None] * 13


def test_derived_points_match_existing_label_studio_canonical_adapter():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / (
        "tools/label_studio/work/label-studio-ml-backend/"
        "label_studio_ml/examples/yolo/control_models/pose13.py"
    )
    spec = importlib.util.spec_from_file_location("canonical_pose13", path)
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    raw = points()
    raw[:5] = [[10, 80, 0.5], [30, 60, 0.9], [100, 100, 0.1], [200, 200, 0.2], [300, 300, 0.3]]
    raw[5] = raw[6] = [25, 40, 0.05]
    raw[7] = [20, 30, 0.049]
    regions = canonical.pose13_results([[0, 0, 400, 400]], [0.9], [raw], 400, 400)
    by_name = {r["value"]["keypointlabels"][0]: r for r in regions if r["type"] == "keypointlabels"}
    converted = pose.coco17_to_pose13(raw)
    for index, name in enumerate(canonical.POSE13_NAMES):
        if name not in by_name:
            assert converted[index] is None
        else:
            region = by_name[name]
            assert converted[index] == pytest.approx(
                [region["value"]["x"] * 4, region["value"]["y"] * 4, region["score"]]
            )


@pytest.mark.parametrize(
    "bad",
    [
        points()[:13],
        [[1, 2]] * 17,
        [[float("nan"), 2, 0.5]] * 17,
        [[1, 2, float("inf")]] * 17,
        [[1, 2, 1.01]] * 17,
        [[1, 2, -0.01]] * 17,
    ],
)
def test_conversion_rejects_wrong_profile_and_nonfinite_data(bad):
    with pytest.raises(ValueError):
        pose.coco17_to_pose13(bad)


def test_prediction_contract_preserves_raw_points_and_unknown_visibility(engine):
    inspector, _ = engine
    output = inspector.predict(np.zeros((80, 80, 3), dtype=np.uint8), 0.25, imgsz=1280)
    row = output["detections"][0]
    assert row["class_id"] == row["model_class_id"] == 0
    assert row["class_name"] == "person"
    assert row["bbox_xyxy"] == [0, 0, 60, 60]
    assert row["confidence"] == 0.9
    assert row["keypoints17"] == points()
    assert row["keypoints13"][0] == [12, 22, 0.8]
    assert row["keypoints13_missing"] == [False] * 13
    assert row["visibility"] is None
    assert output["device"] == "cpu"
    assert output["processing_ms"] >= 0
    assert output["inspector_lock_wait_ms"] >= 0


def test_nms_selects_whole_person_rows_without_mutating_results(engine):
    inspector, model = engine
    raw = result(
        [points(0.6), points(0.7), points(0.8)],
        [[0, 0, 40, 40], [1, 1, 41, 41], [45, 45, 70, 70]],
        [0.5, 0.9, 0.8],
    )
    model.predict = lambda *args, **kwargs: [raw]
    rows = inspector.predict(np.zeros((80, 80, 3)), 0.25, imgsz=640, extra_nms_iou=0.5)[
        "detections"
    ]
    assert [r["confidence"] for r in rows] == [0.9, 0.8]
    assert rows[0]["bbox_xyxy"] == [1, 1, 41, 41]
    assert rows[0]["keypoints17"] == points(0.7)
    assert rows[1]["keypoints17"] == points(0.8)
    assert len(raw.boxes.xyxy) == len(raw.keypoints.data) == 3


def test_empty_predictions_are_allowed(engine):
    inspector, model = engine
    model.predict = lambda *args, **kwargs: [result([])]
    assert inspector.predict(np.zeros((80, 80, 3)), 0.25, imgsz=640)["detections"] == []


def test_missing_mask_is_derived_without_erasing_raw_model_scores(engine):
    inspector, model = engine
    raw = points(0.8)
    raw[5] = [10, 20, 0.049]
    raw[6] = [10, 20, 0.05]
    model.predict = lambda *args, **kwargs: [result([raw])]
    row = inspector.predict(np.zeros((80, 80, 3)), 0.25, imgsz=640)["detections"][0]
    assert row["keypoints17"][5] == [10, 20, 0.049]
    assert row["keypoints13"][1] is None
    assert row["keypoints13"][2] == [10, 20, 0.05]
    assert row["keypoints13_missing"] == [False, True] + [False] * 11
    assert row["visibility"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "detect"),
        ("names", {0: "car"}),
        ("names", {0: "person", 1: "head"}),
        ("model", SimpleNamespace(yaml={"kpt_shape": [13, 3]})),
        ("model", SimpleNamespace(yaml={"kpt_shape": [17, 2]})),
    ],
)
def test_constructor_rejects_wrong_task_names_and_keypoint_profile(
    tmp_path, monkeypatch, field, value
):
    import ultralytics

    weights = tmp_path / "wrong.pt"
    weights.touch()
    model = SimpleNamespace(
        task="pose", names={0: "person"}, model=SimpleNamespace(yaml={"kpt_shape": [17, 3]})
    )
    setattr(model, field, value)
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: model)
    with pytest.raises(ValueError):
        pose.PoseInspector(weights)


def test_missing_weights_fail_before_loading(tmp_path, monkeypatch):
    import ultralytics

    def forbidden(path):
        pytest.fail("Missing local weights must never enter the download-capable loader")

    monkeypatch.setattr(ultralytics, "YOLO", forbidden)
    with pytest.raises(FileNotFoundError):
        pose.PoseInspector(tmp_path / "missing.pt")


def test_loading_existing_weights_disables_install_and_uses_absolute_local_path(
    tmp_path, monkeypatch
):
    import os

    import ultralytics
    from ultralytics.utils import checks

    monkeypatch.setattr(checks, "AUTOINSTALL", True)
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    weights = tmp_path / "yolo26n-pose.pt"
    weights.touch()

    def local_loader(path):
        if not Path(path).is_absolute() or not Path(path).is_file():
            raise RuntimeError("Loader could enter download path")
        if checks.AUTOINSTALL or os.environ["YOLO_AUTOINSTALL"] != "false":
            raise RuntimeError("Loader could enter automatic install path")
        return SimpleNamespace(
            task="pose", names={0: "person"}, model=SimpleNamespace(yaml={"kpt_shape": [17, 3]})
        )

    from pathlib import Path

    monkeypatch.setattr(ultralytics, "YOLO", local_loader)
    inspector = pose.PoseInspector(weights)
    assert inspector.weights == weights.resolve()


@pytest.mark.parametrize("confidence", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_rejects_before_inference(engine, confidence):
    inspector, model = engine
    model.predict = lambda *args, **kwargs: pytest.fail("Invalid confidence reached inference")
    with pytest.raises(ValueError):
        inspector.predict(np.zeros((80, 80, 3)), confidence, imgsz=640)


@pytest.mark.parametrize(
    "options",
    [
        {"imgsz": 0},
        {"imgsz": True},
        {"passes": 3},
        {"extra_nms_iou": float("nan")},
        {"extra_nms_iou": 1.1},
    ],
)
def test_invalid_controls_reject_before_inference(engine, options):
    inspector, model = engine
    model.predict = lambda *args, **kwargs: pytest.fail("Invalid controls reached inference")
    with pytest.raises(ValueError):
        inspector.predict(np.zeros((80, 80, 3)), 0.25, **options)


@pytest.mark.parametrize(
    "corrupt", ["no_keypoints", "count", "profile", "point_nan", "class", "box_nan", "score_nan"]
)
def test_malformed_result_is_rejected(engine, corrupt):
    inspector, model = engine
    raw = result()
    if corrupt == "no_keypoints":
        raw.keypoints = None
    elif corrupt == "count":
        raw.keypoints.data = np.zeros((0, 17, 3))
    elif corrupt == "profile":
        raw.keypoints.data = np.zeros((1, 13, 3))
    elif corrupt == "point_nan":
        raw.keypoints.data[0, 0, 0] = float("nan")
    elif corrupt == "class":
        raw.boxes.cls[0] = 1
    elif corrupt == "box_nan":
        raw.boxes.xyxy[0, 0] = float("nan")
    else:
        raw.boxes.conf[0] = float("nan")
    model.predict = lambda *args, **kwargs: [raw]
    with pytest.raises(ValueError):
        inspector.predict(np.zeros((80, 80, 3)), 0.25, imgsz=640)


def test_drawing_never_joins_people_or_missing_points_and_copies_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    first = [None] * 13
    second = [None] * 13
    first[1:3] = [[10, 10, 0.8], [20, 10, 0.8]]
    second[1:3] = [[70, 70, 0.8], [80, 70, 0.8]]
    drawn = pose.draw_pose(image, [{"keypoints13": first}, {"keypoints13": second}])
    assert image.sum() == 0
    assert drawn[10, 15].any() and drawn[70, 75].any()
    assert not drawn[40, 40].any()
    assert not drawn[0, 0].any()


def test_native_drawing_prefers_display_points_while_raw_rows_still_draw_raw():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    raw, shown = [None] * 13, [None] * 13
    raw[0] = [10, 10, 0.8]
    shown[0] = [40, 40, 0.8]
    drawn = pose.draw_pose(image, [{"keypoints13": raw, "display_keypoints13": shown}])
    assert drawn[40, 40].any()
    assert not drawn[10, 10].any()
    assert pose.draw_pose(image, [{"keypoints13": raw}])[10, 10].any()
    assert image.sum() == 0
