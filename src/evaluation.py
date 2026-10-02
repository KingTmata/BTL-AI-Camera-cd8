"""Common COCO bbox evaluation for COCO80 and project8 checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import io
import json
import math
import platform
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.dataset import NAMES, ROOT, labels, read_manifest, resolve, sha256, validate, verify_release
from src.inference.detector import class_mapping

PROTOCOL = {"task": "bbox", "imgsz": 640, "conf": .001, "iou": .7, "nms": True,
            "max_det": 300, "agnostic_nms": False, "augment": False,
            "iou_thresholds": [round(.5 + .05*i, 2) for i in range(10)],
            "maxDets": [1, 10, 100], "categories": list(NAMES)}


def ground_truth(rows, root=ROOT):
    images, annotations = [], []
    for image_id, row in enumerate(rows, 1):
        width, height = int(row["width"]), int(row["height"])
        images.append({"id": image_id, "file_name": row["image_path"], "width": width, "height": height})
        for category, x, y, w, h in labels(resolve(row["label_path"], root)):
            box = [(x-w/2)*width, (y-h/2)*height, w*width, h*height]
            annotations.append({"id": len(annotations)+1, "image_id": image_id, "category_id": category,
                                "bbox": box, "area": box[2]*box[3], "iscrowd": 0})
    return {"info": {"description": "project8 bbox evaluation"}, "images": images,
            "annotations": annotations, "categories": [{"id": i, "name": name} for i, name in enumerate(NAMES)]}


def map_predictions(image_id, xyxy, scores, raw_ids, mapping):
    predictions = []
    for box, score, raw_id in zip(xyxy, scores, raw_ids):
        if int(raw_id) not in mapping:
            continue
        x1, y1, x2, y2 = map(float, box)
        score = float(score)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)) or x2 <= x1 or y2 <= y1 or not 0 <= score <= 1:
            raise ValueError("Invalid predicted bbox or score")
        predictions.append({"image_id": image_id, "category_id": mapping[int(raw_id)],
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
    return predictions


def score_predictions(gt, predictions):
    image_ids = {image["id"] for image in gt["images"]}
    for prediction in predictions:
        if prediction["image_id"] not in image_ids or prediction["category_id"] not in range(len(NAMES)):
            raise ValueError("Prediction references unknown image/category")
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = copy.deepcopy(gt)
        truth.createIndex()
        if predictions:
            detected = truth.loadRes(copy.deepcopy(predictions))
        else:
            detected = COCO()
            detected.dataset = {"images": copy.deepcopy(gt["images"]), "categories": copy.deepcopy(gt["categories"]), "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = list(range(len(NAMES)))
        evaluator.params.iouThrs = np.array(PROTOCOL["iou_thresholds"])
        evaluator.params.maxDets = PROTOCOL["maxDets"]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    def mean(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None
    precision = evaluator.eval["precision"]  # IoU, recall, class, area, maxDet
    per_class = {}
    for index, name in enumerate(NAMES):
        per_class[name] = {"AP50_95": mean(precision[:, :, index, 0, -1]),
                           "AP50": mean(precision[0, :, index, 0, -1]),
                           "GT_boxes": sum(a["category_id"] == index for a in gt["annotations"])}
    return {"mAP50_95": mean(precision[:, :, :, 0, -1]), "mAP50": mean(precision[0, :, :, 0, -1]),
            "per_class": per_class, "images": len(gt["images"]), "GT_boxes": len(gt["annotations"]),
            "negative_images": sum(not any(a["image_id"] == i for a in gt["annotations"]) for i in image_ids),
            "predictions": len(predictions)}


def evaluate(manifest, weights, output, split="val", device="cpu", nms=True, root=ROOT):
    from ultralytics import YOLO
    manifest, weights, output = Path(manifest), Path(weights), Path(output)
    if not weights.is_file():
        raise ValueError("Checkpoint does not exist; download/copy it explicitly first")
    if output.exists():
        raise ValueError("Use a new output directory for every evaluation")
    release_hash = verify_release(manifest.parent)
    checkpoint_hash = sha256(weights)
    report, rows = validate(read_manifest(manifest), root, release=True)
    if not report["valid"]:
        raise ValueError("\n".join(report["errors"]))
    rows = sorted((row for row in rows if row["split"] == split), key=lambda row: row["image_id"])
    if not rows:
        raise ValueError(f"No images in {split}")
    gt = ground_truth(rows, root)
    model = YOLO(str(weights))
    mapping = class_mapping(model.names)
    config = {**PROTOCOL, "nms": nms, "device": device}
    predict_args = {key: config[key] for key in ("imgsz", "conf", "iou", "nms", "max_det", "agnostic_nms", "augment", "device")}
    predictions = []
    for index, row in enumerate(rows, 1):
        result = model.predict(source=str(resolve(row["image_path"], root)), classes=list(mapping),
                               verbose=False, save=False, **predict_args)[0]
        predictions.extend(map_predictions(index, result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                           result.boxes.cls.cpu().tolist(), mapping))
    if verify_release(manifest.parent) != release_hash:
        raise ValueError("Dataset changed during evaluation")
    if sha256(weights) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation")
    metrics = score_predictions(gt, predictions)
    actual_end2end = bool(model.predictor.model.end2end)
    if actual_end2end != (not nms):
        raise ValueError("Loaded head does not match requested NMS protocol")
    metadata = {"protocol": config, "split": split, "checkpoint": str(weights.resolve()),
                "checkpoint_sha256": checkpoint_hash, "release_sha256": release_hash,
                "manifest_sha256": sha256(manifest), "class_mapping": mapping,
                "model_profile": "coco80" if len(model.names) == 80 else "project8",
                "class_scope_note": "COCO dining table -> table is a proxy; group table includes other table types. book includes notebooks by group annotation policy.",
                "actual_end2end": actual_end2end,
                "versions": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "pycocotools", "numpy")},
                "python": platform.python_version(), "platform": platform.platform()}
    output.mkdir(parents=True)
    for name, value in (("ground_truth.json", gt), ("predictions.json", predictions), ("metrics.json", metrics), ("run.json", metadata)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset/v1/manifest.csv")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo26n.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nms", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.manifest, args.weights, args.output, args.split, args.device, args.nms), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
