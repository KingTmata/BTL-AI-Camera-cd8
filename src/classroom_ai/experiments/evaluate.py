"""COCO bbox evaluator shared by Colab baseline and fine-tuned checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NAMES = ["person", "table", "chair", "laptop", "cell phone", "backpack", "book", "cup"]
COCO_MAP = {0: 0, 60: 1, 56: 2, 63: 3, 67: 4, 24: 5, 73: 6, 41: 7}


def evaluate(manifest: Path, weights: Path, output: Path, device: str) -> dict:
    import numpy as np
    from ultralytics import YOLO

    if output.exists():
        raise ValueError(f"Output already exists; choose a new name: {output}")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "val"]
    rows.sort(key=lambda row: row["image_id"])
    if not rows:
        raise ValueError("Validation manifest has no images.")
    model = YOLO(str(weights))
    names = [model.names[i] for i in range(len(model.names))]
    if names == NAMES:
        class_map = {i: i for i in range(8)}
    elif len(names) == 80 and all(
        model.names[i] == n
        for i, n in {
            0: "person",
            60: "dining table",
            56: "chair",
            63: "laptop",
            67: "cell phone",
            24: "backpack",
            73: "book",
            41: "cup",
        }.items()
    ):
        class_map = COCO_MAP
    else:
        raise ValueError("Weights must match project8 or the known COCO80 mapping.")
    images, annotations, predictions = [], [], []
    annotation_id = 1
    for image_index, row in enumerate(rows, 1):
        width, height = int(row["width"]), int(row["height"])
        images.append(
            {"id": image_index, "file_name": row["image_path"], "width": width, "height": height}
        )
        label_path = manifest.parent / row["label_path"]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, x, y, w, h = line.split()
            cls_id = int(cls)
            if not 0 <= cls_id < 8:
                raise ValueError(f"Invalid project8 class in {label_path}")
            w_px, h_px = float(w) * width, float(h) * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index,
                    "category_id": cls_id + 1,
                    "bbox": [
                        (float(x) - float(w) / 2) * width,
                        (float(y) - float(h) / 2) * height,
                        w_px,
                        h_px,
                    ],
                    "area": w_px * h_px,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        result = model.predict(
            str(manifest.parent / row["image_path"]),
            imgsz=640,
            conf=0.001,
            iou=0.70,
            nms=True,
            max_det=300,
            agnostic_nms=False,
            augment=False,
            classes=list(class_map),
            device=device,
            verbose=False,
            save=False,
        )[0]
        boxes = result.boxes
        for box, score, cls in zip(
            boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
        ):
            project_id = class_map[int(cls)]
            x1, y1, x2, y2 = box
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)) or x2 <= x1 or y2 <= y1:
                continue
            predictions.append(
                {
                    "image_id": image_index,
                    "category_id": project_id + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    actual_end2end = bool(model.predictor.model.end2end)
    if actual_end2end:
        raise ValueError("Protocol yêu cầu NMS=True nhưng model vẫn chạy end-to-end/NMS-free.")
    gt = COCO()
    gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)],
    }
    gt.createIndex()
    pred = gt.loadRes(predictions) if predictions else COCO()
    if not predictions:
        pred.dataset = {"images": images, "annotations": [], "categories": gt.dataset["categories"]}
        pred.createIndex()
    evaluator = COCOeval(gt, pred, "bbox")
    evaluator.params.imgIds = [item["id"] for item in images]
    evaluator.params.catIds = [item["id"] for item in gt.dataset["categories"]]
    evaluator.params.iouThrs = np.array([0.5 + 0.05 * i for i in range(10)])
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    def mean(values):
        values = values[values >= 0]
        return float(values.mean()) if values.size else None

    precision = evaluator.eval["precision"]
    per_class = {
        name: {
            "AP50": mean(precision[0, :, i, 0, -1]),
            "AP50_95": mean(precision[:, :, i, 0, -1]),
            "GT_boxes": sum(a["category_id"] == i + 1 for a in annotations),
        }
        for i, name in enumerate(NAMES)
    }
    report = {
        "protocol": {
            "task": "bbox",
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.70,
            "nms": True,
            "max_det": 300,
            "agnostic_nms": False,
            "augment": False,
            "split": "val",
        },
        "images": len(images),
        "mAP50": mean(precision[0, :, :, 0, -1]),
        "mAP50_95": mean(precision[:, :, :, 0, -1]),
        "per_class": per_class,
        "checkpoint_sha256": __import__("hashlib").sha256(weights.read_bytes()).hexdigest(),
        "actual_end2end": actual_end2end,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.manifest, args.weights, args.output, args.device), indent=2))
