"""Cached confidence and tiled inference experiments, pinned to Ultralytics 8.4.165."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywh2xyxy


def tile_starts(length, size, overlap=0.25):
    if length <= size:
        return [0]
    stride = max(1, round(size * (1 - overlap)))
    return sorted(set(list(range(0, length - size + 1, stride)) + [length - size]))


def merge_boxes(predictions, iou=0.7, max_det=300):
    if not len(predictions):
        return predictions.reshape(0, 6)
    data = torch.as_tensor(predictions, dtype=torch.float32)
    keep = batched_nms(data[:, :4], data[:, 4], data[:, 5], iou)[:max_det]
    return data[keep].numpy()


def ground_truth(label, width, height):
    text = Path(label).read_text(encoding="utf-8").strip()
    labels = np.array(
        [list(map(float, line.split())) for line in text.splitlines()], dtype=np.float32
    ).reshape(-1, 5)
    boxes = xywh2xyxy(labels[:, 1:5] * np.array([width, height, width, height]))
    return labels[:, 0], boxes.astype(np.float32)


def match(predictions, classes, boxes):
    validator = object.__new__(BaseValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    pred = torch.as_tensor(predictions, dtype=torch.float32)
    targets = torch.as_tensor(boxes, dtype=torch.float32)
    return validator.match_predictions(
        pred[:, 5], torch.as_tensor(classes), box_iou(targets, pred[:, :4])
    ).numpy()


def summarize_cache(
    paths, names, merge_iou=None, thresholds=(0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
):
    """AP uses all predictions >=.001. Fixed confidence P/R/F1 are separately re-matched at IoU .5."""
    metrics = DetMetrics(names=dict(enumerate(names)))
    counts = {threshold: np.zeros((len(names), 3), dtype=np.int64) for threshold in thresholds}
    latencies = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            pred, classes, boxes = cache["pred"], cache["classes"], cache["boxes"]
            elapsed = float(cache["seconds"])
        before_merge = time.perf_counter()
        if merge_iou is not None:
            pred = merge_boxes(pred, merge_iou)
        latencies.append(elapsed + time.perf_counter() - before_merge)
        correct = match(pred, classes, boxes)
        metrics.update_stats(
            dict(
                tp=correct,
                conf=pred[:, 4],
                pred_cls=pred[:, 5],
                target_cls=classes,
                target_img=np.unique(classes),
                im_name=path.stem,
            )
        )
        target_count = np.bincount(classes.astype(int), minlength=len(names))
        for threshold in thresholds:
            selected = pred[pred[:, 4] >= threshold]
            tp_mask = match(selected, classes, boxes)[:, 0]
            predicted = np.bincount(selected[:, 5].astype(int), minlength=len(names))
            tp = np.bincount(selected[tp_mask, 5].astype(int), minlength=len(names))
            counts[threshold] += np.stack([tp, predicted - tp, target_count - tp], axis=1)
    metrics.process(plot=False)
    operating_points = []
    for threshold, values in counts.items():
        tp, fp, fn = values.T
        p = np.divide(tp, tp + fp, out=np.zeros(len(names)), where=(tp + fp) > 0)
        r = np.divide(tp, tp + fn, out=np.zeros(len(names)), where=(tp + fn) > 0)
        f1 = np.divide(2 * p * r, p + r, out=np.zeros(len(names)), where=(p + r) > 0)
        totals = values.sum(axis=0)
        micro_p = float(totals[0] / max(totals[0] + totals[1], 1))
        micro_r = float(totals[0] / max(totals[0] + totals[2], 1))
        operating_points.append(
            dict(
                confidence=threshold,
                match_iou=0.5,
                precision_macro=float(p.mean()),
                recall_macro=float(r.mean()),
                f1_macro=float(f1.mean()),
                precision_micro=micro_p,
                recall_micro=micro_r,
                f1_micro=2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12),
                per_class=[
                    dict(
                        name=n,
                        precision=float(a),
                        recall=float(b),
                        f1=float(c),
                        TP=int(v[0]),
                        FP=int(v[1]),
                        FN=int(v[2]),
                    )
                    for n, a, b, c, v in zip(names, p, r, f1, values)
                ],
            )
        )
    return dict(
        mAP50=float(metrics.box.map50),
        mAP50_95=float(metrics.box.map),
        validator_precision_macro=float(metrics.box.mp),
        validator_recall_macro=float(metrics.box.mr),
        validator_f1_macro=float(metrics.box.f1.mean()),
        per_class_ap=json.loads(json.dumps(metrics.summary(), default=lambda v: v.item())),
        confidence_sweep=operating_points,
        mean_seconds=float(np.mean(latencies)),
        p95_seconds=float(np.percentile(latencies, 95)),
        images=len(paths),
        merge_iou=merge_iou,
        ap_confidence_floor=0.001,
        max_det=300,
        note="Single-image inference evaluator; compare tile/full within this protocol, separately from native batch8 validation.",
    )


def inference_campaign(weights, records, data_dir, output, names, deadline):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(weights))
    # Single-image rectangular preprocessing is intentionally recorded separately from native batch8 val.
    detector.predict(
        np.zeros((1280, 1280, 3), np.uint8),
        imgsz=1280,
        device=0,
        conf=0.001,
        rect=True,
        quantize=32,
        max_det=300,
        verbose=False,
    )
    cached = {mode: [] for mode in ("full", "tile512", "tile768")}
    try:
        for index, record in enumerate(records):
            if time.monotonic() >= deadline:
                raise TimeoutError("Campaign wall-clock budget reached during inference")
            image = cv2.imread(str(Path(data_dir) / record["image_path"]))
            if image is None:
                raise ValueError("Unreadable validation image: " + record["image_path"])
            height, width = image.shape[:2]
            classes, boxes = ground_truth(Path(data_dir) / record["label_path"], width, height)
            torch.cuda.synchronize()
            started = time.perf_counter()
            native = (
                detector.predict(
                    image,
                    imgsz=1280,
                    device=0,
                    conf=0.001,
                    rect=True,
                    quantize=32,
                    max_det=300,
                    verbose=False,
                )[0]
                .boxes.data.cpu()
                .numpy()
            )
            torch.cuda.synchronize()
            full_seconds = time.perf_counter() - started
            for mode, tile in [("full", None), ("tile512", 512), ("tile768", 768)]:
                predictions = [native]
                seconds = full_seconds
                if tile is not None:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for y in tile_starts(height, tile):
                        for x in tile_starts(width, tile):
                            if x == y == 0 and width <= tile and height <= tile:
                                continue  # identical full frame, no extra prediction
                            crop = image[y : min(y + tile, height), x : min(x + tile, width)]
                            found = (
                                detector.predict(
                                    crop,
                                    imgsz=1280,
                                    device=0,
                                    conf=0.001,
                                    rect=True,
                                    quantize=32,
                                    max_det=300,
                                    verbose=False,
                                )[0]
                                .boxes.data.cpu()
                                .numpy()
                                .copy()
                            )
                            found[:, [0, 2]] += x
                            found[:, [1, 3]] += y
                            predictions.append(found)
                    torch.cuda.synchronize()
                    seconds += time.perf_counter() - started
                folder = output / "prediction_cache" / mode
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{index:05d}.npz"
                np.savez_compressed(
                    path,
                    pred=np.concatenate(predictions),
                    classes=classes,
                    boxes=boxes,
                    seconds=seconds,
                )
                cached[mode].append(path)
            if (index + 1) % 100 == 0:
                print("INFERENCE_PROGRESS", index + 1, "/", len(records), flush=True)
        result = {}
        for mode, paths in cached.items():
            for merge_iou in [None, 0.6, 0.7, 0.8] if mode == "full" else [0.6, 0.7, 0.8]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Campaign wall-clock budget reached during metrics")
                label = mode + "_merge" + str(merge_iou)
                result[label] = summarize_cache(paths, names, merge_iou)
                (output / "inference_comparison.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                print(
                    "INFERENCE_METRICS",
                    label,
                    {k: v for k, v in result[label].items() if isinstance(v, (float, int))},
                    flush=True,
                )
        return result
    finally:
        del detector
        torch.cuda.empty_cache()
