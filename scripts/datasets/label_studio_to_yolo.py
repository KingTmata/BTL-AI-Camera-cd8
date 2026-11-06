"""Convert user-approved Label Studio image annotations to YOLO Detect or Pose13."""

import argparse
import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path

from PIL import Image

POSE13 = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
FLIP13 = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _check_dimensions(region, frame):
    if (region.get("original_width"), region.get("original_height")) != (
        frame["width"],
        frame["height"],
    ):
        raise ValueError(f"Task {frame['image_id']}: annotation dimensions differ from the image")
    if _number(region.get("image_rotation", 0), "image rotation") != 0:
        raise ValueError(f"Task {frame['image_id']}: image rotation is unsupported")


def _box(region, frame):
    _check_dimensions(region, frame)
    value = region.get("value", {})
    if value.get("rectanglelabels") != ["person"]:
        raise ValueError(f"Task {frame['image_id']}: unsupported bbox label")
    if _number(value.get("rotation", 0), "bbox rotation") != 0:
        raise ValueError(f"Task {frame['image_id']}: rotated bbox is unsupported")
    x, y, w, h = (_number(value.get(k), f"bbox {k}") for k in ("x", "y", "width", "height"))
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 100.000001 or y + h > 100.000001:
        raise ValueError(f"Task {frame['image_id']}: bbox is outside the image or empty")
    return [(x + w / 2) / 100, (y + h / 2) / 100, w / 100, h / 100]


def _label_lines(results, frame, task):
    if not isinstance(results, list):
        raise ValueError(f"Task {frame['image_id']}: annotation result must be a list")
    boxes = [region for region in results if region.get("type") == "rectanglelabels"]
    box_ids = [box.get("id") for box in boxes]
    if any(not ident for ident in box_ids) or len(box_ids) != len(set(box_ids)):
        raise ValueError(f"Task {frame['image_id']}: duplicate or missing bbox id")
    people = {ident: {} for ident in box_ids}
    if task == "pose13":
        for point in (region for region in results if region.get("type") == "keypointlabels"):
            _check_dimensions(point, frame)
            parent = point.get("parentID")
            if parent not in people:
                raise ValueError(f"Task {frame['image_id']}: keypoint parent is missing or invalid")
            value = point.get("value", {})
            names = value.get("keypointlabels", [])
            if len(names) != 1 or names[0] not in POSE13 or names[0] in people[parent]:
                raise ValueError(f"Task {frame['image_id']}: unknown or duplicate keypoint")
            visibility = value.get("visibility")
            if isinstance(visibility, bool) or visibility not in (0, 1, 2):
                raise ValueError(f"Task {frame['image_id']}: explicit visibility 0/1/2 is required")
            x, y = _number(value.get("x"), "keypoint x"), _number(value.get("y"), "keypoint y")
            if not (0 <= x <= 100 and 0 <= y <= 100):
                raise ValueError(f"Task {frame['image_id']}: keypoint outside image")
            people[parent][names[0]] = (x / 100, y / 100, visibility) if visibility else (0, 0, 0)
    lines = []
    for box in boxes:
        values = [0, *_box(box, frame)]
        if task == "pose13":
            for name in POSE13:
                values.extend(people[box["id"]].get(name, (0, 0, 0)))
        lines.append(" ".join(f"{number:.8g}" for number in values))
    return lines


def convert_dataset(tasks, manifest, review, split, images_dir, output, *, task="detect"):
    if task not in ("detect", "pose13"):
        raise ValueError("Unsupported task")
    frames = manifest.get("frames")
    approved = review.get("approved_image_ids")
    if not isinstance(frames, list) or not frames or not isinstance(approved, list):
        raise ValueError("Manifest and review scope are required")
    if not review.get("all_807_user_review_confirmed"):
        raise ValueError("User review confirmation is required")
    frame_by_id = {r["image_id"]: r for r in frames}
    if (
        len(frame_by_id) != len(frames)
        or len(approved) != len(set(approved))
        or set(approved) != set(frame_by_id)
    ):
        raise ValueError("Duplicate image IDs or review scope does not match the manifest")
    train, val = split.get("train_video_sha256"), split.get("val_video_sha256")
    if not train or not val or train == val:
        raise ValueError("Train and validation video split must use different videos")
    grouped = {}
    for item in tasks:
        image_id = item.get("data", {}).get("image_id")
        if image_id in frame_by_id:
            if image_id in grouped:
                raise ValueError(f"Duplicate task for image {image_id}")
            grouped[image_id] = item
    if set(grouped) != set(frame_by_id):
        raise ValueError("Some approved images have no exported annotation task")
    images_dir, output = Path(images_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    records = []
    for ident in approved:
        frame, item = frame_by_id[ident], grouped[ident]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_]+", ident):
            raise ValueError("Unsafe image ID")
        if frame.get("filename") != ident + ".jpg":
            raise ValueError(f"Invalid image filename: {ident}")
        for key in (
            "image_sha256",
            "source_video_sha256",
            "source_frame_index",
            "source_pts",
            "time_base",
        ):
            if item["data"].get(key) != frame.get(key):
                raise ValueError(f"Task {ident}: {key} does not match manifest")
        source = frame.get("source_video_sha256")
        if source not in (train, val):
            raise ValueError(f"Task {ident}: source video is not in the split")
        annotations = [a for a in item.get("annotations", []) if not a.get("was_cancelled", False)]
        if len(annotations) != 1:
            raise ValueError(f"Task {ident}: expected exactly one annotation")
        image = images_dir / frame["filename"]
        if not image.is_file():
            raise ValueError(f"Task {ident}: source image is missing")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != frame.get("image_sha256"):
            raise ValueError(f"Task {ident}: SHA-256 image hash mismatch")
        with Image.open(image) as img:
            if img.size != (frame.get("width"), frame.get("height")):
                raise ValueError(f"Task {ident}: image dimensions do not match manifest")
            img.verify()
        lines = _label_lines(annotations[0].get("result"), frame, task)
        records.append((ident, "train" if source == train else "val", image, lines))
    counts = {part: sum(record[1] == part for record in records) for part in ("train", "val")}
    if not counts["train"] or not counts["val"]:
        raise ValueError("Each split needs at least one reviewed image")
    report = {
        "task": task,
        "counts": counts,
        "boxes": {part: sum(len(r[3]) for r in records if r[1] == part) for part in counts},
        "empty_reviewed_images": sum(not r[3] for r in records),
        "train_video_sha256": train,
        "val_video_sha256": val,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temp.mkdir()
    try:
        for ident, part, image, lines in records:
            image_out = temp / "images" / part / image.name
            label_out = temp / "labels" / part / f"{ident}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, image_out)
            label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
        yaml = "train: images/train\nval: images/val\nnames:\n  0: person\n"
        if task == "pose13":
            yaml += f"kpt_shape: [13, 3]\nflip_idx: {json.dumps(FLIP13)}\n"
        (temp / "data.yaml").write_text(yaml, encoding="utf8")
        (temp / "conversion_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf8"
        )
        if output.exists():
            raise FileExistsError(f"Output appeared during conversion: {output}")
        temp.rename(output)
    except BaseException:
        if (
            temp.exists()
            and temp.resolve().parent == output.parent
            and temp.name.startswith(f".{output.name}.")
        ):
            shutil.rmtree(temp)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "manifest", "review", "split", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task", choices=("detect", "pose13"), default="detect")
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf8"))

    report = convert_dataset(
        read(args.annotations),
        read(args.manifest),
        read(args.review),
        read(args.split),
        args.images,
        args.output,
        task=args.task,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
