"""Fresh-initializer controls for known scene groups shared with holdouts."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from classroom_ai.experiments.experiment_campaign import file_sha
from classroom_ai.experiments.training_subsets import select_train_rows


def select_group_disjoint_train_rows(rows):
    """Remove training members of known val/test groups; preserve every holdout."""
    rows = list(rows)
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(
        row["split"] not in {"train", "val", "test"} for row in rows
    ):
        raise ValueError("Duplicate identities or unknown split roles")
    if any(
        not isinstance(row.get("source_group_key"), str) or not row["source_group_key"].strip()
        for row in rows
    ):
        raise ValueError("Every record needs a known source group key")
    if not any(row["split"] == "val" for row in rows):
        raise ValueError("Missing validation metadata")
    heldout = {row["source_group_key"] for row in rows if row["split"] in {"val", "test"}}
    excluded = {
        row["image_id"]
        for row in rows
        if row["split"] == "train" and row["source_group_key"] in heldout
    }
    return select_train_rows(rows, excluded_ids=excluded)


def prepare_group_controls(data_root, manifest_path, output, expected_manifest, names):
    """Write two train lists after verifying payloads; never read test labels."""
    data_root, manifest_path, output = Path(data_root).resolve(), Path(manifest_path), Path(output)
    if file_sha(manifest_path) != expected_manifest:
        raise ValueError("Manifest bytes differ")
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = select_train_rows(rows)
    clean = select_group_disjoint_train_rows(rows)
    removal_count = len(train) - len(clean)
    if not removal_count:
        raise ValueError("No shared training groups need a removal control")
    random_control = select_train_rows(rows, random_remove=removal_count, seed=42)
    heldout = {row["source_group_key"] for row in rows if row["split"] in {"val", "test"}}
    boxes = {}
    for row in train:
        image, label = data_root / row["image_path"], data_root / row["label_path"]
        if any(not path.resolve().is_relative_to(data_root) for path in (image, label)):
            raise ValueError("Training payload resolves outside dataset")
        if file_sha(image) != row["image_sha256"] or file_sha(label) != row["label_sha256"]:
            raise ValueError("Training payload changed: " + row["image_id"])
        counts = Counter()
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError("Malformed training annotation")
            cls, cx, cy, width, height = values
            if (
                cls != int(cls)
                or not 0 <= cls < len(names)
                or not all(0 <= v <= 1 for v in values[1:])
                or min(width, height) <= 0
            ):
                raise ValueError("Invalid training annotation")
            counts[int(cls)] += 1
        if sum(counts.values()) != int(row["boxes"]):
            raise ValueError("Training GT count differs")
        boxes[row["image_id"]] = counts
    removed = [row for row in train if row["source_group_key"] in heldout]
    report = dict(
        status="prepared_not_trained",
        manifest_sha256=expected_manifest,
        original_train_images=len(train),
        removed_train_images=removal_count,
        removed_train_boxes=sum(int(row["boxes"]) for row in removed),
        removed_group_keys=sorted({row["source_group_key"] for row in removed}),
        validation_images=sum(row["split"] == "val" for row in rows),
        validation_boxes=sum(int(row["boxes"]) for row in rows if row["split"] == "val"),
        labels_changed=False,
        holdout_roles_changed=False,
        test_metadata_used_for_group_exclusion=True,
        test_predictions_used=False,
        test_labels_read=False,
        full_visual_review_complete=False,
        note="Fresh COCO initializer required: removing rows cannot undo warm-start exposure. Known groups only; not a complete near-scene audit. Random control matches images/steps, not box distribution.",
        subsets={},
    )
    # Validate all class coverage before writing either subset.
    selections = {"group_disjoint": clean, "random_removal": random_control}
    counts_by_subset = {
        name: sum((boxes[row["image_id"]] for row in selected), Counter())
        for name, selected in selections.items()
    }
    if any(counts[cls] == 0 for counts in counts_by_subset.values() for cls in range(len(names))):
        raise ValueError("A control loses a target class")
    output.mkdir(parents=True, exist_ok=False)
    for name, selected in selections.items():
        train_list = output / (name + "_train.txt")
        train_list.write_text(
            "\n".join((data_root / row["image_path"]).as_posix() for row in selected) + "\n",
            encoding="utf-8",
        )
        config = dict(
            path=data_root.as_posix(),
            train=train_list.resolve().as_posix(),
            val=(data_root / "images/val").as_posix(),
            nc=len(names),
            names=dict(enumerate(names)),
        )
        config_path = output / (name + ".yaml")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        identities = [
            dict(
                image_id=row["image_id"],
                image_sha256=row["image_sha256"],
                label_sha256=row["label_sha256"],
            )
            for row in selected
        ]
        counts = counts_by_subset[name]
        report["subsets"][name] = dict(
            train_images=len(selected),
            train_boxes=sum(counts.values()),
            class_boxes={names[cls]: counts[cls] for cls in range(len(names))},
            shared_holdout_groups=len({row["source_group_key"] for row in selected} & heldout),
            source_counts=dict(Counter(row["source_id"] for row in selected)),
            train_identity_sha256=hashlib.sha256(
                json.dumps(identities, sort_keys=True).encode()
            ).hexdigest(),
            train_list=str(train_list.resolve()),
            train_list_sha256=file_sha(train_list),
            data_yaml=str(config_path.resolve()),
            data_yaml_sha256=file_sha(config_path),
        )
    return report
