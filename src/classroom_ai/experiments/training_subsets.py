"""Train-only list ablations: immutable source files and validation split."""

import random
from pathlib import PurePosixPath


def select_train_rows(rows, excluded_sources=(), excluded_ids=(), random_remove=0, seed=42):
    """Deterministically select training records without touching holdout roles."""
    train = sorted(
        (row for row in rows if row["split"] == "train"), key=lambda row: row["image_id"]
    )
    ids = [row["image_id"] for row in train]
    if not train or len(ids) != len(set(ids)):
        raise ValueError("Missing train records or duplicate image IDs")
    sources = set(row["source_id"] for row in train)
    if not set(excluded_sources) <= sources or not set(excluded_ids) <= set(ids):
        raise ValueError("Exclusions must identify existing training records")
    if not isinstance(random_remove, int) or random_remove < 0:
        raise ValueError("Random removal count must be a nonnegative integer")
    for row in train:
        paths = []
        for field, directory in (("image_path", "images"), ("label_path", "labels")):
            text = row[field]
            path = PurePosixPath(text)
            if "\\" in text or ":" in text or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe training path")
            if len(path.parts) != 3 or path.parts[:2] != (directory, "train"):
                raise ValueError("Training path must use its original train role")
            paths.append(path)
        if paths[0].stem != paths[1].stem or paths[1].suffix != ".txt":
            raise ValueError("Image/label stems differ")
    selected = [
        row
        for row in train
        if row["source_id"] not in excluded_sources and row["image_id"] not in excluded_ids
    ]
    if random_remove >= len(selected):
        raise ValueError("Ablation must retain training images")
    if random_remove:
        removed = set(
            random.Random(seed).sample([row["image_id"] for row in selected], random_remove)
        )
        selected = [row for row in selected if row["image_id"] not in removed]
    return selected
