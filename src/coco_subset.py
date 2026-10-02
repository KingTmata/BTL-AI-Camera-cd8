"""Pure helpers for deterministic, unstratified COCO subset preparation."""
from __future__ import annotations

import random
import math
from collections import Counter, defaultdict

from src.classes import COCO_NAMES, COCO_TO_PROJECT, PROJECT_NAMES


def category_mapping(categories):
    by_name = {category["name"]: category["id"] for category in categories}
    missing = set(COCO_NAMES.values()) - set(by_name)
    if missing:
        raise ValueError(f"Missing COCO categories: {sorted(missing)}")
    return {by_name[name]: COCO_TO_PROJECT[raw_id] for raw_id, name in COCO_NAMES.items()}


def normalized_box(annotation, image, mapping):
    category = annotation["category_id"]
    if category not in mapping:
        return None
    if annotation.get("iscrowd", 0):
        raise ValueError("target crowd annotation")
    x, y, w, h = map(float, annotation["bbox"])
    width, height = image["width"], image["height"]
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError("invalid geometry")
    if w <= 0 or h <= 0 or x < -1e-6 or y < -1e-6 or x+w > width+1e-6 or y+h > height+1e-6:
        raise ValueError("invalid geometry")
    return (mapping[category], (x+w/2)/width, (y+h/2)/height, w/width, h/height)


def ordered_candidates(images, annotations, mapping, excluded_ids=(), seed=42):
    """Exclude entire images with crowd/invalid/repeated target boxes."""
    images = {image["id"]: image for image in images}
    positive, bad, signatures = set(), {}, set()
    for annotation in annotations:
        image_id = annotation["image_id"]
        if annotation["category_id"] not in mapping:
            continue
        positive.add(image_id)
        try:
            box = normalized_box(annotation, images[image_id], mapping)
            signature = (image_id, *box)
            if signature in signatures:
                raise ValueError("duplicate target box")
            signatures.add(signature)
        except (ValueError, KeyError) as exc:
            bad[image_id] = str(exc)
    candidates = sorted(positive - set(bad) - set(excluded_ids))
    random.Random(seed).shuffle(candidates)
    return candidates, {"target_positive_images": len(positive), "excluded_bad_images": len(bad),
                        "excluded_reasons": dict(Counter(bad.values())),
                        "excluded_reference_images": len(positive.intersection(excluded_ids)),
                        "eligible_images": len(candidates)}


def coverage_review(rows, quotas=(('train', 80), ('val', 20)), seed=42):
    """Select review targets, covering every class before random fill."""
    selected = []
    for split, quota in quotas:
        pool = sorted((r for r in rows if r["split"] == split), key=lambda r: r["image_id"])
        random.Random(seed).shuffle(pool)
        chosen, covered = [], set()
        for index, name in enumerate(PROJECT_NAMES):
            if index in covered:
                continue
            candidate = next((r for r in pool if int(r[name.replace(' ', '_')+'_count']) > 0), None)
            if candidate is None:
                raise ValueError(f"{split} missing class {name}")
            if candidate not in chosen:
                chosen.append(candidate)
                covered.update(i for i, n in enumerate(PROJECT_NAMES) if int(candidate[n.replace(' ', '_')+'_count']) > 0)
        if len(pool) < quota:
            raise ValueError("insufficient images for review quota")
        chosen.extend(r for r in pool if r not in chosen)
        selected.extend(chosen[:quota])
    return selected
