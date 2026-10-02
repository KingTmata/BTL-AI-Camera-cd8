"""Week 2: validate project8 labels, group splits and immutable dataset releases."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import yaml

from src.classes import PROJECT_NAMES as NAMES
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROOT = Path(__file__).resolve().parents[1]


def open_image(path):
    # Ultralytics patches Image.open to auto-install HEIF on ANY decode failure.
    # Our accepted formats exclude HEIF; report corrupt images without downloads.
    patches = sys.modules.get("ultralytics.utils.patches")
    opener = getattr(patches, "_image_open", Image.open)
    return opener(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path, root=ROOT):
    candidate = (Path(root) / path).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path outside project: {path}")
    return candidate


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "image_path", "label_path", "source_id", "session_id", "split_group_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest has no images; templates are not a dataset.")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row has more/fewer fields than its header")
    rows = [{key: value.strip() for key, value in row.items()} for row in rows]
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inventory(image_dir, label_dir, source_id, session_id="unknown", root=ROOT):
    """Generate pending rows; never invent approvals or negative labels."""
    image_dir, label_dir = Path(image_dir).resolve(), Path(label_dir).resolve()
    rows = []
    template = ROOT / "data/templates/manifest.csv"
    with template.open(encoding="utf-8-sig", newline="") as stream:
        fields = next(csv.reader(stream))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
        raise ValueError("source_id must contain only letters, digits, _ or -")
    expected_labels = set()
    for image in sorted(image_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image.relative_to(image_dir)
        label = label_dir / relative.with_suffix(".txt")
        expected_labels.add(label)
        image_id = source_id + "_" + hashlib.sha256(relative.as_posix().encode()).hexdigest()[:16]
        row = dict.fromkeys(fields, "")
        row.update(image_id=image_id, image_path=image.relative_to(Path(root).resolve()).as_posix(),
                   label_path=label.relative_to(Path(root).resolve()).as_posix(), source_id=source_id,
                   original_image_id=relative.as_posix(), session_id=session_id,
                   split_group_id=f"{source_id}:{session_id}", annotation_status="unlabeled",
                   review_status="pending", near_duplicate_status="pending", needs_review="no")
        rows.append(row)
    orphans = set(label_dir.rglob("*.txt")) - expected_labels
    if orphans:
        raise ValueError(f"Orphan labels: {sorted(str(p) for p in orphans)[:10]}")
    if not rows:
        raise ValueError("No images found")
    return rows


def labels(path):
    boxes, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or not re.fullmatch(r"\d+", parts[0]) or not 0 <= int(parts[0]) < len(NAMES):
            raise ValueError(f"line {number}: expected class 0..{len(NAMES)-1} and four coordinates")
        values = tuple(map(float, parts[1:]))
        x, y, w, h = values
        if not all(math.isfinite(v) for v in values) or w <= 0 or h <= 0:
            raise ValueError(f"line {number}: non-finite or non-positive dimensions")
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"line {number}: box extends outside image")
        box = (int(parts[0]), *values)
        if box in seen:
            raise ValueError(f"line {number}: duplicate box")
        seen.add(box)
        boxes.append(box)
    return boxes


def group_keys(row):
    keys = [("split_group", row["split_group_id"])] if row["split_group_id"] else []
    if row["session_id"] and row["session_id"] != "unknown":
        keys.append(("session", row["source_id"], row["session_id"]))
    if row.get("duplicate_group_id"):
        keys.append(("duplicate", row["duplicate_group_id"]))
    return keys


def groups(rows):
    """Union shared sessions, split groups and manually identified duplicates."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for index, row in enumerate(rows):
        for key in group_keys(row):
            if key in owners:
                parents[find(index)] = find(owners[key])
            owners[key] = index
    result = defaultdict(list)
    for index, row in enumerate(rows):
        result[find(index)].append(row)
    return list(result.values())


def validate(rows, root=ROOT, release=False):
    errors, counts, enriched = [], defaultdict(Counter), []
    identifiers, images, hashes, label_paths = set(), set(), {}, set()
    for original in rows:
        row = dict(original)
        prefix = row.get("image_id", "<missing id>")
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
                raise ValueError("image_id must contain only letters, digits, _ or -")
            if prefix.casefold() in identifiers:
                raise ValueError("duplicate image_id")
            identifiers.add(prefix.casefold())
            image, label = resolve(row["image_path"], root), resolve(row["label_path"], root)
            if image.suffix.lower() not in IMAGE_EXTENSIONS or label.suffix != ".txt" or image.stem != label.stem:
                raise ValueError("image/TXT extensions or stems do not match")
            if image in images:
                raise ValueError("same image listed more than once")
            images.add(image)
            if label in label_paths:
                raise ValueError("label assigned to multiple images")
            label_paths.add(label)
            with open_image(image) as opened:
                opened.verify()
            with open_image(image) as opened:
                if opened.getexif().get(274, 1) != 1:
                    raise ValueError("EXIF orientation must be normalized before annotating")
                width, height = opened.size
                opened.load()
            boxes = labels(label)
            image_hash, label_hash = sha256(image), sha256(label)
            if image_hash in hashes:
                raise ValueError(f"exact duplicate image of {hashes[image_hash]}")
            hashes[image_hash] = prefix
            for key, value in (("image_sha256", image_hash), ("label_sha256", label_hash)):
                if row.get(key) and row[key] != value:
                    raise ValueError(f"{key} changed")
                row[key] = value
            row.update(width=width, height=height)
            split = row["split"]
            if split not in ("", "train", "val", "test"):
                raise ValueError("unknown split")
            for key in ("source_id", "split_group_id"):
                if not row[key] or row[key] == "unknown":
                    raise ValueError(f"{key} required; use a reviewed source/scene group")
            if release:
                if not row.get("license_or_consent") or row["license_or_consent"] in ("unknown", "pending"):
                    raise ValueError("source permission not confirmed")
                if row.get("annotation_status") != "approved" or not row.get("annotator"):
                    raise ValueError("annotations need approval and annotator")
                if row.get("near_duplicate_status") != "approved":
                    raise ValueError("merged dataset near-duplicate review not confirmed")
                if row.get("review_status") == "approved":
                    if not row.get("reviewer") or row["reviewer"] == row["annotator"]:
                        raise ValueError("reviewer must differ from annotator")
                elif row.get("review_status") != "pending":
                    raise ValueError("unresolved review")
                if (not boxes or row.get("needs_review") == "yes") and row.get("review_status") != "approved":
                    raise ValueError("negative/suspicious image requires second review")
                if row.get("needs_review") not in ("yes", "no"):
                    raise ValueError("needs_review must be yes/no")
            box_counts = Counter(box[0] for box in boxes)
            counts[split]["images"] += 1
            counts[split]["negative_images"] += int(not boxes)
            for index, name in enumerate(NAMES):
                row[f"{name.replace(' ', '_')}_count"] = box_counts[index]
                counts[split][name] += box_counts[index]
            enriched.append(row)
        except (ValueError, OSError, KeyError, SyntaxError, Image.DecompressionBombError) as exc:
            errors.append(f"{prefix}: {exc}")
    for group in groups(rows):
        split_set = {r["split"] for r in group if r["split"]}
        if len(split_set) > 1:
            errors.append(f"group leakage: {[r['image_id'] for r in group]}")
    if release:
        for split in {r["split"] for r in rows}:
            subset = [r for r in rows if r["split"] == split]
            reviewed = sum(r.get("review_status") == "approved" for r in subset)
            if reviewed / len(subset) < .1:
                errors.append(f"{split or 'unassigned'}: fewer than 10% second reviews")
            # Enforce representation of every present class in the reviewed subset.
            for name in NAMES:
                column = f"{name.replace(' ', '_')}_count"
                present = [r for r in enriched if r["split"] == split and r[column] > 0]
                if present and not any(r.get("review_status") == "approved" for r in present):
                    errors.append(f"{split}: no second review covering {name}")
        for split in ("train", "val"):
            if counts[split]["images"] == 0 or any(counts[split][name] == 0 for name in NAMES):
                errors.append(f"{split}: needs images and examples of all four classes")
        if any(not r["split"] for r in rows):
            errors.append("unassigned split")
    return {"valid": not errors, "images_checked": len(rows), "errors": errors,
            "counts": dict(counts)}, enriched


def assign_splits(rows, val_fraction=.2, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    result = [dict(r) for r in rows]
    buckets = groups(result)
    random.Random(seed).shuffle(buckets)
    buckets.sort(key=len, reverse=True)
    total = sum(len(g) for g in buckets if not any(r["split"] == "test" or r.get("original_split") == "test" for r in g))
    targets = {"train": total*(1-val_fraction), "val": total*val_fraction}
    assigned = Counter()
    pending = []
    for group in buckets:
        fixed = {r["split"] for r in group if r["split"]}
        if len(fixed) > 1:
            raise ValueError("existing split crosses a group; resolve before splitting")
        # Preserve source train/val when supplied (e.g. COCO).
        fixed |= {r.get("original_split") for r in group if r.get("original_split") in ("train", "val", "test")}
        if len(fixed) > 1:
            raise ValueError("original source splits conflict in a group")
        if fixed:
            split = fixed.pop()
            for row in group:
                row["split"] = split
            assigned[split] += len(group)
        else:
            pending.append(group)
    for group in pending:
        split = max(targets, key=lambda s: (targets[s]-assigned[s])/targets[s])
        for row in group:
            row["split"] = split
        assigned[split] += len(group)
    if not assigned["train"] or not assigned["val"]:
        raise ValueError("not enough independent groups for train and val")
    return result


def lock_release(rows, destination, root=ROOT, seed=42):
    report, enriched = validate(rows, root, release=True)
    if not report["valid"]:
        raise ValueError("\n".join(report["errors"]))
    destination = Path(destination).resolve()
    if not destination.is_relative_to(Path(root).resolve()) or destination.exists():
        raise ValueError("release must be a NEW directory inside project")
    destination.mkdir(parents=True)
    try:
        for row in enriched:
            for kind, column, suffix in (("images", "image_path", Path(row["image_path"]).suffix.lower()),
                                         ("labels", "label_path", ".txt")):
                target = destination / kind / row["split"] / (row["image_id"] + suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resolve(row[column], root), target)
                row[column] = target.relative_to(Path(root).resolve()).as_posix()
        write_manifest(destination / "manifest.csv", enriched)
        splits = destination / "splits"
        splits.mkdir()
        for split in sorted({r["split"] for r in enriched}):
            (splits / f"{split}.txt").write_text("".join(r["image_path"]+"\n" for r in enriched if r["split"] == split), encoding="utf-8")
        config = {"path": str(destination), "train": "images/train", "val": "images/val", "names": dict(enumerate(NAMES))}
        if any(r["split"] == "test" for r in enriched):
            config["test"] = "images/test"
        (destination / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        card = {"status": "locked", "seed": seed, "counts": report["counts"],
                "sources": sorted({r['source_id'] for r in enriched}),
                "near_duplicate_review": "human-approved after merge", "review_fraction": sum(r.get('review_status') == 'approved' for r in enriched)/len(enriched),
                "limitations": "Complete source/lighting/pretraining overlap analysis in project data/DATA_CARD.md."}
        (destination / "DATA_CARD.md").write_text("# Dataset release\n\n```json\n"+json.dumps(card, ensure_ascii=False, indent=2)+"\n```\n", encoding="utf-8")
        entries = {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}
        (destination / "checksums.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
        # Verify copied data before declaring a release usable.
        verified, _ = validate(enriched, root, release=True)
        if not verified["valid"]:
            raise ValueError("copy verification failed: "+str(verified["errors"]))
    except Exception:
        # Preserve partial output for inspection; it cannot pass verify_release.
        (destination / "INCOMPLETE").write_text("Release failed; do not use for training.", encoding="utf-8")
        raise
    return report


def verify_release(destination):
    destination = Path(destination)
    if (destination / "INCOMPLETE").exists():
        raise ValueError("incomplete release")
    entries = json.loads((destination / "checksums.json").read_text(encoding="utf-8"))
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file() and p != destination / "checksums.json"}
    if actual != set(entries):
        raise ValueError("release file list changed")
    for name, digest in entries.items():
        if sha256(resolve(name, destination)) != digest:
            raise ValueError(f"checksum changed: {name}")
    return sha256(destination / "checksums.json")
