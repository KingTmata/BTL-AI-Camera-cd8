"""Archive, audit and stage the supplied Roboflow laptop ZIP for project8 review.

This is an intake, not a training release: the source annotates only two classes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.classes import PROJECT_NAMES
from src.dataset import labels, open_image, read_manifest, sha256, validate, write_manifest

SOURCE = "laptop_roboflow_v1"
RAW = ROOT / "data/raw" / SOURCE
DEST = ROOT / "data/dataset" / (SOURCE + "_intake")
URL = "https://universe.roboflow.com/new-workspace-xp2sh/laptop-tgbyh/dataset/1"
MAPPING = {0: 6, 1: 3}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(archive):
    archive = archive.resolve()
    if DEST.exists() or RAW.exists():
        raise ValueError("Intake already exists; inspect its report instead of overwriting")
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename or ":" in member.filename:
                raise ValueError(f"Unsafe archive entry: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains symlink")
        config = yaml.safe_load(zipped.read("data.yaml"))
        if config["names"] != ["Book", "Laptop"] or config["nc"] != 2:
            raise ValueError("Unexpected source mapping")
        RAW.mkdir(parents=True)
        shutil.copy2(archive, RAW / "submission.zip")
        zipped.extractall(RAW / "original")

    DEST.mkdir(parents=True)
    with (ROOT / "data/templates/manifest.csv").open(encoding="utf-8-sig") as stream:
        fields = next(csv.reader(stream))
    existing = read_manifest(ROOT / "data/dataset/coco500_v0.1/manifest.csv")
    existing_hashes = {row["image_sha256"]: row["image_id"] for row in existing}
    pixel_hashes = {}
    for row in existing:
        with open_image(ROOT / row["image_path"]) as img:
            pixel_hashes[pixel_hash(img)] = row["image_id"]
    rows, usable, audit, duplicate_groups = [], [], [], defaultdict(list)
    counts = defaultdict(Counter)
    geometry, widths, heights, areas, short_sides = [], [], [], defaultdict(list), defaultdict(list)
    modes = Counter()
    image_paths = sorted((RAW / "original").glob("*/images/*"))
    expected_label_paths = set()
    for index, image in enumerate(image_paths):
        if not image.is_file() or image.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        split = image.parent.parent.name
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        expected_label_paths.add(label)
        local_id = SOURCE + "_" + hashlib.sha256(image.relative_to(RAW / "original").as_posix().encode()).hexdigest()[:16]
        entry = {"image_id": local_id, "file": image.name, "original_split": split,
                 "errors": [], "flags": [], "boxes": 0, "duplicate_of": ""}
        counts[split]["images"] += 1
        mapped_image = DEST / "images" / split / image.name
        mapped_label = DEST / "labels" / split / label.name
        mapped_image.parent.mkdir(parents=True, exist_ok=True)
        mapped_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, mapped_image)
        row = dict.fromkeys(fields, "")
        row.update(image_id=local_id, image_path=mapped_image.relative_to(ROOT).as_posix(),
                   label_path=mapped_label.relative_to(ROOT).as_posix(), source_id=SOURCE,
                   original_image_id=image.name, original_split=split, source_url=URL,
                   license_or_consent="CC BY 4.0 (source README; original image provenance unverified)",
                   session_id="unknown", split_group_id=SOURCE + ":unknown_sessions",
                   split="", annotation_status="converted", annotator="Roboflow source annotations",
                   review_status="pending", near_duplicate_status="pending", needs_review="yes",
                   notes="Only Book/Laptop annotated; complete all eight classes before splitting/training")
        try:
            if not label.is_file():
                raise ValueError("missing TXT")
            converted = []
            for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5 or parts[0] not in ("0", "1"):
                    raise ValueError(f"source label line {number}: expected IDs 0/1 and four coordinates")
                converted.append(str(MAPPING[int(parts[0])]) + " " + " ".join(parts[1:]))
            mapped_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
            boxes = labels(mapped_label)
            entry["boxes"] = len(boxes)
            if not boxes:
                entry["flags"].append("empty_label_needs_eight_class_review")
            with open_image(image) as img:
                img.verify()
            with open_image(image) as img:
                img.load()
                w, h = img.size
                if img.getexif().get(274, 1) != 1:
                    raise ValueError("EXIF orientation not normalized")
                widths.append(w)
                heights.append(h)
                modes[img.mode] += 1
                gray = np.asarray(img.convert("L"), dtype=np.float32)
                brightness, contrast = float(gray.mean()), float(gray.std())
                entry.update(width=w, height=h, brightness=brightness, contrast=contrast)
                if brightness < 40: entry["flags"].append("dark")
                if brightness > 215: entry["flags"].append("bright")
                if contrast < 20: entry["flags"].append("low_contrast")
                phash = pixel_hash(img)
            digest = sha256(image)
            row.update(image_sha256=digest, label_sha256=sha256(mapped_label), width=w, height=h)
            duplicate_groups[digest].append(local_id)
            if digest in existing_hashes or phash in pixel_hashes:
                entry["duplicate_of"] = existing_hashes.get(digest, pixel_hashes.get(phash))
                entry["flags"].append("exact_byte_or_decoded_pixel_duplicate")
            else:
                existing_hashes[digest] = local_id
                pixel_hashes[phash] = local_id
            box_counts = Counter(box[0] for box in boxes)
            for class_id, name in enumerate(PROJECT_NAMES):
                row[name.replace(" ", "_") + "_count"] = box_counts[class_id]
                counts[split][name + "_boxes"] += box_counts[class_id]
                counts[split][name + "_images"] += int(box_counts[class_id] > 0)
            for class_id, x, y, bw, bh in boxes:
                areas[PROJECT_NAMES[class_id]].append(bw * bh)
                short_sides[PROJECT_NAMES[class_id]].append(min(bw*w, bh*h)*640/max(w,h))
                if min(bw*w, bh*h)*640/max(w,h) < 8 and "tiny_box_at_640" not in entry["flags"]:
                    entry["flags"].append("tiny_box_at_640")
            counts[split]["boxes"] += len(boxes)
            counts[split]["empty_labels"] += int(not boxes)
            if not entry["duplicate_of"]:
                usable.append(row)
        except (ValueError, OSError, SyntaxError) as exc:
            entry["errors"].append(str(exc))
            row["notes"] += "; technical error: " + str(exc)
        rows.append(row)
        audit.append(entry)
        if (index + 1) % 200 == 0:
            print(f"Audited {index+1}/{len(image_paths)} images", flush=True)

    orphan_labels = sorted(str(p.relative_to(RAW / "original")) for p in
                           (RAW / "original").glob("*/labels/*.txt") if p not in expected_label_paths)
    write_manifest(DEST / "inventory_all.csv", rows)
    report, enriched = validate(usable)
    if not report["valid"]:
        save(DEST / "validation.json", report)
        raise ValueError(f"Technical intake validation failed: {report['errors'][:5]}")
    write_manifest(DEST / "manifest.csv", enriched)
    combined_report, combined = validate(existing + enriched)
    if not combined_report["valid"]:
        raise ValueError(f"Combined inventory failed: {combined_report['errors'][:5]}")
    write_manifest(DEST / "combined_inventory.csv", combined)
    save(DEST / "validation.json", {"intake": report, "combined_inventory": combined_report})
    save(DEST / "audit.json", audit)
    review = [{"image_id": e["image_id"], "original_split": e["original_split"], "file": e["file"],
               "reason": "complete_eight_class_annotations; " + "; ".join(e["errors"] + e["flags"]),
               "duplicate_of": e["duplicate_of"], "reviewer": "", "review_status": "pending"} for e in audit]
    write_manifest(DEST / "review.csv", review)
    save(DEST / "mapping.json", {"original": config["names"], "project": list(PROJECT_NAMES),
                                 "mapping": MAPPING, "training_ready": False})
    stats = {"source": URL, "archive_sha256": sha256(archive), "source_config": config,
             "images": len(audit), "original_splits": dict(counts), "boxes": sum(e["boxes"] for e in audit),
             "technical_errors": [e for e in audit if e["errors"]], "orphan_labels": orphan_labels,
             "duplicates": [e for e in audit if e["duplicate_of"]], "usable_unique_pending": len(enriched),
             "combined_unique_pending": len(combined), "color_modes": dict(modes),
             "width": summary(widths), "height": summary(heights),
             "bbox_area_ratio": {name: summary(values) for name, values in areas.items()},
             "bbox_short_side_at_640": {name: summary(values) for name, values in short_sides.items()},
             "quality_flags": dict(Counter(f for e in audit for f in e["flags"])),
             "annotation_complete_project8": False, "session_information": "unknown", "status": "pending_reannotation"}
    save(DEST / "statistics.json", stats)
    contact_sheet(enriched)
    write_report(stats)
    save(DEST / "checksums.json", {p.relative_to(DEST).as_posix(): sha256(p)
                                   for p in sorted(DEST.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    print(json.dumps({key: stats[key] for key in ("images", "boxes", "usable_unique_pending", "combined_unique_pending", "status")}, indent=2))


def pixel_hash(img):
    rgb = img.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def summary(values):
    return {"min": float(min(values)), "median": float(np.median(values)), "max": float(max(values))} if values else None


def contact_sheet(rows):
    selected = rows[::max(1, len(rows)//12)][:12]
    sheet = Image.new("RGB", (1000, 3*285), "white")
    for index, row in enumerate(selected):
        with open_image(ROOT / row["image_path"]) as img:
            tile = img.convert("RGB").resize((240, 240))
        draw = ImageDraw.Draw(tile)
        for class_id, x, y, w, h in labels(ROOT / row["label_path"]):
            color = "red" if class_id == 3 else "blue"
            draw.rectangle(((x-w/2)*240, (y-h/2)*240, (x+w/2)*240, (y+h/2)*240), outline=color, width=2)
            draw.text(((x-w/2)*240, (y-h/2)*240), PROJECT_NAMES[class_id], fill=color)
        left, top = (index%4)*250, (index//4)*285
        sheet.paste(tile, (left, top))
        ImageDraw.Draw(sheet).text((left, top+242), row["original_split"] + " " + row["original_image_id"][:20], fill="black")
    sheet.save(DEST / "contact_sheet.jpg")


def write_report(s):
    lines = ["# Hồ sơ dữ liệu thành viên nộp — Laptop Roboflow v1", "",
             "Ngày tiếp nhận: 02/10/2026. **Dữ liệu chính do thành viên đóng góp**, ngang hàng COCO500; đã gộp vào bản draft project8_v0.2. Bản intake lưu riêng để đối chiếu nguồn. Các thành viên tiếp tục bổ sung nhãn theo phân công.", "",
             "## Nguồn và đặc tính cơ bản", "",
             f"- ZIP: `Laptop.v1-test-dataset.yolov8.zip`; SHA256: `{s['archive_sha256']}`.",
             f"- Nguồn theo README/YAML: [Roboflow Laptop v1]({URL}). Thành viên cung cấp dữ liệu trên mạng; chưa có bằng chứng đây là ảnh tự chụp của nhóm.",
             "- Giấy phép được gói khai báo: CC BY 4.0. Chưa xác minh riêng nguồn/quyền từng ảnh; cần giữ ghi công và thông tin nguồn khi chia sẻ.",
             "- Export nguồn ghi 18/03/2023; đây là ngày export, không phải ngày chụp.",
             "- Tiền xử lý theo README: auto-orientation, bỏ EXIF orientation; resize 416×416 bằng stretch, không augmentation. Stretch có thể làm biến dạng tỷ lệ vật.",
             f"- Thực tế: {s['images']} ảnh; kích thước width {s['width']}, height {s['height']}; chế độ màu {s['color_modes']}.",
             "- Bài toán detection, TXT mỗi ảnh; một dòng là class cx cy w h chuẩn hóa. Không có segmentation, sự kiện video hay session/camera trong gói.", "",
             "## Mapping nhãn", "", "| Nhãn nguồn / ID | Nhãn nhóm / ID |", "|---|---|",
             "| Book / 0 | book / 6 |", "| Laptop / 1 | laptop / 3 |", "",
             "Nguồn không tách vở khỏi sách. Sáu lớp khác chưa được gán nhãn theo xác nhận của người nhận. Không xem chúng là vắng mặt chỉ vì TXT không có lớp đó.", "",
             "## Thống kê thực tế", "", "| Split nguồn | Ảnh | Box book | Ảnh có book | Box laptop | Ảnh có laptop | TXT rỗng |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, count in s["original_splits"].items():
        lines.append(f"| {split} | {count['images']} | {count.get('book_boxes',0)} | {count.get('book_images',0)} | {count.get('laptop_boxes',0)} | {count.get('laptop_images',0)} | {count.get('empty_labels',0)} |")
    lines += ["", f"Tổng {s['boxes']} box đã phân tích hợp lệ. Số ảnh chứa các lớp có thể chồng nhau.", "",
              "## Kiểm tra, lỗi và outlier", "",
              f"- {len(s['technical_errors'])} ảnh/cặp nhãn có lỗi kỹ thuật; {len(s['orphan_labels'])} TXT mồ côi.",
              f"- {len(s['duplicates'])} ảnh trùng bytes hoặc pixel giải mã với ảnh trước đó trong nguồn hoặc COCO500. Giữ nguyên trong raw, loại khỏi manifest ứng viên duy nhất.",
              f"- {s['usable_unique_pending']} ảnh duy nhất đạt kiểm kỹ thuật, vẫn chờ hoàn thiện nhãn và review.",
              "- Kiểm: giải mã ảnh, EXIF, cặp tên, class nguồn 0/1, năm cột, finite, box dương và không vượt biên, box trùng, checksum; đối chiếu trùng với COCO500.",
              "- Không chạy tìm gần trùng bằng embeddings. Chưa chứng minh split nguồn độc lập theo phiên. Bản intake giữ unassigned để truy bước nhập. Trong manifest dataset chính project8_v0.2, đã giữ split nguồn 910 train / 260 val / 130 test, nhóm theo tên gốc trước hậu tố Roboflow; session vẫn unknown, chưa khẳng định độc lập cảnh/phiên.",
              "", "| Cờ cần xem | Số ảnh |", "|---|---:|"]
    for flag, count in s["quality_flags"].items():
        lines.append(f"| {flag} | {count} |")
    lines += ["", "Ngưỡng heuristic: brightness grayscale<40 hoặc>215, std<20; cạnh ngắn box khi resize giữ tỷ lệ về 640<8px. Cờ không tự kết luận nhãn sai hoặc tự loại ảnh; review.csv ghi ca cần xem.", "",
              "Box area/ảnh (min/median/max): " + json.dumps(s["bbox_area_ratio"]), "",
              "Cạnh ngắn box tại 640 (min/median/max px): " + json.dumps(s["bbox_short_side_at_640"]), "",
              "### Quan sát mẫu trực quan", "",
              "Đã xem contact sheet 12 ảnh trải trên các split: có người, bàn, ghế xuất hiện nhưng chỉ có box book/laptop; một số box book bao cụm sách/kệ sách cần người phụ trách rà lại phạm vi nhãn. Đây là quan sát mẫu, không phải review ngữ nghĩa 100% hoặc xác nhận người thứ hai.", "",
              "## Đã thêm vào dự án như thế nào", "",
              "- `data/raw/laptop_roboflow_v1/`: ZIP nguyên bản và toàn bộ nội dung gốc.",
              "- `data/dataset/laptop_roboflow_v1_intake/`: ảnh sao chép, nhãn đổi ID, inventory_all.csv, manifest.csv, combined_inventory.csv, mapping.json, audit.json, validation.json, statistics.json, review.csv, contact_sheet.jpg, checksums.json.",
              f"- Combined inventory: 500 ảnh COCO + {s['usable_unique_pending']} ảnh nguồn này = **{s['combined_unique_pending']} ảnh ứng viên duy nhất** đạt kiểm bytes/pixel và định dạng. Đây không phải số ảnh đã hoàn thiện nhãn/review.",
              "- COCO500 giữ nguyên để truy thí nghiệm. Dataset chính gộp tại `data/dataset/project8_v0.2/`, có YAML tám lớp, manifest, split và checksum riêng; xem [hồ sơ bản gộp](PROJECT8_V0_2_DATA_PROFILE.md). Chưa khóa release hoặc hoàn tất review.", "",
              "## Việc còn cần làm", "",
              "1. Rà từng ảnh, bổ sung đầy đủ person, table, chair, cell phone, backpack, cup và kiểm lại book/laptop.",
              "2. Ghi nguồn gốc ảnh/nhóm/phiên nếu truy được; đối chiếu trùng/gần trùng và split nguồn trước khi chia lại. Giữ test nguồn trong kho, chưa gọi là test độc lập tám lớp.",
              "3. Người thứ hai review theo quy trình; ghi reviewer thật, không tự đánh dấu approved.",
              "4. Sau khi hoàn thiện, tạo bản dataset mới, khóa checksum và chạy baseline theo protocol chung.", ""]
    text = "\n".join(lines)
    (ROOT / "docs/LAPTOP_SUBMISSION_DATA_PROFILE.md").write_text(text, encoding="utf-8")
    (DEST / "DATA_CARD.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    run(parser.parse_args().archive)
