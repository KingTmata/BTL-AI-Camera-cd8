"""Build the primary project8 draft from COCO500 and the member laptop source."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.classes import PROJECT_NAMES
from src.dataset import labels, read_manifest, sha256, validate, write_manifest

DEST = ROOT / "data/dataset/project8_v0.2"


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if DEST.exists():
        raise ValueError("Version exists; do not overwrite a dataset version")
    coco = read_manifest(ROOT / "data/dataset/coco500_v0.1/manifest.csv")
    laptop = read_manifest(ROOT / "data/dataset/laptop_roboflow_v1_intake/manifest.csv")
    rows, source_counts, split_counts = [], Counter(), Counter()
    original_name_groups = defaultdict(set)
    for row in laptop:
        original_name_groups[Path(row["original_image_id"]).stem.split(".rf.")[0]].add(row["original_split"])
    if any(len(splits) > 1 for splits in original_name_groups.values()):
        raise ValueError("Repeated source filename crosses splits; resolve before merging")
    DEST.mkdir(parents=True)
    for original in coco + laptop:
        row = dict(original)
        member = row["source_id"] == "laptop_roboflow_v1"
        if member:
            row["split"] = {"train": "train", "valid": "val", "test": "test"}[row["original_split"]]
            source_stem = Path(row["original_image_id"]).stem.split(".rf.")[0]
            row["split_group_id"] = row["source_id"] + ":original_name:" + source_stem
            row["notes"] = ("Primary member-contributed source; only Book/Laptop annotated; "
                            "other classes assigned to other team members; source splits preserved, "
                            "shooting-session independence unverified")
        old_image, old_label = ROOT / row["image_path"], ROOT / row["label_path"]
        stem = row["image_id"]
        image = DEST / "images" / row["split"] / (stem + old_image.suffix)
        label = DEST / "labels" / row["split"] / (stem + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_image, image)
        shutil.copy2(old_label, label)
        if sha256(image) != row["image_sha256"] or sha256(label) != row["label_sha256"]:
            raise ValueError("Copied media or labels differ from source")
        row.update(image_path=image.relative_to(ROOT).as_posix(), label_path=label.relative_to(ROOT).as_posix())
        rows.append(row)
        source_counts[row["source_id"]] += 1
        split_counts[row["split"]] += 1
    report, rows = validate(rows)
    save(DEST / "validation.json", report)
    if not report["valid"]:
        raise ValueError(f"Merged draft validation failed: {report['errors'][:5]}")
    write_manifest(DEST / "manifest.csv", rows)
    config = {"path": str(DEST), "train": "images/train", "val": "images/val", "test": "images/test",
              "names": dict(enumerate(PROJECT_NAMES))}
    (DEST / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (DEST / "splits").mkdir()
    for split in ("train", "val", "test"):
        (DEST / "splits" / (split + ".txt")).write_text(
            "\n".join(r["image_path"] for r in rows if r["split"] == split) + "\n", encoding="utf-8")
    counts = {name: {"images": 0, "boxes": 0} for name in PROJECT_NAMES}
    for row in rows:
        for name in PROJECT_NAMES:
            count = int(row[name.replace(" ", "_") + "_count"])
            counts[name]["images"] += int(count > 0)
            counts[name]["boxes"] += count
    review = [{"image_id": r["image_id"], "image_path": r["image_path"], "label_path": r["label_path"],
               "source_id": r["source_id"], "split": r["split"], "needs_review": r["needs_review"],
               "reviewer": "", "review_status": "pending", "notes": r["notes"]} for r in rows]
    write_manifest(DEST / "review.csv", review)
    stats = {"version": "project8_v0.2", "status": "primary_draft", "sources": dict(source_counts),
             "images": len(rows), "splits": dict(split_counts), "class_distribution": counts,
             "boxes": sum(c["boxes"] for c in counts.values()),
             "member_annotations": "Book/Laptop only; remaining classes handled by team",
             "source_split_independence": "unknown for Roboflow; original-name groups do not cross splits",
             "locked_release": False}
    save(DEST / "statistics.json", stats)
    text = "\n".join([
        "# Dataset chính — project8_v0.2", "", "Cập nhật 03/10/2026. **Bản draft chính của dự án**, gồm hai nguồn ngang hàng: COCO500 và bộ Roboflow Laptop do thành viên đóng góp.", "",
        f"Tổng **{len(rows)} ảnh duy nhất**, **{stats['boxes']} box**; COCO 500 ảnh, thành viên 1.300 ảnh. Không tính COCO128 hoặc ảnh smoke/mẫu.", "",
        "## Split hiện tại", "", "| Tập | COCO | Thành viên | Tổng |", "|---|---:|---:|---:|",
        "| train | 400 | 910 | 1.310 |", "| validation | 100 | 260 | 360 |", "| test | 0 | 130 | 130 |", "",
        "130 ảnh test nằm ở images/test và labels/test; không tính vào 1.670 ảnh train/validation. Có 215 box book và 142 box laptop. Tổng 1.310 + 360 + 130 = 1.800 ảnh. Giữ split nguồn. Nhóm theo tên ảnh gốc trước hậu tố Roboflow; không thấy nhóm tên gốc xuyên split. Không có metadata phiên chụp nên chưa chứng minh độc lập cảnh/phiên hoặc loại gần trùng. Test hiện chỉ có nhãn book/laptop từ nguồn này, chưa là test đủ tám lớp hay test phòng học độc lập.", "",
        "## Nhãn", "", "Thứ tự: person, table, chair, laptop, cell phone, backpack, book, cup. Book nguồn ID0 → book ID6; Laptop nguồn ID1 → laptop ID3.", "",
        "| Lớp | Ảnh có nhãn | Box |", "|---|---:|---:|",
        *[f"| {name} | {count['images']} | {count['boxes']} |" for name, count in counts.items()], "",
        "Số trên là nhãn đã có, không phải số vật thật xuất hiện. Một người có thể phụ trách nhiều lớp; 1.300 ảnh nguồn thành viên hiện chỉ có nhãn book/laptop. Người phụ trách các lớp còn lại bổ sung trên những ảnh có vật tương ứng. Không tự đánh dấu chúng vắng mặt hoặc tự tạo nhãn.", "",
        "## Kiểm tự động", "",
        "1.800 ảnh giải mã được, cặp TXT đầy đủ, ID nhóm hợp lệ, box hữu hạn/dương/trong ảnh, không box trùng trong TXT, không trùng bytes/pixel trong nguồn mới hoặc với COCO500. Ảnh/nhãn khi gộp được đối chiếu checksum với bản nguồn.", "",
        "Kiểm định dạng thành công không xác nhận đầy đủ/đúng ngữ nghĩa mọi nhãn. Review người thứ hai chưa hoàn tất; checksum là snapshot draft, chưa khóa release. Baseline tám lớp cần ground truth đầy đủ trên tập đánh giá.", "",
        "## Hồ sơ", "",
        "`data/dataset/project8_v0.2/`: images/, labels/, data.yaml, manifest.csv, splits/, validation.json, statistics.json, review.csv, DATA_CARD.md, checksums.json. Raw nguồn và COCO500 giữ nguyên để truy thí nghiệm.", "",
        "Đặc tính chi tiết: [COCO500](COCO500_DATA_PROFILE.md), [bộ Laptop thành viên](LAPTOP_SUBMISSION_DATA_PROFILE.md).", "",
        "Tái tạo bản gộp trên workspace chưa có project8_v0.2: `.venv/Scripts/python.exe scripts/data/merge_project_dataset.py`. Không ghi đè phiên bản đã có; sau chỉnh nhãn tạo phiên bản mới.", ""])
    (ROOT / "docs/PROJECT8_V0_2_DATA_PROFILE.md").write_text(text, encoding="utf-8")
    (DEST / "DATA_CARD.md").write_text(text, encoding="utf-8")
    save(DEST / "checksums.json", {p.relative_to(DEST).as_posix(): sha256(p) for p in sorted(DEST.rglob("*"))
                                   if p.is_file() and p.name != "checksums.json"})
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
