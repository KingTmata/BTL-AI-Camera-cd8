"""Report current dataset counts and storage separately, using local artifacts."""
import argparse
import csv
import json
import os
import sys
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dataset import read_manifest, verify_release

EXCLUDED = {".git", ".venv", ".python", ".tools", ".temp", ".codex", ".opencode", ".uv-cache",
            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", "node_modules"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def storage_inventory():
    counts, archives = Counter(), []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in EXCLUDED]
        for name in names:
            path = Path(base) / name
            counts["files"] += 1
            counts["images"] += path.suffix.lower() in IMAGE_SUFFIXES
            counts["labels"] += path.suffix == ".txt" and "labels" in path.relative_to(ROOT).parts
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    infos = [i for i in archive.infolist() if not i.is_dir()]
                    archives.append({"path": path.relative_to(ROOT).as_posix(),
                                     "images": sum(Path(i.filename).suffix.lower() in IMAGE_SUFFIXES for i in infos)})
    return dict(counts), archives


def table(headers, rows):
    def cell(value):
        return f"{value:,}".replace(",", ".") if isinstance(value, int) else str(value)
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/dataset/project8_v0.3")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/results/project_progress_20261004")
    args = parser.parse_args()
    output, dataset = args.output.resolve(), args.dataset.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Report output must be a new directory inside this checkout")
    snapshot = verify_release(dataset)
    stats = json.loads((dataset / "statistics.json").read_text(encoding="utf-8"))
    validation = json.loads((dataset / "validation.json").read_text(encoding="utf-8"))
    rows = read_manifest(dataset / "inventory_all.csv")
    output.mkdir(parents=True)
    for name in ("REPORT.md", "inventory.json", "summary.csv", "classes.csv"):
        (output / name).touch()
    storage, archives = storage_inventory()
    summary = [["Toàn checkout (loại môi trường/cache/Git)", storage["files"], storage["images"], storage["labels"], "—"],
               ["Dataset chính", sum(p.is_file() for p in dataset.rglob("*")), stats["inventory_images"], stats["inventory_labels"], stats["inventory_boxes"]]]
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        boxes = sum(sum(int(row[name.replace(" ", "_") + "_count"]) for name in stats["class_distribution"]) for row in subset)
        summary.append([split, "—", len(subset), len(subset), boxes])
    summary.append(["Pending", "—", stats["pending_images"], stats["pending_images"], stats["inventory_boxes"] - stats["boxes"]])
    classes = [[name, count["images"], count["boxes"]] for name, count in stats["inventory_class_distribution"].items()]
    sources = [[name, count] for name, count in stats["sources_inventory"].items()]
    headers = ["Phạm vi", "File", "Ảnh", "TXT nhãn", "Box"]
    write_csv(output / "summary.csv", headers, summary)
    write_csv(output / "classes.csv", ["Lớp", "Ảnh có nhãn", "Box"], classes)
    timestamp = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    train_artifacts = [p.relative_to(ROOT).as_posix() for p in (ROOT / "runs").rglob("*")
                       if p.name in {"best.pt", "last.pt", "results.csv", "args.yaml"}]
    text = ["# Báo cáo tiến độ dự án tuần 1–2–3", "", "Cập nhật: " + timestamp + " (Asia/Saigon).", "",
            "## Kiểm kê hiện hành", "", table(headers, summary), "",
            "File trong checkout bao gồm code, tài liệu, metadata và báo cáo. Không cộng ảnh trong ZIP hoặc cộng số ảnh từng lớp thành ảnh độc lập.", "",
            "## Nguồn ảnh chính", "", table(["Nguồn", "Ảnh"], sources), "",
            "Tên nguồn v0.2/COCO/Laptop là provenance lịch sử, không yêu cầu giữ thêm bản sao ảnh. Nhãn nguồn chỉ là draft, chưa được gán đủ mọi vật hoặc review người thứ hai.", "",
            "## Nhãn từng lớp", "", table(["Lớp", "Ảnh có nhãn", "Box"], classes), "",
            "## Gói nguồn còn giữ riêng", "", table(["Archive", "Lượt ảnh"], [[a["path"], a["images"]] for a in archives]), "",
            "Hai gói hành vi gốc phục vụ tuần 4–6 được giữ riêng. 600 ảnh được chọn từ nguồn này đã có trong dataset chính; không cộng lại vào số ảnh chính.", "",
            "## Tiến độ theo tuần", "", table(["Tuần", "Đã có", "Còn thiếu"], [
                [1, "App ảnh/video/webcam và pretrained; giao diện dùng ảnh train chính", "Webcam thiết bị thật và review người; ảnh COCO128/review cũ đã dọn theo yêu cầu"],
                [2, f"Dataset draft {stats['inventory_images']} ảnh, manifest/YAML/checksum/validator/evaluator", "Nhãn đủ tám lớp, review, quyền/gần trùng/phiên, khóa release, video thật và baseline validation"],
                [3, f"Cấu hình và evaluator; {len(train_artifacts)} artifact training trong runs", "Chưa fine-tune 26n/26s hoặc có bảng validation chung"]]), "",
            "## Kiểm tra và dọn dữ liệu", "",
            "Mapping classroom: Student→person, chair→chair, table và with-student→table theo quyết định người dùng. Giữ nguyên tọa độ nguồn; không tạo ground truth giả.",
            "107 ảnh pending cùng TXT đã xóa theo yêu cầu; không đổi media/nhãn/split của 5.002 cặp giữ lại. Metadata nguồn và danh sách xóa được lưu trong audit; dữ liệu chưa được duyệt hoặc train.",
            f"Validator đã kiểm {validation['images_checked']:,} ảnh, valid={validation['valid']}; checksum hiện tại được kiểm lại khi tạo báo cáo. Kiểm gần trùng/độc lập cảnh vẫn cần xác nhận của người.",
            "Tập test hiện chưa có nhãn backpack và cup; chưa đủ cơ sở đánh giá cả tám lớp trên test. Nhãn nguồn còn thiếu vật cũng có thể ảnh hưởng phép đánh giá.",
            "Ảnh mẫu giao diện lấy từ train, không đưa test vào luồng xem mẫu mặc định.", ""]
    cleanup = stats.get("cleanup", {})
    cleanup_rows = [["Bản sao ảnh", cleanup.get("duplicate_images_deleted", 0)],
                    ["Ảnh chờ xử lý", cleanup.get("pending_deleted", 0)],
                    ["Ảnh tham khảo/kết quả thử", cleanup.get("reference_generated_images_deleted", 0)],
                    ["ZIP nguồn đã nhập/tham khảo", cleanup.get("source_archives_deleted", 0)]]
    text += [table(["Đã xóa", "Số lượng"], cleanup_rows), "",
             "Audit dọn dữ liệu và code: [primary_cleanup_20261004](../primary_cleanup_20261004/REPORT.md).", ""]
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    (output / "inventory.json").write_text(json.dumps({"timestamp": timestamp, "storage": storage, "archives": archives,
        "primary": stats, "validation": validation, "dataset_checksum": snapshot, "training_artifacts": train_artifacts}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output / "REPORT.md"), "storage": storage, "primary_images": stats["inventory_images"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
