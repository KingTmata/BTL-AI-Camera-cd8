"""Prepare a deterministic COCO500 draft using FiftyOne download/export APIs."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.classes import COCO_NAMES, PROJECT_NAMES
from src.coco_subset import category_mapping, coverage_review, normalized_box, ordered_candidates
from src.dataset import labels, open_image, sha256, validate, write_manifest

RAW = ROOT / "data/raw/coco500_v0.1"
DEST = ROOT / "data/dataset/coco500_v0.1"
SPLITS = {"train": ("train2017", 400), "val": ("val2017", 100)}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def items(path, key):
    import ijson
    with Path(path).open("rb") as stream:
        yield from ijson.items(stream, key+".item", use_float=True)


def select():
    references = {int(p.stem) for p in (ROOT / "data/reference/coco128/images/train2017").glob("*.jpg")}
    for split, (original, count) in SPLITS.items():
        path = RAW / "annotations" / f"instances_{original}.json"
        selection_path = RAW / f"selection_{split}.json"
        source_hash = sha256(path)
        if selection_path.exists():
            previous = json.loads(selection_path.read_text(encoding="utf-8"))
            if previous["annotation_sha256"] != source_hash:
                raise ValueError("Annotation source changed; use a new dataset version")
            print(f"Selection {split} already recorded", flush=True)
            continue
        categories = list(items(path, "categories"))
        images = list(items(path, "images"))
        mapping = category_mapping(categories)
        ordered, stats = ordered_candidates(images, items(path, "annotations"), mapping, references)
        if len(ordered) < count:
            raise ValueError(f"Too few eligible images for {split}")
        selection = {"split": split, "original_split": original, "seed": 42, "target_count": count,
                     "annotation_sha256": source_hash, "selection_method": "uniform shuffle of eligible sorted IDs; no class balancing",
                     "initial_ids": ordered[:count], "ordered_candidate_ids": ordered, "exclusions": stats,
                     "source_images": len(images), "excluded_coco128_ids": sorted(references)}
        save(selection_path, selection)
        save(RAW / f"image_metadata_{split}.json", {str(image["id"]): image for image in images})
        print(f"Selected {split}: {count}; eligible={len(ordered)}; exclusions={stats}", flush=True)
        del images
        gc.collect()


def subset_annotations(split, selected):
    original = SPLITS[split][0]
    path = RAW / "annotations" / f"instances_{original}.json"
    metadata = json.loads((RAW / f"image_metadata_{split}.json").read_text(encoding="utf-8"))
    selected_set = set(selected)
    data = {"info": {"description": "Selected original COCO2017 annotations"},
            "licenses": list(items(path, "licenses")), "categories": list(items(path, "categories")),
            "images": [metadata[str(i)] for i in selected],
            "annotations": [annotation for annotation in items(path, "annotations") if annotation["image_id"] in selected_set]}
    return data


def configure_fiftyone():
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(ROOT / ".tools/fiftyone-db"))
    os.environ.setdefault("FIFTYONE_CONFIG_DIR", str(ROOT / ".tools/fiftyone-config"))
    os.environ.setdefault("FIFTYONE_DATASET_ZOO_DIR", str(RAW / "zoo"))
    os.environ.setdefault("FIFTYONE_DISABLE_TELEMETRY", "1")


def download():
    configure_fiftyone()
    import fiftyone.utils.coco as fouc
    seen_hashes = {}
    for split, (original, count) in SPLITS.items():
        selection = json.loads((RAW / f"selection_{split}.json").read_text(encoding="utf-8"))
        ledger_path = RAW / f"download_{split}.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"accepted_ids": [], "replacements": []}
        rejected = {entry["image_id"] for entry in ledger["replacements"]}
        candidates = [i for i in selection["ordered_candidate_ids"] if i not in rejected]
        chosen = candidates[:count]
        data_dir = RAW / "fiftyone" / split
        attempts = 0
        while True:
            data = subset_annotations(split, chosen)
            # Preserve source URLs in original JSON; use the official S3 bucket
            # transport URL for the downloader to avoid plain HTTP/TLS host issues.
            transport = json.loads(json.dumps(data))
            for image in transport["images"]:
                image["coco_url"] = f"https://s3.amazonaws.com/images.cocodataset.org/{original}/{image['file_name']}"
            transport_dir = RAW / "fiftyone_transport"
            save(transport_dir / f"instances_{original}.json", transport)
            for attempt in range(2):
                try:
                    fouc.download_coco_dataset_split(str(data_dir), "train" if split == "train" else "validation",
                                                    year="2017", image_ids=chosen, label_types=["detections"],
                                                    raw_dir=str(transport_dir), num_workers=8)
                    break
                except Exception as exc:
                    print(f"{split} download attempt {attempt+1}: {type(exc).__name__}: {exc}", flush=True)
            failed = []
            hashes_this_split = {}
            for image in data["images"]:
                path = data_dir / "data" / image["file_name"]
                try:
                    with open_image(path) as opened:
                        opened.verify()
                    with open_image(path) as opened:
                        opened.load()
                        if opened.size != (image["width"], image["height"]):
                            raise ValueError("image dimensions differ from source annotations")
                        if opened.getexif().get(274, 1) != 1:
                            raise ValueError("non-normalized EXIF orientation")
                    digest = sha256(path)
                    if digest in seen_hashes or digest in hashes_this_split:
                        raise ValueError("exact duplicate bytes")
                    hashes_this_split[digest] = image["id"]
                except (ValueError, OSError, SyntaxError) as exc:
                    failed.append({"image_id": image["id"], "reason": str(exc)})
            if not failed:
                seen_hashes.update(hashes_this_split)
                ledger["accepted_ids"] = chosen
                save(ledger_path, ledger)
                save(RAW / f"selected_original_{split}.json", data)
                print(f"Downloaded and decoded {split}: {len(chosen)} unique images", flush=True)
                break
            ledger["replacements"].extend(failed)
            save(ledger_path, ledger)
            rejected.update(entry["image_id"] for entry in failed)
            chosen = [i for i in selection["ordered_candidate_ids"] if i not in rejected][:count]
            attempts += 1
            if len(chosen) < count or attempts > 10:
                raise ValueError("Download cannot fill quota; see replacement ledger")


def export():
    configure_fiftyone()
    import fiftyone as fo
    from fiftyone import ViewField as F
    if (DEST / "manifest.csv").exists():
        raise ValueError("Dataset already exported; use verify, never overwrite")
    if DEST.exists() and any(DEST.iterdir()):
        raise ValueError("Export destination already has partial data; inspect it before retrying")
    for split in SPLITS:
        dataset_name = f"coco500_v0.1_{split}"
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
        else:
            dataset = fo.Dataset.from_dir(data_path=str(RAW / "fiftyone" / split / "data"),
                                          labels_path=str(RAW / f"selected_original_{split}.json"),
                                          dataset_type=fo.types.COCODetectionDataset,
                                          label_field={"detections": "ground_truth", "coco_id": "coco_id", "license": "license"},
                                          name=dataset_name, label_types=["detections"], include_id=True,
                                          include_license=True, include_annotation_id=True)
            dataset.persistent = True
        source = json.loads((RAW / f"selected_original_{split}.json").read_text(encoding="utf-8"))
        expected = {image["file_name"] for image in source["images"]}
        imported_paths = dataset.values("filepath")
        if {Path(path).name for path in imported_paths} != expected or len(imported_paths) != SPLITS[split][1]:
            raise ValueError("FiftyOne dataset differs from selected ID list")
        view = dataset.filter_labels("ground_truth", F("label").is_in(list(COCO_NAMES.values())))
        view = view.map_labels("ground_truth", {"dining table": "table"})
        view.export(export_dir=str(DEST), dataset_type=fo.types.YOLOv5Dataset, label_field="ground_truth",
                    split=split, classes=list(PROJECT_NAMES), export_media=True)
        print(f"FiftyOne exported {split}: {len(imported_paths)} samples", flush=True)
    build_manifest()


def build_manifest():
    rows = []
    with (ROOT / "data/templates/manifest.csv").open(encoding="utf-8-sig") as stream:
        fields = next(csv.reader(stream))
    for split in SPLITS:
        data = json.loads((RAW / f"selected_original_{split}.json").read_text(encoding="utf-8"))
        license_map = {license["id"]: license for license in data["licenses"]}
        for image in sorted(data["images"], key=lambda image: image["id"]):
            row = dict.fromkeys(fields, "")
            image_id = image["id"]
            licence = license_map.get(image.get("license"), {})
            image_path = DEST / "images" / split / image["file_name"]
            label_path = DEST / "labels" / split / (Path(image["file_name"]).stem+".txt")
            row.update(image_id=f"coco_{split}_{image_id:012d}", image_path=image_path.relative_to(ROOT).as_posix(),
                       label_path=label_path.relative_to(ROOT).as_posix(), source_id=f"coco2017_{split}",
                       original_image_id=str(image_id), original_split=split,
                       source_url=image.get("flickr_url") or image.get("coco_url", ""),
                       license_or_consent=licence.get("url", "unknown"), session_id="unknown",
                       split_group_id=f"coco_image_{image_id}", split=split, annotation_status="converted",
                       annotator="COCO source annotations", review_status="pending", near_duplicate_status="pending",
                       needs_review="no", notes="Source split preserved; scene/session independence unknown. Human review pending.")
            rows.append(row)
    report, rows = validate(rows)
    if not report["valid"]:
        save(RAW / "export_validation_errors.json", report)
        raise ValueError(f"Export validation failed: {report['errors'][:10]}")
    write_manifest(DEST / "manifest.csv", rows)


def summary_numbers(values):
    import numpy as np
    values = list(values)
    if not values:
        return None
    return {"min": float(min(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(max(values))}


def verify():
    import cv2
    import numpy as np
    import yaml
    from src.dataset import read_manifest
    rows = read_manifest(DEST / "manifest.csv")
    report, rows = validate(rows)
    if not report["valid"] or len(rows) != 500:
        raise ValueError(f"Dataset validation failed: {report['errors'][:10]}")
    source = {split: json.loads((RAW / f"selected_original_{split}.json").read_text(encoding="utf-8")) for split in SPLITS}
    annotations = {}
    statistics = {"total_images": 500, "status": "draft_pending_human_review", "seed": 42,
                  "classes": list(PROJECT_NAMES), "split_counts": dict(Counter(r['split'] for r in rows)),
                  "class_distribution": {}, "image_properties": {}, "quality_flags": {}, "selection": {}, "replacements": {}}
    for split, data in source.items():
        mapping = category_mapping(data["categories"])
        image_lookup = {i["id"]: i for i in data["images"]}
        target = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation["category_id"] in mapping:
                box = normalized_box(annotation, image_lookup[annotation["image_id"]], mapping)
                target[annotation["image_id"]].append(box)
        annotations[split] = target
        statistics["selection"][split] = {k: v for k, v in json.loads((RAW / f"selection_{split}.json").read_text()).items()
                                           if k not in ('ordered_candidate_ids', 'excluded_coco128_ids')}
        statistics["replacements"][split] = json.loads((RAW / f"download_{split}.json").read_text())["replacements"]
    metrics, box_by_class, licenses, cooccurrence = [], defaultdict(list), Counter(), Counter()
    image_modes, image_formats, aspect_ratios, capture_dates = Counter(), Counter(), [], []
    for row in rows:
        split, image_id = row["split"], int(row["original_image_id"])
        actual = sorted(labels(ROOT / row["label_path"]))
        expected = sorted(annotations[split][image_id])
        # FiftyOne YOLO exporter writes six decimal places (%f).
        if len(actual) != len(expected) or any(a[0] != e[0] or not np.allclose(a[1:], e[1:], atol=1e-6, rtol=0) for a, e in zip(actual, expected)):
            raise ValueError(f"Export labels differ from original annotations: {row['image_id']}")
        image = cv2.imread(str(ROOT / row["image_path"]))
        with open_image(ROOT / row["image_path"]) as opened:
            image_modes[opened.mode] += 1
            image_formats[opened.format] += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        width, height = int(row["width"]), int(row["height"])
        aspect_ratios.append(width/height)
        original_image = next(i for i in source[split]['images'] if i['id'] == image_id)
        if original_image.get('date_captured'):
            capture_dates.append(original_image['date_captured'])
        brightness, contrast, sharpness = float(gray.mean()), float(gray.std()), float(cv2.Laplacian(gray, cv2.CV_64F).var())
        flags = []
        if brightness < 40: flags.append("dark")
        if brightness > 215: flags.append("bright")
        if contrast < 20: flags.append("low_contrast")
        if sharpness < 50: flags.append("low_laplacian_variance")
        if max(width/height, height/width) > 3: flags.append("extreme_aspect_ratio")
        if min(width, height) < 320: flags.append("low_resolution")
        if len(actual) > 30: flags.append("many_target_boxes")
        present = sorted({b[0] for b in actual})
        for left in present:
            for right in present:
                cooccurrence[(left, right)] += 1
        for category, _, _, w, h in actual:
            bbox = {"split": split, "area_ratio": w*h, "width_px": w*width, "height_px": h*height,
                    "area_px": w*width*h*height, "min_side_at_640": min(w*width, h*height)*640/max(width, height)}
            box_by_class[category].append(bbox)
            if bbox["min_side_at_640"] < 8 and "very_small_object_at_640" not in flags:
                flags.append("very_small_object_at_640")
        if flags:
            row["needs_review"] = "yes"
        metrics.append({"image_id": row["image_id"], "image_path": row["image_path"], "split": split,
                        "width": width, "height": height, "file_bytes": (ROOT / row["image_path"]).stat().st_size,
                        "brightness": round(brightness, 4), "contrast": round(contrast, 4),
                        "laplacian_variance": round(sharpness, 4), "box_count": len(actual),
                        "flags": '|'.join(flags)})
        licenses[row["license_or_consent"]] += 1
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        if len(subset) != SPLITS[split][1]: raise ValueError("Wrong split count")
        statistics["class_distribution"][split] = {}
        for index, name in enumerate(PROJECT_NAMES):
            column = name.replace(' ', '_')+'_count'
            image_count = sum(int(r[column]) > 0 for r in subset)
            if not image_count: raise ValueError(f"{split} missing class {name}; no automatic balancing")
            boxes = [b for b in box_by_class[index] if b["split"] == split]
            statistics["class_distribution"][split][name] = {"images": image_count, "boxes": len(boxes),
                "bbox_area_ratio": summary_numbers(b['area_ratio'] for b in boxes),
                "min_side_at_640": summary_numbers(b['min_side_at_640'] for b in boxes),
                "bbox_pixel_area_groups": dict(Counter('small_bbox' if b['area_px'] < 32**2 else 'medium_bbox' if b['area_px'] < 96**2 else 'large_bbox' for b in boxes))}
    statistics["image_properties"] = {key: summary_numbers(m[key] for m in metrics) for key in
                                       ("width", "height", "file_bytes", "brightness", "contrast", "laplacian_variance", "box_count")}
    statistics["quality_flags"] = dict(Counter(flag for m in metrics for flag in m["flags"].split('|') if flag))
    statistics["flagged_images"] = sum(bool(m['flags']) for m in metrics)
    statistics["licenses"] = dict(licenses)
    statistics["formats"] = dict(image_formats)
    statistics["color_modes"] = dict(image_modes)
    statistics["aspect_ratio_width_over_height"] = summary_numbers(aspect_ratios)
    statistics["source_date_captured"] = {"available": len(capture_dates), "min": min(capture_dates) if capture_dates else None,
                                          "max": max(capture_dates) if capture_dates else None,
                                          "note": "COCO metadata date_captured; not independently verified shooting sessions"}
    statistics["class_cooccurrence_images"] = {PROJECT_NAMES[i]: {PROJECT_NAMES[j]: cooccurrence[(i,j)] for j in range(8)} for i in range(8)}
    write_manifest(DEST / "manifest.csv", rows)
    write_manifest(DEST / "image_properties.csv", metrics)
    review_targets = coverage_review(rows)
    selected_review_ids = {r['image_id'] for r in review_targets}
    review_targets += [r for r in rows if r['needs_review'] == 'yes' and r['image_id'] not in selected_review_ids]
    review = [{"image_id": r['image_id'], "image_path": r['image_path'], "split": r['split'],
               "review_reason": "stratified_class_coverage_20_percent" if r['image_id'] in selected_review_ids else "additional_quality_flag",
               "reviewer": "", "review_date": "", "status": "pending", "resolution": "", "notes": ""} for r in review_targets]
    write_manifest(DEST / "review.csv", review)
    statistics["review"] = {"base_train": 80, "base_val": 20, "additional_flagged": len(review)-100, "total_pending": len(review)}
    config = {"path": str(DEST), "train": "images/train", "val": "images/val", "names": dict(enumerate(PROJECT_NAMES))}
    (DEST / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    split_dir = DEST / "splits"
    split_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        (split_dir / f"{split}.txt").write_text(''.join(r['image_path']+'\n' for r in rows if r['split'] == split), encoding="utf-8")
        (split_dir / f"{split}_coco_ids.txt").write_text(''.join(r['original_image_id']+'\n' for r in rows if r['split'] == split), encoding="utf-8")
    save(DEST / "validation.json", report)
    save(DEST / "statistics.json", statistics)
    write_profile(statistics)
    snapshot = {p.relative_to(DEST).as_posix(): sha256(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'checksums.json'}
    save(DEST / "checksums.json", snapshot)
    print(json.dumps({"status": "verified_draft", "images": len(rows), "splits": statistics['split_counts'],
                      "pending_review": statistics['review'], "manifest_sha256": sha256(DEST / 'manifest.csv')}, indent=2), flush=True)


def write_profile(stats):
    lines = ["# COCO500 — Hồ sơ đặc tính dữ liệu", "", "Phiên bản: `coco500_v0.1`; cập nhật 02/10/2026. **500 ảnh thật, trạng thái draft chờ review người thứ hai.**", "",
             "## Nguồn, loại dữ liệu và cách chọn", "",
             "Nguồn COCO2017, 400 ảnh train2017 và 100 ảnh val2017, seed 42. Chọn ngẫu nhiên không cân bằng từ ảnh có ít nhất một trong tám lớp, loại crowd mục tiêu/annotation lỗi và ID COCO128. Giữ split gốc; nguồn không cung cấp session_id nên không khẳng định độc lập theo phiên/cảnh.", "",
             "Ảnh RGB/JPEG theo nguồn, nhãn gốc COCO JSON có category_id, bbox xywh pixel, area, iscrowd và segmentation. Dataset nhóm xuất YOLO TXT: class_id cx cy w h chuẩn hóa; sử dụng detection bbox, không dùng segmentation. Metadata nguồn gồm image ID, kích thước, URL ảnh/Flickr, license và date_captured khi có. Giữ annotation gốc và danh sách lựa chọn trong data/raw/coco500_v0.1/.", "",
             "Tám lớp: person, table, chair, laptop, cell phone, backpack, book, cup. COCO dining table → table là đối chứng gần đúng; book theo phạm vi sách/vở nhóm nhưng nhãn COCO không tách riêng vở. Ảnh vẫn có thể chứa đối tượng ngoài tám lớp; các đối tượng đó không tham gia nhãn YOLO. Bộ chọn chỉ lấy ảnh dương tính nên không có tập ảnh âm tính chủ động.", "",
             "## Phân bố lớp thực tế", "", "Số ảnh các lớp có thể cộng vượt 500 vì một ảnh chứa nhiều lớp.", "",
             "| Lớp | Train: ảnh / box | Validation: ảnh / box | Tổng ảnh / box |", "|---|---:|---:|---:|"]
    for name in PROJECT_NAMES:
        train, val = stats['class_distribution']['train'][name], stats['class_distribution']['val'][name]
        lines.append(f"| {name} | {train['images']} / {train['boxes']} | {val['images']} / {val['boxes']} | {train['images']+val['images']} / {train['boxes']+val['boxes']} |")
    lines += ["", "## Thuộc tính ảnh", "", "| Thuộc tính | Min | Median | P95 | Max |", "|---|---:|---:|---:|---:|"]
    for key, values in stats['image_properties'].items():
        lines.append(f"| {key} | {values['min']:.2f} | {values['median']:.2f} | {values['p95']:.2f} | {values['max']:.2f} |")
    lines += ["", "Brightness/contrast tính từ grayscale 0–255; laplacian_variance là chỉ báo độ chi tiết phụ thuộc độ phân giải/nội dung, không phải kết luận ảnh mờ. CSV image_properties.csv lưu từng ảnh.", "", "## Kích thước vật thể", "",
              "| Lớp | Box area/ảnh: median | Cạnh ngắn khi resize 640: median px | BBox small / medium / large |", "|---|---:|---:|---:|"]
    lines.insert(lines.index("## Kích thước vật thể"),
                 "Định dạng thực: " + json.dumps(stats['formats']) + "; chế độ màu: " + json.dumps(stats['color_modes']) +
                 "; tỷ lệ width/height: " + json.dumps(stats['aspect_ratio_width_over_height']) +
                 "; date_captured nguồn: " + json.dumps(stats['source_date_captured'], ensure_ascii=False) + "\n")
    for index, name in enumerate(PROJECT_NAMES):
        # Weighted distribution details remain separately available per split.
        train = stats['class_distribution']['train'][name]
        groups = Counter(train['bbox_pixel_area_groups']) + Counter(stats['class_distribution']['val'][name]['bbox_pixel_area_groups'])
        lines.append(f"| {name} | {train['bbox_area_ratio']['median']:.5f} (train) | {train['min_side_at_640']['median']:.2f} (train) | {groups['small_bbox']} / {groups['medium_bbox']} / {groups['large_bbox']} |")
    lines += ["", "Phân nhóm dùng diện tích **bounding box** <32², 32²–96², ≥96² pixel; không đồng nhất với phân nhóm COCO theo diện tích annotation/mask. Chi tiết min/median/p95/max từng lớp/tập và bảng đồng xuất hiện lớp ở statistics.json.", "",
              "## Outlier và lỗi", "", "Các cờ sau chỉ tạo ứng viên để người xem, không tự loại ảnh:", "", "| Cờ | Số ảnh |", "|---|---:|"]
    for flag, count in stats['quality_flags'].items(): lines.append(f"| {flag} | {count} |")
    lines += ["", "Ngưỡng: dark mean<40; bright mean>215; low contrast std<20; Laplacian variance<50; tỷ lệ dài/ngắn>3; cạnh ảnh<320; >30 box mục tiêu; cạnh ngắn vật sau resize640<8px. Một ảnh có thể có nhiều cờ. Đây là heuristic để review; giữ nguyên mẫu ngẫu nhiên.", "",
              f"Có {stats['flagged_images']} ảnh có ít nhất một cờ. Review gốc 100 ảnh (80 train/20 val, đủ đại diện tám lớp), thêm {stats['review']['additional_flagged']} ca có cờ chưa nằm trong danh sách gốc; tổng {stats['review']['total_pending']} ca pending. Không ghi reviewer giả.", "",
              "Kiểm tự động: 500 ảnh giải mã được, đúng kích thước nguồn, 500 TXT hợp lệ, mọi box mục tiêu khớp annotation gốc, không trùng bytes/ID xuyên split, không trùng COCO128, đủ tám lớp trong cả hai split. Máy không xác nhận COCO không bỏ sót vật hoặc mọi nhãn đúng về ngữ nghĩa.", "",
              "### Loại trước chọn và thay thế sau tải"]
    for split in SPLITS:
        lines += ["", f"- {split}: {json.dumps(stats['selection'][split]['exclusions'], ensure_ascii=False)}",
                  f"- Ảnh thay thế sau tải {split}: {len(stats['replacements'][split])}; chi tiết trong download_{split}.json."]
    lines += ["", "## Giấy phép và ghi công", "", "Giấy phép ảnh lấy từ metadata từng ảnh, không coi giấy phép annotation/code/model là giấy phép của mọi ảnh. Manifest lưu URL nguồn và license. Giữ nghĩa vụ attribution/điều khoản của từng nguồn khi chia sẻ.", "", "| URL giấy phép ảnh | Số ảnh |", "|---|---:|"]
    for licence, count in stats['licenses'].items(): lines.append(f"| {licence} | {count} |")
    lines += ["", "## Giới hạn và sử dụng", "",
              "Bộ này là dataset đầu tiên để kiểm công cụ và phát triển, có nhãn nguồn sẵn; chưa chứng minh chất lượng camera phòng học. Pretrained YOLO26 dùng COCO; train2017 có overlap nguồn pretraining, val2017 là benchmark công khai, không phải test phòng học độc lập. Không suy ra fine-tune tốt hơn trên miền mới từ điểm COCO.", "",
              "Phân bố lớp không cân bằng theo thiết kế; ảnh âm tính chưa được lấy chủ động, crowd mục tiêu bị loại, bối cảnh COCO chưa được phân loại thủ công thành ánh sáng/địa điểm/phiên. Mean độ sáng không xác định ngày/đêm. Chưa có review người thứ hai hoặc video thật. Không gọi bản draft là release đã khóa.", "",
              "Khi nhận thêm dữ liệu nhóm, giữ bản này để truy thí nghiệm, kiểm nguồn/nhãn/trùng trước gộp và tạo phiên bản mới. Ghi số ảnh COCO, số ảnh nhóm và tổng ảnh duy nhất riêng.", "",
              "## Hồ sơ bàn giao", "",
              "Trong data/dataset/coco500_v0.1/: images, labels, manifest.csv, data.yaml, splits, checksums.json, DATA_CARD.md, statistics.json, validation.json, image_properties.csv, review.csv. Checksums là snapshot draft; các lần review/sửa cần phiên bản mới trước khi khóa và đo baseline.", "",
              "Nguồn tham khảo: [COCO](https://cocodataset.org/#download), [FiftyOne COCO2017](https://docs.voxel51.com/dataset_zoo/datasets/coco_2017.html).", ""]
    text = '\n'.join(lines)
    (ROOT / "docs/COCO500_DATA_PROFILE.md").write_text(text, encoding="utf-8")
    (DEST / "DATA_CARD.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "download", "export", "verify"))
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == "__main__":
    main()
