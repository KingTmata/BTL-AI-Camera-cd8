"""Tạo bộ 40 ảnh review project8 từ COCO128; không tự duyệt nhãn/dự đoán."""

import argparse
import csv
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.detector import ImageInspector, PROJECT_NAMES, draw_boxes, reference_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/week1/project8_review40"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    candidates = []
    base = ROOT / "data/reference/coco128"
    for image in sorted((base / "images/train2017").glob("*.jpg")):
        label = base / "labels/train2017" / f"{image.stem}.txt"
        if label.is_file():
            refs = reference_boxes(label.read_text(), 1, 1)
            if refs:
                candidates.append((image, label, {row["class_name"] for row in refs}))
    selected = []
    for name in PROJECT_NAMES:
        item = next((c for c in candidates if name in c[2]), None)
        if item is None:
            raise RuntimeError(f"COCO128 has no reference for class {name}")
        if item not in selected:
            selected.append(item)
    selected += [item for item in candidates if item not in selected][:40-len(selected)]
    if len(selected) != 40:
        raise RuntimeError("Need 40 source images with project8 labels")
    engine = ImageInspector(ROOT / "weights/yolo26n.pt")
    output.mkdir(parents=True)
    rows, predictions, cards = [], [], []
    for image, label, names in selected:
        frame = cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f"Unreadable image: {image}")
        refs = reference_boxes(label.read_text(), frame.shape[1], frame.shape[0])
        result = engine.predict(frame, 0.25)
        panels = []
        for caption, canvas in (("COCO reference - 8 project classes", draw_boxes(frame, refs, reference=True)),
                                ("YOLO26n pretrained prediction", draw_boxes(frame, result["detections"]))):
            scale = 500 / max(canvas.shape[:2])
            resized = cv2.resize(canvas, (max(1, round(canvas.shape[1]*scale)), max(1, round(canvas.shape[0]*scale))))
            panel = np.full((540, 510, 3), 255, np.uint8)
            panel[35:35+resized.shape[0], 5:5+resized.shape[1]] = resized
            cv2.putText(panel, caption, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            panels.append(panel)
        preview = f"{image.stem}.jpg"
        if not cv2.imwrite(str(output / preview), np.concatenate(panels, axis=1)):
            raise RuntimeError(f"Cannot write preview: {preview}")
        classes = "|".join(name for name in PROJECT_NAMES if name in names)
        rows.append({"image_id": image.stem, "image_path": image.relative_to(ROOT).as_posix(),
                     "reference_classes": classes, "reference_boxes": len(refs),
                     "predicted_boxes": len(result["detections"]), "reviewer": "", "missing_objects": "",
                     "false_positives": "", "box_or_class_issue": "", "review_status": "pending", "notes": ""})
        predictions.append({"image_id": image.stem, **result})
        cards.append(f'<article><h2>{html.escape(image.stem)} · {html.escape(classes)}</h2><img src="{preview}" alt="Reference and prediction for {image.stem}"></article>')
    with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    page = '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Review tuần 1 · project8</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:0 20px}img{width:100%;height:auto}article{margin:40px 0}h2{font-size:18px}</style><h1>40 ảnh review tuần 1 · tám lớp</h1><p>COCO128 là dữ liệu tham khảo, không phải validation độc lập. Bên trái là nhãn COCO; bên phải là dự đoán pretrained ở confidence 0.25. Người review ghi lỗi và kết quả thật trong review.csv; tất cả ảnh hiện đang pending.</p>' + ''.join(cards) + '</html>'
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Prepared {len(rows)} review images, all pending. Open {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
