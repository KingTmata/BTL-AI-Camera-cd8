"""Detection và xử lý ảnh dùng cho giao diện xem kết quả."""

from __future__ import annotations

import io
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAMES = ("person", "bottle", "cell phone", "laptop")
COCO_TO_PROJECT = {0: 0, 39: 1, 67: 2, 63: 3}


def class_mapping(names: dict[int, str]) -> dict[int, int]:
    """Từ chối checkpoint không đúng profile để không âm thầm đổi ID."""
    if len(names) == 80 and all(names.get(i) == PROJECT_NAMES[j] for i, j in COCO_TO_PROJECT.items()):
        return dict(COCO_TO_PROJECT)
    if len(names) == 4 and [names.get(i) for i in range(4)] == list(PROJECT_NAMES):
        return {i: i for i in range(4)}
    raise ValueError("Checkpoint phải có 80 lớp COCO hoặc đúng thứ tự person, bottle, cell phone, laptop.")


def decode_image(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Ảnh vượt 20 triệu pixel. Hãy giảm kích thước trước khi mở.")
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def crop_box(frame: np.ndarray, box: list[float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Hộp không có diện tích bên trong ảnh.")
    return frame[y1:y2, x1:x2].copy()


def reference_boxes(label_text: str, width: int, height: int) -> list[dict]:
    """Đọc nhãn COCO80; output ID luôn là project4, tọa độ là pixel."""
    rows = []
    for line_number, line in enumerate(label_text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Nhãn dòng {line_number} phải có 5 cột.")
        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])
        if not (0 <= class_id < 80 and all(math.isfinite(v) for v in (x, y, w, h))
                and w > 0 and h > 0 and x - w / 2 >= -0.0001 and y - h / 2 >= -0.0001
                and x + w / 2 <= 1.0001 and y + h / 2 <= 1.0001):
            raise ValueError(f"Nhãn dòng {line_number} có ID hoặc tọa độ không hợp lệ.")
        if class_id in COCO_TO_PROJECT:
            project_id = COCO_TO_PROJECT[class_id]
            rows.append({"class_id": project_id, "class_name": PROJECT_NAMES[project_id],
                         "bbox_xyxy": [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]})
    return rows


def draw_boxes(frame: np.ndarray, rows: list[dict], reference: bool = False) -> np.ndarray:
    canvas = frame.copy()
    color = (95, 155, 25) if reference else (180, 95, 0)
    thickness = max(1, round(max(frame.shape[:2]) / 500))
    for index, row in enumerate(rows, 1):
        x1, y1, x2, y2 = map(lambda v: int(round(v)), row["bbox_xyxy"])
        text = f"#{index} {row['class_name']}"
        if "confidence" in row:
            text += f" {row['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(canvas, text, (max(0, x1), max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, max(1, thickness), cv2.LINE_AA)
    return canvas


class ImageInspector:
    """Một model tái sử dụng, có lock vì cache_resource được dùng giữa các phiên."""

    def __init__(self, weights: Path):
        from ultralytics import YOLO
        if not weights.is_file():
            raise FileNotFoundError(f"Không thấy weights: {weights}")
        self.model = YOLO(str(weights))
        self.mapping = class_mapping(self.model.names)
        self.lock = threading.Lock()

    def predict(self, frame: np.ndarray, confidence: float) -> dict:
        with self.lock:
            started = time.perf_counter()
            result = self.model.predict(frame, device="cpu", imgsz=640, conf=confidence,
                                        classes=list(self.mapping), nms=False, verbose=False)[0]
            rows = []
            for box, score, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                project_id = self.mapping[int(class_id)]
                rows.append({"class_id": project_id, "class_name": PROJECT_NAMES[project_id],
                             "model_class_id": int(class_id), "confidence": float(score), "bbox_xyxy": box})
            duration = (time.perf_counter() - started) * 1000
        return {"detections": rows, "processing_ms": duration}
