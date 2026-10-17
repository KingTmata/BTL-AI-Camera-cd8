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

from classroom_ai.classes import COCO_NAMES, COCO_TO_PROJECT, LEGACY_PROJECT_NAMES, PROJECT_NAMES
from classroom_ai.inference.devices import validate_device


def class_mapping(names: dict[int, str]) -> dict[int, int]:
    """Từ chối checkpoint không đúng profile để không âm thầm đổi ID."""
    if len(names) == 80 and all(names.get(i) == name for i, name in COCO_NAMES.items()):
        return dict(COCO_TO_PROJECT)
    ordered = tuple(names.get(i) for i in range(len(PROJECT_NAMES)))
    if len(names) == len(PROJECT_NAMES) and ordered in (PROJECT_NAMES, LEGACY_PROJECT_NAMES):
        return {i: i for i in range(len(PROJECT_NAMES))}
    raise ValueError(
        "Checkpoint phải là COCO80 hoặc project8 đúng thứ tự: " + ", ".join(PROJECT_NAMES)
    )


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


def reference_boxes(
    label_text: str, width: int, height: int, profile: str = "coco80"
) -> list[dict]:
    """Đọc nhãn theo profile tường minh; output ID project8, tọa độ pixel."""
    if profile not in ("coco80", "project8"):
        raise ValueError("Profile nhãn phải là coco80 hoặc project8")
    mapping = COCO_TO_PROJECT if profile == "coco80" else {i: i for i in range(len(PROJECT_NAMES))}
    class_count = 80 if profile == "coco80" else len(PROJECT_NAMES)
    rows = []
    for line_number, line in enumerate(label_text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Nhãn dòng {line_number} phải có 5 cột.")
        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])
        if not (
            0 <= class_id < class_count
            and all(math.isfinite(v) for v in (x, y, w, h))
            and w > 0
            and h > 0
            and x - w / 2 >= -0.0001
            and y - h / 2 >= -0.0001
            and x + w / 2 <= 1.0001
            and y + h / 2 <= 1.0001
        ):
            raise ValueError(f"Nhãn dòng {line_number} có ID hoặc tọa độ không hợp lệ.")
        if class_id in mapping:
            project_id = mapping[class_id]
            rows.append(
                {
                    "class_id": project_id,
                    "class_name": PROJECT_NAMES[project_id],
                    "bbox_xyxy": [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        (x + w / 2) * width,
                        (y + h / 2) * height,
                    ],
                }
            )
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
        cv2.putText(
            canvas,
            text,
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(1, thickness),
            cv2.LINE_AA,
        )
    return canvas


class ImageInspector:
    """Một model tái sử dụng, có lock vì cache_resource được dùng giữa các phiên."""

    def __init__(self, weights: Path, device: str = "cpu"):
        from ultralytics import YOLO

        if not weights.is_file():
            raise FileNotFoundError(f"Không thấy weights: {weights}")
        self.device = validate_device(device)
        self.model = YOLO(str(weights))
        self.weights = weights
        self.multi_engine = None
        self.mapping = class_mapping(self.model.names)
        self.lock = threading.Lock()

    def predict(
        self,
        frame: np.ndarray,
        confidence: float,
        *,
        imgsz: int = 640,
        extra_nms_iou: float | None = None,
        passes: int = 1,
    ) -> dict:
        if not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("Kích thước suy luận phải là số nguyên dương.")
        if extra_nms_iou is not None and not 0 <= extra_nms_iou <= 1:
            raise ValueError("Ngưỡng NMS phải nằm trong khoảng 0–1.")
        if passes not in (1, 3):
            raise ValueError("Chọn một hoặc ba lượt suy luận.")
        if passes == 3 and extra_nms_iou is None:
            raise ValueError("Ba lượt suy luận cần NMS để gộp kết quả.")
        lock_requested = time.perf_counter()
        with self.lock:
            lock_wait_ms = (time.perf_counter() - lock_requested) * 1000
            started = time.perf_counter()
            engine = self.model
            if passes == 3:
                from ultralytics import YOLO

                from classroom_ai.inference.variants import EndToEndViews

                if self.multi_engine is None:
                    self.multi_engine = YOLO(str(self.weights))
                    native = self.multi_engine.model
                    wrapper = EndToEndViews(
                        [native],
                        views=((1.0, False), (0.83, True), (0.67, False)),
                        iou=extra_nms_iou,
                        conf=confidence,
                    )
                    wrapper.args = native.args
                    wrapper.pt_path = str(self.weights)
                    self.multi_engine.model = wrapper
                wrapper = self.multi_engine.model
                wrapper.iou, wrapper.conf = extra_nms_iou, confidence
                engine = self.multi_engine
            result = engine.predict(
                frame,
                device=self.device,
                imgsz=imgsz,
                conf=confidence,
                classes=list(self.mapping),
                nms=False,
                augment=False,
                verbose=False,
            )[0]
            if extra_nms_iou is not None and passes == 1:
                from classroom_ai.inference.variants import merge_detections

                result.boxes.data = merge_detections(
                    [result.boxes.data], iou=extra_nms_iou, conf=confidence
                )
            rows = []
            for box, score, class_id in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
            ):
                project_id = self.mapping[int(class_id)]
                rows.append(
                    {
                        "class_id": project_id,
                        "class_name": PROJECT_NAMES[project_id],
                        "model_class_id": int(class_id),
                        "confidence": float(score),
                        "bbox_xyxy": box,
                    }
                )
            duration = (time.perf_counter() - started) * 1000
        return {
            "detections": rows,
            "processing_ms": duration,
            "device": self.device,
            "inspector_lock_wait_ms": lock_wait_ms,
        }
