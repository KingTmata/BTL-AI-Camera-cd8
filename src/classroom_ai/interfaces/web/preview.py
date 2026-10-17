"""Preview settings and JPEG rendering; never run inference or mutate worker data."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from classroom_ai.inference.detector import draw_boxes


@dataclass(frozen=True)
class PreviewSettings:
    fps: float = 10
    max_width: int = 640
    jpeg_quality: int = 75

    def __post_init__(self):
        if (
            type(self.fps) not in (int, float)
            or not math.isfinite(self.fps)
            or self.fps <= 0
            or not math.isfinite(1 / self.fps)
        ):
            raise ValueError("runtime.preview_fps phải là số hữu hạn dương.")
        if type(self.max_width) is not int or self.max_width <= 0:
            raise ValueError("runtime.preview_max_width phải là số nguyên dương.")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("runtime.preview_jpeg_quality phải là số nguyên từ 1 đến 100.")


def load_preview_settings(path: Path) -> PreviewSettings:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("runtime", {}), dict):
        raise ValueError("configs/app.yaml cần có runtime dạng mapping.")
    runtime = config.get("runtime", {})
    return PreviewSettings(
        runtime.get("preview_fps", 10),
        runtime.get("preview_max_width", 640),
        runtime.get("preview_jpeg_quality", 75),
    )


def preview_jpeg(frame, rows, settings: PreviewSettings) -> bytes:
    height, width = frame.shape[:2]
    if width > settings.max_width:
        target_height = max(1, round(height * settings.max_width / width))
        frame = cv2.resize(frame, (settings.max_width, target_height), interpolation=cv2.INTER_AREA)
        rows = [
            {
                **row,
                "bbox_xyxy": [
                    v * (settings.max_width / width if i % 2 == 0 else target_height / height)
                    for i, v in enumerate(row["bbox_xyxy"])
                ],
            }
            for row in rows
        ]
    # draw_boxes copies the image; shared camera frames/results stay read-only.
    ok, encoded = cv2.imencode(
        ".jpg", draw_boxes(frame, rows), [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
    )
    if not ok:
        raise ValueError("Không tạo được ảnh JPEG preview.")
    return encoded.tobytes()
