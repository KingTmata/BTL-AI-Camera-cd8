"""Baseline tuần 1: YOLO26n pretrained trên ảnh, video hoặc webcam.

Chạy từ thư mục gốc: python -m src.week1_demo --source <ảnh|video|số camera>
Đây là demo detection; tracking, đếm và cảnh báo thuộc các tuần sau.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
from src.classes import COCO_NAMES, COCO_TO_PROJECT, PROJECT_NAMES
from src.inference.capture import open_camera
COCO_CLASSES = {i: PROJECT_NAMES[j] for i, j in COCO_TO_PROJECT.items()}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Đường dẫn ảnh/video hoặc chỉ số webcam, ví dụ 0")
    parser.add_argument("--weights", default="weights/yolo26n.pt")
    parser.add_argument("--output-dir", default="runs/week1")
    parser.add_argument("--save", action="store_true", help="Lưu ảnh/video đã vẽ; mặc định không ghi hình")
    parser.add_argument("--no-window", action="store_true", help="Không mở cửa sổ; cần --max-frames cho webcam")
    parser.add_argument("--max-frames", type=int, default=0, help="0 là không giới hạn")
    parser.add_argument("--reopen-at", type=int, default=0, help="Đóng và mở lại nguồn trước frame này")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdecimal():
        return open_camera(int(source))
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy nguồn: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Không đọc được video: {path}")
    return capture


def detect(model: YOLO, frame):
    result = model.predict(
        frame,
        imgsz=640,
        device="cpu",
        conf=0.25,
        classes=list(COCO_CLASSES),
        nms=False,
        verbose=False,
    )[0]
    counts = Counter(COCO_CLASSES[int(class_id)] for class_id in result.boxes.cls.tolist())
    return result.plot(), counts


def write_summary(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã ghi: {path}")


def run_image(args: argparse.Namespace, model: YOLO, output_dir: Path) -> None:
    source = Path(args.source).expanduser()
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Không đọc được ảnh: {source}")
    start = time.perf_counter()
    annotated, counts = detect(model, frame)
    processing_ms = (time.perf_counter() - start) * 1000
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{source.stem}_detected.jpg"
        if not cv2.imwrite(str(output), annotated):
            raise RuntimeError(f"Không ghi được ảnh: {output}")
        print(f"Đã ghi: {output}")
    if not args.no_window:
        cv2.imshow("Week 1 YOLO26n - nhan phim bat ky de dong", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    write_summary(output_dir, {
        "source": str(source.resolve()), "source_type": "image",
        "model": str(Path(args.weights).resolve()), "device": "cpu",
        "size": [frame.shape[1], frame.shape[0]],
        "counts": dict(counts), "processing_ms": round(processing_ms, 2),
    })


def run_stream(args: argparse.Namespace, model: YOLO, output_dir: Path) -> None:
    is_camera = args.source.isdecimal()
    if is_camera and args.no_window and args.max_frames <= 0:
        raise ValueError("Webcam --no-window cần --max-frames để tự dừng.")
    capture = open_capture(args.source)
    writer = None
    frames = 0
    reopens = 0
    counts_total = Counter()
    elapsed_processing = 0.0
    width = height = 0
    started = time.perf_counter()
    try:
        while True:
            if args.reopen_at and frames == args.reopen_at and reopens == 0:
                capture.release()
                capture = open_capture(args.source)
                reopens += 1
                print("Đã đóng và mở lại nguồn thành công.")
            ok, frame = capture.read()
            if not ok:
                if is_camera:
                    raise RuntimeError("Webcam ngừng trả hình. Kiểm tra cáp USB và quyền camera rồi chạy lại.")
                break
            height, width = frame.shape[:2]
            start = time.perf_counter()
            annotated, counts = detect(model, frame)
            elapsed_processing += time.perf_counter() - start
            counts_total.update(counts)
            frames += 1
            if args.save:
                if writer is None:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output = output_dir / "detected.mp4"
                    fps_source = capture.get(cv2.CAP_PROP_FPS)
                    fps_output = fps_source if 1 <= fps_source <= 120 else 10
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps_output, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Không ghi được video: {output}")
                writer.write(annotated)
            if not args.no_window:
                cv2.imshow("Week 1 YOLO26n - q: dung, r: mo lai", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    capture.release()
                    capture = open_capture(args.source)
                    reopens += 1
                    print("Đã đóng và mở lại nguồn thành công.")
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    if frames == 0:
        raise RuntimeError("Nguồn mở được nhưng không đọc được frame nào.")
    write_summary(output_dir, {
        "source": args.source, "source_type": "webcam" if is_camera else "video",
        "model": str(Path(args.weights).resolve()), "device": "cpu",
        "size": [width, height], "frames_processed": frames,
        "reopen_count": reopens, "detections_by_class": dict(counts_total),
        "processing_fps": round(frames / elapsed, 2) if elapsed else 0,
        "mean_processing_ms": round(elapsed_processing * 1000 / frames, 2),
        "measurement_note": "FPS từ lúc bắt đầu đọc frame đến khi dừng; không gồm tải model hoặc hiển thị ngoài ứng dụng.",
    })


def main() -> None:
    args = parse_args()
    weights = (ROOT / args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint: {weights}")
    model = YOLO(str(weights))
    for class_id, expected in COCO_NAMES.items():
        actual = model.names.get(class_id)
        if actual != expected:
            raise RuntimeError(f"Class {class_id}: checkpoint có {actual!r}, cần {expected!r}")
    output_dir = (ROOT / args.output_dir).resolve()
    if Path(args.source).suffix.lower() in IMAGE_SUFFIXES:
        run_image(args, model, output_dir)
    else:
        run_stream(args, model, output_dir)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
