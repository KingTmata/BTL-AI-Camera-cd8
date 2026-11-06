"""Person-only COCO17 Pose inference and the project's derived Pose13 schema."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml

from classroom_ai.inference.devices import validate_device

KEYPOINT_NAMES_13 = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
POSE13_NAMES = KEYPOINT_NAMES_13
# Zero-based indices into KEYPOINT_NAMES_13, matching the Label Studio adapter.
POSE_EDGES_13 = (
    (11, 9),
    (9, 7),
    (12, 10),
    (10, 8),
    (7, 8),
    (1, 7),
    (2, 8),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (0, 1),
    (0, 2),
)
POINT_CONFIDENCE_THRESHOLD = 0.05


def _validated_points(keypoints17) -> list[list[float]]:
    try:
        array = np.asarray(keypoints17, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected exactly 17 [x, y, confidence] COCO points") from exc
    if array.shape != (17, 3):
        raise ValueError("Expected exactly 17 [x, y, confidence] COCO points")
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite pose coordinates or confidence")
    if ((array[:, 2] < 0) | (array[:, 2] > 1)).any():
        raise ValueError("Keypoint confidence must be in [0, 1]")
    return array.tolist()


def coco17_to_pose13(keypoints17) -> list[list[float] | None]:
    """Derive head + body5..16, matching Label Studio's Pose13 adapter.

    The head uses the coordinate medians and mean score of face points with
    score >= .5, falling back to all five. A score below .05 is missing;
    overlapping points remain independent. Scores never imply visibility.
    """
    points = _validated_points(keypoints17)
    face = [point for point in points[:5] if point[2] >= 0.5] or points[:5]
    head = [
        statistics.median(p[0] for p in face),
        statistics.median(p[1] for p in face),
        sum(p[2] for p in face) / len(face),
    ]
    return [p if p[2] >= POINT_CONFIDENCE_THRESHOLD else None for p in [head, *points[5:]]]


def draw_pose(image: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Draw valid per-person display points when present, otherwise raw Pose13."""
    canvas = image.copy()
    color = (30, 200, 245)
    thickness = max(1, round(max(image.shape[:2]) / 600))
    for row in rows:
        points = row.get("display_keypoints13", row.get("keypoints13"))
        if points is None:
            continue
        if len(points) != 13:
            raise ValueError("Drawing requires 13 keypoints per person")
        valid = {}
        for index, point in enumerate(points):
            if point is None:
                continue
            if len(point) != 3 or not all(math.isfinite(v) for v in point):
                continue
            x, y, score = point
            if (
                POINT_CONFIDENCE_THRESHOLD <= score <= 1
                and 0 <= x < image.shape[1]
                and 0 <= y < image.shape[0]
            ):
                valid[index] = (round(x), round(y))
        for first, second in POSE_EDGES_13:
            if first in valid and second in valid:
                cv2.line(canvas, valid[first], valid[second], color, thickness, cv2.LINE_AA)
        for point in valid.values():
            cv2.circle(canvas, point, max(2, thickness + 1), color, -1, cv2.LINE_AA)
    return canvas


def _array(data) -> np.ndarray:
    # CUDA tensors must cross the CPU boundary before NumPy conversion.
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    return np.asarray(data, dtype=float)


def _person_rows(result, height: int, width: int) -> list[dict]:
    if result.boxes is None:
        raise ValueError("Pose result is missing person boxes")
    boxes = _array(result.boxes.xyxy)
    scores = _array(result.boxes.conf)
    classes = _array(result.boxes.cls)
    count = len(boxes)
    if boxes.shape != (count, 4) or scores.shape != (count,) or classes.shape != (count,):
        raise ValueError("Boxes, scores and class indices must align")
    if not all(np.isfinite(a).all() for a in (boxes, scores, classes)):
        raise ValueError("Nonfinite person boxes, scores or classes")
    if ((scores < 0) | (scores > 1)).any() or (classes != 0).any():
        raise ValueError("Pose result must contain only person class 0 with valid scores")
    keypoints = result.keypoints
    if keypoints is None:
        if count:
            raise ValueError("Pose model returned boxes without keypoints")
        return []
    poses = _array(keypoints.data)
    if poses.shape != (count, 17, 3):
        raise ValueError("Pose results must align with boxes and use 17 triples")
    rows = []
    for box, score, points in zip(boxes.tolist(), scores.tolist(), poses):
        x1, y1, x2, y2 = box
        if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
            raise ValueError("Person box must lie inside the original image")
        raw = _validated_points(points)
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y, _ in raw):
            raise ValueError("Pose coordinates must lie inside the original image")
        converted = coco17_to_pose13(raw)
        rows.append(
            {
                "class_id": 0,
                "class_name": "person",
                "model_class_id": 0,
                "confidence": score,
                "bbox_xyxy": box,
                "keypoints17": raw,
                "keypoints13": converted,
                "keypoints13_missing": [point is None for point in converted],
                "visibility": None,
            }
        )
    return rows


def _suppress_rows(rows: list[dict], iou: float) -> list[dict]:
    """Select entire person records; never change box/keypoint indexing."""
    pending = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    kept = []
    while pending:
        first = pending.pop(0)
        kept.append(first)
        ax1, ay1, ax2, ay2 = first["bbox_xyxy"]
        remaining = []
        for other in pending:
            bx1, by1, bx2, by2 = other["bbox_xyxy"]
            intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
                0, min(ay2, by2) - max(ay1, by1)
            )
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
            if union <= 0 or intersection / union <= iou:
                remaining.append(other)
        pending = remaining
    return kept


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_state(path: Path) -> tuple[int, int, int]:
    state = path.stat()
    return state.st_size, state.st_mtime_ns, state.st_ctime_ns


@contextmanager
def _export_lock(path: Path, cancel=None):
    """OS lock releases even if an exporting worker exits unexpectedly."""
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 120
        while True:
            if cancel is not None and cancel():
                raise InterruptedError("Đã hủy khi chờ export OpenVINO.")
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Đang có worker khác export OpenVINO; thử lại sau."
                    ) from None
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_pose_ir(directory: Path, spec: dict) -> dict:
    """Reject a stale/modified export before loading executable model artifacts."""
    try:
        manifest = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        hashes = manifest["artifact_hashes"]
        if manifest["spec"] != spec or set(hashes) != {"model.xml", "model.bin", "metadata.yaml"}:
            raise ValueError("provenance mismatch")
        if any(_file_sha256(directory / name) != digest for name, digest in hashes.items()):
            raise ValueError("artifact hash mismatch")
        metadata = yaml.safe_load((directory / "metadata.yaml").read_text(encoding="utf-8"))
        if (
            metadata.get("task") != "pose"
            or metadata.get("names") != {0: "person"}
            or metadata.get("kpt_shape") != [17, 3]
            or metadata.get("end2end") is not False
            or metadata.get("imgsz") != spec["input_shape"]
        ):
            raise ValueError("metadata/head mismatch")
        return manifest
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"OpenVINO cache không hợp lệ: {directory}. {exc}") from exc


class PoseInspector:
    """One reusable local person/COCO17 model, serialized across callers."""

    def __init__(self, weights: Path, device: str = "cpu", *, cache_dir: Path | None = None):
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Không thấy weights: {weights}")
        if weights.suffix.lower() != ".pt":
            raise ValueError("Pose requires local PyTorch .pt weights")
        self.device = validate_device(device)
        # Dependency management belongs to setup, not inference. The checks module
        # may already be imported by the Detect inspector, so update both copies.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLO
        from ultralytics.utils import checks

        checks.AUTOINSTALL = False
        source_sha256 = _file_sha256(weights)
        # An absolute existing path also avoids Ultralytics' bare-name asset
        # download and corrupted-cache replacement behavior.
        self.model = YOLO(str(weights))
        if self.model.task != "pose" or self.model.names != {0: "person"}:
            raise ValueError("Pose weights must have task pose and names {0: person}")
        native = self.model.model
        profiles = []
        if isinstance(getattr(native, "yaml", None), dict):
            profiles.append(native.yaml.get("kpt_shape"))
        if hasattr(native, "kpt_shape"):
            profiles.append(native.kpt_shape)
        layers = getattr(native, "model", None)
        if layers is not None and len(layers) and hasattr(layers[-1], "kpt_shape"):
            profiles.append(layers[-1].kpt_shape)
        if not profiles or any(
            profile is None or tuple(profile) != (17, 3) for profile in profiles
        ):
            raise ValueError("Pose weights must use the COCO17 [17, 3] keypoint profile")
        self.weights = weights
        if _file_sha256(weights) != source_sha256:
            raise ValueError("Weights thay đổi trong khi nạp model; tạo lại job.")
        self.source_sha256 = source_sha256
        self._source_state = _file_state(weights)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(__file__).resolve().parents[3] / "runs/runtime/openvino_pose"
        )
        self._runtime_key = None
        self.runtime_info = {
            "backend": "openvino" if self.device == "intel:gpu" else "pytorch",
            "source_sha256": self.source_sha256,
        }
        if self.device == "intel:gpu" and getattr(native, "end2end", False):
            raise ValueError(
                "OpenVINO Pose yêu cầu weights one-to-many (end2end=False) để giữ nguyên head."
            )
        self.lock = threading.Lock()
        self._cancel = lambda: False

    def set_cancel(self, cancel=None):
        """Set/reset the current job's cancellation check on a cached Inspector.

        Lock waits are interruptible. Native export/compile/predict calls cannot
        be force-interrupted; cancellation is checked as soon as they return.
        """
        self._cancel = cancel if cancel is not None else lambda: False

    def _check_cancel(self):
        if self._cancel():
            raise InterruptedError("Đã hủy xử lý Pose.")

    def _prepare_openvino(self, frame: np.ndarray, imgsz: int) -> list[int]:
        self._check_cancel()
        import openvino as ov
        import torch
        import ultralytics

        # Match PyTorch single-frame rect letterbox: first round the long side
        # to stride, resize with round(), then pad each dimension to stride.
        side = math.ceil(imgsz / 32) * 32
        gain = min(side / frame.shape[0], side / frame.shape[1])
        shape = [math.ceil(round(dimension * gain) / 32) * 32 for dimension in frame.shape[:2]]
        spec = {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "imgsz": imgsz,
            "input_shape": shape,
            "versions": {
                "openvino": ov.__version__,
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
            },
            "task": "pose",
            "names": {"0": "person"},
            "kpt_shape": [17, 3],
            "head": "one-to-many",
            "nms": None,
            "quantize": 32,
        }
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        # A cached Inspector may outlive the source path. Check stat every frame,
        # hashing only on change so large checkpoints do not add per-frame I/O.
        state = _file_state(self.weights)
        if state != self._source_state:
            if _file_sha256(self.weights) != self.source_sha256:
                raise ValueError("Weights gốc đã thay đổi sau khi nạp; tạo job mới.")
            self._source_state = state
        if key == self._runtime_key:
            return shape
        if _file_sha256(self.weights) != self.source_sha256:
            raise ValueError(
                "Weights gốc đã thay đổi sau khi nạp; tạo job mới để dùng weights mới."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Ultralytics identifies the backend from the directory suffix.
        directory = self.cache_dir / f"{key}_openvino_model"
        with _export_lock(self.cache_dir / f"{key}.lock", cancel=self._cancel):
            self._check_cancel()
            cache_hit = directory.exists()
            if not cache_hit:
                with tempfile.TemporaryDirectory(prefix="export-", dir=self.cache_dir) as temporary:
                    source_copy = Path(temporary) / "model.pt"
                    shutil.copyfile(self.weights, source_copy)
                    if _file_sha256(source_copy) != self.source_sha256:
                        raise ValueError("Weights thay đổi trong khi sao chép export.")
                    self._check_cancel()
                    exported = Path(
                        ultralytics.YOLO(str(source_copy)).export(
                            format="openvino",
                            imgsz=shape,
                            batch=1,
                            dynamic=False,
                            quantize=32,
                            nms=None,
                            device="cpu",
                            verbose=False,
                        )
                    )
                    self._check_cancel()
                    manifest = {
                        "spec": spec,
                        "artifact_hashes": {
                            name: _file_sha256(exported / name)
                            for name in ("model.xml", "model.bin", "metadata.yaml")
                        },
                    }
                    (exported / "provenance.json").write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )
                    _validate_pose_ir(exported, spec)
                    os.replace(exported, directory)
            manifest = _validate_pose_ir(directory, spec)
        self._check_cancel()
        model = ultralytics.YOLO(str(directory), task="pose")
        self.runtime_info = {
            "backend": "openvino",
            "cache_key": key,
            "cache_hit": cache_hit,
            "source_sha256": self.source_sha256,
            "artifact_dir": str(directory.resolve()),
            "artifact_hashes": manifest["artifact_hashes"],
            "input_shape": shape,
            "versions": spec["versions"],
            "head": "one-to-many",
            "export_nms": None,
        }

        def ensure_gpu(predictor):
            self._check_cancel()  # called immediately after native compilation
            compiled = getattr(predictor.model, "ov_compiled_model", None)
            actual = (
                list(compiled.get_property("EXECUTION_DEVICES")) if compiled is not None else []
            )
            if not actual or not all(device.startswith("GPU") for device in actual):
                raise RuntimeError(
                    f"OpenVINO không chạy trên GPU Intel ({actual}); không cho phép CPU fallback."
                )
            if predictor.model.end2end is not False:
                raise RuntimeError("OpenVINO Pose head khác weights gốc; dừng inference.")
            self.runtime_info.update(
                execution_devices=actual,
                precision_hint=str(compiled.get_property("INFERENCE_PRECISION_HINT")),
            )

        model.add_callback("on_predict_start", ensure_gpu)
        self.model, self._runtime_key = model, key
        return shape

    def predict(
        self,
        frame: np.ndarray,
        confidence: float,
        *,
        imgsz: int = 640,
        extra_nms_iou: float | None = None,
        passes: int = 1,
    ) -> dict:
        if type(imgsz) is not int or imgsz <= 0:
            raise ValueError("Kích thước suy luận phải là số nguyên dương.")
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Confidence must be in (0, 1]")
        if extra_nms_iou is not None and not 0 <= extra_nms_iou <= 1:
            raise ValueError("Ngưỡng NMS phải nằm trong khoảng 0–1.")
        if type(passes) is not int or passes != 1:
            raise ValueError("Pose supports one inference pass; box-only TTA is not supported")
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) <= 0
        ):
            raise ValueError("Pose requires a nonempty BGR image")
        requested = time.perf_counter()
        with self.lock:
            self._check_cancel()
            lock_wait_ms = (time.perf_counter() - requested) * 1000
            started = time.perf_counter()
            shape = self._prepare_openvino(frame, imgsz) if self.device == "intel:gpu" else imgsz
            options = {"rect": False} if self.device == "intel:gpu" else {}
            result = self.model.predict(
                frame,
                device=self.device,
                imgsz=shape,
                conf=confidence,
                classes=[0],
                nms=False,
                augment=False,
                verbose=False,
                save=False,
                **options,
            )[0]
            self._check_cancel()
            rows = _person_rows(result, *frame.shape[:2])
            if extra_nms_iou is not None:
                rows = _suppress_rows(rows, extra_nms_iou)
            duration = (time.perf_counter() - started) * 1000
        return {
            "detections": rows,
            "processing_ms": duration,
            "device": self.device,
            "inspector_lock_wait_ms": lock_wait_ms,
            "runtime": dict(self.runtime_info),
        }
