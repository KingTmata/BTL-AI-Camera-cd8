"""Build a portable Kaggle Notebook bundle from the approved project8 release."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import yaml

from src.classes import COCO_NAMES, LEGACY_PROJECT_NAMES as PROJECT_NAMES

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/dataset/project8_v0.3"
NOTEBOOK = ROOT / "notebooks/kaggle/project8_training.ipynb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checkpoint(path: Path, expected_names: tuple[str, ...] | None = None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(path))
    names = [model.names[index] for index in range(len(model.names))]
    if model.task != "detect":
        raise ValueError(f"Checkpoint không phải detection: {path}")
    if expected_names and names != list(expected_names):
        raise ValueError(f"Checkpoint không đúng thứ tự project8: {path}")
    if not expected_names:
        if len(names) != 80 or not all(model.names[index] == name for index, name in COCO_NAMES.items()):
            raise ValueError(f"Checkpoint baseline không khớp COCO80 đã biết: {path}")
    return {"task": model.task, "names": names, "sha256": sha256(path), "bytes": path.stat().st_size}


def _source_run_evidence(checkpoint: Path) -> dict | None:
    try:
        run = checkpoint.parents[1]
        run.relative_to(ROOT / "runs/train")
    except (IndexError, ValueError):
        return None
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    verification = json.loads((run / "checkpoint_validation.json").read_text(encoding="utf-8"))
    if (state.get("status") not in {"completed", "stopped"} or state.get("exit_code") != 0
            or not verification.get("valid") or not state.get("checkpoints_verified")):
        raise ValueError("Chỉ đóng gói checkpoint của run đã dừng và được kiểm hợp lệ.")
    expected = next((item for item in verification.get("checkpoints", [])
                     if Path(item.get("path", "")).resolve() == checkpoint.resolve()), None)
    if not expected or expected.get("sha256") != sha256(checkpoint):
        raise ValueError("Hash checkpoint được chọn không khớp bản ghi kiểm cuối của run.")
    return {"run_id": state.get("run_id"), "status": state.get("status"),
            "checkpoint_validation": str((run / "checkpoint_validation.json").relative_to(ROOT)),
            "sha256_at_final_verification": expected["sha256"], "bytes_at_final_verification": expected["bytes"]}


def build_bundle(checkpoint: Path, output_dir: Path) -> dict:
    """Create an unpacked export and ZIP. The approved source release is read-only."""
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise ValueError("Cần chọn một checkpoint .pt đang tồn tại.")
    allowed = [ROOT / "weights", ROOT / "runs/train"]
    if not any(checkpoint.is_relative_to(folder.resolve()) for folder in allowed):
        raise ValueError("Checkpoint phải thuộc weights/ hoặc runs/train/ của dự án.")
    source_evidence = _source_run_evidence(checkpoint)

    marker = json.loads((DATASET / "RELEASE.json").read_text(encoding="utf-8"))
    original_checksums = json.loads((DATASET / "checksums.json").read_text(encoding="utf-8"))
    manifest_path = DATASET / "manifest.csv"
    manifest_hash = sha256(manifest_path)
    if marker.get("status") != "locked" or marker.get("classes") != list(PROJECT_NAMES):
        raise ValueError("Release không còn ở trạng thái khóa đúng tám lớp.")
    if marker.get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest nguồn không khớp hash ghi trong RELEASE.")

    selected = _validate_checkpoint(checkpoint, PROJECT_NAMES)
    baseline_path = ROOT / "weights/yolo26n.pt"
    if not baseline_path.is_file():
        raise ValueError("Thiếu weights/yolo26n.pt để chấm baseline cùng protocol.")
    baseline = _validate_checkpoint(baseline_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle = output_dir / "project8_kaggle"
    bundle.mkdir()
    (bundle / "dataset").mkdir()
    rows = []
    exported_files: dict[str, dict] = {}

    def copy_verified(relative_source: str, relative_destination: str) -> None:
        source = DATASET / relative_source
        expected = original_checksums.get(relative_source)
        if not source.is_file() or not expected or sha256(source) != expected:
            raise ValueError(f"Hash release sai hoặc thiếu: {relative_source}")
        destination = bundle / "dataset" / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        exported_files[f"dataset/{relative_destination}"] = {
            "sha256": expected, "bytes": destination.stat().st_size,
            "source_release_path": relative_source,
        }

    with manifest_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in {"train", "val"}:
                continue
            old_image, old_label = row["image_path"], row["label_path"]
            source_image = Path(old_image).relative_to("data/dataset/project8_v0.3").as_posix()
            source_label = Path(old_label).relative_to("data/dataset/project8_v0.3").as_posix()
            copy_verified(source_image, source_image)
            copy_verified(source_label, source_label)
            row["source_image_path"] = old_image
            row["source_label_path"] = old_label
            row["image_path"] = source_image
            row["label_path"] = source_label
            rows.append(row)

    manifest_out = bundle / "dataset/export_manifest.csv"
    fieldnames = list(rows[0])
    with manifest_out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Preserve release metadata bytes. The full-release checksums also document the
    # intentionally omitted test media without copying any test image or label.
    for name in ("RELEASE.json", "checksums.json", "DATA_CARD.md", "mapping.json",
                 "validation.json", "statistics.json"):
        source = DATASET / name
        expected = original_checksums.get(name)
        if expected and sha256(source) != expected:
            raise ValueError(f"Metadata release không khớp checksums.json: {name}")
        destination = bundle / "dataset" / name
        shutil.copyfile(source, destination)
        exported_files[f"dataset/{name}"] = {"sha256": sha256(destination), "bytes": destination.stat().st_size}
    (bundle / "dataset/runtime_data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/val",
        "names": {i: name for i, name in enumerate(PROJECT_NAMES)},
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    runtime_yaml = bundle / "dataset/runtime_data.yaml"
    exported_files["dataset/runtime_data.yaml"] = {"sha256": sha256(runtime_yaml), "bytes": runtime_yaml.stat().st_size}
    exported_files["dataset/export_manifest.csv"] = {"sha256": sha256(manifest_out), "bytes": manifest_out.stat().st_size}

    (bundle / "checkpoints").mkdir()
    shutil.copyfile(checkpoint, bundle / "checkpoints/initial_project8.pt")
    shutil.copyfile(baseline_path, bundle / "checkpoints/baseline_yolo26n_coco80.pt")
    exported_files["checkpoints/initial_project8.pt"] = {**selected, "source": str(checkpoint.relative_to(ROOT)),
                                                          "run_evidence": source_evidence}
    exported_files["checkpoints/baseline_yolo26n_coco80.pt"] = {**baseline, "source": "weights/yolo26n.pt"}

    shutil.copyfile(NOTEBOOK, bundle / "project8_training.ipynb")
    (bundle / "kaggle_tools").mkdir()
    shutil.copyfile(ROOT / "kaggle_tools/__init__.py", bundle / "kaggle_tools/__init__.py")
    shutil.copyfile(ROOT / "kaggle_tools/evaluate.py", bundle / "kaggle_tools/evaluate.py")
    (bundle / "requirements-kaggle.txt").write_text(
        "# Keep the Kaggle-provided torch/torchvision CUDA build.\n"
        "ultralytics==8.4.165\npycocotools>=2.0.10\n", encoding="utf-8")
    readme = ("# Project8 — Kaggle Notebook handoff\n\n"
              "1. Upload this ZIP as a Kaggle Dataset and attach it to a new Kaggle Notebook.\n"
              "2. Open `project8_training.ipynb`; set CPU/GPU in Notebook Settings and enable Internet if pip must install packages.\n"
              "3. Set `BUNDLE_ZIP` to the attached ZIP path under `/kaggle/input`.\n"
              "4. Run setup and verification cells. Training is guarded by `START_TRAINING = False`; change it only when ready.\n"
              "5. Checkpoints are written each epoch to `/kaggle/working`; use Save Version → Save & Run All or download output to persist it.\n\n"
              "The bundle has train+val only; test media is excluded. The original release metadata/checksums are preserved.\n")
    (bundle / "README.md").write_text(readme, encoding="utf-8")
    for path in (NOTEBOOK, bundle / "kaggle_tools/__init__.py", bundle / "kaggle_tools/evaluate.py",
                 bundle / "requirements-kaggle.txt", bundle / "README.md"):
        relative = ("project8_training.ipynb" if path == NOTEBOOK else path.relative_to(bundle).as_posix())
        exported_files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (bundle / "FILES.json").write_text(json.dumps({
        "schema_version": 1, "files": exported_files,
        "source_release": marker["release_id"], "source_release_sha256": sha256(DATASET / "RELEASE.json"),
        "source_manifest_sha256": manifest_hash, "export_split_policy": "train+val only; test media omitted",
        "classes": list(PROJECT_NAMES), "images": {"train": 3883, "val": 798},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = output_dir / "project8_kaggle_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
    def size_prefix(prefix: str) -> int:
        return sum(record["bytes"] for name, record in exported_files.items() if name.startswith(prefix))

    return {"directory": str(bundle), "zip": str(zip_path), "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": sha256(zip_path), "files": sum(path.is_file() for path in bundle.rglob("*")),
            "sizes": {"train_images": size_prefix("dataset/images/train/"),
                      "train_labels": size_prefix("dataset/labels/train/"),
                      "val_images": size_prefix("dataset/images/val/"),
                      "val_labels": size_prefix("dataset/labels/val/"),
                      "metadata_and_yaml": sum(record["bytes"] for name, record in exported_files.items()
                                                if name.startswith("dataset/") and not name.startswith(("dataset/images/", "dataset/labels/"))),
                      "checkpoints": size_prefix("checkpoints/"),
                      "code_and_notebook": sum(record["bytes"] for name, record in exported_files.items()
                                                if name in {"project8_training.ipynb", "requirements-kaggle.txt", "README.md"}
                                                or name.startswith("kaggle_tools/")),
                      "uncompressed_total": sum(record["bytes"] for record in exported_files.values())},
            "checkpoint": selected, "baseline": baseline, "source_run": source_evidence}


def import_kaggle_result(payload: bytes) -> dict:
    """Verify a notebook result ZIP and register its project8 checkpoint locally."""
    if len(payload) > 128 * 1024 * 1024:
        raise ValueError("Result ZIP vượt giới hạn 128 MiB.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
            raise ValueError("Result ZIP phải chứa các file phẳng, không có đường dẫn con.")
        if any(item.file_size > 64 * 1024 * 1024 for item in archive.infolist()):
            raise ValueError("Một file trong result ZIP vượt giới hạn 64 MiB.")
        required = {"FILES.json", "result.json", "best.pt", "baseline_val_metrics.json", "finetuned_val_metrics.json"}
        if not required.issubset(names):
            raise ValueError("Result ZIP thiếu file provenance, checkpoint hoặc metric validation.")
        files = json.loads(archive.read("FILES.json"))
        result = json.loads(archive.read("result.json"))
        if result.get("schema_version") != 1 or result.get("status") != "completed":
            raise ValueError("Schema hoặc trạng thái Kaggle result không hợp lệ.")
        run_id = result.get("run_id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id):
            raise ValueError("run_id trong result không hợp lệ.")
        if set(files.get("files", {})) != set(names) - {"FILES.json"}:
            raise ValueError("Danh sách file trong FILES.json không khớp ZIP.")
        for name, record in files["files"].items():
            content = archive.read(name)
            if len(content) != record.get("bytes") or hashlib.sha256(content).hexdigest() != record.get("sha256"):
                raise ValueError(f"Hash file result không khớp: {name}")
        checkpoint = archive.read("best.pt")
        checkpoint_hash = hashlib.sha256(checkpoint).hexdigest()
        if checkpoint_hash != result.get("checkpoint_sha256"):
            raise ValueError("Hash best.pt không khớp provenance result.")
        metrics = json.loads(archive.read("finetuned_val_metrics.json"))
        baseline = json.loads(archive.read("baseline_val_metrics.json"))
        expected_protocol = {"task": "bbox", "imgsz": 640, "conf": .001, "iou": .70,
                             "nms": True, "max_det": 300, "agnostic_nms": False,
                             "augment": False, "split": "val"}
        if metrics.get("protocol") != expected_protocol or baseline.get("protocol") != expected_protocol:
            raise ValueError("Baseline và fine-tuned không dùng protocol validation chung đã định.")
        if metrics.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Metric fine-tuned không trỏ tới hash checkpoint đã nạp.")

    source_hashes = {sha256(ROOT / "weights/yolo26n.pt")}
    for candidate in (ROOT / "runs/train").glob("*/weights/*.pt"):
        try:
            run = candidate.parents[1]
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            validation = json.loads((run / "checkpoint_validation.json").read_text(encoding="utf-8"))
            if state.get("status") in {"completed", "stopped"} and validation.get("valid"):
                source_hashes.add(sha256(candidate))
        except (OSError, ValueError):
            continue
    if result.get("source_checkpoint_sha256") not in source_hashes:
        raise ValueError("Checkpoint nguồn trong provenance không khớp nguồn đã kiểm trên máy này.")
    from ultralytics import YOLO
    temporary = _write_temp_checkpoint(checkpoint)
    try:
        loaded = YOLO(str(temporary))
        if loaded.task != "detect" or [loaded.names[i] for i in range(len(loaded.names))] != list(PROJECT_NAMES):
            raise ValueError("Checkpoint Kaggle không phải project8 detect đúng thứ tự.")
        del loaded
    finally:
        temporary.unlink(missing_ok=True)

    destination = ROOT / "weights/kaggle" / f"{run_id}_{checkpoint_hash[:12]}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) != checkpoint_hash:
        raise ValueError("Đích weights đã tồn tại nhưng hash khác; không ghi đè.")
    if not destination.exists():
        temporary = destination.with_suffix(".pt.tmp")
        temporary.write_bytes(checkpoint)
        temporary.replace(destination)
    report_dir = ROOT / "reports/results/kaggle" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "baseline_val_metrics.json").write_bytes(json.dumps(baseline, indent=2).encode("utf-8"))
    (report_dir / "finetuned_val_metrics.json").write_bytes(json.dumps(metrics, indent=2).encode("utf-8"))
    return {"run_id": run_id, "checkpoint": destination, "checkpoint_sha256": checkpoint_hash,
            "result": result, "baseline_metrics": baseline, "metrics": metrics}


def _write_temp_checkpoint(payload: bytes) -> Path:
    directory = ROOT / ".temp"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"project8_kaggle_verify_{hashlib.sha256(payload).hexdigest()[:16]}.pt"
    path.write_bytes(payload)
    return path
