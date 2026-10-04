"""Detached CPU training worker. Invoke only through RunManager."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import os
import platform
import time
import traceback
from pathlib import Path

import psutil
import yaml

from src.classes import PROJECT_NAMES
from src.dataset import read_manifest, sha256, validate, verify_release
from src.inference.detector import class_mapping
from src.training.manager import RunManager, atomic_json, read_json
from src.training.progress import TrainingProgress
from src.training.options import ADVANCED_DEFAULTS


def preflight(request, root):
    root = Path(root).resolve()
    cfg = request["config"]
    RunManager(root).build_config(epochs=cfg["epochs"], batch=cfg["batch"], imgsz=cfg["imgsz"],
                                  workers=cfg["workers"], cache=cfg["cache"], checkpoint=cfg["model"],
                                  advanced={key: cfg[key] for key in ADVANCED_DEFAULTS if key in cfg})
    if cfg["device"] != "cpu" or cfg["fraction"] != 1.0 or cfg["resume"]:
        raise ValueError("Run UI phải dùng CPU, toàn train và khởi tạo run mới.")
    data_file = Path(cfg["data"]).resolve()
    if data_file != root / "configs/data.yaml":
        raise ValueError("YAML dataset phải là configs/data.yaml của dự án.")
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    folder = Path(data["path"]).resolve()
    if not folder.is_relative_to(root / "data/dataset") or folder == root / "data/dataset":
        raise ValueError("Dataset nằm ngoài kho dự án.")
    if data.get("download") or list(data["names"].values()) != list(PROJECT_NAMES):
        raise ValueError("Mapping YAML phải đúng tám lớp; không dùng YAML tự tải dữ liệu.")
    frozen_data = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))
    if data != frozen_data:
        raise ValueError("configs/data.yaml không khớp YAML snapshot đã khóa.")
    marker = read_json(folder / "RELEASE.json")
    if not marker or marker.get("status") != "locked" or marker.get("classes") != list(PROJECT_NAMES):
        raise ValueError("Dataset chưa có snapshot đã khóa đúng tám lớp.")
    release_hash = verify_release(folder)
    manifest = folder / "manifest.csv"
    validation, rows = validate(read_manifest(manifest), root, release=True)
    if not validation["valid"]:
        raise ValueError("Release validation thất bại: " + "; ".join(validation["errors"][:5]))
    if cfg["cache"]:
        estimate = sum(r["split"] in {"train", "val"} for r in rows) * cfg["imgsz"]**2 * 3 * 1.3
        if estimate + 2*1024**3 > psutil.virtual_memory().available:
            raise ValueError("RAM cache ảnh vượt RAM khả dụng. Tắt cache để chạy CPU.")
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT, check_dict_alignment
    from src.evaluation import PROTOCOL
    check_dict_alignment(DEFAULT_CFG_DICT, cfg)
    checkpoint = Path(cfg["model"])
    checkpoint_hash = sha256(checkpoint)
    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError("Checkpoint phải thuộc tác vụ detection.")
    mapping = class_mapping(model.names)
    memory = psutil.virtual_memory()
    metadata = {"dataset": str(folder), "manifest_sha256": sha256(manifest),
                "release_sha256": release_hash, "checkpoint_sha256": checkpoint_hash,
                "class_mapping": mapping, "classes": list(PROJECT_NAMES),
                "images": {s: validation["counts"][s]["images"] for s in validation["counts"]},
                "counts": validation["counts"], "comparison_protocol": PROTOCOL,
                "training_validation_note": "Trainer validation is NMS-free; common comparison protocol uses NMS.",
                "python": platform.python_version(), "platform": platform.platform(),
                "processor": platform.processor(), "ram_total_bytes": memory.total,
                "ram_available_before_training_bytes": memory.available,
                "versions": {n: importlib.metadata.version(n) for n in
                             ("ultralytics", "torch", "torchvision", "psutil", "pycocotools", "numpy")}}
    return metadata, model


def run_worker(run, token):
    request = read_json(Path(run) / "request.json")
    if not request or request.get("token") != token:
        raise ValueError("Request/ownership token không hợp lệ.")
    manager = RunManager(request["root"])
    run, _ = manager.owned_run(run)
    if request["config"]["name"] != run.name or Path(request["config"]["project"]).resolve() != manager.runs:
        raise ValueError("Thư mục output không khớp run đã được cấp.")
    progress = TrainingProgress(run, token, request["config"])
    progress.publish(force=True)
    try:
        print("Preflight: release checksum, human review gates, YAML, checkpoint and trainer schema.", flush=True)
        metadata, model = preflight(request, manager.root)
        import torch
        torch.set_num_threads(min(8, psutil.cpu_count(logical=False) or 4))
        torch.set_num_interop_threads(2)
        metadata["torch_threads"] = torch.get_num_threads()
        atomic_json(run / "environment.json", metadata)
        (run / "effective_config.yaml").write_text(yaml.safe_dump(request["config"], sort_keys=False), encoding="utf-8")
        progress.state["dataset_images"] = metadata["images"]
        for event, callback in progress.callbacks().items():
            model.add_callback(event, callback)
        progress.phase("setup")
        print(f"Training CPU: {metadata['images']}; full train, batch={request['config']['batch']}, "
              f"epochs={request['config']['epochs']}, threads={metadata['torch_threads']}", flush=True)
        model.train(**{k:v for k,v in request["config"].items() if k != "model"})
        progress.phase("verification")
        del model
        gc.collect()
        expected = [run / "weights/last.pt", run / "weights/best.pt"]
        expected += [run / f"weights/epoch{i}.pt" for i in range(progress.state["epochs_completed"])]
        if not progress.state["epochs_completed"]:
            raise RuntimeError("Không có epoch nào được lưu checkpoint.")
        from ultralytics import YOLO
        checkpoint_records = []
        for checkpoint in expected:
            if not checkpoint.is_file():
                raise RuntimeError(f"Thiếu checkpoint: {checkpoint.name}")
            loaded = YOLO(str(checkpoint))
            if list(loaded.names.values()) != list(PROJECT_NAMES) or loaded.task != "detect":
                raise RuntimeError(f"Mapping checkpoint {checkpoint.name} không đúng project8.")
            checkpoint_records.append({"path": str(checkpoint), "sha256": sha256(checkpoint),
                                       "names": loaded.names, "bytes": checkpoint.stat().st_size})
            del loaded
            gc.collect()
        if verify_release(metadata["dataset"]) != metadata["release_sha256"]:
            raise RuntimeError("Snapshot dataset thay đổi trong lúc training.")
        if sha256(request["config"]["model"]) != metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint khởi tạo đã bị sửa.")
        atomic_json(run / "checkpoint_validation.json", {"valid": True, "checkpoints": checkpoint_records,
                    "dataset_unchanged": True, "initial_checkpoint_unchanged": True})
        progress.state.update(status="stopped" if progress.stop_accepted else "completed",
                              finished_at=time.time(), exit_code=0, checkpoints_verified=True)
        progress.phase("finished")
        atomic_json(run / "timings.json", {"seconds_by_phase": progress.state["timings"],
                    "total_seconds": progress.state["elapsed_seconds"],
                    "sampled_process_tree_peak_rss_bytes": progress.state["rss_peak_bytes"],
                    "worker_peak_working_set_bytes": progress.state["worker_peak_working_set_bytes"],
                    "available_ram_min_bytes": progress.state["available_ram_min_bytes"]})
        print("Run finished; project8 checkpoints load correctly and dataset is unchanged.", flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        progress.state.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                              finished_at=time.time(), exit_code=1)
        progress.phase("failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    raise SystemExit(run_worker(args.run, args.token))
