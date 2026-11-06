"""Optional OpenVINO path: cache integrity and device failures, without hardware."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from classroom_ai.inference import devices, pose


def test_missing_openvino_has_setup_error_without_auto_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino", None)
    found, reason = devices.intel_gpu_devices()
    assert not found and "OpenVINO" in reason
    with pytest.raises(ValueError, match="OpenVINO"):
        devices.validate_device("intel:gpu")


def test_cpu_only_openvino_never_offers_gpu(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "openvino",
        SimpleNamespace(Core=lambda: SimpleNamespace(available_devices=["CPU"])),
    )
    found, reason = devices.intel_gpu_devices()
    assert not found and "GPU" in reason
    with pytest.raises(ValueError, match="GPU"):
        devices.validate_device("intel:gpu")


@pytest.fixture
def ov_engine(tmp_path, monkeypatch):
    import ultralytics

    state = SimpleNamespace(exports=[], predicts=0, execution_devices=["GPU.0"])
    weights = tmp_path / "source.pt"
    weights.write_bytes(b"original model")
    monkeypatch.setattr(pose, "validate_device", lambda device: device)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(__version__="test-ov"))

    class Model:
        def __init__(self, path, task=None):
            self.path = Path(path)
            self.task, self.names = "pose", {0: "person"}
            self.model = SimpleNamespace(
                yaml={"kpt_shape": [17, 3]}, end2end=False, stride=np.array([8, 16, 32])
            )
            self.callbacks = []

        def export(self, **kwargs):
            state.exports.append(kwargs)
            directory = self.path.with_name(self.path.stem + "_openvino_model")
            directory.mkdir()
            (directory / "model.xml").write_text("fake XML")
            (directory / "model.bin").write_bytes(b"fake weights")
            (directory / "metadata.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task": "pose",
                        "names": {0: "person"},
                        "kpt_shape": [17, 3],
                        "end2end": False,
                        "imgsz": kwargs["imgsz"],
                    }
                )
            )
            return str(directory)

        def add_callback(self, name, callback):
            assert name == "on_predict_start"
            self.callbacks.append(callback)

        def predict(self, frame, **kwargs):
            compiled = SimpleNamespace(
                get_property=lambda name: (
                    state.execution_devices if name == "EXECUTION_DEVICES" else "float16"
                )
            )
            predictor = SimpleNamespace(
                model=SimpleNamespace(ov_compiled_model=compiled, end2end=False)
            )
            for callback in self.callbacks:
                callback(predictor)
            state.predicts += 1
            count = 1
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=np.array([[0, 0, 30, 30]]), conf=np.array([0.8]), cls=np.array([0])
                    ),
                    keypoints=SimpleNamespace(data=np.full((count, 17, 3), 0.8)),
                )
            ]

    monkeypatch.setattr(ultralytics, "YOLO", Model)
    return weights, tmp_path / "cache", state


def test_gpu_export_is_lazy_same_head_and_cache_reuses_original_provenance(ov_engine):
    weights, cache, state = ov_engine
    inspector = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    assert state.exports == []
    frame = np.zeros((1080, 1920, 3), np.uint8)
    result = inspector.predict(frame, 0.05, imgsz=1280)
    assert state.exports[0]["nms"] is None
    assert state.exports[0]["imgsz"] == [736, 1280]
    assert result["runtime"]["execution_devices"] == ["GPU.0"]
    assert result["runtime"]["source_sha256"] == hashlib.sha256(b"original model").hexdigest()
    assert result["runtime"]["artifact_hashes"]
    other = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    other.predict(frame, 0.05, imgsz=1280)
    assert len(state.exports) == 1
    assert weights.read_bytes() == b"original model"


def test_cache_corruption_is_rejected(ov_engine):
    weights, cache, _ = ov_engine
    frame = np.zeros((90, 160, 3), np.uint8)
    pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(frame, 0.05)
    next(cache.glob("*/model.bin")).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="cache|Cache"):
        pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(frame, 0.05)


def test_cache_wrong_head_is_rejected_even_with_matching_hash(ov_engine):
    weights, cache, _ = ov_engine
    frame = np.zeros((90, 160, 3), np.uint8)
    pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(frame, 0.05)
    metadata = next(cache.glob("*/metadata.yaml"))
    data = yaml.safe_load(metadata.read_text())
    data["end2end"] = True
    metadata.write_text(yaml.safe_dump(data))
    manifest = metadata.parent / "provenance.json"
    data = json.loads(manifest.read_text())
    data["artifact_hashes"]["metadata.yaml"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="head|metadata|cache|Cache"):
        pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(frame, 0.05)


def test_size_and_source_changes_get_separate_cache_keys(ov_engine):
    weights, cache, state = ov_engine
    frame = np.zeros((90, 160, 3), np.uint8)
    inspector = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    a = inspector.predict(frame, 0.05, imgsz=640)["runtime"]["cache_key"]
    b = inspector.predict(frame, 0.05, imgsz=1280)["runtime"]["cache_key"]
    weights.write_bytes(b"another model")
    c = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(frame, 0.05, imgsz=1280)[
        "runtime"
    ]["cache_key"]
    assert len({a, b, c}) == 3 and len(state.exports) == 3


def test_compiled_cpu_fallback_is_rejected_before_inference(ov_engine):
    weights, cache, state = ov_engine
    state.execution_devices = ["CPU"]
    with pytest.raises(RuntimeError, match="GPU|CPU"):
        pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(
            np.zeros((90, 160, 3), np.uint8), 0.05
        )
    assert state.predicts == 0


def test_concurrent_inspectors_only_publish_one_export(ov_engine):
    from concurrent.futures import ThreadPoolExecutor

    weights, cache, state = ov_engine
    a = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    b = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    frame = np.zeros((90, 160, 3), np.uint8)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(inspector.predict, frame, 0.05) for inspector in (a, b)]
        results = [future.result() for future in futures]
    assert len(state.exports) == 1
    assert results[0]["runtime"]["cache_key"] == results[1]["runtime"]["cache_key"]
    assert not list(cache.glob("export-*"))


def test_source_changed_after_load_is_rejected_before_export(ov_engine):
    weights, cache, state = ov_engine
    inspector = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    weights.write_bytes(b"changed after load")
    with pytest.raises(ValueError, match="Weights"):
        inspector.predict(np.zeros((90, 160, 3), np.uint8), 0.05)
    assert not state.exports


def test_runtime_version_isolates_cache_key(ov_engine, monkeypatch):
    weights, cache, _ = ov_engine
    frame = np.zeros((90, 160, 3), np.uint8)
    a = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(frame, 0.05)
    monkeypatch.setattr(sys.modules["openvino"], "__version__", "another-runtime")
    b = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache).predict(frame, 0.05)
    assert a["runtime"]["cache_key"] != b["runtime"]["cache_key"]


def test_source_changed_after_success_cannot_reuse_loaded_ir(ov_engine):
    weights, cache, state = ov_engine
    frame = np.zeros((90, 160, 3), np.uint8)
    inspector = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    inspector.predict(frame, 0.05)
    weights.write_bytes(b"replaced after successful predict")
    with pytest.raises(ValueError, match="Weights"):
        inspector.predict(frame, 0.05)
    assert state.predicts == 1


def test_cancel_before_export_and_reset_for_cached_inspector(ov_engine):
    weights, cache, state = ov_engine
    inspector = pose.PoseInspector(weights, "intel:gpu", cache_dir=cache)
    inspector.set_cancel(lambda: True)
    frame = np.zeros((90, 160, 3), np.uint8)
    with pytest.raises(InterruptedError, match="hủy"):
        inspector.predict(frame, 0.05)
    assert not state.exports
    inspector.set_cancel(lambda: False)
    inspector.predict(frame, 0.05)
    assert state.predicts == 1


def test_waiting_export_lock_checks_cancel_promptly(tmp_path):
    import threading
    import time

    path = tmp_path / "export.lock"
    ready, release, cancel = threading.Event(), threading.Event(), threading.Event()

    def hold_lock():
        with pose._export_lock(path):
            ready.set()
            release.wait(2)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert ready.wait(1)
    timer = threading.Timer(0.05, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError, match="hủy"):
            with pose._export_lock(path, cancel=cancel.is_set):
                pytest.fail("Should not enter a locked export")
        assert time.monotonic() - started < 0.7
    finally:
        release.set()
        thread.join(2)
        timer.cancel()
