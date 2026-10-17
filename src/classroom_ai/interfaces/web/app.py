"""Giao diện web xem ảnh; khởi động bằng streamlit run src/classroom_ai/interfaces/web/app.py."""

from __future__ import annotations

import base64
import csv
import gc
import hashlib
import json
import platform
from functools import lru_cache
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from classroom_ai.inference.camera import CameraSession
from classroom_ai.inference.detector import (
    PROJECT_NAMES,
    ImageInspector,
    crop_box,
    decode_image,
    draw_boxes,
    reference_boxes,
)
from classroom_ai.inference.devices import cuda_devices
from classroom_ai.interfaces.web.preview import PreviewSettings, load_preview_settings, preview_jpeg
from classroom_ai.interfaces.web.training import training_page
from classroom_ai.paths import PROJECT_ROOT as ROOT
from classroom_ai.training.manager import RunManager


@st.cache_resource
def load_inspector(path: str, modified_ns: int, device: str = "cpu") -> ImageInspector:
    return ImageInspector(Path(path), device=device)


@st.cache_data(ttl=30)
def available_gpus():
    return cuda_devices()


def stop_current_webcam():
    camera = st.session_state.get("camera_session")
    if camera and not camera.stop():
        return False
    st.session_state.pop("camera_session", None)
    st.session_state.pop("webcam_preview_cache", None)
    return True


def clear_inference():
    load_inspector.clear()
    st.session_state.pop("inspection", None)
    gc.collect()


@st.fragment(run_every=2)
def recognition_paused(manager):
    if manager.active() is None:
        st.rerun(scope="app")
    st.info("Training đang chạy. Recognition tạm dừng để dành CPU/RAM. Chọn Training để theo dõi.")


def choose_sample(image_id: str) -> None:
    st.session_state.sample_picker = image_id


def sample_source():
    current_dataset = ROOT / "configs/dataset_current.json"
    dataset_root = (
        ROOT / json.loads(current_dataset.read_text(encoding="utf-8"))["path"]
        if current_dataset.is_file()
        else ROOT / "data/dataset/project8_v0.4_merged_20261008"
    )
    manifest = dataset_root / "manifest.csv"
    if not manifest.is_file():
        dataset_root = ROOT
        manifest = ROOT / "data/manifest.csv"
    if not manifest.is_file():
        st.info("Chưa có danh sách ảnh mẫu. Chọn ‘Ảnh của bạn’ để mở ảnh riêng.")
        return None
    with manifest.open(encoding="utf-8-sig") as file:
        available = (
            r
            for r in csv.DictReader(file)
            if r.get("split") == "train" and (dataset_root / r["image_path"]).is_file()
        )
        samples = {r["image_id"]: r for r in islice(available, 50)}
    if not samples:
        st.info("Chưa có ảnh train trong manifest chính. Chọn ‘Ảnh của bạn’ để mở ảnh riêng.")
        return None
    ids = list(samples)
    # Nhãn dataset chính dùng ID project8; không đọc chúng như raw ID COCO80.
    sample_names = {}
    for item, row in samples.items():
        label = dataset_root / row["label_path"]
        names = (
            {r["class_name"] for r in reference_boxes(label.read_text(), 1, 1, profile="project8")}
            if label.is_file()
            else set()
        )
        sample_names[item] = (
            ", ".join(name for name in PROJECT_NAMES if name in names) or "không có tám lớp"
        )
    if st.session_state.get("sample_picker") not in samples:
        st.session_state.sample_picker = ids[0]
    image_id = st.selectbox(
        "Chọn ảnh mẫu", ids, key="sample_picker", format_func=lambda i: f"{i} · {sample_names[i]}"
    )
    with st.expander(f"Duyệt {len(samples)} ảnh thu nhỏ"):
        page = st.selectbox("Trang ảnh", range(1, (len(ids) + 5) // 6 + 1))
        cols = st.columns(3)
        for index, item in enumerate(ids[(page - 1) * 6 : page * 6]):
            with cols[index % 3]:
                st.image(str(dataset_root / samples[item]["image_path"]), width=190)
                st.button(f"Xem {item}", key=f"thumb_{item}", on_click=choose_sample, args=(item,))
    row = samples[image_id]
    frame = decode_image((dataset_root / row["image_path"]).read_bytes())
    label = dataset_root / row["label_path"]
    refs = (
        reference_boxes(label.read_text(), frame.shape[1], frame.shape[0], profile="project8")
        if label.is_file()
        else None
    )
    st.caption(
        "Ảnh train từ dataset chính • ID 7 bottle gồm cup và bottle; bộ ghép đang chờ rà nhãn đủ 8 lớp."
    )
    return frame, image_id, refs


def uploaded_source():
    upload = st.file_uploader("Mở ảnh JPG, PNG hoặc WebP", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.info("Chọn ảnh trên máy. Ảnh được xử lý trong phiên này, không tự ghi vào dataset.")
        return None
    return decode_image(upload.getvalue()), upload.name, None


def video_source():
    files = []
    for folder in (ROOT / "demo/smoke", ROOT / "data/video_dev"):
        if folder.exists():
            files.extend(
                p for p in folder.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            )
    if not files:
        st.info("Đặt video vào data/video_dev rồi tải lại trang để chọn.")
        return None
    file = st.selectbox("Chọn video", sorted(files), format_func=lambda p: str(p.relative_to(ROOT)))
    capture = cv2.VideoCapture(str(file))
    try:
        if not capture.isOpened():
            raise ValueError("Không mở được video này.")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise ValueError("Không xác định được số frame. Hãy thử MP4 hoặc CLI.")
        index = st.slider("Frame", 0, count - 1, key=f"frame_{file.name}") if count > 1 else 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Không đọc được frame {index}.")
    finally:
        capture.release()
    st.caption(
        f"{count} frame • đang xem frame {index}. Xem từng frame không phải benchmark video."
    )
    if "smoke" in file.parts:
        st.caption("Đây là slideshow COCO128 để thử đọc video, chưa phải video camera liên tục.")
    return frame, f"{file.name} / frame {index}", None


def detection_table(rows):
    return pd.DataFrame(
        [
            {
                "STT": i,
                "Lớp": r["class_name"],
                "Confidence (%)": round(r["confidence"] * 100, 2),
                "x1": round(r["bbox_xyxy"][0], 1),
                "y1": round(r["bbox_xyxy"][1], 1),
                "x2": round(r["bbox_xyxy"][2], 1),
                "y2": round(r["bbox_xyxy"][3], 1),
            }
            for i, r in enumerate(rows, 1)
        ]
    )


@lru_cache(maxsize=1)
def webcam_versions():
    return {name: version(name) for name in ("ultralytics", "torch", "opencv-python", "streamlit")}


def render_webcam(
    preview,
    checkpoint,
    confidence,
    visible_classes,
    manager,
    imgsz=640,
    extra_nms_iou=None,
    passes=1,
    device="cpu",
):
    # Register on full app runs, so editing runtime settings takes effect on reload.
    panel = st.fragment(run_every=1)(webcam_panel)
    panel(
        checkpoint,
        confidence,
        visible_classes,
        manager,
        imgsz,
        extra_nms_iou,
        passes,
        preview,
        device=device,
    )
    image = st.fragment(run_every=1 / preview.fps)(webcam_image)
    image(visible_classes, preview)


def webcam_image(visible_classes, preview):
    camera = st.session_state.get("camera_session")
    if camera is None:
        return
    snapshot = camera.snapshot()
    if snapshot["status"] != "running":
        return
    rows = [row for row in snapshot["output"]["detections"] if row["class_name"] in visible_classes]
    token = (
        camera,
        snapshot["starts"],
        snapshot["frames_processed"],
        tuple(visible_classes),
        preview,
    )
    cached = st.session_state.get("webcam_preview_cache")
    if cached is None or cached[0] != token:
        jpeg = preview_jpeg(snapshot["frame"], rows, preview)
        # Inline the bounded JPEG: rapid fragment runs can evict /media URLs
        # before a slow browser fetches them. This also avoids re-encoding in st.image.
        cached = (token, "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
        st.session_state.webcam_preview_cache = cached
    # Fragments clear their contents every tick, so the cached image must still be emitted.
    st.image(cached[1], width="stretch")


def webcam_panel(
    checkpoint,
    confidence,
    visible_classes,
    manager,
    imgsz=640,
    extra_nms_iou=None,
    passes=1,
    preview=PreviewSettings(),
    device="cpu",
):
    # main() checks training; guarded_predict checks it under inference_slot on
    # every frame. Render must not compete for that lock, held throughout inference.
    st.subheader("Webcam trực tiếp")
    st.caption("Mỗi thành viên chạy ứng dụng trên máy của mình để dùng webcam của chính máy đó.")
    index = int(
        st.number_input(
            "Chỉ số webcam",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Thường là 0. Thử 1 hoặc 2 nếu dùng camera USB.",
            key="webcam_index",
        )
    )
    config = (
        index,
        str(checkpoint),
        checkpoint.stat().st_mtime_ns,
        confidence,
        imgsz,
        extra_nms_iou,
        passes,
        device,
    )
    camera = st.session_state.get("camera_session")
    if camera is not None and st.session_state.get("camera_config") != config:
        if not camera.stop():
            st.info("Đang giải phóng webcam trước khi áp dụng cấu hình mới…")
            return
        st.session_state.pop("camera_session", None)
        camera = None
        st.info("Cấu hình đã đổi. Bấm Bật webcam để chạy với cấu hình mới.")
    snapshot = camera.snapshot() if camera else None
    active = snapshot is not None and snapshot["status"] in {"starting", "running", "stopping"}
    start_col, stop_col, reopen_col = st.columns(3)
    start = start_col.button("Bật webcam", key="webcam_start", type="primary", disabled=active)
    stop = stop_col.button("Dừng webcam", key="webcam_stop", disabled=not active)
    reopen = reopen_col.button("Mở lại webcam", key="webcam_reopen", disabled=not active)
    try:
        if stop and camera:
            camera.stop()
        if reopen and camera:
            if camera.stop():
                camera.start()
        if start:
            if camera is None:
                with st.spinner("Đang nạp mô hình…"):
                    inspector = load_inspector(
                        str(checkpoint), checkpoint.stat().st_mtime_ns, device
                    )

                def guarded_predict(frame, threshold):
                    with manager.inference_slot() as manager_timing:
                        output = inspector.predict(
                            frame,
                            threshold,
                            imgsz=imgsz,
                            extra_nms_iou=extra_nms_iou,
                            passes=passes,
                        )
                    return {**output, **manager_timing}

                camera = CameraSession(index, guarded_predict, confidence)
                st.session_state.camera_session = camera
                st.session_state.camera_config = config
            camera.start()
        if start or stop or reopen:
            st.rerun()
    except Exception as exc:
        st.error(f"Không bật được webcam: {exc}")
        return
    if snapshot is None:
        st.info("Bấm Bật webcam. Hình được xử lý cục bộ; ứng dụng không tự lưu ảnh hay video.")
        return
    status = snapshot["status"]
    if status == "error":
        st.error(snapshot["error"])
    elif status == "starting":
        st.info("Đang mở webcam và xử lý khung hình đầu tiên…")
    elif status == "stopping":
        st.info("Đang dừng xử lý và giải phóng webcam…")
    elif status == "stopped":
        st.info("Webcam đã dừng. Bấm Bật webcam để mở lại.")
    else:
        rows = [
            row for row in snapshot["output"]["detections"] if row["class_name"] in visible_classes
        ]
        counts = st.columns(4)
        for i, name in enumerate(PROJECT_NAMES):
            counts[i % 4].metric(name, sum(row["class_name"] == name for row in rows))
        fps_col, latency_col, frame_col = st.columns(3)
        fps_col.metric("FPS gần nhất · 5 giây", f"{snapshot['recent_processing_fps']:.1f}")
        latency_col.metric("p95 callback", f"{snapshot['callback_p95_ms']:.0f} ms")
        frame_col.metric("Frame đã xử lý", snapshot["frames_processed"])
        st.caption(
            f"FPS trung bình từ lúc mở camera: {snapshot['processing_fps']:.1f}. "
            "Số đối tượng hiện tại; callback p95 đo sau khi đọc frame đến lúc có dự đoán, "
            "chưa gồm thời gian đọc camera hay trình duyệt."
        )
        stages = (
            ("đọc camera", snapshot["capture_read_p95_ms"]),
            ("predict + NMS", snapshot["processing_p95_ms"]),
            ("chờ khóa training", snapshot["manager_lock_wait_p95_ms"]),
            ("kiểm tra training", snapshot["manager_active_check_p95_ms"]),
            ("chờ khóa model", snapshot["inspector_lock_wait_p95_ms"]),
        )
        stage_text = " · ".join(
            f"{label} {value:.1f} ms" if value is not None else f"{label} —"
            for label, value in stages
        )
        result_age = snapshot["latest_result_age_ms"]
        st.caption(
            f"p95 theo stage: {stage_text}. Tuổi kết quả mới nhất: {result_age:.0f} ms"
            if result_age is not None
            else f"p95 theo stage: {stage_text}."
        )
        frame_size = snapshot["frame_size"]
        frame_text = (
            f"{frame_size['width_px']}×{frame_size['height_px']} px"
            if frame_size is not None
            else "không có"
        )
        st.caption(f"Kích thước frame camera: {frame_text}.")
        st.caption(
            f"Preview mục tiêu {preview.fps:g} FPS · JPEG tối đa {preview.max_width}px · "
            "FPS hiển thị thực phụ thuộc xử lý, truyền ảnh và trình duyệt."
        )
    if snapshot["frames_processed"]:
        report = {key: value for key, value in snapshot.items() if key not in {"frame", "output"}}
        report.update(
            {
                "weights": str(checkpoint.relative_to(ROOT)),
                "confidence": confidence,
                "device": device,
                "imgsz": imgsz,
                "nms": extra_nms_iou is not None,
                "extra_class_nms_iou": extra_nms_iou,
                "model_passes": passes,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "versions": webcam_versions(),
                "preview_target_fps": preview.fps,
                "preview_interval_s": 1 / preview.fps,
                "preview_max_width": preview.max_width,
                "preview_jpeg_quality": preview.jpeg_quality,
                "measurement_note": "processing_fps trung bình từ lúc mở camera; recent_processing_fps đo cửa sổ 5 giây. Callback p95 tính từ sau capture.read() đến kết quả, chưa gồm thời gian đọc camera, truyền ảnh hoặc render trong trình duyệt. Các stage p95 dùng tối đa 300 mẫu gần nhất.",
            }
        )
        st.download_button(
            "Tải kết quả kiểm webcam",
            json.dumps(report, ensure_ascii=False, indent=2),
            "webcam_summary.json",
            "application/json",
            key="webcam_report",
            on_click="ignore",
        )


def main():
    st.set_page_config(page_title="Camera AI · Xem kết quả", layout="wide")
    manager = RunManager(ROOT)
    with st.sidebar:
        page = st.selectbox("Màn hình", ["Recognition", "Training"], key="workspace_page")
    if page == "Training":
        stop_current_webcam()
        training_page(manager, stop_current_webcam, clear_inference)
        return
    try:
        if manager.active():
            stop_current_webcam()
            st.title("Recognition")
            recognition_paused(manager)
            return
    except ValueError as exc:
        stop_current_webcam()
        st.error(str(exc))
        return
    st.title("Phòng quan sát AI")
    st.caption("Nhận dạng tám lớp trên ảnh, video hoặc webcam của máy đang chạy ứng dụng.")
    with st.sidebar:
        st.header("Nguồn & mô hình")
        source = st.radio(
            "Nguồn dữ liệu",
            ["Ảnh dữ liệu chính", "Ảnh của bạn", "Khung hình video", "Webcam trực tiếp"],
        )
        weights = sorted((ROOT / "weights").rglob("*.pt"))
        if not weights:
            st.error("Chưa có checkpoint. Tải YOLO26n theo README.")
            st.stop()
        preferred = ROOT / "weights/campaign_20261010/R04_YOLO26n_Project8_v05_finetuned_default.pt"
        weight_labels = {
            "E2_YOLO26n_Project8_v05_from_CP07_v04_best19_MuSGD_lr0p0003_bestEpoch10_20261012.pt": "E2 · YOLO26n · Project8 v0.5 · từ CP07 v0.4 best19 · MuSGD LR 0.0003 · best epoch 10/10 · 12/10/2026 · thử nghiệm",
            "E3_YOLO26n_Project8_v05_from_CP07_v04_best19_AdamW_lr0p0001_bestEpoch10_20261012.pt": "E3 · YOLO26n · Project8 v0.5 · từ CP07 v0.4 best19 · AdamW LR 0.0001 · best epoch 10/10 · 12/10/2026 · thử nghiệm",
            "E3_INVERT_seed2026_E2_best.pt": "E3 · INVERT augmentation · seed 2026 · 2 epoch · AdamW LR 0.0001 · imgsz 1280 · Project8 v0.5 · thử nghiệm nhãn chưa review · mAP50 76.56% · mAP50-95 55.10% · Recall 69.94% · F1 macro 73.90%",
            "R04_YOLO26n_Project8_v05_finetuned_default.pt": "R04 · YOLO26n · fine-tune v05 · mặc định",
            "Baseline_YOLO26n_Project8_v05_20261009.pt": "Baseline v05 · YOLO26n · 09/10/2026",
            "R01_YOLO26n_Project8_v05_recovered_epoch0.pt": "R01 · YOLO26n · checkpoint epoch0 khôi phục",
        }
        checkpoint = st.selectbox(
            "Checkpoint",
            weights,
            index=weights.index(preferred) if preferred in weights else 0,
            format_func=lambda p: weight_labels.get(p.name, str(p.relative_to(ROOT))),
        )
        gpus, gpu_reason = available_gpus()
        device_labels = {"cpu": "CPU", **gpus}
        if st.session_state.get("inference_device", "cpu") not in device_labels:
            stop_current_webcam()
            st.session_state.inference_device = "cpu"
            st.session_state.pop("inspection", None)
            st.warning("GPU đã chọn không còn khả dụng. Đã chuyển về CPU.")
        device = st.selectbox(
            "Thiết bị xử lý",
            list(device_labels),
            key="inference_device",
            format_func=device_labels.get,
            help="CPU mặc định. GPU chỉ hiện khi PyTorch chạy được phép tính CUDA trên máy chạy ứng dụng.",
        )
        if gpu_reason:
            st.info(gpu_reason)
        if st.button("Kiểm tra lại GPU"):
            available_gpus.clear()
            st.rerun()
        confidence = st.slider("Ngưỡng confidence", 0.05, 0.95, 0.25, 0.05)
        imgsz = st.selectbox(
            "Kích thước suy luận",
            [512, 640, 768, 960, 1280],
            index=1,
            help="640 là mặc định. Kích thước lớn hơn thường tốn thời gian hơn; 1280 chưa có số đo webcam thực.",
        )
        passes = st.selectbox(
            "Số lượt suy luận mỗi ảnh",
            [1, 3],
            format_func=lambda n: (
                "1 lượt — ưu tiên tốc độ" if n == 1 else "3 lượt rồi gộp — ưu tiên chất lượng"
            ),
        )
        extra_nms_iou = st.selectbox(
            "NMS theo lớp",
            [0.5, 0.6] if passes == 3 else [None, 0.5, 0.6],
            format_func=lambda v: "Tắt" if v is None else f"IoU {v:.1f}",
            help="NMS không đổi trọng số. 0,5 lọc mạnh hơn 0,6; ba lượt cần NMS để gộp kết quả.",
        )
        if passes == 3:
            st.info(
                "Ba lượt luôn dùng NMS để gộp khung. Webcam có thể giảm FPS; chưa có benchmark webcam cho chế độ này."
            )
        visible_classes = st.multiselect(
            "Lớp hiển thị", list(PROJECT_NAMES), default=list(PROJECT_NAMES)
        )
        st.caption(
            f"{device_labels[device]} · kích thước {imgsz} · {passes} lượt suy luận · "
            + (
                f"NMS theo lớp {extra_nms_iou:.2f}"
                if extra_nms_iou is not None
                else "không thêm NMS"
            )
        )
        with st.expander("Hướng dẫn kích thước và NMS"):
            st.markdown(
                "**Mặc định:** R04, 640, một lượt, NMS tắt. Để thử cấu hình đã chấm, chọn **IoU 0,6** trong NMS; "
                "chọn **1280** nếu muốn so chất lượng ở kích thước đó.\n\n"
                "NMS giữ khung có confidence cao hơn trong các khung cùng lớp chồng lên nhau. "
                "Bước lọc có thêm chi phí, tùy số khung; không tự tăng số lượt chạy model. "
                "Kích thước lớn hơn cũng có thể làm giảm FPS.\n\n"
                "Đổi cấu hình ảnh thì bấm **Phân tích ảnh** lại. Đổi khi đang dùng webcam sẽ dừng phiên cũ; "
                "bấm **Bật webcam** để áp dụng. Thời gian xử lý có tính bước NMS khi bật.\n\n"
                "Thiết bị đã chọn áp dụng cho ảnh, video và webcam. Điểm GPU FP16 trong báo cáo không bảo đảm kết quả số học hoặc tốc độ "
                "giống hệt tại đây. Ba lượt dùng tỷ lệ 1,0 / 0,83 lật ngang / 0,67 trên cùng trọng số. "
                "Mốc mAP50 khoảng 80,5% là R04 ở1280, ba lượt, NMS0,5; không bảo đảm điểm đó ở640 hoặc mọi ảnh."
            )
        st.caption("Confidence là điểm của dự đoán, không phải độ chính xác của toàn mô hình.")
    try:
        if source == "Webcam trực tiếp":
            preview = load_preview_settings(ROOT / "configs/app.yaml")
            render_webcam(
                preview,
                checkpoint,
                confidence,
                visible_classes,
                manager,
                imgsz,
                extra_nms_iou,
                passes,
                device,
            )
            return
        camera = st.session_state.get("camera_session")
        if camera is not None:
            camera.stop()
        selected = {
            "Ảnh dữ liệu chính": sample_source,
            "Ảnh của bạn": uploaded_source,
            "Khung hình video": video_source,
        }[source]()
        if selected is None:
            st.stop()
        frame, source_name, refs = selected
        fingerprint = hashlib.sha256(
            frame.tobytes()
            + str(frame.shape).encode()
            + str(checkpoint).encode()
            + str(checkpoint.stat().st_mtime_ns).encode()
            + str((confidence, imgsz, extra_nms_iou, passes, device)).encode()
            + source_name.encode()
        ).hexdigest()
        st.subheader(source_name)
        st.caption(
            f"Kích thước gốc {frame.shape[1]} × {frame.shape[0]} pixel. Nút phóng to nằm ở góc ảnh."
        )
        if st.button("Phân tích ảnh", type="primary", key="analyze"):
            st.session_state.pop("inspection", None)
            with st.spinner("Đang chạy YOLO26n…"):
                engine = load_inspector(str(checkpoint), checkpoint.stat().st_mtime_ns, device)
                with manager.inference_slot():
                    output = engine.predict(
                        frame, confidence, imgsz=imgsz, extra_nms_iou=extra_nms_iou, passes=passes
                    )
                st.session_state.inspection = {"fingerprint": fingerprint, **output}
        output = st.session_state.get("inspection")
        valid = output is not None and output["fingerprint"] == fingerprint
        original_col, result_col = st.columns(2)
        with original_col:
            st.markdown("**Ảnh gốc**")
            st.image(frame, channels="BGR", width="stretch")
        with result_col:
            st.markdown("**Dự đoán YOLO**")
            if valid:
                rows = [r for r in output["detections"] if r["class_name"] in visible_classes]
                annotated = draw_boxes(frame, rows)
                st.image(annotated, channels="BGR", width="stretch")
            else:
                st.info(
                    "Bấm ‘Phân tích ảnh’. Khi đổi nguồn, model, thiết bị, confidence, kích thước, số lượt hoặc NMS, cần chạy lại để xem kết quả mới."
                )
        if valid:
            counts = st.columns(4)
            for index, name in enumerate(PROJECT_NAMES):
                column = counts[index % 4]
                column.metric(name, sum(r["class_name"] == name for r in rows))
            st.metric("Xử lý ảnh", f"{output['processing_ms']:.0f} ms")
            st.caption(
                "Thời gian suy luận và NMS; lần đầu có thể gồm warmup và nạp model ba lượt. Không phải độ trễ webcam."
            )
            if rows:
                table_col, crop_col = st.columns([3, 2])
                with table_col:
                    st.markdown("**Chọn một dòng để xem vùng ảnh cắt**")
                    selection_key = fingerprint + "_" + "_".join(visible_classes)
                    event = st.dataframe(
                        detection_table(rows),
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        key=selection_key,
                    )
                with crop_col:
                    chosen = event.selection.rows
                    if chosen and chosen[0] < len(rows):
                        row = rows[chosen[0]]
                        st.image(crop_box(frame, row["bbox_xyxy"]), channels="BGR", width=260)
                        st.write(f"**{row['class_name']}** · confidence {row['confidence']:.1%}")
                        st.caption(
                            f"ID dự án: {row['class_id']} · ID checkpoint: {row['model_class_id']}"
                        )
                    else:
                        st.info("Chọn một đối tượng trong bảng bên trái.")
            else:
                st.info(
                    "Không có đối tượng ở ngưỡng và lớp hiển thị hiện tại. Đây không phải bằng chứng ảnh không có vật thể."
                )
            export = {
                "source": source_name,
                "weights": str(checkpoint.relative_to(ROOT)),
                "confidence_threshold": confidence,
                "device": output["device"],
                "imgsz": imgsz,
                "nms": False,
                "extra_class_nms_iou": extra_nms_iou,
                "model_passes": passes,
                "visible_classes": visible_classes,
                "processing_ms": output["processing_ms"],
                "detections": rows,
            }
            encoded_ok, encoded = cv2.imencode(".png", annotated)
            if not encoded_ok:
                raise ValueError("Không tạo được ảnh tải xuống.")
            downloads = st.columns(3)
            downloads[0].download_button(
                "Tải ảnh kết quả", encoded.tobytes(), "detection.png", "image/png"
            )
            downloads[1].download_button(
                "Tải JSON",
                json.dumps(export, ensure_ascii=False, indent=2),
                "detection.json",
                "application/json",
            )
            downloads[2].download_button(
                "Tải bảng CSV",
                detection_table(rows).to_csv(index=False).encode("utf-8-sig"),
                "detection.csv",
                "text/csv",
            )
        if refs is not None:
            with st.expander("Đối chiếu nhãn dataset"):
                reference = [r for r in refs if r["class_name"] in visible_classes]
                st.image(draw_boxes(frame, reference, reference=True), channels="BGR", width=640)
                st.write(f"Nhãn tham khảo: {len(reference)} hộp thuộc các lớp đang hiển thị.")
                st.caption(
                    "Khung xanh lá là nhãn tham khảo. So bằng mắt để tìm lỗi; chưa tính TP/FP/FN hay mAP."
                )
    except Exception as exc:
        st.error(f"Không xử lý được nguồn: {exc}")


if __name__ == "__main__":
    main()
