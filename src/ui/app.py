"""Giao diện web xem ảnh; khởi động bằng streamlit run src/app.py."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.inference.detector import ImageInspector, PROJECT_NAMES, crop_box, decode_image, draw_boxes, reference_boxes
from src.inference.camera import CameraSession


@st.cache_resource
def load_inspector(path: str, modified_ns: int) -> ImageInspector:
    return ImageInspector(Path(path))


def choose_sample(image_id: str) -> None:
    st.session_state.sample_picker = image_id


def sample_source():
    manifest = ROOT / "data/week1_coco128_manifest.csv"
    if not manifest.is_file():
        st.info("Chưa có danh sách ảnh mẫu. Chọn ‘Ảnh của bạn’ để mở ảnh riêng.")
        return None
    with manifest.open(encoding="utf-8") as file:
        samples = {r["image_id"]: r for r in csv.DictReader(file) if (ROOT / r["image_path"]).is_file()}
    if not samples:
        st.info("Chưa tải COCO128. Xem lệnh tải trong README, hoặc chọn ‘Ảnh của bạn’.")
        return None
    ids = list(samples)
    # Manifest tuần 1 cũ có bốn lớp; lấy tên tám lớp từ nhãn gốc để hiển thị đúng.
    sample_names = {}
    for item, row in samples.items():
        label = ROOT / row["label_path"]
        names = {r["class_name"] for r in reference_boxes(label.read_text(), 1, 1)} if label.is_file() else set()
        sample_names[item] = ", ".join(name for name in PROJECT_NAMES if name in names) or "không có tám lớp"
    if st.session_state.get("sample_picker") not in samples:
        st.session_state.sample_picker = "000000000283" if "000000000283" in samples else ids[0]
    image_id = st.selectbox("Chọn ảnh mẫu", ids, key="sample_picker",
                            format_func=lambda i: f"{i} · {sample_names[i]}")
    with st.expander(f"Duyệt {len(samples)} ảnh thu nhỏ"):
        page = st.selectbox("Trang ảnh", range(1, (len(ids)+5)//6+1))
        cols = st.columns(3)
        for index, item in enumerate(ids[(page-1)*6:page*6]):
            with cols[index % 3]:
                st.image(str(ROOT / samples[item]["image_path"]), width=190)
                st.button(f"Xem {item}", key=f"thumb_{item}", on_click=choose_sample, args=(item,))
    row = samples[image_id]
    frame = decode_image((ROOT / row["image_path"]).read_bytes())
    label = ROOT / row["label_path"]
    refs = reference_boxes(label.read_text(), frame.shape[1], frame.shape[0]) if label.is_file() else None
    st.caption("COCO128 • nhãn gốc để đối chiếu • chưa phải dữ liệu phòng học đã review")
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
            files.extend(p for p in folder.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
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
        index = st.slider("Frame", 0, count-1, key=f"frame_{file.name}") if count > 1 else 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Không đọc được frame {index}.")
    finally:
        capture.release()
    st.caption(f"{count} frame • đang xem frame {index}. Xem từng frame không phải benchmark video.")
    if "smoke" in file.parts:
        st.caption("Đây là slideshow COCO128 để thử đọc video, chưa phải video camera liên tục.")
    return frame, f"{file.name} / frame {index}", None


def detection_table(rows):
    return pd.DataFrame([{"STT": i, "Lớp": r["class_name"], "Confidence (%)": round(r["confidence"]*100, 2),
                          "x1": round(r["bbox_xyxy"][0], 1), "y1": round(r["bbox_xyxy"][1], 1),
                          "x2": round(r["bbox_xyxy"][2], 1), "y2": round(r["bbox_xyxy"][3], 1)}
                         for i, r in enumerate(rows, 1)])


@st.fragment(run_every=0.3)
def webcam_panel(checkpoint, confidence, visible_classes):
    st.subheader("Webcam trực tiếp")
    st.caption("Mỗi thành viên chạy ứng dụng trên máy của mình để dùng webcam của chính máy đó.")
    index = int(st.number_input("Chỉ số webcam", min_value=0, max_value=20, value=0, step=1,
                                help="Thường là 0. Thử 1 hoặc 2 nếu dùng camera USB.", key="webcam_index"))
    config = (index, str(checkpoint), checkpoint.stat().st_mtime_ns, confidence)
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
                    inspector = load_inspector(str(checkpoint), checkpoint.stat().st_mtime_ns)
                camera = CameraSession(index, inspector.predict, confidence)
                st.session_state.camera_session = camera
                st.session_state.camera_config = config
            camera.start()
        if start or stop or reopen:
            st.rerun()
        snapshot = camera.snapshot() if camera else None
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
        rows = [row for row in snapshot["output"]["detections"] if row["class_name"] in visible_classes]
        st.image(draw_boxes(snapshot["frame"], rows), channels="BGR", width="stretch")
        counts = st.columns(4)
        for i, name in enumerate(PROJECT_NAMES):
            counts[i % 4].metric(name, sum(row["class_name"] == name for row in rows))
        fps_col, latency_col, frame_col = st.columns(3)
        fps_col.metric("FPS xử lý", f"{snapshot['processing_fps']:.1f}")
        latency_col.metric("p95 xử lý frame", f"{snapshot['app_latency_p95_ms']:.0f} ms")
        frame_col.metric("Frame đã xử lý", snapshot["frames_processed"])
        st.caption("Số đối tượng trong frame hiện tại. p95 đo từ lúc Python nhận frame đến khi có dự đoán "
                   "(tối đa 300 frame gần nhất), chưa gồm trễ camera và hiển thị trình duyệt.")
    if snapshot["frames_processed"]:
        report = {key: value for key, value in snapshot.items() if key not in {"frame", "output"}}
        report.update({"weights": str(checkpoint.relative_to(ROOT)), "confidence": confidence,
                       "device": "cpu", "imgsz": 640, "nms": False,
                       "platform": platform.platform(), "python": platform.python_version(),
                       "versions": {name: version(name) for name in ("ultralytics", "torch", "opencv-python", "streamlit")},
                       "measurement_note": "FPS gồm mở camera và xử lý; p95 từ nhận frame đến kết quả, chưa gồm camera/browser."})
        st.download_button("Tải kết quả kiểm webcam", json.dumps(report, ensure_ascii=False, indent=2),
                           "webcam_summary.json", "application/json", key="webcam_report", on_click="ignore")


def main():
    st.set_page_config(page_title="Camera AI · Xem kết quả", layout="wide")
    st.title("Phòng quan sát AI")
    st.caption("Nhận dạng tám lớp trên ảnh, video hoặc webcam của máy đang chạy ứng dụng.")
    with st.sidebar:
        st.header("Nguồn & mô hình")
        source = st.radio("Nguồn dữ liệu", ["Bộ mẫu COCO128", "Ảnh của bạn", "Khung hình video", "Webcam trực tiếp"])
        weights = sorted((ROOT / "weights").glob("*.pt"))
        weights += sorted((ROOT / "runs/train").glob("*/weights/best.pt"))
        if not weights:
            st.error("Chưa có checkpoint. Tải YOLO26n theo README.")
            st.stop()
        checkpoint = st.selectbox("Checkpoint", weights, format_func=lambda p: str(p.relative_to(ROOT)))
        confidence = st.slider("Ngưỡng confidence", 0.05, 0.95, 0.25, 0.05)
        visible_classes = st.multiselect("Lớp hiển thị", list(PROJECT_NAMES), default=list(PROJECT_NAMES))
        st.caption("CPU · ảnh suy luận 640 · NMS-free")
        st.caption("Confidence là điểm của dự đoán, không phải độ chính xác của toàn mô hình.")
    try:
        if source == "Webcam trực tiếp":
            webcam_panel(checkpoint, confidence, visible_classes)
            return
        camera = st.session_state.get("camera_session")
        if camera is not None:
            camera.stop()
        selected = {"Bộ mẫu COCO128": sample_source, "Ảnh của bạn": uploaded_source,
                    "Khung hình video": video_source}[source]()
        if selected is None:
            st.stop()
        frame, source_name, refs = selected
        fingerprint = hashlib.sha256(frame.tobytes() + str(frame.shape).encode()
                                     + str(checkpoint).encode() + str(checkpoint.stat().st_mtime_ns).encode()
                                     + str(confidence).encode() + source_name.encode()).hexdigest()
        st.subheader(source_name)
        st.caption(f"Kích thước gốc {frame.shape[1]} × {frame.shape[0]} pixel. Nút phóng to nằm ở góc ảnh.")
        if st.button("Phân tích ảnh", type="primary", key="analyze"):
            st.session_state.pop("inspection", None)
            with st.spinner("Đang chạy YOLO26n…"):
                engine = load_inspector(str(checkpoint), checkpoint.stat().st_mtime_ns)
                output = engine.predict(frame, confidence)
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
                st.info("Bấm ‘Phân tích ảnh’. Khi đổi nguồn, model hoặc confidence, cần chạy lại để xem kết quả mới.")
        if valid:
            counts = st.columns(4)
            for index, name in enumerate(PROJECT_NAMES):
                column = counts[index % 4]
                column.metric(name, sum(r["class_name"] == name for r in rows))
            st.metric("Xử lý ảnh", f"{output['processing_ms']:.0f} ms")
            st.caption("Thời gian predict sau khi nạp model; lần đầu có thể gồm warmup. Không phải độ trễ webcam.")
            if rows:
                table_col, crop_col = st.columns([3, 2])
                with table_col:
                    st.markdown("**Chọn một dòng để xem vùng ảnh cắt**")
                    selection_key = fingerprint + "_" + "_".join(visible_classes)
                    event = st.dataframe(detection_table(rows), hide_index=True, on_select="rerun",
                                         selection_mode="single-row", key=selection_key)
                with crop_col:
                    chosen = event.selection.rows
                    if chosen and chosen[0] < len(rows):
                        row = rows[chosen[0]]
                        st.image(crop_box(frame, row["bbox_xyxy"]), channels="BGR", width=260)
                        st.write(f"**{row['class_name']}** · confidence {row['confidence']:.1%}")
                        st.caption(f"ID dự án: {row['class_id']} · ID checkpoint: {row['model_class_id']}")
                    else:
                        st.info("Chọn một đối tượng trong bảng bên trái.")
            else:
                st.info("Không có đối tượng ở ngưỡng và lớp hiển thị hiện tại. Đây không phải bằng chứng ảnh không có vật thể.")
            export = {"source": source_name, "weights": str(checkpoint.relative_to(ROOT)), "confidence_threshold": confidence,
                      "device": "cpu", "imgsz": 640, "nms": False, "visible_classes": visible_classes,
                      "processing_ms": output["processing_ms"], "detections": rows}
            encoded_ok, encoded = cv2.imencode(".png", annotated)
            if not encoded_ok:
                raise ValueError("Không tạo được ảnh tải xuống.")
            downloads = st.columns(3)
            downloads[0].download_button("Tải ảnh kết quả", encoded.tobytes(), "detection.png", "image/png")
            downloads[1].download_button("Tải JSON", json.dumps(export, ensure_ascii=False, indent=2), "detection.json", "application/json")
            downloads[2].download_button("Tải bảng CSV", detection_table(rows).to_csv(index=False).encode("utf-8-sig"), "detection.csv", "text/csv")
        if refs is not None:
            with st.expander("Đối chiếu nhãn gốc COCO128"):
                reference = [r for r in refs if r["class_name"] in visible_classes]
                st.image(draw_boxes(frame, reference, reference=True), channels="BGR", width=640)
                st.write(f"Nhãn tham khảo: {len(reference)} hộp thuộc các lớp đang hiển thị.")
                st.caption("Khung xanh lá là nhãn tham khảo. So bằng mắt để tìm lỗi; chưa tính TP/FP/FN hay mAP.")
    except Exception as exc:
        st.error(f"Không xử lý được nguồn: {exc}")


if __name__ == "__main__":
    main()
