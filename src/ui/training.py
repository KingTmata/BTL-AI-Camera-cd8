"""CPU training controls and process-backed monitoring for Streamlit."""

import time
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st
import yaml

from src.training.manager import checkpoint_label, read_json
from src.training.options import ADVANCED_DEFAULTS, mosaic_active_epochs

PHASES = {"preflight": "Kiểm dữ liệu và checkpoint", "setup": "Chuẩn bị trainer",
          "train": "Train", "val": "Validation", "final_val": "Final validation",
          "checkpoint": "Lưu checkpoint", "verification": "Kiểm checkpoint và snapshot",
          "finalizing": "Hoàn thiện kết quả", "finished": "Đã kết thúc", "failed": "Lỗi"}


def duration(seconds):
    seconds = int(seconds or 0)
    return f"{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}"


def advanced_controls():
    values = {}
    with st.expander("Nâng cao · tăng cường dữ liệu, cân bằng lớp và loss"):
        st.caption("Các thay đổi áp dụng cho run mới. Bấm Xem trước cấu hình để cập nhật lịch augmentation; "
                   "chưa bắt đầu train cho đến khi bấm Bắt đầu training.")
        st.markdown("**Tăng cường dữ liệu (augmentation)** — tạo biến thể ảnh và biến đổi nhãn tương ứng khi train.")
        specs = [
            ("mosaic", "Mosaic · xác suất ghép 4 ảnh", "1.0 = 100% cơ hội ghép; 0 = tắt. Giúp đa dạng bối cảnh/kích thước, không tự cân bằng lớp."),
            ("mixup", "MixUp · xác suất trộn 2 ảnh", "Trộn ảnh và nhãn. 0.1 = 10% cơ hội áp dụng, không phải tỷ lệ pha ảnh. Bật mạnh có thể làm khó học."),
            ("cutmix", "CutMix · xác suất ghép vùng ảnh", "Cắt vùng ảnh này ghép vào ảnh khác; nhãn được xử lý tương ứng. 0 = tắt."),
            ("fliplr", "Lật ngang · xác suất", "0.5 = 50% cơ hội lật trái/phải. Chọn phép biến đổi phù hợp góc nhìn camera."),
            ("flipud", "Lật dọc · xác suất", "0 = tắt; ảnh bị lật trên/dưới có thể không giống cảnh camera thực tế."),
        ]
        cols = st.columns(2)
        for i, (key, label, help_text) in enumerate(specs):
            values[key] = cols[i % 2].number_input(label, min_value=0.0, max_value=1.0,
                value=ADVANCED_DEFAULTS[key], step=0.05, key=f"train_{key}", help=help_text)
        values["degrees"] = cols[1].number_input("Xoay · biên độ độ (degrees)", min_value=0.0,
            max_value=180.0, value=ADVANCED_DEFAULTS["degrees"], step=5.0, key="train_degrees",
            help="Ví dụ 10 = góc ngẫu nhiên trong khoảng −10° đến +10°; 0 = không xoay.")
        values["close_mosaic"] = st.number_input("Tắt Mosaic trong N epoch cuối (close_mosaic)",
            min_value=0, max_value=1000, value=ADVANCED_DEFAULTS["close_mosaic"], key="train_close_mosaic",
            help="Đây là số epoch, không phải xác suất. 0 = không đóng; nếu N ≥ số epoch thì đóng ngay từ đầu.")
        st.caption("close_mosaic đóng đồng thời Mosaic, MixUp, CutMix và Copy-Paste (khi tác vụ hỗ trợ). "
                   "Xoay, lật và HSV vẫn có thể hoạt động. Với 100 epoch / close_mosaic=10: 90 epoch đầu cho phép ghép, 10 epoch cuối tắt.")
        st.markdown("**Cân bằng lớp** — tăng mức đóng góp loss của lớp ít bounding box (instance).")
        values["cls_pw"] = st.number_input("Mức weighting theo tần suất lớp (cls_pw)", min_value=0.0,
            max_value=1.0, value=ADVANCED_DEFAULTS["cls_pw"], step=0.25, key="train_cls_pw",
            help="0 = không weighting; 0.25 = nhẹ; 0.5 = mạnh hơn; 1 = inverse-frequency đầy đủ. "
                 "Trainer đếm box tập train, tính (1 / số box)^cls_pw rồi chuẩn hóa trung bình weight = 1.")
        st.caption("cls_pw không nhân bản ảnh hoặc sửa split. Lớp hiếm có nhãn sai vẫn cần kiểm tra; "
                   "weighting không bảo đảm Recall tăng. So A=0, B=0.25, C=0.5 từ cùng checkpoint và cùng cấu hình còn lại.")
        st.markdown("**Trọng số loss** — điều chỉnh mức ưu tiên từng thành phần khi tối ưu; không phải điểm chất lượng.")
        cols = st.columns(3)
        for col, key, label, help_text in zip(cols, ("box", "cls", "dfl"),
                ("Box · trọng số định vị", "CLS · trọng số phân loại", "L1 · trọng số khoảng cách (dfl)"),
                ("box: mức ưu tiên độ khớp bounding box.",
                 "cls: mức ưu tiên phân loại chung cho tất cả lớp; khác cls_pw ưu tiên tương đối từng lớp.",
                 "YOLO26 dùng L1 với tên tham số dfl; không dùng DFL như các kiến trúc cũ.")):
            values[key] = col.number_input(label, min_value=0.0, value=ADVANCED_DEFAULTS[key],
                step=0.1, key=f"train_{key}", help=help_text + " 0 = bỏ thành phần loss này; nên giữ mặc định để làm đối chứng.")
        st.caption("Đổi loss weight làm thay đổi thang loss: không so trực tiếp loss giữa hai cấu hình khác weight. "
                   "Quyết định bằng Precision/Recall/AP từng lớp và mAP trên cùng validation.")
        st.info("Confidence dùng để lọc dự đoán: chỉnh ở Recognition (mặc định 0.25). "
                "Validation trainer đang dùng ngưỡng đầu vào 0.001 để đo đường Precision–Recall; "
                "đổi slider Recognition không đổi số đo training. YOLO26 anchor-free nên không có ô chỉnh anchor boxes.")
    return values


@st.fragment(run_every=2)
def monitor(manager):
    try:
        active = manager.active()
        busy = active is not None
        previous = st.session_state.get("training_busy", busy)
        st.session_state.training_busy = busy
        if previous != busy:
            st.rerun(scope="app")
        history = manager.history()
        if not history:
            st.info("Chưa có run. Cấu hình mặc định chạy một epoch bằng toàn bộ train.")
            return
        ids = [item["run_id"] for item in history]
        if st.session_state.get("training_run") not in ids:
            st.session_state.training_run = active["run_id"] if active else ids[0]
        selected = st.selectbox("Run đang xem", ids, key="training_run")
        snapshot = next(item for item in history if item["run_id"] == selected)
        running = busy and active["run_id"] == selected
        status = snapshot.get("status", "starting")
        phase = snapshot.get("phase", "preflight")
        st.subheader(PHASES.get(phase, phase))
        st.caption(f"{selected} · {status} · tiến trình {'đang hoạt động' if running else 'đã kết thúc'}")
        if status == "failed":
            st.error(snapshot.get("error", "Run gặp lỗi. Xem log để tìm nguyên nhân."))
        elif status in {"completed", "stopped"}:
            st.success(f"Đã lưu {snapshot.get('epochs_completed', 0)} epoch. "
                       + ("Checkpoint và snapshot đã được kiểm." if snapshot.get("checkpoints_verified") else ""))
        epoch = min(snapshot.get("epoch", 0), snapshot["config"]["epochs"])
        batch, batches = snapshot.get("batch", 0), snapshot.get("batches", 0)
        if batches and phase in {"train", "val", "final_val"}:
            phase_label = PHASES[phase]
            images = snapshot.get("train_images" if phase == "train" else "val_images")
            phase_batch = snapshot.get("phase_batch_size")
            st.progress(min(1., batch/batches),
                        text=f"{phase_label} · Epoch {epoch}/{snapshot['config']['epochs']} · batch {batch}/{batches}")
            if images is not None and phase_batch is not None:
                st.caption(f"{phase_label}: {images:,} ảnh · tối đa {phase_batch} ảnh/batch · "
                           "batch cuối có thể ít ảnh hơn. Test không chạy trong lượt train này.")
        elapsed = snapshot.get("elapsed_seconds", 0)
        if running and snapshot.get("started_at"):
            elapsed = time.time()-snapshot["started_at"]
        stats = st.columns(4)
        stats[0].metric("Epoch đã lưu", f"{snapshot.get('epochs_completed', 0)}/{snapshot['config']['epochs']}")
        stats[1].metric("Thời gian run", duration(elapsed))
        stats[2].metric("RSS tiến trình train", f"{snapshot.get('process_tree_rss_bytes', 0)/1024**3:.2f} GiB")
        stats[3].metric("RAM khả dụng", f"{psutil.virtual_memory().available/1024**3:.2f} GiB")
        st.caption(f"Batch train thực: {snapshot.get('batch_size', snapshot['config']['batch'])} · "
                   f"batch của giai đoạn hiện tại: {snapshot.get('phase_batch_size', 'chưa đo')} · "
                   f"đỉnh RSS đã đo: {snapshot.get('rss_peak_bytes', 0)/1024**3:.2f} GiB")
        if st.button("Dừng sau epoch hiện tại", key="training_stop",
                     disabled=not running or snapshot["stop_requested"]):
            manager.request_stop(snapshot["run_dir"])
            st.rerun(scope="fragment")
        if snapshot["stop_requested"]:
            st.info("Đã gửi yêu cầu. Trainer sẽ dừng sau khi validation và checkpoint của epoch hoàn tất.")
        losses = snapshot.get("losses", {})
        metrics = snapshot.get("final_metrics") or snapshot.get("val_metrics", {})
        if losses or metrics:
            st.dataframe(pd.DataFrame([{"Chỉ số": k, "Giá trị đã đo": v} for k,v in {**losses, **metrics}.items()]),
                         hide_index=True, width="stretch")
        rows = snapshot["metrics_rows"]
        if rows:
            with st.expander("Loss và validation theo epoch", expanded=True):
                frame = pd.DataFrame(rows)
                st.dataframe(frame, hide_index=True, width="stretch")
                columns = [c for c in frame if "loss" in c]
                if columns:
                    st.line_chart(frame.set_index("epoch")[columns])
        with st.expander("Thời gian từng giai đoạn và cấu hình"):
            st.json({"seconds_by_phase": snapshot.get("timings", {}), "config": snapshot["config"]})
        with st.expander("Log gần nhất", expanded=status == "failed"):
            st.code(snapshot["log_tail"] or "Đang chờ log…", language="text")
        if snapshot["checkpoints"]:
            st.markdown("**Checkpoint đã lưu**")
            for item in snapshot["checkpoints"]:
                path = Path(item)
                # best/last may be rewritten by final_eval. Epoch files are retained.
                enabled = not running or path.name.startswith("epoch")
                st.download_button(f"Tải {checkpoint_label(path)}", data=lambda p=path: p.read_bytes(),
                                   file_name=f"{selected}_{path.name}", mime="application/octet-stream",
                                   disabled=not enabled, key=f"download_{selected}_{path.name}", on_click="ignore")
        st.caption(f"Artefact: {snapshot['run_dir']}")
        with st.expander("Lịch sử run"):
            st.dataframe(pd.DataFrame([{"Run": r["run_id"], "Trạng thái": r.get("status"),
                                        "Epoch đã lưu": r.get("epochs_completed", 0),
                                        "Thời gian": duration(r.get("elapsed_seconds"))} for r in history]),
                         hide_index=True, width="stretch")
    except (ValueError, OSError) as exc:
        st.error(f"Không đọc được trạng thái training: {exc}")


def training_page(manager, stop_webcam, clear_inference):
    st.title("Huấn luyện YOLO")
    st.caption("CPU local · run mới từ checkpoint · tiếp tục chạy khi tải lại hoặc đóng trình duyệt")
    try:
        active = manager.active()
        data = yaml.safe_load((manager.root / "configs/data.yaml").read_text(encoding="utf-8"))
        marker = read_json(Path(data["path"]) / "RELEASE.json")
        if not marker:
            st.error("Dataset chưa khóa. Hoàn tất review và khóa snapshot trước khi training.")
            return
        columns = st.columns(4)
        for column, split in zip(columns[:3], ("train", "val", "test")):
            column.metric(f"Ảnh {split}", marker["counts"][split]["images"])
        columns[3].metric("RAM khả dụng", f"{psutil.virtual_memory().available/1024**3:.2f} GiB")
        st.caption("Tám lớp project8 · test nằm ngoài huấn luyện/tối ưu. Preflight kiểm checksum trước khi chạy.")
        weights = sorted({* (manager.root / "weights").glob("*.pt"), *manager.runs.glob("*/weights/*.pt")})
        weights = [p for p in weights if p.is_file()]
        preferred = manager.root / "weights/yolo26n.pt"
        with st.form("training_form"):
            checkpoint = st.selectbox("Checkpoint khởi tạo", weights,
                                      index=weights.index(preferred) if preferred in weights else 0,
                                      format_func=lambda p: str(p.relative_to(manager.root)), key="train_checkpoint")
            cols = st.columns(3)
            epochs = cols[0].number_input("Số epoch", min_value=1, max_value=1000, value=1, key="train_epochs")
            batch = cols[1].number_input("Batch size", min_value=1, max_value=16, value=2, key="train_batch")
            imgsz = cols[2].selectbox("Kích thước ảnh", [320, 416, 512, 640, 768, 960, 1280], index=3, key="train_imgsz")
            cols = st.columns(2)
            workers = cols[0].number_input("Data workers", min_value=0, max_value=4, value=0, key="train_workers")
            cache = cols[1].checkbox("Cache ảnh trong RAM", value=False, key="train_cache",
                                     help="Cache không tạo bản sao media. Preflight từ chối nếu RAM khả dụng không đủ.")
            name = st.text_input("Tên run", value="project8_v03_26n_cpu", key="train_name")
            advanced = advanced_controls()
            st.caption("MuSGD · lr 0.001 · patience 10 · seed 42 · toàn bộ train · lưu từng epoch. "
                       "Checkpoint đã train dùng để khởi tạo run mới; bản này không resume optimizer.")
            st.form_submit_button("Xem trước cấu hình", key="train_preview")
            submitted = st.form_submit_button("Bắt đầu training", type="primary", disabled=active is not None, key="train_start")
        allowed = mosaic_active_epochs(int(epochs), int(advanced["close_mosaic"]))
        if allowed == 0:
            st.warning(f"Cấu hình form: {epochs} epoch, close_mosaic={advanced['close_mosaic']} → "
                       "Mosaic/MixUp/CutMix tắt từ epoch đầu, dù xác suất trong form lớn hơn 0.")
        else:
            st.caption(f"Cấu hình form: cho phép Mosaic/MixUp/CutMix trong {allowed}/{epochs} epoch đầu "
                       "theo xác suất đã chọn. Giá trị 0 vẫn tắt riêng phép biến đổi đó.")
        with st.expander("Phân phối box tập train · xem trước class weights"):
            counts = [marker["counts"]["train"].get(name, 0) for name in data["names"].values()]
            weights_preview = [(1 / max(count, 1)) ** advanced["cls_pw"] for count in counts]
            mean_weight = sum(weights_preview) / len(weights_preview)
            st.dataframe(pd.DataFrame({"Lớp": list(data["names"].values()), "Box train": counts,
                "Weight dự kiến": [round(w / mean_weight, 3) for w in weights_preview]}), hide_index=True, width="stretch")
            st.caption("Tính từ RELEASE đã khóa; preflight kiểm lại checksum. Đây là phép tính xem trước, "
                       "không phải metric đã train. Log trainer là nguồn weight thực tế khi cls_pw > 0.")
        if submitted:
            if not stop_webcam():
                st.error("Webcam đang giải phóng thiết bị. Chờ dừng hoàn toàn rồi bắt đầu training.")
            else:
                clear_inference()
                run = manager.start(epochs=int(epochs), batch=int(batch), imgsz=int(imgsz),
                                    workers=int(workers), cache=cache, run_name=name, checkpoint=checkpoint,
                                    advanced=advanced)
                st.session_state.training_run = run.name
                st.rerun()
        if active:
            st.info("Một run đang hoạt động trên máy. Recognition tạm dừng; không mở thêm run.")
        monitor(manager)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Không khởi động training: {exc}")
