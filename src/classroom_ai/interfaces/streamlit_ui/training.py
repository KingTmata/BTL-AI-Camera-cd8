"""CPU training controls and process-backed monitoring for Streamlit."""

import datetime
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st
import yaml

from classroom_ai.training.kaggle_export import NOTEBOOK, build_bundle, import_kaggle_result
from classroom_ai.training.manager import (
    ADVANCED_DEFAULTS,
    checkpoint_label,
    mosaic_active_epochs,
    read_json,
)

PHASES = {
    "preflight": "Kiểm dữ liệu và checkpoint",
    "setup": "Chuẩn bị trainer",
    "train": "Train",
    "val": "Validation",
    "final_val": "Final validation",
    "checkpoint": "Lưu checkpoint",
    "verification": "Kiểm checkpoint và snapshot",
    "finalizing": "Hoàn thiện kết quả",
    "finished": "Đã kết thúc",
    "failed": "Lỗi",
}


def duration(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


def advanced_controls():
    values = {}
    with st.expander("Nâng cao · tăng cường dữ liệu, cân bằng lớp và loss"):
        st.caption(
            "Các thay đổi áp dụng cho run mới. Bấm Xem trước cấu hình để cập nhật lịch augmentation; "
            "chưa bắt đầu train cho đến khi bấm Bắt đầu training."
        )
        st.markdown(
            "**Tăng cường dữ liệu (augmentation)** — tạo biến thể ảnh và biến đổi nhãn tương ứng khi train."
        )
        specs = [
            (
                "mosaic",
                "Mosaic · xác suất ghép 4 ảnh",
                "1.0 = 100% cơ hội ghép; 0 = tắt. Giúp đa dạng bối cảnh/kích thước, không tự cân bằng lớp.",
            ),
            (
                "mixup",
                "MixUp · xác suất trộn 2 ảnh",
                "Trộn ảnh và nhãn. 0.1 = 10% cơ hội áp dụng, không phải tỷ lệ pha ảnh. Bật mạnh có thể làm khó học.",
            ),
            (
                "cutmix",
                "CutMix · xác suất ghép vùng ảnh",
                "Cắt vùng ảnh này ghép vào ảnh khác; nhãn được xử lý tương ứng. 0 = tắt.",
            ),
            (
                "fliplr",
                "Lật ngang · xác suất",
                "0.5 = 50% cơ hội lật trái/phải. Chọn phép biến đổi phù hợp góc nhìn camera.",
            ),
            (
                "flipud",
                "Lật dọc · xác suất",
                "0 = tắt; ảnh bị lật trên/dưới có thể không giống cảnh camera thực tế.",
            ),
        ]
        cols = st.columns(2)
        for i, (key, label, help_text) in enumerate(specs):
            values[key] = cols[i % 2].number_input(
                label,
                min_value=0.0,
                max_value=1.0,
                value=ADVANCED_DEFAULTS[key],
                step=0.05,
                key=f"train_{key}",
                help=help_text,
            )
        values["degrees"] = cols[1].number_input(
            "Xoay · biên độ độ (degrees)",
            min_value=0.0,
            max_value=180.0,
            value=ADVANCED_DEFAULTS["degrees"],
            step=5.0,
            key="train_degrees",
            help="Ví dụ 10 = góc ngẫu nhiên trong khoảng −10° đến +10°; 0 = không xoay.",
        )
        values["close_mosaic"] = st.number_input(
            "Tắt Mosaic trong N epoch cuối (close_mosaic)",
            min_value=0,
            max_value=1000,
            value=ADVANCED_DEFAULTS["close_mosaic"],
            key="train_close_mosaic",
            help="Đây là số epoch, không phải xác suất. 0 = không đóng; nếu N ≥ số epoch thì đóng ngay từ đầu.",
        )
        st.caption(
            "close_mosaic đóng đồng thời Mosaic, MixUp, CutMix và Copy-Paste (khi tác vụ hỗ trợ). "
            "Xoay, lật và HSV vẫn có thể hoạt động. Với 100 epoch / close_mosaic=10: 90 epoch đầu cho phép ghép, 10 epoch cuối tắt."
        )
        st.markdown("**Cân bằng lớp** — tăng mức đóng góp loss của lớp ít bounding box (instance).")
        values["cls_pw"] = st.number_input(
            "Mức weighting theo tần suất lớp (cls_pw)",
            min_value=0.0,
            max_value=1.0,
            value=ADVANCED_DEFAULTS["cls_pw"],
            step=0.25,
            key="train_cls_pw",
            help="0 = không weighting; 0.25 = nhẹ; 0.5 = mạnh hơn; 1 = inverse-frequency đầy đủ. "
            "Trainer đếm box tập train, tính (1 / số box)^cls_pw rồi chuẩn hóa trung bình weight = 1.",
        )
        st.caption(
            "cls_pw không nhân bản ảnh hoặc sửa split. Lớp hiếm có nhãn sai vẫn cần kiểm tra; "
            "weighting không bảo đảm Recall tăng. So A=0, B=0.25, C=0.5 từ cùng checkpoint và cùng cấu hình còn lại."
        )
        st.markdown(
            "**Trọng số loss** — điều chỉnh mức ưu tiên từng thành phần khi tối ưu; không phải điểm chất lượng."
        )
        cols = st.columns(3)
        for col, key, label, help_text in zip(
            cols,
            ("box", "cls", "dfl"),
            (
                "Box · trọng số định vị",
                "CLS · trọng số phân loại",
                "L1 · trọng số khoảng cách (dfl)",
            ),
            (
                "box: mức ưu tiên độ khớp bounding box.",
                "cls: mức ưu tiên phân loại chung cho tất cả lớp; khác cls_pw ưu tiên tương đối từng lớp.",
                "YOLO26 dùng L1 với tên tham số dfl; không dùng DFL như các kiến trúc cũ.",
            ),
        ):
            values[key] = col.number_input(
                label,
                min_value=0.0,
                value=ADVANCED_DEFAULTS[key],
                step=0.1,
                key=f"train_{key}",
                help=help_text + " 0 = bỏ thành phần loss này; nên giữ mặc định để làm đối chứng.",
            )
        st.caption(
            "Đổi loss weight làm thay đổi thang loss: không so trực tiếp loss giữa hai cấu hình khác weight. "
            "Quyết định bằng Precision/Recall/AP từng lớp và mAP trên cùng validation."
        )
        st.info(
            "Confidence dùng để lọc dự đoán: chỉnh ở Recognition (mặc định 0.25). "
            "Validation trainer đang dùng ngưỡng đầu vào 0.001 để đo đường Precision–Recall; "
            "đổi slider Recognition không đổi số đo training. YOLO26 anchor-free nên không có ô chỉnh anchor boxes."
        )
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
        st.caption(
            f"{selected} · {status} · tiến trình {'đang hoạt động' if running else 'đã kết thúc'}"
        )
        if status == "failed":
            st.error(snapshot.get("error", "Run gặp lỗi. Xem log để tìm nguyên nhân."))
        elif status in {"completed", "stopped"}:
            st.success(
                f"Đã lưu {snapshot.get('epochs_completed', 0)} epoch. "
                + (
                    "Checkpoint và snapshot đã được kiểm."
                    if snapshot.get("checkpoints_verified")
                    else ""
                )
            )
        epoch = min(snapshot.get("epoch", 0), snapshot["config"]["epochs"])
        batch, batches = snapshot.get("batch", 0), snapshot.get("batches", 0)
        if batches and phase in {"train", "val", "final_val"}:
            phase_label = PHASES[phase]
            images = snapshot.get("train_images" if phase == "train" else "val_images")
            phase_batch = snapshot.get("phase_batch_size")
            st.progress(
                min(1.0, batch / batches),
                text=f"{phase_label} · Epoch {epoch}/{snapshot['config']['epochs']} · batch {batch}/{batches}",
            )
            if images is not None and phase_batch is not None:
                st.caption(
                    f"{phase_label}: {images:,} ảnh · tối đa {phase_batch} ảnh/batch · "
                    "batch cuối có thể ít ảnh hơn. Test không chạy trong lượt train này."
                )
        elapsed = snapshot.get("elapsed_seconds", 0)
        if running and snapshot.get("started_at"):
            elapsed = time.time() - snapshot["started_at"]
        stats = st.columns(4)
        stats[0].metric(
            "Epoch đã lưu", f"{snapshot.get('epochs_completed', 0)}/{snapshot['config']['epochs']}"
        )
        stats[1].metric("Thời gian run", duration(elapsed))
        stats[2].metric(
            "RSS tiến trình train", f"{snapshot.get('process_tree_rss_bytes', 0) / 1024**3:.2f} GiB"
        )
        stats[3].metric("RAM khả dụng", f"{psutil.virtual_memory().available / 1024**3:.2f} GiB")
        st.caption(
            f"Batch train thực: {snapshot.get('batch_size', snapshot['config']['batch'])} · "
            f"batch của giai đoạn hiện tại: {snapshot.get('phase_batch_size', 'chưa đo')} · "
            f"đỉnh RSS đã đo: {snapshot.get('rss_peak_bytes', 0) / 1024**3:.2f} GiB"
        )
        if st.button(
            "Dừng sau epoch hiện tại",
            key="training_stop",
            disabled=not running or snapshot["stop_requested"],
        ):
            manager.request_stop(snapshot["run_dir"])
            st.rerun(scope="fragment")
        if snapshot["stop_requested"]:
            st.info(
                "Đã gửi yêu cầu. Trainer sẽ dừng sau khi validation và checkpoint của epoch hoàn tất."
            )
        losses = snapshot.get("losses", {})
        metrics = snapshot.get("final_metrics") or snapshot.get("val_metrics", {})
        if losses or metrics:
            st.dataframe(
                pd.DataFrame(
                    [{"Chỉ số": k, "Giá trị đã đo": v} for k, v in {**losses, **metrics}.items()]
                ),
                hide_index=True,
                width="stretch",
            )
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
                st.download_button(
                    f"Tải {checkpoint_label(path)}",
                    data=lambda p=path: p.read_bytes(),
                    file_name=f"{selected}_{path.name}",
                    mime="application/octet-stream",
                    disabled=not enabled,
                    key=f"download_{selected}_{path.name}",
                    on_click="ignore",
                )
        st.caption(f"Artefact: {snapshot['run_dir']}")
        with st.expander("Lịch sử run"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Run": r["run_id"],
                            "Trạng thái": r.get("status"),
                            "Epoch đã lưu": r.get("epochs_completed", 0),
                            "Thời gian": duration(r.get("elapsed_seconds")),
                        }
                        for r in history
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    except (ValueError, OSError) as exc:
        st.error(f"Không đọc được trạng thái training: {exc}")


def training_page(manager, stop_webcam, clear_inference):
    st.title("Huấn luyện YOLO")
    st.caption(
        "CPU local chạy trên máy này; Kaggle nhận gói/notebook để bạn tải lên và bấm chạy trên tài khoản của mình."
    )
    st.info(
        "Màn hình này dùng dataset v0.3 và nhãn cup. Dataset v0.4 12k với nhãn bottle dùng notebook project8_v04_12k_from_best50.ipynb; Recognition đã dùng best v0.4."
    )
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
        columns[3].metric("RAM khả dụng", f"{psutil.virtual_memory().available / 1024**3:.2f} GiB")
        st.caption(
            "Tám lớp project8 · test nằm ngoài huấn luyện/tối ưu. Preflight kiểm checksum trước khi chạy."
        )
        weights = sorted(
            {*(manager.root / "weights").rglob("*.pt"), *manager.runs.glob("*/weights/*.pt")}
        )
        weights = [p for p in weights if p.is_file()]
        preferred = manager.root / "weights/yolo26n.pt"
        if not weights:
            st.error("Chưa có checkpoint .pt trong weights/ hoặc runs/train/.")
            return
        verified = []
        for candidate in manager.runs.glob("*/weights/best.pt"):
            state = read_json(candidate.parents[1] / "state.json")
            validation = read_json(candidate.parents[1] / "checkpoint_validation.json")
            if (
                state
                and validation
                and state.get("status") in {"completed", "stopped"}
                and validation.get("valid")
            ):
                verified.append((state.get("finished_at", 0), candidate))
        preferred = max(
            verified, default=(0, manager.root / "weights/yolo26n.pt"), key=lambda item: item[0]
        )[1]
        selected_checkpoint = st.selectbox(
            "Checkpoint khởi tạo",
            weights,
            index=weights.index(preferred) if preferred in weights else 0,
            format_func=lambda p: str(p.relative_to(manager.root)),
            key="train_checkpoint",
        )
        execution_mode = st.radio(
            "Nơi huấn luyện",
            ["CPU local", "Kaggle Notebook"],
            horizontal=True,
            key="training_execution_mode",
        )
        if execution_mode == "Kaggle Notebook":
            st.markdown("### Chuẩn bị gói chuyển sang Kaggle")
            st.write(
                "Tạo ZIP train+val, checkpoint đã chọn, baseline pretrained, manifest/hash và notebook. "
                "Nút này **không điều khiển accelerator hoặc mở job cloud**; tải ZIP lên Kaggle Dataset, gắn vào notebook rồi chạy cell trên Kaggle."
            )
            st.caption(
                "Nếu run local đang hoạt động, tạo gói sẽ bị khóa để tránh đọc/nén dataset trong lúc CPU train. "
                "Chỉ checkpoint từ run completed/stopped đã qua kiểm cuối mới được đóng gói."
            )
            notebook_bytes = NOTEBOOK.read_bytes()
            download_col, open_col = st.columns(2)
            download_col.download_button(
                "Tải notebook Kaggle",
                notebook_bytes,
                "project8_training.ipynb",
                "application/x-ipynb+json",
                key="kaggle_notebook_download",
            )
            open_col.link_button("Mở Kaggle Notebooks", "https://www.kaggle.com/code/new")
            st.markdown(
                "1. Tải ZIP lên Kaggle như một Dataset; tạo notebook Kaggle và gắn Dataset ở Input.\n"
                "2. Trên thanh đầu trang mở Settings → Accelerator → None (CPU) hoặc GPU; bật Internet nếu cần pip.\n"
                "3. Mở notebook đã tải, sửa `BUNDLE_ZIP` theo `/kaggle/input/<slug>/...`, chạy các cell kiểm tra.\n"
                "4. Cell train mặc định tắt; bật `START_TRAINING` khi sẵn sàng. Đầu ra nằm ở `/kaggle/working`.\n"
                "5. Dùng Save Version → Save & Run All để lưu output của version đã chạy xong, hoặc tải result ZIP/checkpoint về trước khi đóng session."
            )
            st.info(
                "CPU/GPU trong giao diện chỉ chọn nơi chuẩn bị gói; app này không khởi động hoặc điều khiển Kaggle. "
                "GPU/quota phụ thuộc tài khoản và thời điểm. Gói chứa train+val, không có test media."
            )
            export_key = "kaggle_export_result"
            if st.button(
                "Tạo gói Kaggle",
                type="primary",
                disabled=active is not None,
                key="kaggle_export_create",
            ):
                try:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    target = manager.root / ".temp" / f"kaggle_export_{stamp}"
                    with st.spinner("Đang xác minh checkpoint/hash và đóng gói train+val…"):
                        result = build_bundle(selected_checkpoint, target)
                    st.session_state[export_key] = result
                except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
                    st.error(f"Không tạo được gói Kaggle: {exc}")
            result = st.session_state.get(export_key)
            if result:
                sizes = result["sizes"]
                rows = []
                for label, key in (
                    ("Ảnh train", "train_images"),
                    ("Nhãn train", "train_labels"),
                    ("Ảnh validation", "val_images"),
                    ("Nhãn validation", "val_labels"),
                    ("Metadata + YAML", "metadata_and_yaml"),
                    ("Checkpoint project8 + baseline", "checkpoints"),
                    ("Code + notebook", "code_and_notebook"),
                    ("Tổng chưa nén", "uncompressed_total"),
                ):
                    count = sizes[key]
                    rows.append(
                        {
                            "Thành phần": label,
                            "Byte": count,
                            "MB": round(count / 1_000_000, 3),
                            "MiB": round(count / 1_048_576, 3),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(
                    f"ZIP: {result['zip_bytes']:,} byte · {result['zip_bytes'] / 1_000_000:.3f} MB · "
                    f"{result['zip_bytes'] / 1_048_576:.3f} MiB · SHA-256 `{result['zip_sha256']}` · "
                    f"{result['files']:,} file trong gói. Bản gốc release không bị sửa; export chỉ có train/val."
                )
                st.download_button(
                    "Tải ZIP dữ liệu + checkpoint + code",
                    Path(result["zip"]).read_bytes(),
                    "project8_kaggle_bundle.zip",
                    "application/zip",
                    key="kaggle_bundle_download",
                )
            st.markdown("### Nhập kết quả Kaggle về ứng dụng")
            st.caption(
                "Tải `<run_id>_result.zip` từ notebook output. Ứng dụng kiểm SHA-256, protocol val, "
                "checkpoint detect và thứ tự lớp trước khi lưu weights vào weights/kaggle/."
            )
            result_upload = st.file_uploader(
                "Result ZIP do notebook tạo", type=["zip"], key="kaggle_result_upload"
            )
            if st.button(
                "Xác minh và nhập checkpoint",
                disabled=active is not None or result_upload is None,
                key="kaggle_result_import",
            ):
                try:
                    with st.spinner("Đang xác minh hash và nạp checkpoint trên CPU…"):
                        imported = import_kaggle_result(result_upload.getvalue())
                    st.session_state["kaggle_imported_result"] = imported
                except (
                    ValueError,
                    OSError,
                    KeyError,
                    json.JSONDecodeError,
                    zipfile.BadZipFile,
                ) as exc:
                    st.error(f"Không nhập được kết quả: {exc}")
            imported = st.session_state.get("kaggle_imported_result")
            if imported:
                st.success(
                    f"Đã nhập `{imported['run_id']}` · SHA-256 `{imported['checkpoint_sha256']}` · "
                    f"weights: `{imported['checkpoint']}`. Chọn checkpoint này trong Recognition hoặc run local."
                )
                metrics = imported["metrics"]
                baseline = imported["baseline_metrics"]
                st.write(
                    f"Validation NMS chung · mAP50 {metrics['mAP50']:.4f} · "
                    f"mAP50–95 {metrics['mAP50_95']:.4f}; baseline mAP50 {baseline['mAP50']:.4f} · "
                    f"mAP50–95 {baseline['mAP50_95']:.4f}. Giá trị được nạp từ result đã hash; đây không phải kết quả app tự chạy."
                )
                per_class = []
                for class_name, values in metrics["per_class"].items():
                    base = baseline["per_class"][class_name]
                    per_class.append(
                        {
                            "Lớp": class_name,
                            "GT val": values["GT_boxes"],
                            "AP50 fine-tuned": values["AP50"],
                            "AP50 baseline": base["AP50"],
                            "AP50–95 fine-tuned": values["AP50_95"],
                            "AP50–95 baseline": base["AP50_95"],
                        }
                    )
                st.dataframe(pd.DataFrame(per_class), hide_index=True, width="stretch")
            st.markdown(
                "Nguồn: [Kaggle Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage) · "
                "[Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)"
            )
            if active:
                st.warning(
                    "Training CPU local đang chạy; chờ run terminal trước khi tạo gói để tránh tranh tài nguyên."
                )
            monitor(manager)
            return
        with st.form("training_form"):
            cols = st.columns(3)
            epochs = cols[0].number_input(
                "Số epoch", min_value=1, max_value=1000, value=1, key="train_epochs"
            )
            batch = cols[1].number_input(
                "Batch size", min_value=1, max_value=16, value=2, key="train_batch"
            )
            imgsz = cols[2].selectbox(
                "Kích thước ảnh", [320, 416, 512, 640, 768, 960, 1280], index=3, key="train_imgsz"
            )
            cols = st.columns(2)
            workers = cols[0].number_input(
                "Data workers", min_value=0, max_value=4, value=0, key="train_workers"
            )
            cache = cols[1].checkbox(
                "Cache ảnh trong RAM",
                value=False,
                key="train_cache",
                help="Cache không tạo bản sao media. Preflight từ chối nếu RAM khả dụng không đủ.",
            )
            name = st.text_input("Tên run", value="project8_v03_26n_cpu", key="train_name")
            advanced = advanced_controls()
            st.caption(
                "MuSGD · lr 0.001 · patience 10 · seed 42 · toàn bộ train · lưu từng epoch. "
                "Checkpoint đã train dùng để khởi tạo run mới; bản này không resume optimizer."
            )
            st.form_submit_button("Xem trước cấu hình", key="train_preview")
            submitted = st.form_submit_button(
                "Bắt đầu training", type="primary", disabled=active is not None, key="train_start"
            )
        allowed = mosaic_active_epochs(int(epochs), int(advanced["close_mosaic"]))
        if allowed == 0:
            st.warning(
                f"Cấu hình form: {epochs} epoch, close_mosaic={advanced['close_mosaic']} → "
                "Mosaic/MixUp/CutMix tắt từ epoch đầu, dù xác suất trong form lớn hơn 0."
            )
        else:
            st.caption(
                f"Cấu hình form: cho phép Mosaic/MixUp/CutMix trong {allowed}/{epochs} epoch đầu "
                "theo xác suất đã chọn. Giá trị 0 vẫn tắt riêng phép biến đổi đó."
            )
        with st.expander("Phân phối box tập train · xem trước class weights"):
            counts = [marker["counts"]["train"].get(name, 0) for name in data["names"].values()]
            weights_preview = [(1 / max(count, 1)) ** advanced["cls_pw"] for count in counts]
            mean_weight = sum(weights_preview) / len(weights_preview)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Lớp": list(data["names"].values()),
                        "Box train": counts,
                        "Weight dự kiến": [round(w / mean_weight, 3) for w in weights_preview],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Tính từ RELEASE đã khóa; preflight kiểm lại checksum. Đây là phép tính xem trước, "
                "không phải metric đã train. Log trainer là nguồn weight thực tế khi cls_pw > 0."
            )
        if submitted:
            if not stop_webcam():
                st.error(
                    "Webcam đang giải phóng thiết bị. Chờ dừng hoàn toàn rồi bắt đầu training."
                )
            else:
                clear_inference()
                run = manager.start(
                    epochs=int(epochs),
                    batch=int(batch),
                    imgsz=int(imgsz),
                    workers=int(workers),
                    cache=cache,
                    run_name=name,
                    checkpoint=selected_checkpoint,
                    advanced=advanced,
                )
                st.session_state.training_run = run.name
                st.rerun()
        if active:
            st.info("Một run đang hoạt động trên máy. Recognition tạm dừng; không mở thêm run.")
        monitor(manager)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Không khởi động training: {exc}")
