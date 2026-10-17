> **Dataset hiện hành08/10/2026:**v0.5 đủ15.000 ảnh (12.000train/2.500val/500test), tất cảfull_review_pending, chưatrainv05. Weight/metricv04 bên dưới làlịch sử. XemREADME gốc vàtasks/plan.md.

**Cập nhật v0.4 — 08/10/2026:** đã nhập ZIP full train 20/50 epoch; dùng `weights/BEST_Project8_v04_20261008.pt` (best epoch 19), alias `weights/BEST_Project8.pt`. Trainer 1280/NMS-free: 75,356% / 53,488%; cùng protocol COCO 640/NMS=True trên val v0.4: trước 47,593% / 31,047% → mới 70,706% / 50,694%. Trên val v0.3: 56,727% / 35,896% → 55,279% / 35,049%, nên giữ best cũ để đối chứng. Dataset mới 12.000 ảnh (9.540/1.754/706); ID 7 bottle gồm cup+bottle. Recognition đã đổi mapping; local Training/export v0.3 vẫn giữ riêng. [Báo cáo và hash](../archive/completed/reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/report.md).

# Trọng số

## Model v0.3 — đối chứng lịch sử

Để đối chiếu bản v0.3 trong Streamlit, chọn **`weights/BEST_Project8_20261008.pt`**. [Mở trọng số](../../weights/BEST_Project8_20261008.pt). Đây là bản v0.3 giữ để đối chứng; alias BEST_Project8.pt đã được cập nhật sang v0.4 như ghi ở đầu tài liệu. Recognition và Training liệt kê đệ quy `weights/**/*.pt`; rerun/tải lại trang nếu danh sách đang cũ.

K3 `project8_26n_kaggle_20261008_040051` hoàn tất 20/20 epoch từ K2 best, `resume=False`; **best epoch 19**, khác last epoch 20. Model 5.547.077 byte; SHA-256 `d4244d26e69d78a066944b22ba64f80ea68602f2f994d6cf6bbae7970ff123f9`. Đã kiểm hash/CRC ZIP, đúng detect/tám lớp, stride 8/16/32, tensor hữu hạn, CSV/metadata khớp và smoke prediction CPU đạt.

Evaluator chung 640/NMS=True: mAP50 **56,727%**, mAP50–95 **35,896%**; tăng **2,879/2,049 điểm %** so K2. Đây là validation, khác trainer validation NMS-free 1280. [Báo cáo đầy đủ K3](../archive/completed/reports/results/kaggle/project8_26n_kaggle_20261008_040051/report.md) · [Bảng 50 epoch trong 6 run](../archive/completed/reports/results/training_50_epochs_20261008/report.md).

## Bản cũ và artifact nguồn

- [K2 best ngày 06/10](../../weights/kaggle/BEST_Project8_20261006.pt): hash `ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf`, 5.546.053 byte. Giữ để quay lui/đối chiếu; [báo cáo K2](../archive/completed/reports/results/kaggle/project8_26n_kaggle_20261006_083226/report.md).
- K1 và pretrained `weights/yolo26n.pt` vẫn được giữ. Pretrained COCO80 là model đối chứng, khác model Project8 tám lớp.
- ZIP gốc, best/last và log K3 ở `runs/kaggle/project8_26n_kaggle_20261008_040051/`; CSV, cấu hình, metrics và audit ở `reports/results/kaggle/project8_26n_kaggle_20261008_040051/`. Markdown ở `README_DA_HOAN_THANH/reports/results/`.

Đổi tên hoặc chép file không đổi tensor; tên có ngày xác định phiên bản, tên `BEST_Project8.pt` là alias cập nhật. Checkpoint stripped có `epoch=-1`, `optimizer=None`: dùng inference hoặc khởi tạo fine-tune mới, không khẳng định đủ trạng thái resume optimizer. Weights bị Git ignore, cần sao lưu riêng. Chưa đánh giá test độc lập/FPS camera thật.
