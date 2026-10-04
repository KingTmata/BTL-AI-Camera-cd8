> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

# Kết quả lượt YOLO26n CPU một epoch

Run `project8_v03_26n_cpu_20261004_181257_1eb3de94` hoàn tất; bắt đầu 04/10/2026 18:12:58, kết thúc 04/10/2026 18:35:04 (Asia/Saigon). Đây là lượt thử đầu trên toàn train, chưa nghiệm thu model cuối hoặc toàn bộ G3.

## Cấu hình và dữ liệu

- Dataset duy nhất project8_v0.3, đã được người dùng duyệt và khóa tại chỗ: 5.002 ảnh, 26.566 hộp; train/val/test = 3.883/798/321. Train đủ 1.942 batch (batch cuối một ảnh); validation và final validation mỗi lần 200 batch, batch size 4 theo trainer.
- Tám lớp ID 0–7: person, table, chair, laptop, cell phone, backpack, book, cup. Test chưa có ground truth backpack/cup; không dùng test trong lượt này.
- YOLO26n pretrained có sẵn; CPU, epochs 1, imgsz 640, batch 2, workers 0, cache False, fraction 1.0; MuSGD, lr0 0.001, patience 10, seed 42. PyTorch dùng 8 luồng CPU.
- save=True, save_period=1, resume=False. Khởi tạo run mới từ checkpoint; không tiếp tục optimizer của run cũ.
- PyTorch 2.14.0+cpu, Ultralytics 8.4.165, Python 3.11.16, psutil 7.2.2; không dùng GPU, SSL hoặc YOLO26s.

## Thời gian và bộ nhớ thực đo

| Giai đoạn | Giây |
|---|---:|
| Kiểm dữ liệu/checkpoint | 80.578 |
| Chuẩn bị | 18.578 |
| Train | 1088.297 |
| Validation chọn checkpoint | 65.125 |
| Lưu checkpoint | 1.250 |
| Final validation | 39.547 |
| Hoàn thiện trainer | 1.843 |
| Kiểm checkpoint/hash | 30.719 |
| **Tổng worker** | **1325.937** |

Tổng khoảng 22.10 phút; riêng train 18.14 phút, khoảng 3.57 ảnh/giây. Thời gian CSV 1153.5 giây là train + validation của epoch, không phải tổng worker.

- RAM máy: 13.79 GiB; khả dụng trước train: 6.86 GiB.
- Đỉnh RSS cây tiến trình lấy mẫu: 0.961 GiB (1,031,905,280 byte).
- Peak working set worker theo Windows: 1.098 GiB (1,179,152,384 byte).
- RAM khả dụng thấp nhất được lấy mẫu: 6.145 GiB. Lượt batch 2/cache tắt hoàn tất, không có lỗi thiếu bộ nhớ. Số đo RSS không phải toàn bộ RAM hệ thống và mẫu có thể bỏ lỡ đỉnh ngắn; peak working set được ghi riêng.

## Loss và validation

| Chỉ số | Giá trị |
|---|---:|
| train/box_loss (CSV epoch 1) | 1.57004 |
| train/cls_loss (CSV epoch 1) | 5.70516 |
| train/l1_loss (CSV epoch 1) | 0.01751 |
| val/box_loss (CSV epoch 1) | 1.45456 |
| val/cls_loss (CSV epoch 1) | 5.08331 |
| val/l1_loss (CSV epoch 1) | 0.01417 |
| metrics/precision(B) (final validation) | 19.9791% |
| metrics/recall(B) (final validation) | 11.3472% |
| metrics/mAP50(B) (final validation) | 7.0884% |
| metrics/mAP50-95(B) (final validation) | 4.9520% |

Metric trên là validation của trainer NMS-free, chưa phải bảng so sánh evaluator chung. mAP50 khoảng 7,09%, mAP50–95 khoảng 4,95%: lượt đầu chạy được nhưng chất lượng còn thấp, chưa dùng kết quả này để kết luận đã hoàn thành fine-tune hoặc nghiệm thu tuần 3.

AP50 theo lớp do final validator in trong console (số làm tròn): person 17,9%; table 0,879%; chair 9,44%; laptop 28,4%; cell phone/backpack/cup 0%; book 0,12%. Không suy ra nguyên nhân từ một epoch; cần baseline và phân tích TP/FP/FN trước khi đánh giá cải thiện.

## Checkpoint và tính toàn vẹn

Cả ba checkpoint đã được worker nạp bằng YOLO và xác nhận detect, đúng mapping tám lớp. Hash file được kiểm lại khi lập báo cáo. `epoch0.pt` tương ứng epoch 1. Dataset và pretrained khởi tạo không đổi; không sao chép hoặc chỉnh split/media/TXT.

| Checkpoint | Dung lượng byte | SHA-256 |
|---|---:|---|
| [last.pt](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/last.pt) | 5,362,949 | `1b80d8e02380215271a2f3606f66c6b9840f37da5bdedb474cf3e36f42eba495` |
| [best.pt](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt) | 5,362,949 | `c36a5ebe74f48bb116415168c42e0c52de3c8f813702eba030e36fbdfa87c53b` |
| [epoch0.pt](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/epoch0.pt) | 15,524,899 | `616999481a1922fe9a64811dd1272ea7519e3eb35bd38e6f438fd505f440fcbf` |

- Snapshot SHA-256: `e7a891899268d614fbb58eafb093fe6b37a0ec9367c78baf60a392cd31dc2b0a`.
- Pretrained SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

## Kiểm thử và giao diện

44/44 kiểm thử đạt, không skip, trước khi chạy thật; compileall và git diff --check đạt. Kiểm trên trình duyệt: mặc định CPU đúng, Start bị khóa khi active, Recognition tạm dừng, tải lại/đóng/mở lại tab vẫn cùng worker; final validation không cộng thêm epoch. Sau kết thúc nút Stop bị khóa và cả ba checkpoint tải được; nút best.pt đã tải thực và checksum khớp file gốc. Không thấy lỗi console trình duyệt ở các lượt kiểm.

Camera trong kiểm thử hồi quy là camera mô phỏng; không thay kiểm thiết bị thật trên từng máy thành viên. [Bằng chứng UI/test](../../../../runs/week3_ui/verification.json).

## Artefact và bước tiếp theo

- [State](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/state.json), [timings/RAM](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/timings.json), [environment/hash](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/environment.json), [config thực](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/args.yaml).
- [Results CSV](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/results.csv), [console log](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/console.log), [checkpoint validation](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/checkpoint_validation.json), [loss/validation plot](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/results.png).
- [Hướng dẫn Training UI](../../../../TRIEN_KHAI_TUAN_3_TIEP_TUC.md). App local tại http://127.0.0.1:8501.
- Chưa chạy baseline evaluator chung, bảng pretrained/fine-tuned hoặc fine-tune dài. Chỉ một epoch được chạy; việc học thêm 2–3 epoch/tăng giới hạn chờ người dùng chọn sau khi xem kết quả.
