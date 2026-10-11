> Hồ sơ v0.3 trước lượt v0.4: số liệu và checkpoint có ngày ở đây được giữ nguyên. Alias `weights/BEST_Project8.pt` nay đã đổi sang best v0.4; dùng `weights/BEST_Project8_20261008.pt` để tái lập đối chứng v0.3.

# Kaggle Project8 — 20 epoch, trọng số best và đối soát ngày 08/10/2026

Ngày kiểm: **08/10/2026**, Asia/Saigon. Nguồn: result ZIP, checkpoint, CSV và log đã tải từ [Kaggle version 356261309](https://www.kaggle.com/code/awm3636/btn-ai-cd8/log?scriptVersionId=356261309). Không chạy train hay chấm toàn bộ validation lại trên máy local.

## Kết quả và model để chọn trong Streamlit

- Hoàn tất **20/20 epoch**; chọn **best ở epoch 19**. Lượt này là K3, nâng lịch sử lên **50 epoch thực thi trong 6 run**, không phải 50 lượt train hay một run resume liên tục.
- Chọn [weights/BEST_Project8_20261008.pt](../../../../../weights/BEST_Project8_20261008.pt) trong Recognition hoặc Training. [weights/BEST_Project8.pt](../../../../../weights/BEST_Project8.pt) là bản cùng byte, trỏ tới best hiện hành. Tên có ngày giúp phân biệt phiên bản; hai file này cho cùng model.
- Bản cũ K2 được bảo toàn tại [weights/kaggle/BEST_Project8_20261006.pt](../../../../../weights/kaggle/BEST_Project8_20261006.pt), hash nguồn `ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf`.
- Evaluator chung: mAP50 **53.848% → 56.727%**, tăng **2.879 điểm phần trăm**; mAP50–95 **33.847% → 35.896%**, tăng **2.049 điểm phần trăm**.

## Cấu hình, dữ liệu và thời gian

Run ID: `project8_26n_kaggle_20261008_040051`. Khởi tạo từ K2 best, `resume=False`; optimizer/scheduler/warmup được khởi tạo lại. YOLO26n detect, Ultralytics 8.4.165, Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8, T4 ×2 (`device=0,1`). `epochs=20`, `imgsz=1280`, `batch=16`, `optimizer=MuSGD`, `lr0=0.001`, `patience=5`, `seed=42`, `mosaic=0.5`, `close_mosaic=5`, `save_period=1`, `nms=False`. Xem [args.yaml gốc](../../../../../reports/results/kaggle/project8_26n_kaggle_20261008_040051/args.yaml).

Release `project8_v0.3-approved-20261004`, source manifest SHA-256 `bdcc6d2f5cee1ce7e973a453cc0f53e533829a94fbfdfd18a618077c8bd8fc64`. Train **3.883 ảnh**, validation **798 ảnh / 5.382 box**. Log xác nhận đã stage và kiểm hash 9.377 file train/val; test không được đưa vào lượt train/đánh giá này. Không suy diễn từ kiểm train/val thành kiểm lại toàn bộ dataset local.

CSV ghi thời gian tích lũy **2.104,05 giây ≈ 35 phút 04 giây** cho train/val theo epoch. Toàn bộ notebook được Kaggle xác nhận **Successfully ran in 2286.9s ≈ 38 phút 07 giây**, gồm chuẩn bị, validation cuối, evaluator và lưu output. Đây là tổng thời gian cuối; mốc 36 phút 25 giây trong thông báo trước chỉ là thời điểm log khi notebook còn xử lý tiếp.

## Đánh giá chung để so model

| Model | mAP50 (%) | mAP50–95 (%) |
|---|---:|---:|
| Pretrained COCO80 | 28.176 | 18.895 |
| K2 best — 06/10/2026 | 53.848 | 33.847 |
| K3 best — 08/10/2026 | 56.727 | 35.896 |


Protocol giống K2: evaluator COCO bbox trên val; `imgsz=640`, `conf=0.001`, `iou=0.7`, `nms=True`, `max_det=300`, `agnostic_nms=False`, `augment=False`; `actual_end2end=False` được ghi trong JSON. COCO AP dùng maxDets=100. Đây là số so sánh giữa checkpoint; không trộn với trainer validation NMS-free ở 1280. Evaluator JSON không có P/R, nên không gán P/R của trainer vào bảng này. Xem [finetuned_val_metrics.json](../../../../../reports/results/kaggle/project8_26n_kaggle_20261008_040051/finetuned_val_metrics.json) và [baseline_val_metrics.json](../../../../../reports/results/kaggle/project8_26n_kaggle_20261008_040051/baseline_val_metrics.json).

| Lớp | GT box | K2 AP50 (%) | K3 AP50 (%) | Δ AP50 (điểm %) | K2 AP50–95 (%) | K3 AP50–95 (%) | Δ AP50–95 (điểm %) |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 1242 | 82.944 | 84.143 | +1.199 | 53.435 | 54.742 | +1.307 |
| table | 222 | 73.613 | 74.433 | +0.820 | 49.187 | 50.569 | +1.382 |
| chair | 290 | 52.443 | 54.979 | +2.536 | 37.130 | 38.747 | +1.617 |
| laptop | 279 | 83.469 | 84.314 | +0.845 | 56.597 | 58.772 | +2.175 |
| cell phone | 579 | 32.179 | 35.547 | +3.368 | 14.823 | 16.412 | +1.589 |
| backpack | 110 | 19.577 | 25.568 | +5.992 | 8.653 | 12.002 | +3.349 |
| book | 2458 | 55.362 | 61.463 | +6.101 | 30.114 | 33.705 | +3.591 |
| cup | 202 | 31.196 | 33.370 | +2.174 | 20.840 | 22.221 | +1.381 |


Bảng trên giữ toàn bộ tám lớp, kể cả lớp giảm hoặc tăng ít. Các điểm AP vẫn là validation, chưa chứng minh chất lượng trên test độc lập hay camera thực tế.

## Trainer validation ở 1280, NMS-free

| Mốc | P (%) | R (%) | mAP50 (%) | mAP50–95 (%) |
|---|---:|---:|---:|---:|
| K2 best — dòng CSV epoch 14 | 58.718 | 54.028 | 56.546 | 35.617 |
| K3 best — dòng CSV epoch 19 | 66.132 | 56.650 | 61.644 | 39.033 |
| K3 last — dòng CSV epoch 20 | 64.944 | 57.748 | 61.436 | 38.933 |
| K3 best — validation cuối từ log (làm tròn) | 66.6 | 56.6 | 61.6 | 39.1 |

`best.pt.train_metrics` khớp dòng epoch 19: fitness **0.39033**, cao hơn epoch 20 **0.38933**. `train_results` trong best chứa đủ 20 dòng sau strip; không dùng số dòng này để suy ra epoch chọn best. Log validation cuối của best sau nạp/fuse có sai khác nhỏ với CSV; giữ riêng số gốc, không thay CSV bằng log đã làm tròn.

## Kiểm ZIP, trọng số và vị trí file

| Artifact | Vị trí / giá trị |
|---|---|
| Result ZIP | [runs/kaggle/project8_26n_kaggle_20261008_040051/project8_26n_kaggle_20261008_040051_result.zip](../../../../../runs/kaggle/project8_26n_kaggle_20261008_040051/project8_26n_kaggle_20261008_040051_result.zip) |
| ZIP bytes / SHA-256 | 4,878,270 / `73f824e711e23469a0828860d71de5f19926a8d1215908fee29de0ddbc3a5966` |
| Best bytes / SHA-256 | 5,547,077 / `d4244d26e69d78a066944b22ba64f80ea68602f2f994d6cf6bbae7970ff123f9` |
| Last local | [runs/kaggle/project8_26n_kaggle_20261008_040051/last.pt](../../../../../runs/kaggle/project8_26n_kaggle_20261008_040051/last.pt) |
| Last SHA-256 | `26f3c459c692b0cbca38771a63b722dfb996dfc03fc92ee91a4b294e2215eb15` |
| Log gốc | [runs/kaggle/project8_26n_kaggle_20261008_040051/kaggle.log](../../../../../runs/kaggle/project8_26n_kaggle_20261008_040051/kaggle.log) |
| Audit local | [checkpoint_validation.json](../../../../../reports/results/kaggle/project8_26n_kaggle_20261008_040051/checkpoint_validation.json) |

ZIP gồm **7 file**, CRC đạt; **6 payload** trong FILES.json đạt cả byte count và SHA-256. Hash best trùng manifest, result.json, metric JSON và log Kaggle. Hash checkpoint nguồn trùng bản K2 local trước khi cập nhật alias. Metadata được chép bằng byte gốc vào `runs/kaggle/project8_26n_kaggle_20261008_040051/` và `reports/results/kaggle/project8_26n_kaggle_20261008_040051/`.

Đã nạp best và last bằng Ultralytics local 8.4.165: task detect, thứ tự lớp `person/table/chair/laptop/cell phone/backpack/book/cup`, stride 8/16/32, **2.506.920 tham số**, tensor hữu hạn; best là end-to-end. `train_results` khớp toàn bộ 20 dòng CSV. Đã chạy smoke prediction một ảnh đen 640×640 trên CPU, output hữu hạn; đây là kiểm khả năng chạy, không phải phép đo chất lượng hay FPS camera.

**Best và last khác nhau:** cùng kích thước file nhưng khác hash và **523/708 tensor state_dict** khác byte/giá trị. Dùng best epoch 19 để inference/fine-tune tiếp. Cả hai đã strip: `epoch=-1`, `optimizer=None`; dùng khởi tạo run mới, không khẳng định còn đủ trạng thái resume optimizer.

Streamlit Recognition và Training liệt kê đệ quy `weights/**/*.pt`; hai tên best đã nằm trong đúng nguồn lựa chọn. Kiểm bằng mã discovery và model local; không khẳng định đã kiểm webcam thật. Tải lại/rerun trang app nếu danh sách đang giữ trạng thái cũ.

Đã tải result ZIP, best, last và log. Checkpoint `epoch0.pt`–`epoch19.pt` hiện có trong output Kaggle nhưng **chưa tải local**; chưa tải full output, ảnh train/val và plots. Không ghi các artifact đó là đã đối soát local.

## Đủ 20 epoch mới

Số liệu giữ từ [results.csv gốc](../../../../../reports/results/kaggle/project8_26n_kaggle_20261008_040051/results.csv). Giây là chênh thời gian tích lũy, gồm train/val/overhead ghi bởi trainer. STT 31–50 nối với lịch sử 30 epoch trước.

| STT | Run | Epoch | imgsz | Giây | Train box | Train cls | Train L1 | Val box | Val cls | Val L1 | P (%) | R (%) | mAP50 (%) | mAP50–95 (%) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | K3 | 1 | 1280 | 152.126 | 1.27642 | 1.36048 | 0.01061 | 1.33951 | 1.61582 | 0.01614 | 59.817 | 53.985 | 56.054 | 35.331 |
| 32 | K3 | 2 | 1280 | 103.828 | 1.24818 | 1.33252 | 0.01053 | 1.35153 | 1.61828 | 0.01645 | 60.924 | 52.690 | 56.318 | 35.438 |
| 33 | K3 | 3 | 1280 | 100.305 | 1.24778 | 1.31891 | 0.01077 | 1.38855 | 1.61617 | 0.01665 | 58.049 | 54.255 | 55.869 | 34.821 |
| 34 | K3 | 4 | 1280 | 102.640 | 1.25628 | 1.29466 | 0.00986 | 1.37982 | 1.59988 | 0.0166 | 60.132 | 52.414 | 56.281 | 34.788 |
| 35 | K3 | 5 | 1280 | 101.722 | 1.23957 | 1.28821 | 0.01049 | 1.36047 | 1.57899 | 0.0163 | 61.346 | 53.631 | 57.550 | 35.805 |
| 36 | K3 | 6 | 1280 | 103.317 | 1.24819 | 1.27695 | 0.01023 | 1.36732 | 1.5845 | 0.01681 | 59.799 | 55.312 | 56.544 | 35.106 |
| 37 | K3 | 7 | 1280 | 104.880 | 1.23722 | 1.21336 | 0.00994 | 1.36197 | 1.55578 | 0.01648 | 61.829 | 52.930 | 57.349 | 35.775 |
| 38 | K3 | 8 | 1280 | 105.274 | 1.25374 | 1.24042 | 0.00987 | 1.36476 | 1.54479 | 0.01661 | 62.442 | 53.497 | 57.631 | 36.324 |
| 39 | K3 | 9 | 1280 | 104.032 | 1.22752 | 1.22797 | 0.00972 | 1.36046 | 1.55415 | 0.0162 | 60.960 | 56.135 | 58.121 | 36.307 |
| 40 | K3 | 10 | 1280 | 104.176 | 1.23604 | 1.252 | 0.01044 | 1.37052 | 1.52626 | 0.01631 | 66.024 | 53.337 | 58.168 | 36.419 |
| 41 | K3 | 11 | 1280 | 104.440 | 1.24827 | 1.2283 | 0.00983 | 1.3436 | 1.53412 | 0.01605 | 66.916 | 53.993 | 59.461 | 37.381 |
| 42 | K3 | 12 | 1280 | 104.390 | 1.23522 | 1.19854 | 0.00983 | 1.34201 | 1.52571 | 0.016 | 65.850 | 54.191 | 59.254 | 37.552 |
| 43 | K3 | 13 | 1280 | 103.630 | 1.22264 | 1.19579 | 0.00982 | 1.35061 | 1.48876 | 0.01624 | 62.265 | 56.967 | 59.534 | 37.665 |
| 44 | K3 | 14 | 1280 | 103.080 | 1.22474 | 1.19295 | 0.0104 | 1.34199 | 1.47681 | 0.01606 | 64.656 | 56.095 | 60.567 | 38.210 |
| 45 | K3 | 15 | 1280 | 102.160 | 1.21418 | 1.16162 | 0.00944 | 1.34442 | 1.4933 | 0.01575 | 62.419 | 56.309 | 59.789 | 38.229 |
| 46 | K3 | 16 | 1280 | 103.250 | 1.18501 | 1.04566 | 0.01098 | 1.33098 | 1.47563 | 0.01542 | 64.932 | 56.278 | 60.471 | 38.198 |
| 47 | K3 | 17 | 1280 | 100.140 | 1.17283 | 1.02174 | 0.01105 | 1.34067 | 1.46099 | 0.01554 | 66.241 | 56.217 | 60.748 | 38.235 |
| 48 | K3 | 18 | 1280 | 100.300 | 1.20229 | 1.03425 | 0.01078 | 1.34229 | 1.49086 | 0.01558 | 68.486 | 54.089 | 60.890 | 38.210 |
| 49 | K3 | 19 | 1280 | 100.400 | 1.19006 | 1.06203 | 0.0106 | 1.32706 | 1.44308 | 0.01539 | 66.132 | 56.650 | 61.644 | 39.033 |
| 50 | K3 | 20 | 1280 | 99.960 | 1.18695 | 1.03122 | 0.01057 | 1.31705 | 1.45118 | 0.01536 | 64.944 | 57.748 | 61.436 | 38.933 |


Xem [bảng toàn bộ 50 epoch trong 6 run](../../training_50_epochs_20261008/REPORT.md). Chưa dùng test để chọn checkpoint; chưa nghiệm thu FPS, webcam thật hoặc G3.
