# Project8 v0.4 — kết quả 20 epoch và đối chứng trước/sau

Cập nhật 08/10/2026, Asia/Saigon. Lượt full train dừng thủ công sau **20/50 epoch**, không ghi là completed 50 epoch. Fine-tune mới từ best K3, reset optimizer/scheduler (`resume=False`); 50 epoch lịch sử + 20 epoch full run mới = 70 epoch chính, không phải resume liên tục. Smoke được tách riêng và không có trong ZIP này.

**Checkpoint chọn dùng: best epoch 19**, SHA256 `d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb`; trùng byte `epoch18.pt` (tên file đánh số từ 0). [Trọng số](../../../../../../../weights/BEST_Project8_v04_20261008.pt), alias [BEST_Project8.pt](../../../../../../../weights/BEST_Project8.pt). Checkpoint đầu của lượt này [best v0.3](../../../../../../../weights/BEST_Project8_20261008.pt) được giữ để đối chứng/quay lui.

## Kết quả chính

| Bộ chấm cùng protocol | Model | Ảnh val | mAP50 (%) | mAP50–95 (%) |
|---|---|---:|---:|---:|
| v03 | Trọng số đầu / best v0.3 | 798 | 56.727 | 35.896 |
| v03 | Best mới v0.4 epoch 19 | 798 | 55.279 | 35.049 |
| v04 | Trọng số đầu / best v0.3 | 1754 | 47.593 | 31.047 |
| v04 | Best mới v0.4 epoch 19 | 1754 | 70.706 | 50.694 |

Chấm mới cả bốn phép đo trên CPU, Ultralytics 8.4.165, Torch 2.14.0+cpu, pycocotools 2.0.11. Cùng imgsz=640, conf=0.001, iou=0.7, NMS=True, max_det=300, augment=False; COCO maxDets=[1,10,100]. Thứ tự ảnh cố định, lưu ground truth/predictions/run/metrics. Hộp zero-area sau clip được bỏ và ghi riêng, đúng chính sách evaluator Kaggle trước đó; dự đoán NaN/Inf gây lỗi. Kết quả best cũ tái đo trên v0.3 khớp báo cáo cũ trong sai số làm tròn.

Bộ mới: tăng **23.113/19.647 điểm phần trăm**. Bộ cũ: thay đổi **-1.448/-0.847 điểm phần trăm**. Không tuyên bố best mới vượt best cũ trong mọi miền dữ liệu.

Trainer validation tại imgsz=1280/NMS-free chọn epoch 19: P=77.882%, R=67.352%, **mAP50=75.356%, mAP50–95=53.488%**. Highest mAP50 ở epoch 18: 75.415%, nhưng mAP50–95 chỉ 52.792%; vì fitness chọn theo mAP50–95 nên dùng epoch 19. Last epoch 20: 75.024% / 53.004%. Không trộn số trainer này vào bảng COCO 640/NMS=True.

## AP50 từng lớp — cùng protocol

| ID / lớp | v0.3 đầu | v0.3 mới | v0.4 đầu | v0.4 mới | GT box val v0.4 |
|---|---:|---:|---:|---:|---:|
| 0 / person | 84.143 | 82.648 | 62.518 | 77.521 | 1443 |
| 1 / table | 74.433 | 76.222 | 57.778 | 73.962 | 241 |
| 2 / chair | 54.979 | 54.216 | 43.993 | 57.806 | 339 |
| 3 / laptop | 84.314 | 81.611 | 80.204 | 80.204 | 274 |
| 4 / cell phone | 35.547 | 35.927 | 22.768 | 49.881 | 1139 |
| 5 / backpack | 25.568 | 17.529 | 22.631 | 79.916 | 408 |
| 6 / book | 61.463 | 64.841 | 59.293 | 64.738 | 2439 |
| 7 / bottle | 33.370 | 29.236 | 31.559 | 81.620 | 578 |

ID 7 trên v0.3 có GT cup; v0.4 là lớp hợp nhất cup+bottle. Chấm trên v0.3 vẫn giữ tên GT cup, ánh xạ số ID 7 của cả hai model. Không suy diễn AP lớp gộp thành AP riêng cho chai.

## Kiểm artifact và dữ liệu

- ZIP nguồn [project8_result.zip](../../../../../../../project8_result.zip), SHA256 `fad53a993a74ea339957f9035e4b23d336b5761a558e1f023f14a989484d02ec`; CRC toàn bộ 29 entry đạt, giải nén an toàn vào [run gốc](../../../../../../../runs/kaggle/project8_v04_12k_from_best50_20261008_151936).
- CSV có đủ 20 dòng; best/last khác nhau; best=epoch18.pt; tensor best/last hữu hạn và đúng 8 lớp detect. Checkpoint chưa stripped sau khi dừng, giữ optimizer/state gốc để có thể kiểm việc resume. Bản alias giữ nguyên byte.
- Dataset: 12.000 ảnh, 37.453 box; train/val/test 9.540/1.754/706; box 28.443/6.861/2.149. SHA256 manifest khớp gói chuẩn bị. Không dùng test trong bốn phép so sánh.
- Đối chiếu hash ảnh: 3.827 train cũ còn ở train mới; 779 val cũ còn ở val mới; 313 test cũ còn ở test mới. Bị loại tương ứng 56/19/8. Không có ảnh byte-identical cũ chuyển từ val/test sang train hoặc từ train sang val/test.
- args.yaml ghi device=0,1, batch=32 (tổng batch), workers=2, cache=False, imgsz=1280, optimizer=MuSGD, lr0=0.001, epochs=50, patience=5, close_mosaic=10, mosaic=0.5, cls_remap=False. Device config xác nhận yêu cầu hai GPU; không có telemetry để kết luận mức sử dụng thực tế.
- Recognition đã nhận checkpoint bottle, giữ ID 0–7; default chọn alias best mới, ảnh mẫu dùng manifest v0.4. Hợp đồng v0.3 và evaluator cũ được giữ riêng; local Training/export vẫn dành cho v0.3 và có cảnh báo phiên bản.
- Backend đã khởi động tại http://127.0.0.1:8501, health trả `ok`; giao diện mở được ảnh mẫu v0.4 và hiển thị đúng 8 lớp. Đã sửa đọc manifest UTF-8 BOM bằng `utf-8-sig` và đường dẫn ảnh thu nhỏ; kiểm thử trực tiếp hàm đọc mẫu đạt trên ảnh/nhãn thật.

## Vấn đề và bước tiếp theo

1. **Cell phone còn yếu:** AP50 mới trên v0.4 49.881%; cần rà các FN/FP, điện thoại nhỏ/che khuất, box sai và nhãn bỏ sót trước khi tăng số ảnh.
2. **Khác biệt miền dữ liệu:** backpack v0.4 đạt 79.916% nhưng v0.3 còn 17.529% (đầu 25.568%). Không gọi lớp này đã ổn trong cảnh camera cũ; cần bổ sung cảnh đúng mục tiêu và kiểm theo nguồn.
3. **Review chưa đủ:** 7.081/12.000 ảnh (59.008%) có annotation_review=requires_full_class_review. Nguồn chỉ gán một lớp có thể bỏ sót các lớp còn lại; đây là nguy cơ cần audit ảnh, không kết luận mọi ảnh đều sai.
4. **Chưa có test đủ độc lập:** near-duplicate/cùng cảnh chưa kiểm đầy đủ, không có ảnh negative trong val; test lớp 7 chỉ 11 ảnh/38 box. Không dùng test này để tối ưu hoặc tuyên bố nghiệm thu.
5. **Dừng theo giới hạn người dùng:** scheduler và warmup được cấu hình cho 50 epoch; close_mosaic=10 chỉ dự kiến bắt đầu ở epoch 41, nên lượt dừng ở 20 chưa qua pha đóng mosaic. Nếu lượt sau chọn 20 epoch ngay từ đầu thì cấu hình epochs=20, close_mosaic=5 hoặc 10 và tạo run mới; không chỉ sửa biến khi subprocess đang chạy.
6. **Chưa đủ dấu hiệu kết luận overfit:** last giảm so best sau một epoch; cần đường cong dài hơn và kiểm lỗi theo lớp. Model hiện là YOLO26n; cân nhắc YOLO26s sau khi xử lý nhãn/miền dữ liệu nếu mục tiêu độ chính xác còn chưa đạt.

## Bảng 20 epoch

| Epoch mới | STT lịch sử (không tính smoke) | P (%) | R (%) | mAP50 (%) | mAP50–95 (%) | Train box | Train cls | Train L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51 | 70.386 | 57.287 | 63.899 | 43.535 | 1.18944 | 1.90658 | 0.01147 |
| 2 | 52 | 74.475 | 59.061 | 67.420 | 45.860 | 1.19594 | 1.54038 | 0.01161 |
| 3 | 53 | 73.038 | 62.575 | 69.539 | 47.140 | 1.17546 | 1.32228 | 0.01161 |
| 4 | 54 | 72.804 | 62.433 | 69.450 | 47.337 | 1.20785 | 1.2893 | 0.0116 |
| 5 | 55 | 74.103 | 63.064 | 70.674 | 48.334 | 1.17766 | 1.2095 | 0.01162 |
| 6 | 56 | 75.973 | 63.757 | 71.511 | 48.512 | 1.15597 | 1.15701 | 0.0112 |
| 7 | 57 | 74.686 | 64.804 | 72.050 | 49.256 | 1.1538 | 1.12427 | 0.01146 |
| 8 | 58 | 76.380 | 63.901 | 72.447 | 50.139 | 1.16034 | 1.12697 | 0.01102 |
| 9 | 59 | 77.143 | 64.597 | 73.060 | 50.648 | 1.12285 | 1.07729 | 0.01098 |
| 10 | 60 | 77.191 | 64.999 | 73.703 | 51.453 | 1.14572 | 1.09432 | 0.01094 |
| 11 | 61 | 80.188 | 63.100 | 73.521 | 51.184 | 1.14065 | 1.05364 | 0.01032 |
| 12 | 62 | 77.289 | 66.151 | 73.903 | 51.322 | 1.12103 | 1.01887 | 0.01069 |
| 13 | 63 | 80.051 | 65.401 | 74.965 | 52.109 | 1.1229 | 1.02958 | 0.01053 |
| 14 | 64 | 79.433 | 67.496 | 75.109 | 52.295 | 1.11561 | 1.01721 | 0.01079 |
| 15 | 65 | 78.857 | 66.175 | 74.738 | 52.664 | 1.1012 | 0.98767 | 0.01023 |
| 16 | 66 | 78.033 | 66.697 | 75.103 | 52.386 | 1.09995 | 0.97005 | 0.01 |
| 17 | 67 | 77.826 | 67.734 | 74.860 | 52.567 | 1.10142 | 0.94417 | 0.01016 |
| 18 | 68 | 78.080 | 67.979 | 75.415 | 52.792 | 1.08331 | 0.93774 | 0.01 |
| 19 | 69 | 77.882 | 67.352 | 75.356 | 53.488 | 1.10099 | 0.95641 | 0.00968 |
| 20 | 70 | 76.166 | 68.194 | 75.024 | 53.004 | 1.09528 | 0.91354 | 0.00962 |

Thời gian CSV đến epoch 20: 79.47 phút. Đây là thời gian trainer, không cộng các smoke, stage dữ liệu, chấm so sánh CPU hoặc tải ZIP.

[CSV gốc](../../../../../../../reports/archive/results/kaggle/project8_v04_12k_from_best50_20261008_151936/results.csv) · [args.yaml](../../../../../../../reports/archive/results/kaggle/project8_v04_12k_from_best50_20261008_151936/args.yaml) · [audit](../../../../../../../reports/archive/results/kaggle/project8_v04_12k_from_best50_20261008_151936/audit.json) · [bốn phép chấm](../../../../../../../reports/archive/results/kaggle/project8_v04_12k_from_best50_20261008_151936/comparison.json). Các thư mục before/after_on_v03/v04_val cạnh audit chứa predictions/ground_truth và thông tin protocol/hash.
