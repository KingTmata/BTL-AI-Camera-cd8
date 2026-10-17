# Kết quả hoàn tất E1–E6 từ checkpoint 07

Ngày 12/10/2026. Phiên Kaggle Version 2 hoàn tất thành công sau 18.687 giây (~5 giờ 11 phút), Tesla T4 x2, tổng batch 32. Mỗi run đủ 10 epoch, patience=3. Sáu ZIP tải về Downloads, checksum SHA256 khớp log Kaggle và ZIP không hỏng. best.pt/last.pt, results.csv, args.yaml đã trích vào reports/classroom07_matrix_20261012/artifacts; xem download_manifest.json.

Checkpoint gốc: YOLO26n Project8 v0.4, phiên Kaggle 08/10/2026, best epoch19; SHA256 d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb. Mọi run cùng bắt đầu từ checkpoint này.

## Bảng đánh giá chung

Validation 2500 ảnh/8359 boxes; imgsz1280, batch8, device0, conf0.001, iou0.7, nms=False, max_det300. Đơn vị %. Số liệu log làm tròn.

| Model | Cấu hình | Best epoch | Precision | Recall | mAP50 | mAP50–95 | F1 xấp xỉ |
|---|---|---:|---:|---:|---:|---:|---:|
| Gốc07 v0.4 | initializer | 19 | 73.5 | 66.3 | 71.8 | 50.8 | 69.71 |
| E1 | MuSGD lr0.0001 | 8 | 80.4 | 68.1 | 76.1 | 54.6 | 73.74 |
| E2 | MuSGD lr0.0003 | 10 | 82.0 | 67.6 | 76.6 | 55.2 | 74.11 |
| E3 | AdamW lr0.0001 | 10 | 79.8 | 69.7
| 77.2 | 55.7 | 74.41 |
| E4 | AdamW lr0.0003 | 10 | 79.7 | 69.4 | 76.7 | 54.8 | 74.19 |
| E5 | MuSGD lr0.0001 scale0.2 | 8 | 80.6 | 67.7 | 76.0 | 54.9 | 73.59 |
| E6 | MuSGD lr0.0001 freeze backbone0–10 | 10 | 79.1 | 66.8 | 74.7 | 53.7 | 72.43 |

E3 tốt nhất theo native mAP50–95; tăng4.9 điểm phần trăm so checkpoint gốc. E2 Precision cao nhất. F1=2*meanP*meanR/(meanP+meanR), xấp xỉ từ log; không macroF1.

## Trainer final — chỉ báo riêng

| Run | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| E1 | 82.0 | 71.1 | 78.5 | 56.2 |
| E2 | 81.9 | 71.6 | 79.1 | 56.8 |
| E3 | 83.6 | 71.0 | 79.5 | 57.2 |
| E4 | 83.5 | 71.2 | 79.2 | 56.7 |
| E5 | 82.4 | 71.1 | 78.5 | 56.1 |
| E6 | 80.8 | 69.3 | 77.1 | 55.0 |

Không trộn trainer final với native validation. Chưa xác định đầy đủ nguyên nhân chênh lệch protocol.

## Phạm vi hoàn thành và việc còn lại

Đã hoàn thành training sáu cấu hình, tải ZIP, kiểm checksum/tính toàn vẹn, bảng native/trainer. Nhãn v0.5 chưa review; chỉ thử phương pháp, một seed. Chưa COCO640/application, đối chứng YOLO26n pretrained gốc, test độc lập. Chưa kết luận cải thiện lớp học, chưa promote/thay model ứng dụng. Plan và todo vẫn ở tasks vì phạm vi đánh giá rộng hơn chưa xong. Lịch sử progress chuyển vào thư mục này; báo cáo hoàn tất chỉ bao gồm phần đã xác nhận.
