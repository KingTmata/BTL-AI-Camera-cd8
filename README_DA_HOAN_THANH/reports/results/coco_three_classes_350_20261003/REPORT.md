> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

> **Báo cáo lịch sử ngày 03/10/2026.** Dữ liệu đã được gộp và dọn sau thời điểm này; ảnh nguồn/bản sao/ảnh pending được nhắc bên dưới có thể đã xóa. Xem [báo cáo hiện hành 04/10/2026](../project_progress_20261004/REPORT.md) và [audit dọn](../primary_cleanup_20261004/REPORT.md).

> Số liệu lịch sử trước lọc sách/điện thoại. Tập ứng viên hiện là 600 ảnh; tổng detection hiện là 2.966 file. Xem reports/results/book_phone_balance_20261003/REPORT.md.

# Bổ sung COCO cho ba lô, cốc và ghế — 03/10/2026

## Kết quả

Đã tải 566 ảnh COCO2017 khác nhau, đủ 350 ảnh có nhãn cho mỗi lớp backpack, cup và chair. Một ảnh có thể chứa nhiều lớp nên không cộng 350 × 3 thành số ảnh riêng. Tải thành công 566/566, không có lỗi.

| Lớp ưu tiên | Train | Validation | Tổng ảnh có nhãn |
|---|---:|---:|---:|
| backpack | 280 | 70 | 350 |
| cup | 280 | 70 | 350 |
| chair | 280 | 70 | 350 |

433 ảnh train2017 và 133 ảnh val2017; giữ split nguồn. Không chọn dining table làm tiêu chí, nhưng giữ toàn bộ nhãn của tám lớp mục tiêu xuất hiện trong ảnh.

| Lớp | Ảnh có nhãn trong tập mới |
|---|---:|
| person | 441 |
| table | 236 |
| chair | 350 |
| laptop | 69 |
| cell phone | 72 |
| backpack | 350 |
| book | 84 |
| cup | 350 |

## Dung lượng và kiểm tra

Ảnh: 93.24 MB; TXT: 0.29 MB; tổng ảnh + TXT: 93.53 MB. MB thập phân.

- Kiểm 566 ảnh và nhãn bằng validator dự án: đạt, không lỗi.
- Loại ID ảnh COCO128 và COCO đã có trước khi chọn. Kiểm SHA-256 không có ảnh trùng bytes giữa cả ba manifest detection.
- Chưa kiểm gần trùng hoặc độc lập phiên/cảnh, chưa review người thứ hai; đây là draft, không tự ghi approved.
- Tên/thứ tự lớp dùng chung src/classes.py; nhãn xuất YOLO detection TXT.
- Lấy nhãn và metadata từ annotation COCO2017 gốc có sẵn; tải ảnh qua bucket HTTPS chính thức. License/URL nguồn ghi theo từng ảnh trong manifest.
- Model pretrained đã học COCO; không dùng tập này để khẳng định chất lượng trên phòng học độc lập.
- Một số phạm vi nhãn nguồn khác phạm vi nhóm: dining table chỉ bàn ăn, book không khẳng định bao phủ vở.

## Tổng dữ liệu detection hiện có

Có 4,962 file ảnh: 1.800 ảnh chính + 2.596 ứng viên nhãn một phần + 566 COCO mới. Train/val/test tổng: 3,681/862/419. Đây chưa phải số ảnh gốc hợp lệ đạt chỉ tiêu; tập ứng viên có augmentation và nhãn thiếu.

| Lớp | Ảnh có nhãn, tất cả split |
|---|---:|
| person | 866 |
| table | 322 |
| chair | 425 |
| laptop | 887 |
| cell phone | 1865 |
| backpack | 378 |
| book | 2979 |
| cup | 408 |

## File sử dụng

manifest.csv: nguồn, hash, split và trạng thái review.
selection.json: seed 42, ID nguồn, bounding box và metadata để tái lập.
statistics.json và validation.json: số liệu/kiểm kỹ thuật.
data.yaml: cấu hình tám lớp của riêng draft này; chưa gộp vào project8_v0.2.
