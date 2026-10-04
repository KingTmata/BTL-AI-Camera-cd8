> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

# Báo cáo tiến độ dự án tuần 1–2–3

Cập nhật: 2026-10-04T00:42:21+07:00 (Asia/Saigon).

> Snapshot trước xác nhận review/khóa của người dùng. Trạng thái mới nhất xem [readiness tuần 3](../week3_readiness_20261004/REPORT.md).

## Kiểm kê tại thời điểm báo cáo

| Phạm vi | File | Ảnh | TXT nhãn | Box |
|---|---|---|---|---|
| Toàn checkout (loại môi trường/cache/Git) | 10.230 | 5.002 | 5.002 | — |
| Dataset chính | 10.017 | 5.002 | 5.002 | 26.566 |
| train | — | 3.883 | 3.883 | 19.718 |
| val | — | 798 | 798 | 5.382 |
| test | — | 321 | 321 | 1.466 |
| Pending | — | 0 | 0 | 0 |

File trong checkout bao gồm code, tài liệu, metadata và báo cáo. Không cộng ảnh trong ZIP hoặc cộng số ảnh từng lớp thành ảnh độc lập.

## Nguồn ảnh chính

| Nguồn | Ảnh |
|---|---|
| project8_v0.2 | 1.800 |
| coco_three_classes_350_v0.1 | 566 |
| student_behaviour_book_phone_600_v0.2 | 600 |
| classroom_roboflow_v1 | 2.036 |

Tên nguồn v0.2/COCO/Laptop là provenance lịch sử, không yêu cầu giữ thêm bản sao ảnh. Nhãn nguồn chỉ là draft, chưa được gán đủ mọi vật hoặc review người thứ hai.

## Nhãn từng lớp

| Lớp | Ảnh có nhãn | Box |
|---|---|---|
| person | 1.346 | 8.169 |
| table | 1.017 | 2.479 |
| chair | 1.286 | 2.173 |
| laptop | 887 | 1.525 |
| cell phone | 711 | 2.012 |
| backpack | 378 | 525 |
| book | 1.214 | 8.651 |
| cup | 408 | 1.032 |

## Gói nguồn còn giữ riêng

| Archive | Lượt ảnh |
|---|---|
| data/raw/week456_behavior/student_behaviour_detection_v6/Student Behaviour Detection.v6i.yolo26.zip | 4.065 |
| data/raw/week456_behavior/student_classroom_activity_v6/student-classroom-activity.v6i.yolo26.zip | 779 |

Hai gói hành vi gốc phục vụ tuần 4–6 được giữ riêng. 600 ảnh được chọn từ nguồn này đã có trong dataset chính; không cộng lại vào số ảnh chính.

## Tiến độ theo tuần

| Tuần | Đã có | Còn thiếu |
|---|---|---|
| 1 | App ảnh/video/webcam và pretrained; giao diện dùng ảnh train chính | Webcam thiết bị thật và review người; ảnh COCO128/review cũ đã dọn theo yêu cầu |
| 2 | Dataset draft 5002 ảnh, manifest/YAML/checksum/validator/evaluator | Nhãn đủ tám lớp, review, quyền/gần trùng/phiên, khóa release, video thật và baseline validation |
| 3 | Cấu hình và evaluator; 0 artifact training trong runs | Chưa fine-tune 26n/26s hoặc có bảng validation chung |

## Kiểm tra và dọn dữ liệu

Mapping classroom: Student→person, chair→chair, table và with-student→table theo quyết định người dùng. Giữ nguyên tọa độ nguồn; không tạo ground truth giả.
107 ảnh pending cùng TXT đã xóa theo yêu cầu; không đổi media/nhãn/split của 5.002 cặp giữ lại. Metadata nguồn và danh sách xóa được lưu trong audit; dữ liệu chưa được duyệt hoặc train.
Validator đã kiểm 5,002 ảnh, valid=True; checksum hiện tại được kiểm lại khi tạo báo cáo. Kiểm gần trùng/độc lập cảnh vẫn cần xác nhận của người.
Tập test hiện chưa có nhãn backpack và cup; chưa đủ cơ sở đánh giá cả tám lớp trên test. Nhãn nguồn còn thiếu vật cũng có thể ảnh hưởng phép đánh giá.
Ảnh mẫu giao diện lấy từ train, không đưa test vào luồng xem mẫu mặc định.

| Đã xóa | Số lượng |
|---|---|
| Bản sao ảnh | 6.566 |
| Ảnh chờ xử lý | 107 |
| Ảnh tham khảo/kết quả thử | 189 |
| ZIP nguồn đã nhập/tham khảo | 3 |

Audit dọn dữ liệu và code: [primary_cleanup_20261004](../primary_cleanup_20261004/REPORT.md).
