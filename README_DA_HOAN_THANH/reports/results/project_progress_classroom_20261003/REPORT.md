> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

> **Báo cáo lịch sử ngày 03/10/2026.** Dữ liệu đã được gộp và dọn sau thời điểm này; ảnh nguồn/bản sao/ảnh pending được nhắc bên dưới có thể đã xóa. Xem [báo cáo hiện hành 04/10/2026](../project_progress_20261004/REPORT.md) và [audit dọn](../primary_cleanup_20261004/REPORT.md).

# Báo cáo tiến độ tuần 1–2–3 và bổ sung classroom

Cập nhật: 2026-10-03T23:52:24+07:00 (Asia/Saigon).

Đã gộp detection và classroom vào project8_v0.3; giữ nguồn cũ và hai ZIP hành vi riêng. Đây là draft chưa duyệt, không phải release sẵn sàng train.

## Bảng kiểm kê

| Phạm vi | File | Ảnh | TXT nhãn | Box | Ghi chú |
|---|---:|---:|---:|---:|---|
| Toàn checkout sau gộp (loại môi trường/cache/Git) | 23.405 | 11.864 | 11.311 | — | Gồm raw, phiên bản cũ, bản mới, ảnh tham khảo và output; không phải số ảnh duy nhất |
| Ảnh nằm trong ZIP | — | 8.415 | 8.415 | — | Số lần xuất hiện trong gói; không cộng với ảnh giải nén thành dữ liệu độc lập |
| Detection trước gộp | — | 2.966 | 2.966 | 18.867 | 1.800 chính + 566 COCO + 600 sách/điện thoại |
| Classroom nguồn bổ sung | 4.291 | 2.143 | 2.143 | 7.734 | 862 nhóm tên gốc; archive có 3 data.yaml giống nhau, 4.289 đường dẫn riêng |
| Kho data chính project8_v0.3 | 10.232 | 5.109 | 5.109 | 26.601 | Bao gồm ảnh pending; bytes/pixel duy nhất trong kho, không khẳng định độc lập cảnh |
| Train | — | 3.883 | 3.883 | 19.718 | Split draft |
| Validation | — | 798 | 798 | 5.382 | Split draft |
| Test | — | 321 | 321 | 1.466 | Giữ holdout nguồn; chưa phải test phòng học độc lập đủ tám lớp |
| Tổng danh sách train/val/test | — | 5.002 | 5.002 | 26.566 | Không có pending trong YAML; còn thiếu nhãn/review trước training chính thức |
| Pending trong kho chính | — | 107 | 107 | 35 | 72 ảnh nhãn rỗng + 35 bản ảnh xuyên split; giữ nguyên dữ liệu, chưa dùng train/val/test |
| Hai ZIP hành vi giữ riêng | — | 4.844 | 4.844 | — | Ngoài phạm vi gộp; 600 ảnh chọn từ nguồn đã có trong detection, không cộng lại |

Trước gộp: 13.164 file và 6.755 file ảnh đã giải nén; ZIP chứa 8.415 lượt ảnh. Số 2.966 chỉ là các bộ detection đang dùng, không phải toàn bộ ảnh lưu trữ. Không cộng ảnh raw, bản intake, dataset và ZIP thành số ảnh mới.

## Nhãn theo lớp

| Lớp | Ảnh có nhãn trong kho | Box trong kho | Ảnh trong split | Box trong split |
|---|---:|---:|---:|---:|
| person | 1.346 | 8.169 | 1.346 | 8.169 |
| table | 1.017 | 2.479 | 1.017 | 2.479 |
| chair | 1.321 | 2.208 | 1.286 | 2.173 |
| laptop | 887 | 1.525 | 887 | 1.525 |
| cell phone | 711 | 2.012 | 711 | 2.012 |
| backpack | 378 | 525 | 378 | 525 |
| book | 1.214 | 8.651 | 1.214 | 8.651 |
| cup | 408 | 1.032 | 408 | 1.032 |

Ảnh có thể chứa nhiều lớp; không cộng số ảnh từng lớp để suy ra số ảnh riêng. Số box là số nhãn có sẵn, không xác nhận đã gán đủ mọi vật.

## Tiến độ theo tuần

| Tuần | Bằng chứng hiện có | Còn thiếu / kết luận |
|---|---|---|
| 1 | App ảnh/video/webcam, pretrained YOLO26n; 30 test phần mềm trong artifact lịch sử; bộ review 40 ảnh | Webcam thật và review người còn pending; G1 chưa nghiệm thu |
| 2 | Kho chính 5.109 ảnh; 5.002 có split, 107 pending; manifest, YAML, checksum, validator/evaluator và smoke phần mềm | Nhãn tám lớp chưa đầy đủ, review/quyền/gần trùng/phiên chưa xác nhận, chưa khóa release, chưa có video_dev/test thật và baseline validation; G2 chưa nghiệm thu |
| 3 | Cấu hình và hướng dẫn train/evaluator có sẵn; 0 artifact best.pt/last.pt/results.csv/args.yaml trong runs | Chưa có run fine-tune 26n/26s, weights/log/so sánh validation; G3 chưa hoàn thành |

## Mapping, kiểm tra và những việc cần xử lý

- Student 0→person 0; chair 1→chair 2; table 2 và with-student 3→table 1 theo quyết định người dùng. Giữ tọa độ nhãn nguồn; không tạo dự đoán hoặc ground truth giả.
- Bàn: từ 322 ảnh / 460 box trong detection cũ lên 1.017 ảnh / 2.479 box; classroom thêm 695 ảnh / 2.019 box, thuộc 289 nhóm tên gốc. Chưa khẳng định tất cả là cảnh bàn học độc lập.
- 13 nhóm classroom xuyên split: giữ holdout nguồn theo test > val > train; 35 bản còn lại giữ pending. 72 nhãn rỗng chưa coi là ảnh âm tính đã duyệt.
- Toàn bộ 5.109 ảnh pass giải mã/cặp TXT/ID/hình học/checksum; không trùng bytes/pixel trong kho mới và không nhóm tên gốc xuyên danh sách split. Kiểm gần trùng hoặc độc lập video/phiên vẫn chờ người xác nhận. Đã chạy 33/33 kiểm thử mã thành công, gồm ba ca importer classroom.
- Tất cả nhãn/review còn pending; cần kiểm hộp with-student, thêm nhãn vật thuộc các lớp khác và review người thứ hai. Chưa khởi chạy training.

Hồ sơ chính: data/dataset/project8_v0.3/{inventory_all.csv,manifest.csv,pending_review.csv,review.csv,statistics.json,validation_all.json,validation.json,checksums.json,data.yaml}. Cấu hình local configs/data.yaml đã trỏ vào phiên bản mới.
