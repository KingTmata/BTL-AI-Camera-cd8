# Hồ sơ dữ liệu thành viên nộp — Laptop Roboflow v1

Ngày tiếp nhận: 02/10/2026. **Dữ liệu chính do thành viên đóng góp**, ngang hàng COCO500; đã gộp vào bản draft project8_v0.2. Bản intake lưu riêng để đối chiếu nguồn. Các thành viên tiếp tục bổ sung nhãn theo phân công.

## Nguồn và đặc tính cơ bản

- ZIP: `Laptop.v1-test-dataset.yolov8.zip`; SHA256: `26db26b4bada9453f89677448775edd1cf1949a1a1aebc61f7b4f4760b762c39`.
- Nguồn theo README/YAML: [Roboflow Laptop v1](https://universe.roboflow.com/new-workspace-xp2sh/laptop-tgbyh/dataset/1). Thành viên cung cấp dữ liệu trên mạng; chưa có bằng chứng đây là ảnh tự chụp của nhóm.
- Giấy phép được gói khai báo: CC BY 4.0. Chưa xác minh riêng nguồn/quyền từng ảnh; cần giữ ghi công và thông tin nguồn khi chia sẻ.
- Export nguồn ghi 18/03/2023; đây là ngày export, không phải ngày chụp.
- Tiền xử lý theo README: auto-orientation, bỏ EXIF orientation; resize 416×416 bằng stretch, không augmentation. Stretch có thể làm biến dạng tỷ lệ vật.
- Thực tế: 1300 ảnh; kích thước width {'min': 416.0, 'median': 416.0, 'max': 416.0}, height {'min': 416.0, 'median': 416.0, 'max': 416.0}; chế độ màu {'RGB': 1300}.
- Bài toán detection, TXT mỗi ảnh; một dòng là class cx cy w h chuẩn hóa. Không có segmentation, sự kiện video hay session/camera trong gói.

## Mapping nhãn

| Nhãn nguồn / ID | Nhãn nhóm / ID |
|---|---|
| Book / 0 | book / 6 |
| Laptop / 1 | laptop / 3 |

Nguồn không tách vở khỏi sách. Sáu lớp khác chưa được gán nhãn theo xác nhận của người nhận. Không xem chúng là vắng mặt chỉ vì TXT không có lớp đó.

## Thống kê thực tế

| Split nguồn | Ảnh | Box book | Ảnh có book | Box laptop | Ảnh có laptop | TXT rỗng |
|---|---:|---:|---:|---:|---:|---:|
| test | 130 | 215 | 45 | 142 | 85 | 0 |
| train | 910 | 1795 | 353 | 987 | 557 | 0 |
| valid | 260 | 440 | 102 | 256 | 158 | 0 |

Tổng 3835 box đã phân tích hợp lệ. Số ảnh chứa các lớp có thể chồng nhau.

## Kiểm tra, lỗi và outlier

- 0 ảnh/cặp nhãn có lỗi kỹ thuật; 0 TXT mồ côi.
- 0 ảnh trùng bytes hoặc pixel giải mã với ảnh trước đó trong nguồn hoặc COCO500. Giữ nguyên trong raw, loại khỏi manifest ứng viên duy nhất.
- 1300 ảnh duy nhất đạt kiểm kỹ thuật, vẫn chờ hoàn thiện nhãn và review.
- Kiểm: giải mã ảnh, EXIF, cặp tên, class nguồn 0/1, năm cột, finite, box dương và không vượt biên, box trùng, checksum; đối chiếu trùng với COCO500.
- Không chạy tìm gần trùng bằng embeddings. Chưa chứng minh split nguồn độc lập theo phiên. Bản intake giữ unassigned để truy bước nhập. Trong manifest dataset chính project8_v0.2, đã giữ split nguồn 910 train / 260 val / 130 test, nhóm theo tên gốc trước hậu tố Roboflow; session vẫn unknown, chưa khẳng định độc lập cảnh/phiên.

| Cờ cần xem | Số ảnh |
|---|---:|
| tiny_box_at_640 | 13 |
| dark | 22 |
| bright | 4 |
| low_contrast | 2 |

Ngưỡng heuristic: brightness grayscale<40 hoặc>215, std<20; cạnh ngắn box khi resize giữ tỷ lệ về 640<8px. Cờ không tự kết luận nhãn sai hoặc tự loại ảnh; review.csv ghi ca cần xem.

Box area/ảnh (min/median/max): {"book": {"min": 0.00015168500369822484, "median": 0.019180207562869825, "max": 0.9975975984652365}, "laptop": {"min": 0.00015024038461538462, "median": 0.08595483542899407, "max": 0.9975975984652365}}

Cạnh ngắn box tại 640 (min/median/max px): {"book": {"min": 1.5384615384615385, "median": 63.07692307692308, "max": 639.2307692307693}, "laptop": {"min": 6.153846153846154, "median": 158.46153846153845, "max": 639.2307692307693}}

### Quan sát mẫu trực quan

Đã xem contact sheet 12 ảnh trải trên các split: có người, bàn, ghế xuất hiện nhưng chỉ có box book/laptop; một số box book bao cụm sách/kệ sách cần người phụ trách rà lại phạm vi nhãn. Đây là quan sát mẫu, không phải review ngữ nghĩa 100% hoặc xác nhận người thứ hai.

## Đã thêm vào dự án như thế nào

- `data/raw/laptop_roboflow_v1/`: ZIP nguyên bản và toàn bộ nội dung gốc.
- `data/dataset/laptop_roboflow_v1_intake/`: ảnh sao chép, nhãn đổi ID, inventory_all.csv, manifest.csv, combined_inventory.csv, mapping.json, audit.json, validation.json, statistics.json, review.csv, contact_sheet.jpg, checksums.json.
- Combined inventory: 500 ảnh COCO + 1300 ảnh nguồn này = **1800 ảnh ứng viên duy nhất** đạt kiểm bytes/pixel và định dạng. Đây không phải số ảnh đã hoàn thiện nhãn/review.
- COCO500 giữ nguyên để truy thí nghiệm. Dataset chính gộp tại `data/dataset/project8_v0.2/`, có YAML tám lớp, manifest, split và checksum riêng; xem [hồ sơ bản gộp](PROJECT8_V0_2_DATA_PROFILE.md). Chưa khóa release hoặc hoàn tất review.

## Việc còn cần làm

1. Rà từng ảnh, bổ sung đầy đủ person, table, chair, cell phone, backpack, cup và kiểm lại book/laptop.
2. Ghi nguồn gốc ảnh/nhóm/phiên nếu truy được; đối chiếu trùng/gần trùng và split nguồn trước khi chia lại. Giữ test nguồn trong kho, chưa gọi là test độc lập tám lớp.
3. Người thứ hai review theo quy trình; ghi reviewer thật, không tự đánh dấu approved.
4. Sau khi hoàn thiện, tạo bản dataset mới, khóa checksum và chạy baseline theo protocol chung.
