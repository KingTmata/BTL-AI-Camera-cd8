# Dữ liệu

**Phạm vi hiện tại 03/10/2026:** tám lớp `person, table, chair, laptop, cell phone, backpack, book, cup`. Dataset chính draft ở `dataset/project8_v0.2/`; [hồ sơ 1.800 ảnh](../docs/PROJECT8_V0_2_DATA_PROFILE.md). Thực hiện [workflow tuần 2](../docs/WEEK2_WORKFLOW.md); `DATA_CARD.md` đã có số liệu và hạn chế của draft hiện tại. Chưa khóa release. Hồ sơ COCO128 tuần 1 dưới đây là lịch sử kỹ thuật; không coi là dataset/review hiện hành.

`dataset/images/{train,val,test}` chứa ảnh; `dataset/labels/{train,val,test}` chứa nhãn YOLO cùng tên. `raw/` giữ dữ liệu gốc được phép sử dụng. `video_dev/` và `video_test/` là các phiên độc lập. `manifest.csv` cần ghi nguồn, quyền, session và split cho từng ảnh.

Hiện có **1.800 ảnh duy nhất không tính ảnh mẫu**: 500 COCO và 1.300 ảnh Roboflow Laptop do thành viên cung cấp. Split chính: 1.310 train, 360 validation, 130 test. Nguồn thành viên chỉ có nhãn book/laptop; nhãn các lớp khác được hoàn thiện theo phân công. YAML tại `dataset/project8_v0.2/data.yaml`; manifest, review và checksum cùng thư mục. Giữ COCO500 và bản intake riêng để truy nguồn. Ảnh/nhãn/raw được Git bỏ qua; hồ sơ `.md` và script được lưu trong repo.

Tuần 1 đã tải COCO128 vào `reference/coco128/` (không commit), tạo danh sách 50 ảnh ở `week1_coco128_manifest.csv` phục vụ UI. Bộ review hiện hành có 40 ảnh tám lớp theo [bàn giao tuần 1](../docs/WEEK1_HANDOFF.md); danh sách 20 ảnh bốn lớp cũ chưa có kết quả đã được bỏ. Đây là ảnh tham khảo, chưa phải ảnh tự thu trong phòng học. Gói COCO128 tải về có 128 ảnh và 128 nhãn nhưng chỉ 126 cặp trùng tên; hai nhãn `000000000656`, `000000000659` không có ảnh tương ứng và hai ảnh `000000000250`, `000000000508` không có nhãn tương ứng. Danh sách 50 ảnh chỉ lấy các cặp đầy đủ.
