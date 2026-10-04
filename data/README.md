# Dữ liệu hiện hành

Cập nhật 04/10/2026: chỉ giữ **5.002 ảnh / 5.002 TXT / 26.566 box** ở `dataset/project8_v0.3/`. Train/val/test: **3.883 / 798 / 321**. [Hồ sơ đầy đủ](../docs/PROJECT8_V0_3_DATA_PROFILE.md).

- `dataset/project8_v0.3/images/{train,val,test}`: một bản chuẩn của mỗi ảnh.
- `dataset/project8_v0.3/labels/{train,val,test}`: nhãn project8 cùng stem.
- `manifest.csv`: manifest chính cho CLI/UI mặc định; `statistics.json`, review, validation và checksum nằm trong phiên bản chính.
- `raw/week456_behavior/`: giữ hai ZIP hành vi gốc cho tuần 4–6, tổng 4.844 lượt ảnh. 600 ảnh chọn từ nguồn đó đã có trong dataset chính; không cộng lại.
- `video_dev/`, `video_test/`: chờ video thật từ các phiên độc lập.
- `templates/`: mẫu metadata, review và video/sự kiện.

Các bản raw/intake/dataset cũ bị lặp, ảnh tham khảo/kết quả thử, ZIP classroom/Laptop/COCO128 và 107 ảnh pending đã xóa theo yêu cầu. Thư mục lịch sử còn metadata để truy nguồn; không coi là dataset đầy đủ có thể train.

Mapping classroom: Student→person, chair→chair, table và with-student→table theo quyết định người dùng. Nhãn còn cần bổ sung đủ vật và review; chưa khóa release hoặc train. Giao diện dùng ảnh train trong kho chính, không cần thư viện ảnh mẫu riêng.
