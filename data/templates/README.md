# Mẫu bàn giao dữ liệu — không phải dataset thật

Các CSV chỉ có header, chưa có bản ghi hay kết quả review. Sao mẫu ra `data/sources.csv`, `data/manifest.csv`, `data/review.csv`, `data/videos.csv` khi bắt đầu bàn giao. Giữ file mẫu nguyên để thành viên khác dùng.

- `sources.csv`: một dòng mỗi nguồn/phiên bản. `class_mapping_file` trỏ tới bảng ID nguồn → ID nhóm đã được kiểm. `pretraining_overlap` dùng `yes`, `no_verified` hoặc `unknown`, không tự mặc định `no`.
- `manifest.csv`: một dòng mỗi ảnh; đường dẫn tương đối repo, dùng `/`. `split` là `train`/`val`/`test` khi đã chốt; trước đó để trống. `session_id` không rõ dùng `unknown`; `split_group_id` là nhóm thực dùng để chia, gồm ràng buộc phiên và gần trùng.
- `annotation_status`: `unlabeled`, `converted`, `needs_fix`, `approved`. `review_status`: `pending`, `needs_fix`, `approved`. Chỉ điền số box sau khi kiểm; không điền 0 để thay số chưa biết. Đếm ảnh âm tính chỉ khi có nhãn hợp lệ rỗng và đã xác nhận.
- `review.csv`: log người thứ hai; trường `status` như trên. Người gán không tự điền tên mình vào vai reviewer thứ hai.
- `videos.csv`: `split` chỉ `dev` hoặc `test`; `events_file` trỏ tới file sự kiện có quy ước cụ thể của chức năng đếm/vùng. Khung này chưa định nghĩa evaluator tracking.

`image_sha256` và `label_sha256` ghi checksum thật sau chuẩn hóa. Header này là hợp đồng bàn giao dự kiến; chưa có script tự đọc/kiểm toàn bộ schema. Tài liệu quy trình: [WEEK2_G2.md](../../WEEK2_G2.md).
