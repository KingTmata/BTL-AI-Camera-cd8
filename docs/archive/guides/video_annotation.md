# Bàn giao video phát triển và test

Mỗi clip thuộc một phiên nguồn; dev và test không cùng phiên và không trùng phiên ảnh test/train trái quy tắc độc lập đã chốt. Kê vào `data/videos.csv` theo `data/templates/videos.csv`, ghi SHA-256 thật, duration/fps, quyền sử dụng, người gán/review và đường dẫn sự kiện.

File sự kiện dùng header tại `data/templates/events.csv`. `event_type` nhận `line_crossing` hoặc `zone_alert`; đối tượng nghiệp vụ hiện là person. `object_local_id` chỉ là mã tạm để người gán phân biệt người trong clip, không phải danh tính thật hay GT tracking theo frame.

- Với vượt vạch: điền `timestamp_s` theo timeline nguồn tại lúc vượt, `direction` là A_to_B hoặc B_to_A và `line_or_zone_id` theo file cấu hình. Ghi cách đặt A/B, anchor và vùng đệm trong notes/config; không lấy thời gian xử lý của máy.
- Với vùng: điền khoảng `start_s`, `end_s` nhìn thấy vật trong vùng và `timestamp_s` lúc đạt thời gian quan sát yêu cầu. Quãng mất dấu/che khuất không được cộng thành thời gian quan sát; tách các khoảng và ghi lý do.
- `config_path` trỏ tới cấu hình vạch/vùng thực tế của clip. Reviewer khác annotator xem lại trước `status=approved`. Clip không có sự kiện vẫn cần xác nhận âm tính ở manifest video và file sự kiện header rỗng, không bỏ khỏi đánh giá.

Đây là quy ước bàn giao nhãn sự kiện. Bộ kiểm/matching tracking sẽ triển khai theo chức năng tuần 4–5; không dùng nhãn này để công bố HOTA/IDF1/MOTA. Chưa có clip thật hoặc sự kiện đã duyệt trong repo.
