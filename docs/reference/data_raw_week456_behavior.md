# Dữ liệu giữ riêng cho tuần 4–6

Giữ nguyên hai ZIP và toàn bộ annotation nguồn. Không lấy các ID nhãn từ những nguồn này làm ID tám lớp dự án.

- student_behaviour_detection_v6: 4.065 file ảnh; nhãn đồ vật book/phone và hành vi giơ tay, đọc, viết... Có augmentation; không coi số file là số ảnh gốc. Nhãn hành vi có thể phục vụ kiểm nhận biết tư thế sau review, nhưng không có keypoint hoặc chuỗi sự kiện video được xác nhận.
- student_classroom_activity_v6: 779 file ảnh; phone/sleep/study là nhãn hành vi. Mẫu phone bao người dùng điện thoại, không phải hộp chiếc điện thoại.

Ảnh dùng cho SSL phải ở phần phát triển, tách khỏi validation/test và được kiểm trùng; nhãn hành vi không tự trở thành nhãn tám lớp.

600 ảnh book/cell phone đã chọn nằm trong dataset chính data/dataset/project8_v0.3; bản sao ứng viên cũ đã dọn. Nhãn còn một phần, cần bổ sung các lớp mục tiêu khác và review trước train. Hai ZIP nguồn ở đây vẫn giữ nguyên.
