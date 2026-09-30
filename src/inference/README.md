# Inference = dùng model để nhận dạng

Đầu vào: weights `.pt` và ảnh/frame. Đầu ra: hộp, lớp, confidence hoặc ảnh đã vẽ. Trọng số không thay đổi khi chạy.

- `detector.py`: bộ nhận dạng dùng trong web, mapping COCO80/project4, đọc nhãn tham khảo, vẽ/crop ảnh.
- `demo.py`: baseline CLI đọc ảnh/video/webcam, vòng lặp frame, giải phóng nguồn và ghi `summary.json`.

Chạy CLI: `.venv\Scripts\python.exe -m src.inference.demo --source <ảnh|video|0>` từ gốc repo. Lệnh cũ `-m src.week1_demo` vẫn hoạt động. UI gọi bộ nhận dạng này; không gọi train.
