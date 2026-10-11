> **08/10/2026:** dùng best v0.4 epoch 19; nhận cả tên checkpoint v0.3 `cup` và v0.4 `bottle` ở đúng ID 7. Recognition hiển thị tên hợp nhất `bottle`; evaluator v0.3 vẫn giữ tên GT cũ để so sánh.

# Inference = dùng model để nhận dạng

Đầu vào: weights `.pt` và ảnh/frame. Đầu ra: hộp, lớp, confidence hoặc ảnh đã vẽ. Trọng số không thay đổi khi chạy.

- `detector.py`: bộ nhận dạng dùng trong web, mapping COCO80/project8, đọc nhãn tham khảo, vẽ/crop ảnh.
- `demo.py`: baseline CLI đọc ảnh/video/webcam, vòng lặp frame, giải phóng nguồn và ghi `summary.json`.

Chạy CLI: `.venv\Scripts\python.exe -m src.inference.demo --source <ảnh|video|0>` từ gốc repo. UI gọi bộ nhận dạng này; không gọi train. Lối vào CLI cũ tuần 1 đã được bỏ.
