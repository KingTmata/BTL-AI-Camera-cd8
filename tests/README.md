# Kiểm thử

Đã thêm `test_dataset.py` cho validator/chia nhóm/lock/checksum và `test_evaluation.py` cho mapping, AP, ảnh âm tính, hộp sai/dư và empty predictions. Giao diện/inference hiện dùng project8.

Chạy từ gốc repo: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

`test_inspection.py` kiểm mapping, nhãn, pixel và crop. `test_app.py` chạy giao diện Streamlit bằng AppTest và suy luận thật; tự skip nếu thiếu dữ liệu/weights tương ứng. `test_camera.py` kiểm camera mô phỏng: backend fallback, bật/dừng/mở lại, độc quyền phiên, mất kết nối và hết heartbeat. Test UI webcam dùng camera mô phỏng + YOLO thật. Tổng 30 test đã pass trên máy chuẩn; không thay kiểm thiết bị thật hoặc chất lượng test set. Hướng dẫn ở [docs/WEEK1_HANDOFF.md](../docs/WEEK1_HANDOFF.md).
