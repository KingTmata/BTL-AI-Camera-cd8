# Kiểm thử

Đã thêm `test_dataset.py` cho validator/chia nhóm/lock/checksum và `test_evaluation.py` cho mapping, AP, ảnh âm tính, hộp sai/dư và empty predictions. Giao diện/inference hiện dùng project8.

Chạy từ gốc repo: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

`test_inspection.py` kiểm mapping, profile nhãn COCO80/project8, pixel và crop. `test_app.py` dùng manifest chính và YOLO thật; tự skip nếu thiếu dữ liệu/weights. `test_camera.py` kiểm camera mô phỏng: backend fallback, bật/dừng/mở lại, độc quyền phiên, mất kết nối và hết heartbeat. Các test gắn với script chuẩn bị COCO/gộp classroom đã bỏ cùng code cũ. **27/27 test hiện hành pass, không skip**, sau khi dọn dữ liệu/code ngày 04/10/2026; không thay kiểm thiết bị thật hoặc chất lượng test set. Hướng dẫn ở [docs/WEEK1_HANDOFF.md](../docs/WEEK1_HANDOFF.md).
