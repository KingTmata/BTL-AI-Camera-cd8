# Kiểm thử

Đã thêm `test_dataset.py` cho validator/chia nhóm/lock/checksum và `test_evaluation.py` cho mapping, AP, ảnh âm tính, hộp sai/dư và empty predictions. Giao diện/inference hiện dùng project8.

Chạy từ gốc repo: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

`test_inspection.py` kiểm mapping, profile nhãn COCO80/project8, pixel và crop. `test_app.py` dùng manifest chính và YOLO thật; tự skip nếu thiếu dữ liệu/weights. `test_camera.py` kiểm camera mô phỏng: backend fallback, bật/dừng/mở lại, độc quyền phiên, mất kết nối và hết heartbeat. Các test gắn với script chuẩn bị COCO/gộp classroom đã bỏ cùng code cũ. **29/29 test hiện hành pass, không skip**, sau khi dọn dữ liệu/code ngày 04/10/2026; không thay kiểm thiết bị thật hoặc chất lượng test set. Hướng dẫn ở [docs/WEEK1_HANDOFF.md](../docs/WEEK1_HANDOFF.md).

Verifier snapshot tại chỗ có hai ca kiểm cache dẫn xuất: chỉ bỏ qua labels/train.cache, val.cache, test.cache; file thêm khác và thay TXT vẫn bị phát hiện.

Training UI bổ sung kiểm cấu hình/form, chặn inference, chống chạy đồng thời, PID bị tái sử dụng, worker chết, theo dõi sau tải lại, bảo toàn run/checkpoint, checksum sai và dừng sau checkpoint. `test_training.py` dùng tiến trình fixture riêng trong thư mục tạm; không train dữ liệu chính hoặc tạo kết quả model giả trong `runs/train`. Sau tích hợp Training ngày 04/10/2026: **44/44 đạt, không skip**. Xem [Training UI](../../TRIEN_KHAI_TUAN_3_TIEP_TUC.md).
