# UI = giao diện web

`app.py` chứa các điều khiển Streamlit: nguồn ảnh/video, confidence, gallery, nút phân tích, bảng đối tượng, ảnh cắt và tải kết quả. File này gọi `src/classroom_ai/inference/detector.py` để chạy YOLO.

Khởi động từ gốc repo: `.venv\Scripts\python.exe -m streamlit run src/classroom_ai/interfaces/web/app.py`. Mở `http://127.0.0.1:8501` trong trình duyệt. Terminal giữ server chạy; đóng terminal hoặc Ctrl+C sẽ dừng web.

Không đặt thuật toán train hay dữ liệu vào thư mục này. Khi sửa bố cục/nút/hiển thị, sửa UI. Khi sửa mapping/đầu ra YOLO, sửa inference.

Webcam preview đọc `runtime.preview_fps`, `preview_max_width` và `preview_jpeg_quality`
trong `configs/app.yaml`. Reload trang sau khi sửa cấu hình. Xem
[cách đo FPS hiển thị và giới hạn còn lại](../guides/webcam_performance.md).

Chọn **Thiết bị xử lý** dưới Checkpoint để dùng CPU hoặc GPU NVIDIA khả dụng.
Xem [cài đặt và kiểm tra GPU](../guides/gpu_setup.md).
