# UI = giao diện web

`app.py` chứa các điều khiển Streamlit: nguồn ảnh/video, confidence, gallery, nút phân tích, bảng đối tượng, ảnh cắt và tải kết quả. File này gọi `src/inference/detector.py` để chạy YOLO.

Khởi động từ gốc repo: `.venv\Scripts\python.exe -m streamlit run src/app.py`. Mở `http://127.0.0.1:8501` trong trình duyệt. Terminal giữ server chạy; đóng terminal hoặc Ctrl+C sẽ dừng web.

Không đặt thuật toán train hay dữ liệu vào thư mục này. Khi sửa bố cục/nút/hiển thị, sửa UI. Khi sửa mapping/đầu ra YOLO, sửa inference.
