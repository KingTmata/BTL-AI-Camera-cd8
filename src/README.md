# Mã nguồn: chọn đúng phần cần sửa

| Thư mục/file | Dùng để làm gì | Có huấn luyện model không? |
|---|---|---|
| `inference/` | Đọc model có sẵn, đưa ảnh/frame vào, lấy hộp/lớp/confidence. CLI ảnh/video/webcam nằm ở đây. | Không; chỉ dự đoán |
| `ui/` | Web Streamlit: chọn ảnh, nút chạy, bảng/crop/ảnh kết quả, tải kết quả. Gọi `inference/`. | Không |
| `training/` | Nơi dành cho code điều phối train về sau; hiện có README chỉ tới lệnh train Ultralytics và cấu hình. | Hiện chưa có trainer tự viết |
| `app.py` | File khởi động web rất ngắn, chuyển việc cho `ui/app.py`. | Không |
| `week1_demo.py` | Giữ lệnh cũ hoạt động, chuyển việc cho `inference/demo.py`. | Không |

Hướng dẫn đầy đủ: [cấu trúc và luồng](../docs/CAU_TRUC_VA_LUONG.md). Không đặt ảnh, weights hay kết quả run vào `src/`.
