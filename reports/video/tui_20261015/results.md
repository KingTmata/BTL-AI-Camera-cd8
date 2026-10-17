# TUI video tối giản — 15/10/2026

TUI dùng **Textual 8.2.8**, bổ sung vào `.venv` dự án qua `requirements/video_tui.txt`; không cài textual-dev, không đổi PyTorch/Ultralytics/OpenCV. `pip check` không phát hiện dependency hỏng. CLI cũ không cần import Textual khi chạy video trực tiếp.

## Đã có

- Chọn/dán đường dẫn video, Fast/Full, Chạy/Dừng, tiến độ và log tối đa 200 dòng.
- Chuột, Tab/Enter; Ctrl+R chạy, Ctrl+D dừng, Ctrl+Q/Ctrl+C thoát. Cây thư mục nằm trong terminal, lọc file video.
- Backend là subprocess riêng, dùng argv không shell. Windows không mở thêm console backend; cửa sổ video OpenCV vẫn mở bình thường. TUI không import torch/cv2/streamlit, không nhận hoặc render frame.
- Log đọc tối đa 64 KiB mỗi nhịp, một nhịp/giây; tắt animation/con trỏ nhấp nháy. Không hiển thị phần trăm giả khi load.
- Dừng dùng cờ file, backend kiểm mỗi 200 ms. Thoát đợi tối đa 10 giây rồi thu dọn cây process riêng. Force-stop không hứa có báo cáo hoàn chỉnh.

## Kiểm chứng

Textual Pilot kiểm click Chạy/Dừng, lỗi nguồn chưa chọn, browse/đóng cây thư mục và phím thoát. Kiểm argv an toàn với tên file có dấu/khoảng trắng/ký tự `&`, ngăn start trùng, parser tiến độ. Có integration chạy backend thật và dừng ngay lúc preflight; các test video cũ tiếp tục được chạy cùng nhóm.

Đo ngắn frontend bằng **Pilot headless 100×30**, idle 8 giây; đây không phải phép đo render thực trên mọi terminal. Phiên thử frontend điều khiển backend phát test pattern 640×360, 30 FPS, 5 giây, không YOLO. Số liệu và điều kiện đo trong [checks.json](checks.json); [preview.svg](preview.svg) là ảnh xuất từ renderer Textual.

- Frontend RSS **43.712.512 byte (~41,7 MiB)**; trước khi import Textual là **25.686.016 byte**. Idle khoảng **1,37% một core CPU** trong cửa sổ đo tám giây, chỉ là mẫu ngắn trên môi trường kiểm thử.
- Backend có TUI điều khiển: **29,994 FPS**, hiển thị đủ **150/150 frame**, không bỏ frame trong clip năm giây. Phép thử không chứa YOLO và không thay thế benchmark dài với model.
- **25/25 test video/TUI đạt**, gồm sáu test bổ sung cho runner/TUI. Suite toàn dự án: **96/97 đạt**, một lỗi augmentation cũ do thiếu `albumentations`; không xóa hoặc skip test.
- `compileall` và `pip check` đạt. CLI help hoạt động; import TUI riêng không kéo `torch`, `cv2`, `streamlit` vào process giao diện.

Không suy từ phép thử ngắn này ra FPS trên lớp học dài, tốc độ GPU hoặc phần trăm overhead cố định. TUI và pipeline video vẫn có tài nguyên riêng; lựa chọn process riêng và chỉ truyền log chữ giúp tránh đưa đường phát ảnh vào terminal.

Hướng dẫn chạy và giải thích terminal/TUI/backend/OpenCV: [VIDEO_NATIVE.md](../../../docs/guides/video_processing.md).
