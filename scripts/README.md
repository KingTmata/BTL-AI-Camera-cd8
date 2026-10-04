# Tiện ích đang dùng

Dataset chính: [project8_v0.3](../docs/PROJECT8_V0_3_DATA_PROFILE.md), 5.002 ảnh / 26.566 box; không còn pending.

- `setup_demo.py`: tạo môi trường và chạy kiểm môi trường cho demo Windows; được `setup-demo.cmd` gọi.
- `data/week2.py`: inventory/check/split/lock/verify theo workflow dataset tám lớp. Manifest mặc định là `data/manifest.csv`; nhãn/review phải hoàn thiện trước lock.
- `data/smoke_week2.py`: kiểm phần mềm dataset/evaluator bằng dữ liệu nhân tạo tách biệt; không phải baseline chất lượng. Chỉ chạy khi chủ động cần kiểm lại pipeline.
- `data/report_project_progress.py`: xuất bảng Markdown/CSV/JSON từ manifest, statistics, checksum và artifact đang có; `--output` phải là thư mục mới.

Các script nhập Laptop, gộp v0.2/classroom, tải/chọn COCO500 và tạo mẫu/review COCO128 đã được bỏ sau khi chốt kho chính. CLI cũ tuần 1 được thay bằng `python -m src.inference.demo`. Không cần tải lại dữ liệu tham khảo để mở giao diện.
