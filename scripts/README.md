# Scripts = tiện ích chuẩn bị, không phải ứng dụng

Dataset chính hiện tại: [project8_v0.2](../docs/PROJECT8_V0_2_DATA_PROFILE.md), 1.800 ảnh.

- `data/import_laptop_submission.py <ZIP>`: lưu raw, kiểm ảnh/nhãn, đổi Book 0→book 6 và Laptop 1→laptop 3, lập inventory/review và hồ sơ nguồn. Chạy một lần trên đường dẫn mới; không ghi đè bản đã nhập.
- `data/merge_project_dataset.py`: gộp COCO500 và nguồn thành viên thành `data/dataset/project8_v0.2/`, giữ split nguồn, tạo YAML tám lớp, manifest, thống kê và checksum. Đây là draft, chưa xác nhận review/nhãn đầy đủ.

Tuần 2: `data/week2.py` cung cấp inventory/check/split/lock/verify cho dataset tám lớp. Hướng dẫn từng lệnh và trường manifest ở [docs/WEEK2_WORKFLOW.md](../docs/WEEK2_WORKFLOW.md). `data/smoke_week2.py` kiểm toàn luồng bằng dữ liệu nhân tạo và pretrained thật; không sinh baseline dataset nhóm.

`data/prepare_smoke_dataset.py` tìm các cặp ảnh/nhãn COCO128 và tạo list/YAML cho một lần train thử nhỏ. Script này **không train và không chạy nhận dạng**.

Chạy: `.venv\Scripts\python.exe scripts/data/prepare_smoke_dataset.py --limit 8`. Đầu ra nằm trong `runs/week1/coco128_smoke/`. `--limit 0` lấy mọi cặp đầy đủ. Dữ liệu đó dùng chung train/val nên chỉ kiểm pipeline, không chứng minh chất lượng.
