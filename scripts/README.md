# Scripts = tiện ích chuẩn bị, không phải ứng dụng

`data/prepare_smoke_dataset.py` tìm các cặp ảnh/nhãn COCO128 và tạo list/YAML cho một lần train thử nhỏ. Script này **không train và không chạy nhận dạng**.

Chạy: `.venv\Scripts\python.exe scripts/data/prepare_smoke_dataset.py --limit 8`. Đầu ra nằm trong `runs/week1/coco128_smoke/`. `--limit 0` lấy mọi cặp đầy đủ. Dữ liệu đó dùng chung train/val nên chỉ kiểm pipeline, không chứng minh chất lượng.
