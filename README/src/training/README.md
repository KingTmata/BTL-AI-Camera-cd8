# Training = cập nhật trọng số model từ dữ liệu có nhãn

`manager.py` quản lý cấu hình, khóa chạy và tiến trình; `worker.py` kiểm release rồi gọi trainer Ultralytics; `progress.py` theo dõi callback và dừng sau khi lưu epoch. Không sao chép thuật toán YOLO vào repo.

Ngay bây giờ đã có thể dùng bộ huấn luyện Ultralytics qua `.venv\Scripts\yolo.exe detect train cfg=configs/train_n.yaml ...`. Tham số ở `configs/train_n.yaml`, dữ liệu ở `data/dataset/`, checkpoint và log tạo tại `runs/train/`.

Chạy và theo dõi qua [Training UI](../../../TRIEN_KHAI_TUAN_3_TIEP_TUC.md). Quy trình CLI và đánh giá: [docs/TRAIN_VA_DANH_GIA.md](../../docs/TRAIN_VA_DANH_GIA.md). UI khởi tạo run mới từ checkpoint, chưa resume nguyên trạng optimizer.
