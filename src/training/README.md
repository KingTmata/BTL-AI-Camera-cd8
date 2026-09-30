# Training = cập nhật trọng số model từ dữ liệu có nhãn

Thư mục này dành cho code điều phối train khi nhóm cần viết thêm. **Hiện chưa có `train.py` của nhóm và chưa có run fine-tune chính thức.**

Ngay bây giờ đã có thể dùng bộ huấn luyện Ultralytics qua `.venv\Scripts\yolo.exe detect train cfg=configs/train_n.yaml ...`. Tham số ở `configs/train_n.yaml`, dữ liệu ở `data/dataset/`, checkpoint và log tạo tại `runs/train/`.

Quy trình, lệnh CPU/Colab, resume và validation: [docs/TRAIN_VA_DANH_GIA.md](../../docs/TRAIN_VA_DANH_GIA.md). Thêm script điều phối mới ở đây sau này nếu cần quản lý run/metadata; không sao chép nguyên thuật toán YOLO vào repo.
