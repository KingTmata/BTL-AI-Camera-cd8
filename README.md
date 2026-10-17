# Classroom AI

Ứng dụng Python local để nhận diện, theo dõi người và xử lý video lớp học. Có cửa sổ OpenCV, giao diện terminal tùy chọn và giao diện Streamlit riêng.

Tên đề tài và mục tiêu chính thức nằm tại [tasks/plan.md](tasks/plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026). Đọc [ghi nhớ dự án](tasks/project_memory.md) trước khi thay đổi dự án. Các báo cáo cũ là bằng chứng theo ngày và phạm vi của từng lượt.

## Chạy trên Windows

Chạy từ thư mục gốc. Môi trường hiện tại đã cài package editable; máy mới dùng Python 3.11 và `setup_demo.cmd` để cài demo CPU. Video cần `ffmpeg`, `ffprobe` trong PATH.

```powershell
# TUI: chọn video, Full/Fast và tùy chọn Bám người rồi bấm Chạy.
.\run_video.cmd

# CLI: nhận diện toàn bộ, xuất xong mới phát.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode full

# Web Streamlit riêng.
.\run_demo.cmd
```

Nếu cài phần video trên môi trường đã có thư viện nền:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[terminal,tracking,dev]"
.venv\Scripts\python.exe -m classroom_ai.video --help
```

GPU có [quy trình cài PyTorch riêng](docs/guides/gpu_setup.md). Không chạy lại bootstrap CPU trên môi trường GPU đang dùng.

Full nhận diện mọi frame rồi xuất MP4; fast lấy mẫu và phát cuốn chiếu. Tracking person chỉ áp dụng full, cần bật tường minh. Video mặc định conf 0,05, imgsz 1280, checkpoint E3 hiện hành. [Hướng dẫn video](docs/guides/video_processing.md) có cấu hình demo LIGHT_CONTRAST seed123 và giải thích OpenCV/backend/TUI.

## Cấu trúc

```text
src/classroom_ai/     Package ứng dụng
tests/               unit/ và integration/
scripts/             Cài đặt, tạo notebook, benchmark
configs/             Cấu hình thực thi
requirements/        Dependencies và lock theo môi trường
docs/                Hướng dẫn, kiến trúc, nghiên cứu và lịch sử
tasks/               Mục tiêu chính thức, ghi nhớ và kế hoạch
notebooks/           Notebook thí nghiệm
reports/             Bằng chứng và báo cáo theo nhóm
data/, weights/      Dữ liệu và checkpoint local
runs/                Đầu ra từng lượt chạy
```

Ưu tiên ít module có trách nhiệm rõ, không máy móc tách một class thành một file. [Cấu trúc và quy ước](docs/architecture/project_structure.md) giải thích cách tổ chức và kiểm thử. [Bảng đường dẫn cũ–mới](docs/architecture/path_migration_20261015.md) bao gồm cả tài liệu lịch sử; dữ liệu/checkpoint và số đo gốc được bảo toàn.

## Kiểm tra

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\ruff.exe format --check src tests scripts
```

Môi trường local còn thiếu `albumentations` cho một test tăng cường dữ liệu nghiên cứu; xem [báo cáo tái cấu trúc](reports/maintenance/restructure_20261015.md). Test này vẫn được giữ và báo thất bại.

[Danh mục tài liệu](docs/index.md) · [README trước tái cấu trúc](docs/archive/readme_snapshot_20261015.md)
