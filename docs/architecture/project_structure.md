# Cấu trúc dự án và quy ước

Quyết định ngày 15/10/2026: một package `classroom_ai` với src layout, ưu tiên ít file code có trách nhiệm rõ. Không tạo sẵn API/service/repository khi chương trình chưa cần chúng. Đây là ứng dụng local; backend không đòi hỏi HTTP server.

## Phân chia code

| Vị trí trong `src/classroom_ai/` | Trách nhiệm |
|---|---|
| `classes.py`, `paths.py` | Schema lớp và tìm tài nguyên |
| `datasets.py`, `evaluation.py` | Kiểm dữ liệu/release; đánh giá detector |
| `inference/` | Model, thiết bị, webcam và biến thể inference |
| `training/` | Điều phối job, khóa Training, worker, tiến độ và xuất Kaggle |
| `video/core.py` | Cấu hình, frame, trạng thái giới hạn và đồng hồ |
| `video/pipeline.py` | Điều phối phiên full/fast và ghi báo cáo |
| `video/media.py`, `playback.py` | Decode/probe/encode; điều khiển và tua MP4 đã xuất |
| `video/tracking.py`, `rendering.py`, `tracked_processing.py` | Ghép person, dựng box và full hai lượt |
| `video/metrics.py` | Đo công đoạn, tài nguyên và tiến độ |
| `interfaces/` | CLI, TUI/process, HighGUI và web Streamlit |
| `experiments/` | Quy trình thực nghiệm riêng, tái sử dụng trong notebook |

Không gộp mọi thực nghiệm vào một file: chúng có mục đích, đầu vào và cách chạy khác nhau. Không tạo module chỉ để bọc một hàm dùng đúng một chỗ. Cấu hình, cấu trúc dữ liệu và helper nhỏ có thể ở cùng module liên quan. Chỉ tách khi có trách nhiệm độc lập hoặc việc đọc/kiểm thử thực sự khó hơn.

```text
CLI / TUI / Streamlit / HighGUI
              ↓
     video pipeline / training manager
              ↓
   inference / tracking / media / dữ liệu
```

Model, tracker và encoder không sở hữu widget. TUI gửi cấu hình cho process video và nhận log, không nhận frame. `run_video()` nhận dependency có thể thay bằng fake khi test; bộ hiển thị mặc định được chọn ở điểm khởi tạo. Inference không import thực nghiệm hoặc Streamlit.

## Tên và số lượng file

- Python: `snake_case.py`, hàm/biến `snake_case`, class `PascalCase`, hằng số `UPPER_SNAKE_CASE`. `__init__.py`, `__main__.py` là tên chuẩn của Python.
- Test: `test_<nội_dung>.py` trong `tests/unit/` hoặc `tests/integration/`. Giữ test `unittest.TestCase` có giá trị: pytest chạy được, không viết lại chỉ để đổi phong cách.
- Launcher: `run_video.cmd`, `run_demo.cmd`, `setup_demo.cmd`. Tên gạch ngang cũ chỉ chuyển tiếp.
- Không đổi ID checkpoint, tên dataset hoặc lượt thực nghiệm để áp PEP 8. Chúng là định danh artifact và bằng chứng.
- Ít file là ưu tiên về khả năng đọc và quản lý, không đặt hạn ngạch cứng hoặc gộp phần không liên quan. `__init__.py` khai báo package, không phải tầng nghiệp vụ.

PEP 8 quy định phong cách Python, không quy định một cây thư mục duy nhất. Cấu trúc áp dụng [PEP 8](https://peps.python.org/pep-0008/), [src layout của PyPA](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) và [quy ước pytest](https://docs.pytest.org/en/stable/explanation/goodpractices.html).

## Phát triển và kiểm chứng

```powershell
.venv\Scripts\python.exe -m pip install -e ".[terminal,tracking,dev]"
.venv\Scripts\python.exe -m classroom_ai.video --help
.venv\Scripts\python.exe -m classroom_ai.interfaces.web
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pytest -m unit
.venv\Scripts\python.exe -m pytest -m integration
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\ruff.exe format --check src tests scripts
```

`pyproject.toml` quản lý package, console scripts, pytest và Ruff. Lock nằm trong `requirements/`; không cài lock CPU lên môi trường GPU đang dùng. Một test tăng cường dữ liệu cần `albumentations` trong môi trường nghiên cứu phù hợp. Local chưa có gói này; không âm thầm cài OpenCV headless làm mất HighGUI.

Package cài editable nên không chèn `sys.path` theo vị trí module. `CLASSROOM_AI_ROOT` chỉ rõ thư mục chứa tài nguyên nếu cài code ngoài checkout. Wheel không đóng gói dataset/checkpoint local. `scripts/setup_demo.py` tìm root riêng vì phải chạy trước khi package được cài.

## Tương thích và bảo toàn

- `python -m src.video`, `streamlit run src/app.py` là adapter cũ; lệnh chuẩn là `python -m classroom_ai.video`, `run_demo.cmd`.
- Import cũ `src.inference`, `src.training`, `kaggle_tools` được cập nhật trong repository. Không giữ hai bản logic ứng dụng.
- `tasks/plan.md`, `tasks/project_memory.md` giữ vị trí theo AGENTS.md. Không nhân bản mục tiêu chính thức.
- JSON/CSV/checkpoint/ảnh/video lịch sử chuyển vị trí vẫn giữ bytes. Đường dẫn bên trong bằng chứng giữ nguyên; tra [ánh xạ](path_migration_20261015.md) khi đọc lại.
- Không xóa assertion, skip test hoặc hạ tiêu chí để đạt kiểm tra. Test thiếu dependency vẫn được báo thất bại.
