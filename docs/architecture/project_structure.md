# Cấu trúc dự án và quy ước

Một package `classroom_ai` với src layout, ưu tiên ít file code có trách nhiệm rõ. Từ ngày 02/11/2026, web local là điểm vào chính cho video, Detect/Pose/tracking và duyệt timeline; web gọi lại pipeline đang có. CLI/TUI/OpenCV vẫn dùng trực tiếp backend. Không tạo tầng API/service/repository dự phòng.

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
| `interfaces/web.py`, `web.html` | Web local chính: cấu hình lượt video, xem kết quả, duyệt sự kiện và tổng hợp |
| `interfaces/streamlit_ui/` | Giao diện ảnh/webcam/Training được giữ cho công cụ nghiên cứu trước đây |
| `interfaces/terminal.py`, `terminal_process.py`, `native_video.py` | TUI/process và hiển thị OpenCV dùng chung backend video |
| `experiments/` | Quy trình thực nghiệm riêng, tái sử dụng trong notebook |

Không gộp mọi thực nghiệm vào một file: chúng có mục đích, đầu vào và cách chạy khác nhau. Không tạo module chỉ để bọc một hàm dùng đúng một chỗ. Cấu hình, cấu trúc dữ liệu và helper nhỏ có thể ở cùng module liên quan. Chỉ tách khi có trách nhiệm độc lập hoặc việc đọc/kiểm thử thực sự khó hơn.

```text
Web local / CLI / TUI / Streamlit nghiên cứu / HighGUI
              ↓
     video pipeline / training manager
              ↓
   inference / tracking / media / dữ liệu
```

Model, tracker và encoder không sở hữu widget. TUI gửi cấu hình cho process video và nhận log, không nhận frame. `run_video()` nhận dependency có thể thay bằng fake khi test; bộ hiển thị mặc định được chọn ở điểm khởi tạo. Inference không import thực nghiệm hoặc Streamlit.

## Tên và số lượng file

- Python: `snake_case.py`, hàm/biến `snake_case`, class `PascalCase`, hằng số `UPPER_SNAKE_CASE`. `__init__.py`, `__main__.py` là tên chuẩn của Python.
- Test: `test_<nội_dung>.py` trong `tests/unit/` hoặc `tests/integration/`. Giữ test `unittest.TestCase` có giá trị: pytest chạy được, không viết lại chỉ để đổi phong cách.
- Launcher: `run_demo.cmd` mở web local tại cổng 8090; `run_video.cmd` mở CLI/TUI; `setup_demo.cmd` chuẩn bị môi trường CPU. Tên gạch ngang cũ đã bỏ ngày 20/10/2026.
- `classroom-web` và `classroom-demo` cùng gọi `interfaces.web:main`; `classroom-streamlit` gọi `interfaces.streamlit_ui.__main__:main`.
- Không đổi ID checkpoint, tên dataset hoặc lượt thực nghiệm để áp PEP 8. Chúng là định danh artifact và bằng chứng.
- Ít file là ưu tiên về khả năng đọc và quản lý, không đặt hạn ngạch cứng hoặc gộp phần không liên quan. `__init__.py` khai báo package, không phải tầng nghiệp vụ.

PEP 8 quy định phong cách Python, không quy định một cây thư mục duy nhất. Cấu trúc áp dụng [PEP 8](https://peps.python.org/pep-0008/), [src layout của PyPA](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) và [quy ước pytest](https://docs.pytest.org/en/stable/explanation/goodpractices.html).

## Phát triển và kiểm chứng

```powershell
.venv\Scripts\python.exe -m pip install -e ".[terminal,tracking,dev]"
.venv\Scripts\python.exe -m pip install -r requirements/web.txt
.venv\Scripts\python.exe -m classroom_ai.video --help
.venv\Scripts\python.exe -m classroom_ai.interfaces.web --port 8090
.venv\Scripts\python.exe -m classroom_ai.interfaces.streamlit_ui
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pytest -m unit
.venv\Scripts\python.exe -m pytest -m integration
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\ruff.exe format --check src tests scripts
node --test tests/frontend/web_review_ui.cjs tests/frontend/web_prefetch_ui.cjs
```

`pyproject.toml` quản lý package, console scripts, pytest và Ruff. Lock nằm trong `requirements/`; không cài lock CPU lên môi trường GPU đang dùng. Test frontend dùng Node built-in VM/test, không cần npm package hoặc bước build. Một test tăng cường dữ liệu cần `albumentations` trong môi trường nghiên cứu phù hợp; đã kiểm bằng môi trường riêng, không cài OpenCV headless đè môi trường chạy chính có HighGUI.

Package cài editable nên không chèn `sys.path` theo vị trí module. `CLASSROOM_AI_ROOT` chỉ rõ thư mục chứa tài nguyên nếu cài code ngoài checkout. Wheel không đóng gói dataset/checkpoint local. `scripts/setup_demo.py` tìm root riêng vì phải chạy trước khi package được cài.

## Tương thích và bảo toàn

- `python -m src.video`, `streamlit run src/app.py` là adapter cũ; lệnh chính là `run_demo.cmd` cho web video và `python -m classroom_ai.video` cho CLI/TUI. Streamlit cũ chạy bằng `python -m classroom_ai.interfaces.streamlit_ui`.
- Import cũ `src.inference`, `src.training`, `kaggle_tools` được cập nhật trong repository. Không giữ hai bản logic ứng dụng.
- `tasks/plan.md`, `tasks/project_memory.md` giữ vị trí theo AGENTS.md. Không nhân bản mục tiêu chính thức.
- JSON/CSV/checkpoint/ảnh/video lịch sử chuyển vị trí vẫn giữ bytes. Đường dẫn bên trong bằng chứng giữ nguyên; tra [ánh xạ](path_migration_20261015.md) khi đọc lại.
- Ánh xạ ngày 15/10 vẫn giữ nguyên hồ sơ. Đổi tên code ngày 02/11: `interfaces/web/` → `interfaces/streamlit_ui/`, `interfaces/demo.py` → `interfaces/web.py`, `interfaces/demo.html` → `interfaces/web.html`; test toàn vẹn tra vị trí Python hiện tại trước khi kiểm artifact lịch sử.
- Không xóa assertion, skip test hoặc hạ tiêu chí để đạt kiểm tra. Test thiếu dependency vẫn được báo thất bại.
