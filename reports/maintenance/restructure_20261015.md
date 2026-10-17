# Nhật ký tái cấu trúc dự án

Bắt đầu 15/10/2026, hoàn tất kiểm chứng 16/10/2026, Asia/Saigon. Phạm vi: mã nguồn, kiểm thử, dependency manifests, launcher, tài liệu hiện hành và lịch sử. Ưu tiên bổ sung của người dùng: ít file code, không tách nhỏ theo máy móc.

## Kết quả

- Một package cài được `classroom-ai`, namespace `classroom_ai`, src layout. Cấu hình package/pytest/Ruff trong `pyproject.toml`; định dạng trong `.editorconfig`, `.gitattributes`.
- Code ứng dụng còn **45 file Python**, trong đó 7 `__init__.py` khai báo package. Đã gộp 11 file so với bản phân tách trung gian của đợt này. Không tuyên bố giảm so với cây ban đầu: cấu trúc mới còn có khai báo package, tìm root và adapter phục vụ migration.
- Gộp helper/cấu hình nhỏ: video core, playback, camera acquisition, Training options; CLI ngay trong `video/__main__.py`. Dataset/evaluation là module phẳng, không đặt trong thư mục chỉ có một module.
- Giữ phần model, tracking, media, UI và điều phối có trách nhiệm riêng. Không tạo API web mới hoặc các tầng dự phòng.
- Import trong source, test, script và cell code notebook chuyển sang namespace chuẩn. Biến thể inference dùng trong production nằm ở `inference/variants.py`, không phụ thuộc package điều phối thực nghiệm.
- Lệnh mới: `run_video.cmd`, `run_demo.cmd`, `setup_demo.cmd`; lệnh gạch ngang cũ chuyển tiếp. Giữ adapter `python -m src.video`, `streamlit run src/app.py`. Không duy trì các import nội bộ cũ.
- Tài liệu được phân nhóm vào `docs/`, kế hoạch vào `tasks/`, bằng chứng vào `reports/`. Mục tiêu chính thức vẫn tại `tasks/plan.md`; ghi nhớ vẫn tại `tasks/project_memory.md`.

[Cấu trúc và quy ước](../../docs/architecture/project_structure.md) · [903 mục ánh xạ](../../docs/architecture/path_migration_20261015.md) · [Manifest hash trước chuyển](../../docs/architecture/path_migration_20261015.json).

## Bảo toàn

- **506 artifact** không phải source/tài liệu/notebook được đối chiếu SHA-256 và giữ nguyên bytes. Bao gồm dữ liệu JSON/CSV và media trong tập file di chuyển; checkpoint/dataset hiện có không bị đổi tên theo PEP 8.
- Markdown cập nhật link/path và khoảng trắng; cell code notebook cập nhật import/path, không sửa output thí nghiệm. README gốc được giữ thành [snapshot](../../docs/archive/readme_snapshot_20261015.md).
- Đối soát **497 liên kết vốn tồn tại: 0 lỗi do migration**. Có 19 tham chiếu đã không có file ở thời điểm bắt đầu, chủ yếu checkpoint/ZIP/script lịch sử. Không tạo file giả hoặc sửa số đo để che chúng. [Chi tiết liên kết](documentation_links_20261015.json).
- Chỉ dọn `.pyc` cũ và thư mục legacy đã rỗng sau khi kiểm đường dẫn trong workspace. Không xóa dữ liệu/ảnh/model để giảm số file. [Đối soát cấu trúc](restructure_audit_20261015.json).
- Bản sao trước chuyển từng nằm ở `.temp/restructure_backup/`; đã xóa trong đợt dọn 16/10/2026 theo yêu cầu người dùng. Manifest ánh xạ và bằng chứng bàn giao vẫn còn. Working tree đã có thay đổi từ trước; không reset, commit hay push.

## Kiểm thử

| Kiểm tra | Kết quả |
|---|---|
| Baseline trước migration | 117 qua, 2 lỗi / 119 test |
| Suite cuối | **123 qua, 1 lỗi / 124 test**, 58,09 giây |
| Ruff lint | Qua |
| Ruff format | 80 file source/test/script đúng định dạng |
| Cài editable không kéo thêm dependency | Qua |
| `pip check` | Không phát hiện dependency đã cài bị xung đột |
| CLI chuẩn, CLI cũ và trợ giúp launcher web | Qua |
| Import package ngoài checkout | Qua trong suite đầy đủ |
| Mapping đích tồn tại và hash artifact | Qua |

Hai lỗi baseline là thiếu `albumentations` và timer TUI có thể truy vấn widget sau khi màn hình đóng. Đã thêm kiểm tra vòng đời `is_running` trước khi đọc log/truy vấn widget; test hồi quy chứng minh bỏ guard thì thất bại. Các assertion gốc được giữ. Không skip/xfail test thiếu dependency.

Lỗi còn lại:

```text
tests/unit/test_e3_improvement_eval.py:17
ModuleNotFoundError: No module named 'albumentations'
```

Đây là thư viện tăng cường ảnh phục vụ test cấu hình training nghiên cứu. Môi trường local chưa cài; không phải lỗi đọc MP4 hoặc tracking. Không tự cài dependency mặc định của gói vì có thể đưa OpenCV headless vào môi trường cần HighGUI. `pip check` chỉ kiểm quan hệ giữa các gói đã cài, không chứng minh đủ mọi dependency cho nghiên cứu.

Một lần chạy lại test cấu trúc riêng gặp quyền tạo thư mục temp của sandbox; đã chạy lại với TEMP/TMP và `--basetemp` nằm trong workspace. Suite đầy đủ dùng cấu hình temp này và quyền loopback cần cho Streamlit/Textual.

## Smoke YOLO thật

Trích một giây đầu của `data/videos/samples/screen_recording_20261015_204628.mp4` bằng FFmpeg sang `.temp/restructure_smoke_input.mp4`, giữ audio đầu tiên. Chạy:

```powershell
.venv\Scripts\python.exe -m classroom_ai.video --source .temp/restructure_smoke_input.mp4 --mode full --weights weights/seed_repeats_20261014/E3_LIGHT_CONTRAST_seed123_E2_best.pt --confidence 0.05 --imgsz 1280 --tracking bytetrack --device cpu --no-display --output-dir runs/restructure_smoke
```

Kết quả `completed`: **30 frame đọc / 30 inference / 30 tracking / 30 frame xuất**, 30 FPS, 1914×844, audio AAC. YOLO không chạy lại trong lượt dựng; hash checkpoint `5487978ad53e7c5fc9d5cbe0c043edab8497f4e83dc248a7c0f785775654e816`. Bốn track xác nhận, một box nội suy. Tổng13,29s gồm import/runtime; RAM peak lấy mẫu443,8MiB. Đây là số đo chức năng ngắn trên CPU, không phải chất lượng GT hoặc thời gian xử lý chung cho mọi máy.

[Metadata và report smoke](restructure_smoke_20261015.json) · [Video local](../../runs/restructure_smoke/20261015_235133_cbe6b6dec1/annotated.mp4).

Không mở cửa sổ trong lượt này, nên FPS hiển thị và latency hiển thị là null. Kiểm ba phút HighGUI trước migration giữ vai trò bằng chứng lịch sử riêng, không gọi đó là benchmark lặp lại sau migration. Chưa chạy lại GPU, soak30phút/sáu giờ hoặc đánh giá Recall/IDF1 có ground truth.

## Lệnh kiểm chứng

```powershell
$env:PYTHONUTF8 = '1'
$env:TEMP = (Resolve-Path .temp/testtmp).Path
$env:TMP = $env:TEMP
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.venv\Scripts\python.exe -m pytest -q --basetemp=.temp/pytest_restructure_final
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\ruff.exe format --check src tests scripts
```

Không thay đổi mục tiêu nghiên cứu, mặc định trọng số/confidence, khóa Training, FPS timeline, quy tắc tracking hay tiêu chí nghiệm thu chỉ để hoàn tất việc sắp xếp.

## Dọn tệp phụ — 16/10/2026

Theo yêu cầu xóa tệp thừa và không tạo thêm file: đã xóa **544 file**, giải phóng **34,56 MiB**, gồm script dùng một lần, log kiểm tra đã hoàn tất, cache Python/pytest/Ruff/pip, bản sao trước migration và video tổng hợp tạm. Các đường dẫn `.temp/` trong lệnh kiểm chứng phía trên là lịch sử và có thể không còn tồn tại. Giữ log backend/Streamlit hiện hành, dữ liệu thực nghiệm, kết quả bàn giao, trọng số và báo cáo.

Không tạo file mới hoặc chạy lại công cụ sinh cache. Đối chiếu hash trong RAM xác nhận **135 file source/test/script/notebook/config/dependency giữ nguyên bytes**; cú pháp Python hợp lệ. Chỉ cập nhật nhật ký và ghi nhớ đã có.
