> **Dataset hiện hành08/10/2026:**v0.5 đủ15.000 ảnh (12.000train/2.500val/500test), tất cảfull_review_pending, chưatrainv05. Weight/metricv04 bên dưới làlịch sử. XemREADME gốc vàtasks/plan.md.

> **Cập nhật v0.4 — 08/10/2026:** Recognition mặc định dùng `weights/BEST_Project8.pt`, alias của best v0.4 epoch 19. Tám nhãn hiện hành kết thúc bằng `bottle` (gộp cup+bottle). Dataset 12k: 9.540 train / 1.754 val / 706 test. Training local và exporter v0.3 vẫn dùng hợp đồng cũ `cup`; notebook v0.4 và `configs/data_v04.yaml` dành cho bộ mới. Các phần v0.3 bên dưới là lịch sử hoặc hướng dẫn cho bộ cũ.

> **Đối soát 05/10/2026:** xem [trạng thái hiện hành](../archive/project_readme.md#trạng-thái-theo-tuần). Dataset project8_v0.3 đã duyệt/khóa; R1 và R2 hoàn tất tổng 5 epoch. Những mô tả trước đó trong tài liệu là lịch sử hoặc đề xuất; G1/G2/G3 chưa đủ nghiệm thu toàn bộ.

# BTL AI Camera — chạy, xem kết quả và chuẩn bị train

> **Dataset chính hiện hành 04/10/2026: [project8_v0.3](../archive/guides/project8_v0_3_data_profile.md).** Chỉ giữ **5.002 ảnh / 5.002 TXT / 26.566 box**, không còn pending. Train/val/test: **3.883 / 798 / 321**. `table` và `with-student` classroom cùng map về `table`; lớp bàn có **1.017 ảnh / 2.479 box**. Bản sao raw/dataset cũ, ảnh tham khảo/kết quả thử và ZIP sao lưu đã dọn theo yêu cầu; giữ riêng hai ZIP hành vi cho tuần 4–6. Nhãn chưa duyệt đầy đủ, chưa khóa release hoặc train. Số liệu v0.2 và thử nghiệm tuần 1 bên dưới là lịch sử.

- [Bảng báo cáo tiến độ sau khi dọn dữ liệu/code](../archive/completed/reports/results/project_progress_20261004/report.md)
- [Hồ sơ dataset chính mới](../archive/guides/project8_v0_3_data_profile.md)

> **Lịch sử 03/10/2026:** dataset `project8_v0.2` từng có **1.800 ảnh**: 500 COCO + 1.300 ảnh Roboflow do thành viên đóng góp; 1.310 train / 360 validation / 130 test theo split nguồn. Ảnh và nhãn nay đã gộp vào v0.3; không còn bộ media v0.2 riêng. Các kết quả tuần 1 bên dưới là lịch sử theo phạm vi bốn lớp cũ.

- [Thực hiện tuần 2 từ đầu đến cuối](../archive/guides/week2_workflow.md)
- [Protocol đánh giá](../research/evaluation_protocol.md)
- [Hồ sơ lịch sử v0.2 — 1.800 ảnh](../archive/completed/docs/project8_v0_2_data_profile.md)
- [Đặc tính bộ Laptop thành viên](../archive/completed/docs/laptop_submission_data_profile.md)
- [Đặc tính COCO500](../archive/completed/docs/coco500_data_profile.md)
- [Báo cáo triển khai tuần 2](../archive/guides/week2_implementation_report.md)

Repo hiện có **baseline tuần 1** theo [kế hoạch](../archive/plans/week1_2_3_plan.md): YOLO26n pretrained trên ảnh/video/webcam cho tám lớp hiện hành `person`, `table`, `chair`, `laptop`, `cell phone`, `backpack`, `book`, `cup`. Giao diện đã thêm webcam trực tiếp với bật/dừng/mở lại, chọn camera, FPS/p95 và tải kết quả kiểm. Có `setup_demo.cmd` và `run_demo.cmd` cho từng máy Windows. Xem [bàn giao webcam tuần 1](../archive/guides/week1_handoff.md): phần mềm đã kiểm, G1 còn chờ webcam thật trên máy thành viên và review người. Đã có Training UI và [lượt thử fine-tune CPU một epoch](../archive/completed/reports/results/project8_v03_26n_cpu_20261004_181257_1eb3de94/report.md), chất lượng chưa đủ nghiệm thu. Tracking, đếm lượt và cảnh báo chưa triển khai.

## Bắt đầu từ đâu?

| Bạn muốn làm gì? | Dùng phần nào? | Hướng dẫn |
|---|---|---|
| Chọn ảnh, xem hộp, bấm xem từng đối tượng | Web `src/ui/`, dùng model ở `src/inference/` | [Chạy web và test](../archive/guides/kiem_thu.md) |
| Hiểu từng folder và luồng hoạt động | Tài liệu sơ đồ, bảng đầu vào/đầu ra | [Cấu trúc và luồng](../archive/guides/cau_truc_va_luong.md) |
| Chạy ảnh/video/webcam bằng terminal | `src/inference/demo.py` | [Lệnh CLI](../archive/guides/kiem_thu.md#3-test-ảnh-bằng-cli) |
| Kiểm code có hoạt động đúng không | `tests/` | [Kiểm thử tự động](../archive/guides/kiem_thu.md#2-kiểm-mã-tự-động) |
| Chuẩn bị dữ liệu để train thử | `scripts/data/` | [Train thử và fine-tune về sau](../archive/guides/train_va_danh_gia.md) |
| Fine-tune model mới | Ultralytics trainer + `configs/train_n.yaml`; `src/training/` chưa có trainer tự viết | [CPU, Colab, resume, validation](../archive/guides/train_va_danh_gia.md) |

Mở giao diện trên máy:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/classroom_ai/interfaces/web/app.py
```

Mở **http://127.0.0.1:8501**, chọn Recognition để xem ảnh/video/webcam hoặc Training để huấn luyện trên CPU. Có thư viện ảnh thu nhỏ, upload ảnh, nhãn project8 để đối chiếu và tải PNG/JSON/CSV. [Training UI](../archive/plans/week3_handoff.md) cho phép chỉnh cấu hình, theo dõi tiến trình riêng và tải checkpoint từng epoch; tải lại/đóng trình duyệt không dừng train. Terminal giữ server chạy; Ctrl+C để dừng server.

## Môi trường đã chọn

- Windows, Python **3.11** trong `.venv/` riêng của repo.
- PyTorch và torchvision bản **CPU**, Ultralytics YOLO26n, OpenCV, Streamlit, NumPy, Pandas, Matplotlib và PyYAML.
- `yolo26n.pt` pretrained dùng làm baseline. `yolo26s.pt` là đối chứng về sau; chưa tải vì không cần cho môi trường demo CPU ban đầu.
- ByteTrack dùng bản tích hợp Ultralytics; `configs/bytetrack.yaml` được sao từ chính gói đã cài.

Các phiên bản cài thực tế nằm trong `requirements/demo_cpu.lock.txt` sau bước chuẩn bị. Không cài song song `opencv-python` và `opencv-python-headless` trong cùng môi trường. Chưa cài CUDA, Docker, FastAPI, database hoặc công cụ gán nhãn vì chưa cần cho bước này.

Trên máy đã chuẩn bị, `.venv` dùng Python 3.11.16; các gói chính là `torch 2.14.0+cpu`, `ultralytics 8.4.165`, `opencv-python 5.0.0.93`, `streamlit 1.64.0`. Máy này không cần thêm Python vào PATH: gọi trực tiếp `.venv\Scripts\python.exe` như các lệnh bên dưới. Checkpoint YOLO26n có SHA-256 `9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF`.

## Cấu trúc hiện tại

```text
.
├── .python-version                 # Yêu cầu Python 3.11
├── .venv/                          # Môi trường ảo cục bộ, không commit
├── configs/
│   ├── app.yaml                    # Tham số dự kiến cho ứng dụng
│   ├── bytetrack.yaml              # Sao từ Ultralytics đã cài
│   ├── data.yaml.example           # Mẫu cấu hình dataset tám lớp hiện hành
│   ├── data.yaml                   # Đường dẫn dataset của máy hiện tại, không commit
│   └── train_n.yaml                # Cấu hình huấn luyện dự kiến
├── data/
│   ├── dataset/
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   ├── raw/
│   ├── video_dev/
│   └── video_test/
├── demo/
├── reports/{results,errors}/
├── runs/
├── src/
│   ├── inference/                 # CODE CHẠY nhận dạng: detector.py, demo.py
│   ├── ui/                        # CODE WEB: chọn ảnh, bảng, crop, download
│   ├── training/                  # Chỗ dành cho code train sau này; hiện có README
│   ├── app.py                     # Điểm khởi động web, gọi ui/app.py
│   └── week1_demo.py               # Lệnh CLI cũ, chuyển sang inference/demo.py
├── scripts/data/                  # CODE CHUẨN BỊ dữ liệu, không train
├── tests/                         # CODE KIỂM THỬ; không phải dataset test
├── docs/                          # Cấu trúc/luồng, chạy test, train về sau
├── .streamlit/config.toml          # Web local, giới hạn ảnh và giao diện
├── weights/yolo26n.pt             # Checkpoint pretrained, không commit
├── requirements/base.in
├── requirements/demo_cpu.lock.txt
└── THIRD_PARTY.md
```

`data/reference/` đã có COCO128 và `runs/week1/` có kết quả thử trên máy này; dataset chính project8_v0.2 đã có 1.800 ảnh nguồn công khai; chưa có test phòng học độc lập. Git không lưu thư mục rỗng hoặc dữ liệu/weights đã được ignore. `configs/app.yaml` là đặc tả app đầy đủ trong tương lai, chưa được UI/CLI hiện tại đọc. `configs/train_n.yaml` có thể truyền cho lệnh Ultralytics train khi dữ liệu sẵn sàng; chưa có run fine-tune.

## Tái tạo môi trường trên máy Windows khác

Mở PowerShell tại thư mục gốc repo. Cài Python 3.11 trước nếu máy chưa có; `py -3.11 --version` phải trả về 3.11.x. Sau đó:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements/base.in
```

Muốn tái lập **đúng phiên bản của máy đã chuẩn bị**, dùng `requirements/demo_cpu.lock.txt` sau khi cài PyTorch CPU. Lock này ghi cả dependencies phụ và chỉ áp dụng cho Windows/Python 3.11 CPU; môi trường GPU/Colab cần lock riêng sau khi kiểm chạy.

## Dùng giữa các thành viên và Colab

Chạy demo tuần 1 trên Windows: mở `setup_demo.cmd` một lần, sau đó mở `run_demo.cmd` và chọn **Webcam trực tiếp**. Hướng dẫn camera, macOS/Linux, log từng máy và review ảnh ở [docs/WEEK1_HANDOFF.md](../archive/guides/week1_handoff.md).

- Chọn **một máy cá nhân làm máy demo chuẩn**. Chạy thử camera, tốc độ và độ trễ trên chính máy đó; không lấy số đo trên Colab hoặc máy bạn khác để gọi là tốc độ máy demo.
- Các máy Windows CPU dùng cùng `requirements/base.in` hoặc lock CPU ở trên, nhưng tự tạo `.venv`. Không sao chép thư mục `.venv` qua máy khác.
- Nếu thành viên có GPU NVIDIA tương thích, cài PyTorch theo cấu hình GPU của máy đó rồi cài dependencies dự án. Ghi riêng phiên bản PyTorch/CUDA và thiết bị; không trộn lock GPU với lock CPU.
- Colab Free chỉ dùng để fine-tune khi có GPU. Mỗi phiên Colab là môi trường Linux riêng, cần cài gói trong notebook và lưu checkpoint/log ra nơi bền vững sau mỗi đợt; không dùng `.venv` Windows hoặc lock Windows trực tiếp.
- YOLO26n pretrained → fine-tuned là đối chứng chính. YOLO26s chỉ là đối chứng mở rộng khi đủ tài nguyên. Nếu Colab ngắt, dùng `last.pt` để tiếp tục; không báo kết quả của run chưa hoàn thành như kết quả cuối.

Tải checkpoint chính thức qua Ultralytics (chỉ cần internet lần đầu):

```powershell
& .\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
Move-Item -LiteralPath .\yolo26n.pt -Destination .\weights\yolo26n.pt
```

Sao cấu hình tracker từ đúng phiên bản cài đặt:

```powershell
Copy-Item -LiteralPath .\.venv\Lib\site-packages\ultralytics\cfg\trackers\bytetrack.yaml -Destination .\configs\bytetrack.yaml
```

Tạo `configs/data.yaml` từ `configs/data.yaml.example`, sửa `path` thành đường dẫn tuyệt đối của `data/dataset` trên máy đang dùng. Không đưa ảnh test từ cùng phiên quay với ảnh train vào dataset; kế hoạch yêu cầu chia theo phiên và ghi nguồn trong `data/manifest.csv`.

## Kiểm tra môi trường

```powershell
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\python.exe -c "import torch, cv2, ultralytics, streamlit, pandas, yaml; print(torch.__version__, cv2.__version__, ultralytics.__version__, streamlit.__version__)"
& .\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO('weights/yolo26n.pt'); print({i:m.names[i] for i in (0,60,56,63,67,24,73,41)})"
```

Kỳ vọng tám ID COCO lần lượt là person, dining table, chair, laptop, cell phone, backpack, book, cup; dining table map về table của nhóm. Lệnh này chỉ xác nhận có thể import và load model; **chưa xác nhận webcam, FPS, tracking hay chất lượng dự đoán**.

## Việc nhóm còn phải làm

1. Xin giảng viên duyệt phạm vi và xác nhận sĩ số, hạn nộp, phương án pretrained/fine-tune.
2. Nhận ảnh/clip có quyền sử dụng, gán đủ tám lớp, lập manifest và chia theo nhóm bằng workflow tuần 2.
3. Viết ứng dụng theo các module ở mục 10 của kế hoạch; kiểm camera, tracking, đếm và cảnh báo.
4. Fine-tune, đánh giá trên test độc lập, đo FPS/độ trễ và ghi kết quả thật vào báo cáo.

Baseline ảnh/video/webcam chạy bằng `python -m src.inference.demo --source <ảnh|video|0>`. Web `streamlit run src/classroom_ai/interfaces/web/app.py` dùng ảnh train của dataset chính, upload hoặc frame video. Lối vào CLI tuần 1 cũ và script chuẩn bị dữ liệu cũ đã bỏ. Các lệnh `python -m src.cli`, `src.train`, `src.evaluate` trong kế hoạch vẫn chưa có; hướng dẫn train hiện dùng CLI Ultralytics.

Chạy một ảnh đã có trong dataset chính, không cần tải thư viện ảnh tham khảo riêng:

```powershell
& .\.venv\Scripts\python.exe -m src.inference.demo --source data/dataset/project8_v0.3/images/train/coco_train_000000002782.jpg --no-window
```

Hai MP4 trong `demo/smoke/` là slideshow kỹ thuật được tạo cục bộ từ ảnh COCO128 và không nằm trong Git. Chúng không thay video quay thật cần cho G1.

## Nhật ký theo tuần

### Tuần 1 — Baseline detection với YOLO26n pretrained

- Đã chuẩn bị môi trường Python 3.11 CPU, checkpoint YOLO26n pretrained COCO80; phạm vi hiện hành là tám lớp theo src/classroom_ai/classes.py.
- Đã chạy thử nhận dạng trên ảnh COCO128 và hai video slideshow kỹ thuật; có kết quả phát hiện người/chai, nhưng bỏ sót một số điện thoại/laptop. Đã thêm giao diện web để chọn ảnh/frame, xem bounding box, crop và tải kết quả.
- Sáu bài kiểm thử mã đã chạy thành công. Chưa xác nhận webcam trên máy demo, video quay thật, dữ liệu tự thu hoặc chất lượng trên test độc lập. Chi tiết: [bàn giao G1](../archive/guides/kiem_thu.md).

### Tuần 2 — Chuẩn bị dataset v1

- Đã ghi quy trình nhận ảnh từ thành viên, kiểm trùng, chuẩn hóa nhãn, review chéo và chia train/validation theo nguồn hoặc phiên; thêm các file CSV mẫu để bàn giao dữ liệu.
- Mục tiêu là ít nhất 2.500 ảnh phát triển hợp lệ; hiện có 4.681 file train/validation và 321 test trong tổng 5.002 ảnh. Augmentation/frame video chưa chứng minh số cảnh độc lập; nhãn và test đủ tám lớp còn cần bổ sung, chưa khóa release hoặc fine-tune. Chi tiết: [kế hoạch G2](../archive/plans/week2_g2.md).
