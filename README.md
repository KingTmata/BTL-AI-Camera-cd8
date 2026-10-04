# BTL AI Camera — chạy, xem kết quả và chuẩn bị train

> **Dataset chính hiện hành 04/10/2026: [project8_v0.3](docs/PROJECT8_V0_3_DATA_PROFILE.md).** Chỉ giữ **5.002 ảnh / 5.002 TXT / 26.566 box**, không còn pending. Train/val/test: **3.883 / 798 / 321**. `table` và `with-student` classroom cùng map về `table`; lớp bàn có **1.017 ảnh / 2.479 box**. Bản sao raw/dataset cũ, ảnh tham khảo/kết quả thử và ZIP sao lưu đã dọn theo yêu cầu; giữ riêng hai ZIP hành vi cho tuần 4–6. Nhãn chưa duyệt đầy đủ, chưa khóa release hoặc train. Số liệu v0.2 và thử nghiệm tuần 1 bên dưới là lịch sử.

- [Bảng báo cáo tiến độ sau khi dọn dữ liệu/code](reports/results/project_progress_20261004/REPORT.md)
- [Hồ sơ dataset chính mới](docs/PROJECT8_V0_3_DATA_PROFILE.md)

> **Lịch sử 03/10/2026:** dataset `project8_v0.2` từng có **1.800 ảnh**: 500 COCO + 1.300 ảnh Roboflow do thành viên đóng góp; 1.310 train / 360 validation / 130 test theo split nguồn. Ảnh và nhãn nay đã gộp vào v0.3; không còn bộ media v0.2 riêng. Các kết quả tuần 1 bên dưới là lịch sử theo phạm vi bốn lớp cũ.

- [Thực hiện tuần 2 từ đầu đến cuối](docs/WEEK2_WORKFLOW.md)
- [Protocol đánh giá](EVALUATION_PROTOCOL.md)
- [Hồ sơ lịch sử v0.2 — 1.800 ảnh](docs/PROJECT8_V0_2_DATA_PROFILE.md)
- [Đặc tính bộ Laptop thành viên](docs/LAPTOP_SUBMISSION_DATA_PROFILE.md)
- [Đặc tính COCO500](docs/COCO500_DATA_PROFILE.md)
- [Báo cáo triển khai tuần 2](docs/WEEK2_IMPLEMENTATION_REPORT.md)

Repo hiện có **baseline tuần 1** theo [kế hoạch](WEEK1_2_3_PLAN.md): YOLO26n pretrained trên ảnh/video/webcam cho tám lớp hiện hành `person`, `table`, `chair`, `laptop`, `cell phone`, `backpack`, `book`, `cup`. Giao diện đã thêm webcam trực tiếp với bật/dừng/mở lại, chọn camera, FPS/p95 và tải kết quả kiểm. Có `setup-demo.cmd` và `run-demo.cmd` cho từng máy Windows. Xem [bàn giao webcam tuần 1](docs/WEEK1_HANDOFF.md): phần mềm đã kiểm, G1 còn chờ webcam thật trên máy thành viên và review người. Chưa có fine-tune, tracking, đếm lượt hay cảnh báo.

## Bắt đầu từ đâu?

| Bạn muốn làm gì? | Dùng phần nào? | Hướng dẫn |
|---|---|---|
| Chọn ảnh, xem hộp, bấm xem từng đối tượng | Web `src/ui/`, dùng model ở `src/inference/` | [Chạy web và test](docs/KIEM_THU.md) |
| Hiểu từng folder và luồng hoạt động | Tài liệu sơ đồ, bảng đầu vào/đầu ra | [Cấu trúc và luồng](docs/CAU_TRUC_VA_LUONG.md) |
| Chạy ảnh/video/webcam bằng terminal | `src/inference/demo.py` | [Lệnh CLI](docs/KIEM_THU.md#3-test-ảnh-bằng-cli) |
| Kiểm code có hoạt động đúng không | `tests/` | [Kiểm thử tự động](docs/KIEM_THU.md#2-kiểm-mã-tự-động) |
| Chuẩn bị dữ liệu để train thử | `scripts/data/` | [Train thử và fine-tune về sau](docs/TRAIN_VA_DANH_GIA.md) |
| Fine-tune model mới | Ultralytics trainer + `configs/train_n.yaml`; `src/training/` chưa có trainer tự viết | [CPU, Colab, resume, validation](docs/TRAIN_VA_DANH_GIA.md) |

Mở giao diện trên máy:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/app.py
```

Mở **http://127.0.0.1:8501**, chọn ảnh → **Phân tích ảnh** → chọn dòng đối tượng để xem crop. Có thư viện ảnh thu nhỏ, upload ảnh, chọn frame video, nhãn COCO để đối chiếu, tải PNG/JSON/CSV. Terminal giữ server chạy; Ctrl+C để dừng. Giao diện hiện chỉ chạy inference, không có nút train.

## Môi trường đã chọn

- Windows, Python **3.11** trong `.venv/` riêng của repo.
- PyTorch và torchvision bản **CPU**, Ultralytics YOLO26n, OpenCV, Streamlit, NumPy, Pandas, Matplotlib và PyYAML.
- `yolo26n.pt` pretrained dùng làm baseline. `yolo26s.pt` là đối chứng về sau; chưa tải vì không cần cho môi trường demo CPU ban đầu.
- ByteTrack dùng bản tích hợp Ultralytics; `configs/bytetrack.yaml` được sao từ chính gói đã cài.

Các phiên bản cài thực tế nằm trong `requirements-demo-cpu.lock.txt` sau bước chuẩn bị. Không cài song song `opencv-python` và `opencv-python-headless` trong cùng môi trường. Chưa cài CUDA, Docker, FastAPI, database hoặc công cụ gán nhãn vì chưa cần cho bước này.

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
├── requirements.in
├── requirements-demo-cpu.lock.txt
└── THIRD_PARTY.md
```

`data/reference/` đã có COCO128 và `runs/week1/` có kết quả thử trên máy này; dataset chính project8_v0.2 đã có 1.800 ảnh nguồn công khai; chưa có test phòng học độc lập. Git không lưu thư mục rỗng hoặc dữ liệu/weights đã được ignore. `configs/app.yaml` là đặc tả app đầy đủ trong tương lai, chưa được UI/CLI hiện tại đọc. `configs/train_n.yaml` có thể truyền cho lệnh Ultralytics train khi dữ liệu sẵn sàng; chưa có run fine-tune.

## Tái tạo môi trường trên máy Windows khác

Mở PowerShell tại thư mục gốc repo. Cài Python 3.11 trước nếu máy chưa có; `py -3.11 --version` phải trả về 3.11.x. Sau đó:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements.in
```

Muốn tái lập **đúng phiên bản của máy đã chuẩn bị**, dùng `requirements-demo-cpu.lock.txt` sau khi cài PyTorch CPU. Lock này ghi cả dependencies phụ và chỉ áp dụng cho Windows/Python 3.11 CPU; môi trường GPU/Colab cần lock riêng sau khi kiểm chạy.

## Dùng giữa các thành viên và Colab

Chạy demo tuần 1 trên Windows: mở `setup-demo.cmd` một lần, sau đó mở `run-demo.cmd` và chọn **Webcam trực tiếp**. Hướng dẫn camera, macOS/Linux, log từng máy và review ảnh ở [docs/WEEK1_HANDOFF.md](docs/WEEK1_HANDOFF.md).

- Chọn **một máy cá nhân làm máy demo chuẩn**. Chạy thử camera, tốc độ và độ trễ trên chính máy đó; không lấy số đo trên Colab hoặc máy bạn khác để gọi là tốc độ máy demo.
- Các máy Windows CPU dùng cùng `requirements.in` hoặc lock CPU ở trên, nhưng tự tạo `.venv`. Không sao chép thư mục `.venv` qua máy khác.
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

Baseline ảnh/video/webcam chạy bằng `python -m src.inference.demo --source <ảnh|video|0>`. Web `streamlit run src/app.py` dùng ảnh train của dataset chính, upload hoặc frame video. Lối vào CLI tuần 1 cũ và script chuẩn bị dữ liệu cũ đã bỏ. Các lệnh `python -m src.cli`, `src.train`, `src.evaluate` trong kế hoạch vẫn chưa có; hướng dẫn train hiện dùng CLI Ultralytics.

Chạy một ảnh đã có trong dataset chính, không cần tải thư viện ảnh tham khảo riêng:

```powershell
& .\.venv\Scripts\python.exe -m src.inference.demo --source data/dataset/project8_v0.3/images/train/coco_train_000000002782.jpg --no-window
```

Hai MP4 trong `demo/smoke/` là slideshow kỹ thuật được tạo cục bộ từ ảnh COCO128 và không nằm trong Git. Chúng không thay video quay thật cần cho G1.

## Nhật ký theo tuần

### Tuần 1 — Baseline detection với YOLO26n pretrained

- Đã chuẩn bị môi trường Python 3.11 CPU, checkpoint YOLO26n pretrained COCO80; phạm vi hiện hành là tám lớp theo src/classes.py.
- Đã chạy thử nhận dạng trên ảnh COCO128 và hai video slideshow kỹ thuật; có kết quả phát hiện người/chai, nhưng bỏ sót một số điện thoại/laptop. Đã thêm giao diện web để chọn ảnh/frame, xem bounding box, crop và tải kết quả.
- Sáu bài kiểm thử mã đã chạy thành công. Chưa xác nhận webcam trên máy demo, video quay thật, dữ liệu tự thu hoặc chất lượng trên test độc lập. Chi tiết: [bàn giao G1](docs/KIEM_THU.md).

### Tuần 2 — Chuẩn bị dataset v1

- Đã ghi quy trình nhận ảnh từ thành viên, kiểm trùng, chuẩn hóa nhãn, review chéo và chia train/validation theo nguồn hoặc phiên; thêm các file CSV mẫu để bàn giao dữ liệu.
- Mục tiêu là ít nhất 2.500 ảnh phát triển hợp lệ; hiện có 4.681 file train/validation và 321 test trong tổng 5.002 ảnh. Augmentation/frame video chưa chứng minh số cảnh độc lập; nhãn và test đủ tám lớp còn cần bổ sung, chưa khóa release hoặc fine-tune. Chi tiết: [kế hoạch G2](WEEK2_G2.md).
