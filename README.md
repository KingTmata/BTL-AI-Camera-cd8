# BTL AI Camera — chạy, xem kết quả và chuẩn bị train

Repo hiện có **baseline tuần 1** theo [kế hoạch](Ke_hoach_De_8_Camera_AI.md): YOLO26n pretrained trên ảnh/video/webcam cho bốn lớp `person`, `bottle`, `cell phone`, `laptop`. Chưa có ứng dụng hoàn chỉnh, dữ liệu tự thu, fine-tune, tracking, đếm hay cảnh báo. Xem [bàn giao G1](WEEK1_G1.md) để biết phần đã kiểm và phần còn thiếu.

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
│   ├── data.yaml.example           # Mẫu cấu hình dataset bốn lớp
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

`data/reference/` đã có COCO128 và `runs/week1/` có kết quả thử trên máy này; dataset phòng học chưa có. Git không lưu thư mục rỗng hoặc dữ liệu/weights đã được ignore. `configs/app.yaml` là đặc tả app đầy đủ trong tương lai, chưa được UI/CLI hiện tại đọc. `configs/train_n.yaml` có thể truyền cho lệnh Ultralytics train khi dữ liệu sẵn sàng; chưa có run fine-tune.

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
& .\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO('weights/yolo26n.pt'); print({i:m.names[i] for i in (0,39,67,63)})"
```

Kỳ vọng bốn ID COCO lần lượt là `person`, `bottle`, `cell phone`, `laptop`. Lệnh này chỉ xác nhận có thể import và load model; **chưa xác nhận webcam, FPS, tracking hay chất lượng dự đoán**.

## Việc nhóm còn phải làm

1. Xin giảng viên duyệt phạm vi và xác nhận sĩ số, hạn nộp, phương án pretrained/fine-tune.
2. Thu ảnh/clip có quyền sử dụng, gán nhãn bốn lớp, lập manifest và chia phiên train/val/test.
3. Viết ứng dụng theo các module ở mục 10 của kế hoạch; kiểm camera, tracking, đếm và cảnh báo.
4. Fine-tune, đánh giá trên test độc lập, đo FPS/độ trễ và ghi kết quả thật vào báo cáo.

Baseline ảnh/video/webcam chạy bằng `python -m src.inference.demo --source <ảnh|video|0>`; lệnh cũ `-m src.week1_demo` vẫn dùng được. Web `streamlit run src/app.py` đã triển khai để xem ảnh/frame. Các lệnh `python -m src.cli`, `src.train`, `src.evaluate` trong kế hoạch vẫn chưa có; hướng dẫn train hiện dùng CLI Ultralytics.

Ảnh tham khảo COCO128 không nằm trong Git. Để chạy lại mẫu tuần 1 trên máy khác, tải [COCO128 theo tài liệu Ultralytics](https://docs.ultralytics.com/datasets/detect/coco128/) rồi giải nén đúng vị trí:

```powershell
New-Item -ItemType Directory -Path data\reference -Force | Out-Null
Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip' -OutFile data\reference\coco128.zip
Expand-Archive -LiteralPath data\reference\coco128.zip -DestinationPath data\reference -Force
& .\.venv\Scripts\python.exe -m src.week1_demo --source data\reference\coco128\images\train2017\000000000283.jpg --no-window --save --output-dir runs\week1\image_bottle
```

Hai MP4 trong `demo/smoke/` là slideshow kỹ thuật được tạo cục bộ từ ảnh COCO128 và không nằm trong Git. Chúng không thay video quay thật cần cho G1.

## Nhật ký theo tuần

### Tuần 1 — Baseline detection với YOLO26n pretrained

- Đã chuẩn bị môi trường Python 3.11 CPU, tải checkpoint YOLO26n và chốt bốn lớp: `person`, `bottle`, `cell phone`, `laptop`.
- Đã chạy thử nhận dạng trên ảnh COCO128 và hai video slideshow kỹ thuật; có kết quả phát hiện người/chai, nhưng bỏ sót một số điện thoại/laptop. Đã thêm giao diện web để chọn ảnh/frame, xem bounding box, crop và tải kết quả.
- Sáu bài kiểm thử mã đã chạy thành công. Chưa xác nhận webcam trên máy demo, video quay thật, dữ liệu tự thu hoặc chất lượng trên test độc lập. Chi tiết: [bàn giao G1](WEEK1_G1.md).

### Tuần 2 — Chuẩn bị dataset v1

- Đã ghi quy trình nhận ảnh từ thành viên, kiểm trùng, chuẩn hóa nhãn, review chéo và chia train/validation theo nguồn hoặc phiên; thêm các file CSV mẫu để bàn giao dữ liệu.
- Mục tiêu hiện tại là ít nhất 2.500 ảnh gốc hợp lệ cho train và validation; bộ test sẽ được thu riêng gần cuối dự án. Chưa có dataset v1 được khóa hoặc lần fine-tune nào. Chi tiết: [kế hoạch G2](WEEK2_G2.md).
