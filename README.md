# BTL AI Camera — chuẩn bị môi trường

Repo hiện có **baseline tuần 1** theo [kế hoạch](Ke_hoach_De_8_Camera_AI.md): YOLO26n pretrained trên ảnh/video/webcam cho bốn lớp `person`, `bottle`, `cell phone`, `laptop`. Chưa có ứng dụng hoàn chỉnh, dữ liệu tự thu, fine-tune, tracking, đếm hay cảnh báo. Xem [bàn giao G1](WEEK1_G1.md) để biết phần đã kiểm và phần còn thiếu.

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
├── src/week1_demo.py              # Baseline detection tuần 1; chưa là app hoàn chỉnh
├── tests/                         # Chưa có test vì chưa có logic
├── weights/yolo26n.pt             # Checkpoint pretrained, không commit
├── requirements.in
├── requirements-demo-cpu.lock.txt
└── THIRD_PARTY.md
```

Các thư mục ảnh/video và `runs/` đang trống; Git không lưu thư mục rỗng. Các tệp README trong thư mục cấp trên giữ lại ý nghĩa của từng nơi. File cấu hình ứng dụng và train là **đặc tả** từ kế hoạch, chưa phải lệnh chạy được của dự án.

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

Baseline ảnh/video/webcam tuần 1 chạy bằng `python -m src.week1_demo --source <ảnh|video|0>` từ thư mục gốc. Lệnh kiểm camera chi tiết ở [WEEK1_G1.md](WEEK1_G1.md). Các lệnh `python -m src.cli`, `src.train`, `src.evaluate` và `streamlit run src/app.py` trong kế hoạch vẫn là **giao diện dự kiến** cho các tuần sau.

Ảnh tham khảo COCO128 không nằm trong Git. Để chạy lại mẫu tuần 1 trên máy khác, tải [COCO128 theo tài liệu Ultralytics](https://docs.ultralytics.com/datasets/detect/coco128/) rồi giải nén đúng vị trí:

```powershell
New-Item -ItemType Directory -Path data\reference -Force | Out-Null
Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip' -OutFile data\reference\coco128.zip
Expand-Archive -LiteralPath data\reference\coco128.zip -DestinationPath data\reference -Force
& .\.venv\Scripts\python.exe -m src.week1_demo --source data\reference\coco128\images\train2017\000000000283.jpg --no-window --save --output-dir runs\week1\image_bottle
```

Hai MP4 trong `demo/smoke/` là slideshow kỹ thuật được tạo cục bộ từ ảnh COCO128 và không nằm trong Git. Chúng không thay video quay thật cần cho G1.
