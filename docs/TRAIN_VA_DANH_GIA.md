# Hướng dẫn train về sau

> Cập nhật 02/10/2026: dùng project8 và release YAML data/dataset/v1/data.yaml; pipeline/evaluator tuần 2 đã có. Xem [workflow hiện hành](WEEK2_WORKFLOW.md). Các lệnh/kết quả smoke COCO128 cũ là lịch sử kỹ thuật, không phải baseline dataset nhóm.

**Hiện tại chưa chạy fine-tune.** Các lệnh bên dưới là quy trình để dùng khi dữ liệu đã sẵn sàng. Mở web/chạy predict không cập nhật weights.

## 1. Ai thực hiện và file nào liên quan?

- Bộ train nằm trong thư viện **Ultralytics đã cài**. Hiện nhóm dùng `yolo.exe detect train`; chưa có `src.training.train` của nhóm.
- `src/training/` dành cho script điều phối khi cần thêm sau này.
- `configs/train_n.yaml` chứa hyperparameter; `configs/data.yaml` chứa vị trí và thứ tự lớp.
- `scripts/data/week2.py` kiểm/khóa dataset; `scripts/data/smoke_week2.py` kiểm phần mềm khi cần, không phải baseline.
- `weights/yolo26n.pt` là điểm bắt đầu. Kết quả train đi vào `runs/train/<tên_run>/`, không ghi đè weights gốc.

## 2. Chuẩn bị dataset phòng học tám lớp

```text
data/dataset/
├── images/train/s01_0001.jpg
├── images/val/s10_0001.jpg
├── images/test/s15_0001.jpg
├── labels/train/s01_0001.txt
├── labels/val/s10_0001.txt
└── labels/test/s15_0001.txt
```

Một ảnh tương ứng một TXT cùng stem. Dùng ID **0 person, 1 table, 2 chair, 3 laptop, 4 cell phone, 5 backpack, 6 book, 7 cup**. Một hộp là `class_id x_center y_center width height` chuẩn hóa 0–1. Ảnh âm tính có TXT rỗng theo quy ước nhóm. Thiếu TXT không được tự hiểu là ảnh âm tính.

Chia theo phiên quay trước khi train: cùng phiên/gần trùng không đi qua nhiều split. Chỉ augmentation tập train. Test chưa dùng để chọn confidence, epoch, model hay sửa logic. Ghi nguồn, quyền, session và hash trong manifest, review nhãn theo [LABELING_GUIDE.md](../LABELING_GUIDE.md).

`configs/data.yaml` đã có đường dẫn tuyệt đối cho máy hiện tại. Trên máy khác/Colab sao từ `.example` và sửa `path` tới thư mục dataset thực. Không lấy file Windows có `C:/...` dùng nguyên trong Colab. COCO128 gốc dùng 80 ID, không đưa trực tiếp vào YAML tám lớp. Chuyển sang tám lớp đòi hỏi lọc/đổi ID nhãn và giữ đủ mọi hộp thuộc tám lớp.

## 3. Kiểm dataset trước khi chủ động train

Thư viện ảnh và script smoke COCO128 đã dọn theo yêu cầu. Kiểm dataset chính bằng manifest hiện hành:

```powershell
& .\.venv\Scripts\python.exe scripts/data/week2.py check --manifest data/manifest.csv
```

Check chỉ xác nhận kỹ thuật; cần hoàn thiện nhãn tám lớp, review và khóa release trước training chính thức. Không dùng test để kiểm/chọn cấu hình train. Các kết quả smoke cũ là lịch sử, không phải baseline chất lượng.

## 4. Fine-tune dữ liệu phòng học trên máy CPU

Khi train/val có dữ liệu thật đã kiểm:

```powershell
& .\.venv\Scripts\yolo.exe detect train cfg=configs/train_n.yaml device=cpu workers=0 batch=2 cache=False nms=False
```

`cfg` dùng model/data, 50 epoch tối đa, ảnh 640, seed 42, optimizer MuSGD và tên run đã ghi trong YAML. `batch=2` ở lệnh ghi đè batch 8 trong file để bắt đầu trên máy RAM thấp; không bảo đảm tốc độ/đủ RAM cho mọi máy. Sau 1–2 epoch, xem thời gian mỗi epoch để quyết định chuyển sang GPU. `patience=10` có thể dừng sớm; không cần chạy đủ 50 để gọi là thành công.

MuSGD là lựa chọn tường minh theo kế hoạch. Nếu thí nghiệm dùng AdamW, tạo tên run và ghi lý do khác; không thay optimizer giữa các run rồi mô tả như cùng cấu hình. File `args.yaml` của mỗi run ghi cấu hình thực sau override và mặc định.

Nếu máy thành viên có PyTorch GPU phù hợp, kiểm `torch.cuda.is_available()` rồi dùng `device=0`, batch theo bộ nhớ. Cài CPU hiện tại trả False; đổi `device=0` trên bản CPU không tạo ra GPU.

## 5. Fine-tune trên Colab GPU

Đưa **mã, configs, weights và dataset được phép dùng** sang Colab/Drive; loại `.venv`, `.python`, cache và video không cần thiết. Mã và tài liệu được push trên nhánh `codex/week1-baseline`; clone đúng nhánh này để lấy cập nhật tuần 2. Dataset và weights được Git bỏ qua, cần chuyển riêng sang Colab/Drive.

Trong notebook, chọn runtime GPU, cài Ultralytics cùng phiên bản rồi kiểm thiết bị:

```python
%pip install ultralytics==8.4.165
import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.cuda.is_available(), "Phiên này chưa có GPU; kiểm runtime trước khi train"
print(torch.cuda.get_device_name(0))
```

Ví dụ nếu mã đã ở `/content/BTL-AI-Camera-cd8` và dataset ở `/content/classroom_v1`:

```python
from pathlib import Path
import yaml
from google.colab import drive
drive.mount('/content/drive')
%cd /content/BTL-AI-Camera-cd8

data_cfg = yaml.safe_load(Path('configs/data.yaml.example').read_text())
data_cfg['path'] = '/content/classroom_v1'
Path('configs/data.yaml').write_text(yaml.safe_dump(data_cfg, sort_keys=False))

from ultralytics import YOLO
args = yaml.safe_load(Path('configs/train_n.yaml').read_text())
model_path = args.pop('model')
args.update(device=0, batch=8, workers=2, cache=False, nms=False, save_period=5,
            project='/content/drive/MyDrive/BTL-AI-Camera/runs', name='classroom_v1_26n_gpu_seed42')
model = YOLO(model_path)
model.train(**args)
```

Đường dẫn trong ví dụ phải thay theo nơi bạn thực sự đặt dữ liệu. Drive giữ output qua các lần runtime bị xóa; ghi ra Drive có thể chậm. Kiểm `last.pt`/`best.pt` thật sự có ở Drive sau epoch, lưu `args.yaml`, `results.csv`, phiên bản dataset và `pip freeze` của Colab riêng. Không dùng lock Windows CPU để ghi đè PyTorch GPU trong notebook. Colab Free không bảo đảm GPU/thời gian chạy.

## 6. Resume và dùng model sau train

Nếu run **bị ngắt trước khi hoàn tất**, dùng `last.pt` còn trạng thái optimizer/epoch. Trên máy local, ví dụ:

```powershell
& .\.venv\Scripts\yolo.exe detect train model=runs/train/classroom_v1_26n_seed42/weights/last.pt resume=True device=cpu workers=0
```

Lấy đúng tên thư mục run thực vì Ultralytics có thể thêm hậu tố để tránh ghi đè. Với Colab dùng `YOLO('/content/drive/MyDrive/.../weights/last.pt').train(resume=True, device=0)`. Nếu đường dẫn dataset thay sau restart, tạo lại đúng đường dẫn hoặc truyền `data=` trỏ tới **cùng phiên bản dữ liệu**. Checkpoint run đã hoàn tất có thể đã bỏ optimizer; tiếp tục fine-tune từ weights đó là run mới, không gọi là resume đầy đủ.

`best.pt` là checkpoint tốt nhất theo tiêu chí validation của trainer; `last.pt` là trạng thái gần nhất. Để xem `best.pt` trong web, đặt nó dưới `runs/train/<tên_run>/weights/best.pt` hoặc copy thành tên rõ ràng trong `weights/`, rồi tải lại trang và chọn checkpoint. App kiểm bốn tên/lớp trước khi suy luận.

## 7. Validation và test sau fine-tune

Với checkpoint **đã train đúng project8**, chấm validation:

```powershell
& .\.venv\Scripts\yolo.exe detect val model=runs/train/classroom_v1_26n_seed42/weights/best.pt data=configs/data.yaml split=val imgsz=640 device=cpu nms=False project=runs/val name=classroom_v1_val
```

Chỉ sau khi khóa model/ngưỡng/protocol mới đổi `split=test` và đặt tên output khác. Ghi `imgsz`, head `nms`, conf dùng cho AP, `max_det` và phiên bản evaluator. Không lấy conf hiển thị 0.25 làm conf tính AP một cách tự động.

**Không chấm `yolo26n.pt` 80 lớp trực tiếp trên YAML tám lớp rồi so với checkpoint4:** ID các lớp khác nhau. Evaluator chung đã có ở src/evaluation.py; dùng lệnh trong docs/WEEK2_WORKFLOW.md với manifest release đã khóa để so sánh pretrained/fine-tuned. Web hiện chỉ phục vụ xem và tìm lỗi, không cung cấp mAP.

Nguồn tham khảo: [Ultralytics Train](https://docs.ultralytics.com/modes/train/), [Validation](https://docs.ultralytics.com/modes/val/), [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/), [Colab FAQ](https://research.google.com/colaboratory/faq.html). Cú pháp đã đối chiếu với Ultralytics 8.4.165 đang cài; chất lượng, thời gian train và cấu hình GPU chưa được kiểm chứng bằng run thật của nhóm.
