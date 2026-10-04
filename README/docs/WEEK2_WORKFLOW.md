# Tuần 2 — Hướng dẫn nhận, kiểm và khóa dataset

Quyết định 02/10/2026: tám lớp, thứ tự cố định bên dưới; `table` bao gồm các loại bàn, `book` gồm sách và vở. Mục tiêu hiện có trong `WEEK2_G2.md` là ≥2.500 ảnh phát triển hợp lệ, train/val khoảng 80/20; Hiện giữ riêng 130 ảnh test nguồn Laptop; test đủ tám lớp/phòng học độc lập tiếp tục bổ sung cuối dự án. Tỷ lệ có thể lệch vì giữ nguyên phiên/nhóm. Dataset chính draft hiện có 1.800 ảnh: 1.310 train, 360 validation và 130 test; nhãn/review chưa hoàn tất.

| ID nhóm | Nhãn | COCO80 ID | Phạm vi |
|---:|---|---:|---|
| 0 | person | 0 | Người |
| 1 | table | 60 dining table | Mọi loại bàn theo quy chuẩn nhóm; COCO là đối chứng gần đúng |
| 2 | chair | 56 | Ghế |
| 3 | laptop | 63 | Laptop |
| 4 | cell phone | 67 | Điện thoại |
| 5 | backpack | 24 | Balo |
| 6 | book | 73 | Sách/vở theo quy chuẩn nhóm |
| 7 | cup | 41 | Cốc; không gồm chai |

Không dùng trực tiếp nhãn COCO80 trong dataset nhóm; nhãn cuối phải dùng ID 0–7. Nhãn cũ bốn lớp có bottle phải được gán/chuyển lại, không coi đổi YAML là đã đổi nhãn.

## 1. Nhận từng nguồn/phiên

```text
data/raw/member_a/session_01/
  images/anh001.jpg
  labels/anh001.txt
  SOURCE.md
```

Nhóm chuẩn hóa nhãn về YOLO detection: `class_id cx cy width height`, tọa độ 0–1. Mỗi ảnh có TXT cùng tên; TXT rỗng chỉ dùng khi đã xác nhận ảnh âm tính. Chuẩn hóa hướng EXIF trước khi gán nhãn, không xoay ảnh sau khi đã vẽ hộp. `SOURCE.md` ghi URL/nguồn, quyền, người giao, phiên, cách gán nhãn và split gốc.

Tạo manifest chờ duyệt cho từng nguồn/phiên:

```powershell
& .\.venv\Scripts\python.exe scripts/data/week2.py inventory --images data/raw/member_a/session_01/images --labels data/raw/member_a/session_01/labels --source-id member_a --session-id session_01 --output data/raw/member_a/session_01/manifest.csv
```

Lệnh không tạo nhãn hay tự đánh dấu approved. Nếu có nhiều phiên, chạy riêng từng phiên rồi gộp CSV thành `data/manifest.csv`, chỉ giữ một header. Ảnh thiếu nhãn vẫn được kê và bước check sẽ báo lỗi; nhãn không có ảnh bị inventory từ chối.

Điền metadata thật: `source_id`, `license_or_consent`, `session_id`, `split_group_id`, `annotator`, `annotation_status`, `review_status`, `reviewer`, `near_duplicate_status`, `needs_review`. `near_duplicate_status=approved` chỉ sau khi nhóm đã xem trùng/gần trùng trên dữ liệu đã gộp. Không bắt buộc viết/chạy công cụ tìm gần trùng.

Các ảnh cùng phiên phải có cùng source/session; các nguồn có chung phiên hoặc ảnh liên quan cần chung `split_group_id` hoặc `duplicate_group_id`. Script hợp các ràng buộc thành nhóm liên thông. Nếu nguồn không có session thì ghi `unknown` và giữ nhóm nguồn/cảnh đã kiểm; không tự bịa phiên. Nguồn quá lớn chỉ có một nhóm sẽ không chia được train/val: cần cung cấp phiên/cảnh độc lập hoặc giữ split chuẩn của nguồn.

## 2. Kiểm toàn bộ ảnh trong manifest

```powershell
& .\.venv\Scripts\python.exe scripts/data/week2.py check --manifest data/manifest.csv --output runs/week2/validation.json
```

Kiểm giải mã ảnh, cặp tên ảnh/TXT, ID nguyên 0–7, năm cột nhãn, số hữu hạn, hộp có diện tích và nằm trong ảnh, hộp lặp y hệt, ảnh trùng tuyệt đối, đường dẫn trong repo, checksum có sẵn và rò rỉ nhóm giữa split. Báo số ảnh/box từng lớp và ảnh âm tính. Exit code 1 khi có lỗi; file header mẫu không được coi là dataset hợp lệ.

Phạm vi là 100% ảnh **được kê trong manifest**. Dùng inventory để kê toàn bộ thư mục bàn giao; check không tự tìm ảnh bị bỏ sót ngoài danh sách. Không tự xóa ảnh trùng: đọc lỗi, giữ bản có nguồn/nhãn đúng rồi sửa manifest.

Máy không xác nhận được nhãn thiếu hoặc hộp bao đúng vật. Người thứ hai review mục tiêu 20%, tối thiểu 10% mỗi split; phải có mẫu review cho mọi lớp xuất hiện. Review thêm tất cả ảnh âm tính và `needs_review=yes`. Reviewer phải khác annotator. Ghi log thật ở `data/review.csv`, nguồn ở `data/sources.csv`; bản phát hành kiểm trạng thái trên manifest, không xác thực danh tính hay thay thế log review.

## 3. Chia theo nhóm

```powershell
& .\.venv\Scripts\python.exe scripts/data/week2.py split --manifest data/manifest.csv --output data/manifest_split.csv --val-fraction 0.2 --seed 42
```

Giữ nguyên phiên, nhóm gần trùng được thành viên xác định và split chuẩn trong `original_split` (`train`, `val`, `test`). Split hiện có không bị tự ghi đè; xung đột phải xử lý trước. Thuật toán cân bằng số ảnh theo nhóm; không bảo đảm tỷ lệ chính xác hay đủ lớp. Bước lock sẽ từ chối nếu train hoặc val thiếu bất kỳ lớp nào. Nếu cần điều chỉnh, chỉnh cả nhóm và chạy check lại.

## 4. Khóa một phiên bản mới

```powershell
& .\.venv\Scripts\python.exe scripts/data/week2.py lock --manifest data/manifest_split.csv --release data/dataset/v1 --seed 42
& .\.venv\Scripts\python.exe scripts/data/week2.py verify --release data/dataset/v1
```

Lock chỉ nhận ảnh/nhãn hợp lệ, có quyền, annotation approved, xác nhận gần trùng sau gộp và review đạt yêu cầu. Nó sao chép ảnh/nhãn sang `images/{train,val}` và `labels/{train,val}`, đổi stem theo image_id để tránh ghi đè, lưu manifest có hash, danh sách split, YAML, data card thống kê và checksum toàn bộ file. Không ghi đè release đã tồn tại. Release lỗi giữ dấu `INCOMPLETE` để người dùng kiểm; verify từ chối dùng.

Checksum phát hiện file bị sửa, thêm hoặc xóa; không chứng minh nhãn đúng hay nguồn độc lập. Hash gốc của `checksums.json` phải được lưu vào hồ sơ run/bàn giao để nhận diện phiên bản. Khi sửa dữ liệu tạo v2 và chấm lại, không sửa v1 rồi giữ metric cũ.

YAML sinh nằm ở `data/dataset/v1/data.yaml`, chỉ thêm `test` khi release có test thật. Dùng đúng YAML này cho train thay vì `configs/data.yaml` chưa gắn release:

```powershell
& .\.venv\Scripts\yolo.exe detect train cfg=configs/train_n.yaml data=data/dataset/v1/data.yaml epochs=1 batch=2 device=cpu workers=0 name=dataset_v1_smoke
```

Train thử thuộc tuần 3 và hiện chưa chạy. Dataset chính project8_v0.2 đã có 1.800 ảnh, còn hoàn thiện nhãn/review trước khi khóa release; lệnh trên dành cho release v1 sau khi lock, không phải bằng chứng đã train.

## 5. Chấm baseline validation

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.lock.txt
& .\.venv\Scripts\python.exe -m src.evaluation --manifest data/dataset/v1/manifest.csv --weights weights/yolo26n.pt --output reports/results/v1_pretrained_val --split val
```

Evaluator kiểm checksum và release trước/sau inference, giữ ảnh âm tính, chuyển COCO80 hoặc project8 về cùng ID nhóm rồi chấm COCO bbox. Output gồm GT, predictions, metrics và run metadata. Không ghi đè thư mục đánh giá cũ. Checkpoint phải có sẵn; lệnh không tự tải.

Sau fine-tune, dùng cùng lệnh với `--weights .../best.pt` và output mới. Cùng manifest, protocol và head để so sánh. `--no-nms` là thí nghiệm head khác, ghi thành bảng riêng. Chi tiết ở `EVALUATION_PROTOCOL.md`.

## 6. Hồ sơ và video

Điền `data/DATA_CARD.md` bằng nguồn/quyền, bối cảnh, thống kê, cách review, split và hạn chế thật. Data card sinh trong release có thống kê nhưng chưa thay phần phân tích nguồn/bối cảnh/pretraining overlap của nhóm.

Chuẩn bị 3–5 video_dev quay liên tục thật và sau này 5–10 video_test từ phiên độc lập. Kê vào `data/videos.csv` theo mẫu và gán sự kiện theo `docs/VIDEO_ANNOTATION.md`. Slideshow và dữ liệu nhân tạo chỉ kiểm công cụ, không phải bằng chứng hoàn thành G2.
