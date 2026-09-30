# Bàn giao G1 — tuần 1

Mốc này chỉ được đánh dấu hoàn thành khi có bằng chứng thật. Xem [kế hoạch](Ke_hoach_De_8_Camera_AI.md) mục 13.

| Điều kiện | Bằng chứng hiện tại | Trạng thái |
|---|---|---|
| Repo, Python 3.11, OpenCV, Ultralytics, YOLO26n | `.venv`, `requirements-demo-cpu.lock.txt`, `weights/yolo26n.pt` | Đã chuẩn bị và import/load được |
| Bốn lớp | `LABELING_GUIDE.md`, `configs/data.yaml.example` | Đã chốt trong tài liệu |
| Ảnh baseline | `runs/week1/image_bottle/` và `image_phone/` có summary/ảnh đã vẽ | Đã chạy; nhận chai, thấy 3 người ở ảnh khác nhưng bỏ sót điện thoại |
| Video baseline | `runs/week1/video_smoke_1/` và `video_smoke_2/` | Đã chạy; clip 1 đóng/mở lại nguồn thành công; chưa thay video quay thật |
| Web xem kết quả | `src/ui/app.py` + `src/inference/detector.py` | Có gallery/upload ảnh, xem frame video, bảng/crop, nhãn tham khảo và tải kết quả; không phải livestream camera |
| Hướng dẫn và test mã | `docs/`, `tests/` | 6 bài kiểm tự động pass; train chỉ được hướng dẫn, chưa chạy |
| Webcam chạy, dừng, mở lại | Lệnh `--source 0`; máy chạy công cụ không thấy webcam 0–2 | Chưa xác nhận trên máy có webcam |
| 20 ảnh hai người gán thử | `data/week1_review_20.csv` | Chờ hai người gán và review thật |
| 30–50 ảnh thử | `data/week1_coco128_manifest.csv` liệt kê 50 ảnh COCO128 | Có mẫu kỹ thuật; kiểm tự động 50/50 file/box hợp lệ. Chưa có ảnh tự thu/review. COCO128 tải về có 4 file không ghép được cặp ảnh/nhãn; đã loại khỏi danh sách |
| Hai video mẫu khác phiên | Chưa được cung cấp | Chưa có |
| Đề cương một trang | `DE_CUONG_TUAN1.md` | Bản nháp, chờ thông tin nhóm/giảng viên |
| Phân công, phê duyệt, hạn nộp | `PHAN_CONG.md` là bảng vai trò chờ tên; chưa có phản hồi giảng viên | Chưa xác nhận |
| Thông số máy, FPS ban đầu | Máy chạy công cụ: Windows 10 build 26200, 16 CPU logic, RAM 13.8 GiB, PyTorch CPU; slideshow 640×480 đạt 13.83 và 12.93 FPS | Chỉ là phép thử kỹ thuật trên máy hiện tại; webcam/máy bảo vệ chưa đo |

YOLO26n pretrained trên COCO128 ở `conf=0.25` đã phát hiện được người/chai trong mẫu thử, nhưng bỏ sót điện thoại và laptop trong các ảnh hiếm đã kiểm. Đây là quan sát phát triển, không phải phép đánh giá chất lượng cuối kỳ. COCO128 chỉ có 5 ảnh điện thoại và 2 ảnh laptop có nhãn trong các cặp đầy đủ của bản tải này.

## Cách chạy G1 trên máy có webcam

Tại thư mục gốc repo:

```powershell
& .\.venv\Scripts\python.exe -m src.week1_demo --source 0
```

Đưa lần lượt người, chai, điện thoại, laptop trước camera. Phím `r` đóng và mở lại camera; phím `q` dừng và giải phóng camera. Chạy lại lệnh để xác nhận camera mở được lần nữa. Mặc định không ghi video; `--save` mới lưu vào `runs/week1/`. Khi đo tự động không mở cửa sổ: `--source 0 --no-window --max-frames 50 --reopen-at 20`.

Với ảnh/video, dùng `--source <đường_dẫn> --no-window --save --output-dir runs/week1/<tên_run>`. Mỗi run nên có thư mục riêng để không ghi đè `summary.json`. `processing_fps` là thông số của **máy và nguồn đang chạy**, không phải lời hứa FPS trên máy khác. Ghi CPU/GPU/RAM, độ phân giải, số frame, FPS và lỗi thực vào bảng bàn giao của nhóm.

## Những việc con người cần xác nhận

1. Gửi [đề cương](DE_CUONG_TUAN1.md) cho giảng viên và ghi lại câu trả lời, sĩ số, hạn nộp.
2. Hai người gán độc lập 20 ảnh, điền [bảng review](data/week1_review_20.csv); bổ sung ảnh tự thu, đặc biệt điện thoại/laptop.
3. Thu hai video ngắn khác phiên trong bối cảnh dự án, kiểm quyền sử dụng và chạy bằng lệnh trên.
4. Quay màn hình demo camera chạy–dừng–mở lại trên máy demo, ghi FPS và thông số máy đó.
