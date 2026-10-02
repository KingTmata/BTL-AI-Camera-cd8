# Protocol đánh giá detection v1 — project8

Trạng thái: công cụ đã triển khai; chưa có baseline validation dataset nhóm. Áp dụng quyết định lớp ngày 02/10/2026. Hướng dẫn lệnh: [workflow tuần 2](docs/WEEK2_WORKFLOW.md).

## Dataset và nhãn

Chỉ chấm release đã khóa, checksum hợp lệ và nhãn project8 theo thứ tự `person, table, chair, laptop, cell phone, backpack, book, cup`. Validation dùng chọn mô hình/cấu hình; test độc lập dùng sau khi khóa các lựa chọn. Giữ cả ảnh âm tính. Không augmentation validation/test.

Pretrained COCO80 mapping `0→0, 60→1, 56→2, 63→3, 67→4, 24→5, 73→6, 41→7`. Model fine-tuned phải có đúng tám tên/thứ tự, mapping identity. Profile khác bị từ chối. COCO `dining table` chỉ là đối chứng gần đúng cho lớp `table` rộng hơn; không tuyên bố hai phạm vi đồng nghĩa. Nhóm chọn `book` gồm sách/vở; ghi khác biệt nguồn nhãn nếu ảnh nguồn chỉ gán sách.

## Inference cố định cho bảng chính

`imgsz=640`, `conf=0.001`, `nms=True`, `iou=0.70`, `max_det=300`, `agnostic_nms=False`, `augment=False`; ghi device và phiên bản thực tế. Model gọi predict riêng từng ảnh để giữ thứ tự cố định và không giữ toàn bộ ảnh trong RAM. Hộp xyxy pixel chuyển sang xywh pixel, raw class chuyển về project ID trước khi chấm.

Confidence giao diện 0.25 không dùng thay confidence tính AP. NMS-free (`--no-nms`) là một thí nghiệm khác; cả pretrained và fine-tuned phải dùng cùng head trong cùng bảng. Phiên bản Ultralytics đã cài dùng `nms` để chọn head trước fusion; xem [tài liệu Ultralytics](https://docs.ultralytics.com/guides/end2end-detection/).

## Bộ chấm

COCO bbox evaluator `pycocotools==2.0.11`: tám catIds 0–7, IoU 0.50:0.05:0.95, area all, maxDets `[1,10,100]`. Inference tối đa 300 hộp nhưng metric chính giới hạn 100 hộp/ảnh/lớp theo COCO. Xuất mAP50, mAP50–95, AP từng lớp, số GT box, ảnh âm tính và số dự đoán. Lớp không có GT trả `null`, không ngầm tính AP=0 hay tuyên bố đủ dữ liệu tám lớp. [COCOeval chính thức](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

## Hồ sơ mỗi lần đo

`ground_truth.json`, `predictions.json`, `metrics.json`, `run.json` chứa mapping, protocol, split, checkpoint SHA-256, manifest SHA-256, release SHA-256, Python/OS và phiên bản Ultralytics/Torch/NumPy/COCO evaluator. Lưu nguồn checkpoint và phân tích overlap pretrained trong data card. Kết quả trên ảnh thuộc nguồn pretrained đã học không chứng minh tổng quát sang phòng học.

Đã kiểm bằng ca nhân tạo: đúng hộp tất cả lớp AP≈1; sai lớp AP=0; empty predictions hợp lệ; FP trên ảnh âm tính giảm AP; hộp dư không tạo thêm TP; mapping COCO/project cho cùng vật thống nhất. Dữ liệu nhân tạo không đưa vào bảng baseline thật.

Protocol này chấm detection ảnh. Tracking/đếm/cảnh báo cần protocol sự kiện riêng và video thật; chưa công bố IDF1/HOTA/MOTA vì chưa có GT ID/hộp theo frame.
