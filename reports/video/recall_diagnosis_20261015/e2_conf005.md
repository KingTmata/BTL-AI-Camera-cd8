# Chạy đối chứng E2 confidence 0,05 — 15/10/2026

> Đính chính phạm vi sau phản hồi người dùng: bảng dưới chỉ gồm sáu file đã có trong `weights/` lúc kiểm, chưa bao gồm sáu ZIP train thêm hai epoch đang ở Downloads. Người dùng cung cấp LIGHT_CONTRAST seed123 có person Recall 78,61%, cao nhất trong nhóm E3 gốc và sáu bản đó. Đã tìm ZIP và trích đúng best.pt để chạy đối chứng riêng; chưa tìm thấy bảng per-class local để xác minh độc lập 78,61%. Không dùng bảng tổng trainer dưới đây để chọn model có person Recall cao nhất.

Người dùng yêu cầu thử trọng số có Recall cao nhất đang có với confidence 0,05. Không đồng nhất Recall tổng với Recall lớp person hoặc Recall trên video lớp học.

## Cơ sở chọn lượt thử

Đã đọc trực tiếp `train_metrics` của sáu checkpoint trong `weights/`. Recall tổng (trainer, không phải Recall tại ngưỡng ứng dụng 0,05):

| Checkpoint | Recall lưu trong checkpoint |
|---|---:|
| E2 MuSGD lr0,0003 | 71,741% |
| E3 AdamW lr0,0001 | 71,019% |
| E3 INVERT seed2026 epoch2 | 70,664% |
| R04 | 67,980% |
| Baseline v05 | 67,013% |
| R01 recovered | 66,653% |

E2/E3 cùng nhóm matrix từ CP07, imgsz1280, cùng dữ liệu trainer. Các checkpoint lịch sử khác có khác biệt protocol; bảng là inventory metadata, không phải chấm lại tất cả bằng một evaluator. Bảng native validation riêng trong báo cáo matrix lại ghi E3 cao hơn E2 (69,7% so 67,6%); không trộn native với trainer để tuyên bố một model thắng tuyệt đối.

Log trainer theo lớp person: E3 R=77,7%, E2 R=77,6%; chưa có số per-class tương đương cho tất cả checkpoint. Vì vậy **không khẳng định E2 có person Recall cao nhất**, chỉ chọn E2 làm ứng viên Recall tổng trainer cao nhất trong sáu file dưới weights có metadata đã đọc. E3 vẫn là đối chứng và mặc định hiện tại.

## Lượt chạy thực tế

Chạy full trên cùng video lớp học 898×506/25FPS/541frame, CPU, E2, imgsz1280, confidence0,05, không thêm NMS. Metadata đã kiểm đúng tên checkpoint và tham số; không âm thầm hạ imgsz. CLI có hiển thị OpenCV sau khi xuất/kiểm tra xong:

```powershell
.\run_video.cmd --source "stock-footage-mature-teacher-monitoring-students-as-they-write-an-exam-in-a-classroom-at-secondary-school-female.webm" --mode full --weights "weights/checkpoint07_matrix_20261012/E2_YOLO26n_Project8_v05_from_CP07_v04_best19_MuSGD_lr0p0003_bestEpoch10_20261012.pt" --confidence 0.05 --imgsz 1280 --device cpu
```

Lượt do trợ lý khởi chạy: `runs/video_recall_e2_conf005/20261015_220139_4a8b891c3e`; console ở `.temp/video_e2_conf005.log`. MP4 chỉ xuất hiện khi finalize thành công, report cuối ghi khi đóng cửa sổ. Không sửa video nguồn, không thay mặc định E3, không sửa checkpoint, không train.

Số box dự đoán và quan sát trên vài frame không đủ để tuyên bố Recall video. Cần ground truth như kế hoạch trong báo cáo chẩn đoán chính.

## Kết quả lượt thử

Đã nhận diện đủ **541/541 frame**, xuất và kiểm tra thành công `annotated.mp4`. Bắt đầu phát ở khoảng **67,7 giây** từ khởi chạy; vẫn dùng cửa sổ native để người dùng xem/tua, chưa coi thời gian đóng cửa sổ là thời gian xử lý.

Tại frame 0/250/500 (0/10/20 giây), E2/conf0,05 trả **0/0/0 box person**. E3 cùng imgsz1280/conf0,05 trong chẩn đoán trước trả **1/2/0**. Đây là số dự đoán, không phải Recall; kết quả không cho thấy lợi ích đổi E2 trên ba mẫu và không đủ để chọn E2 làm mặc định. Giữ E3 mặc định; không che kết quả kém chỉ vì E2 có Recall tổng trainer cao hơn.
