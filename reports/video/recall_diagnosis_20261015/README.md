# Chẩn đoán bỏ sót người trong video — 15/10/2026

## Phát hiện đã kiểm chứng

Người dùng báo cả full và fast bỏ sót gần như mọi người. Đọc đoạn chat đính kèm, kiểm đường inference/vẽ và tái hiện trên video lớp học gần nhất trong workspace.

Lượt đối chiếu: `runs/video_tui/20261015_214645_d3837d4332/20261015_214646_55d9ce2305`. Nguồn `stock-footage-mature-teacher-monitoring-students-as-they-write-an-exam-in-a-classroom-at-secondary-school-female.webm`, 898×506, 25 FPS, 541 frame. Metadata xác nhận **full, E3, imgsz 1280, confidence 0,25**, không NMS bổ sung, CPU. Cấu hình thực của lượt này chưa phải confidence 0,05 dù mặc định mã hiện tại là 0,05; có thể do TUI giữ đối số từ lúc mở trước đó hoặc override, chưa xác minh nguyên nhân đối số 0,25.

JSONL tại frame 0/250/500 đều không có box người; tổng số box mọi lớp lớn nhất trong cả lượt chỉ 3. Report ghi nhận diện 100% frame. Vì vậy hiện tượng này không chỉ là fast lấy mẫu, GUI không vẽ box hoặc giới hạn 300 box. Class mapping E3 đúng: person=0, đủ 8 lớp; video không lọc riêng bỏ lớp person.

## Thử đối chứng trên ảnh nguồn

Đọc lại trực tiếp frame 0/250/500 bằng OpenCV; gọi Ultralytics độc lập với TUI/backend/vẽ. E3 SHA256 giữ nguyên `9474d7f2b4c869e2101cf9f4f64b0f7f0411b859a8d31d3b2425aba48f479a13`. CPU, một lượt, không augment, lấy kết quả ở conf 0,001 rồi đếm theo từng ngưỡng. Mỗi nhánh nms dùng model mới để tránh nhánh đã fuse. Chi tiết trong `comparison.json`.

Số **box dự đoán person**, không phải số người phát hiện đúng/Recall:

| Frame / giây | Nhánh | conf 0,25 | conf 0,05 | conf 0,01 |
|---|---|---:|---:|---:|
| 0 / 0 | one-to-one, nms=False | 0 | 1 | 3 |
| 250 / 10 | one-to-one, nms=False | 0 | 2 | 4 |
| 500 / 20 | one-to-one, nms=False | 0 | 0 | 1 |
| 0 / 0 | one-to-many, nms=True | 0 | 1 | 3 |
| 250 / 10 | one-to-many, nms=True | 0 | 1 | 4 |
| 500 / 20 | one-to-many, nms=True | 0 | 0 | 1 |

Đổi nhánh không giải quyết được trên mẫu này. Khác với thêm NMS vào đầu ra one-to-one: NMS bổ sung chỉ loại box, không tạo lại ứng viên đã mất. Không đổi cơ chế mặc định theo giả thuyết chưa có bằng chứng.

Đối chứng kích thước trên frame 250, nms=False (chi tiết `sizes_models.json`):

| Model | imgsz | person tại conf 0,25 | person tại conf 0,05 |
|---|---:|---:|---:|
| E3 | 640 | 2 | 3 |
| E3 | 960 | 0 | 2 |
| E3 | 1280 | 0 | 2 |
| E3 | 1920 | 0 | 0 |
| R04 | 640 | 1 | 3 |
| R04 | 960 | 0 | 0 |
| R04 | 1280 | 0 | 1 |
| R04 | 1920 | 0 | 0 |

Nguồn chỉ 898×506; tăng imgsz là phóng lớn, không khôi phục chi tiết quang học đã mất. 640 tốt hơn trên frame này, chưa đủ để chọn mặc định cho toàn bộ video. Hạ conf tăng cả dự đoán sai/box sai vị trí, thấy trực tiếp trong ảnh `E3_640.jpg` và `frame_250_nms_False.jpg`; không coi số box nhiều hơn là Recall tốt hơn.

Log validation lịch sử E3 trong `reports/training/classroom07_matrix_20261012/kaggle_evidence/E3_AdamW_lr1e4_training.log` ghi person P=0,839, R=0,777, mAP50=0,865 trên 282 ảnh/1.491 instances. Đây là chỉ số của protocol/tập validation lịch sử, không phải Recall tại conf 0,05 hoặc trên video này. Kế hoạch dữ liệu lịch sử ghi semantic review chưa hoàn tất và chưa được kết luận cải thiện lớp học.

## Kết luận và bước hợp lý tiếp theo

- Đã tái hiện điểm số person thấp và bỏ sót ngay trong đầu ra model, độc lập OpenCV/TUI; chưa chứng minh được nguyên nhân sâu là riêng domain shift, nhãn hay quá trình fine-tune. Không đổ cho CPU/Streamlit hoặc khẳng định E3 hỏng chỉ từ ba frame.
- Mở lại TUI với `--confidence 0.05` tường minh để loại đối số cũ; việc này không bảo đảm hết bỏ sót. Video đã xuất không tự đổi nhận diện khi sửa mặc định.
- Tạo tập 50–100 frame đại diện nhiều video/cảnh, gán người thật có quy tắc người bị che/cắt khung nhất quán; đo person P/R/F1 tại IoU công bố. So E3/R04 với pretrained COCO person n/s cùng protocol, quét conf và imgsz. Không chọn dựa chỉ số box hoặc một ảnh đẹp.
- Với số liệu đã gán nhãn, phân tích FN theo kích thước, che khuất, blur và góc nhìn. Sửa nhãn/dữ liệu lớp học; chia train/val/test theo video/cảnh để tránh rò rỉ frame lân cận.
- Nếu chọn s, phải dùng pretrained s và train/evaluate phù hợp, không đổi tên hay nhét trọng số E3-n vào s. Chưa chạy training/tải model mới trong chẩn đoán này.
- Tracking/ReID giúp nối theo thời gian nhưng không bảo đảm phát hiện người chưa từng có bằng chứng hoặc ID vĩnh viễn. Detection recall, event recall và ID metrics phải đo riêng. Chỉ quantize sau khi có baseline đạt yêu cầu và đánh giá lại P/R/F1/latency sau export.
- Recall 90–95% là mục tiêu đề xuất, chưa được chứng minh. Không bỏ các trường hợp khó sau khi nhìn kết quả để làm đẹp Recall; định nghĩa vùng/đối tượng đủ điều kiện và báo tỷ lệ loại trừ từ trước.

Không sửa source hoặc mặc định inference trong lượt chẩn đoán này; chỉ thêm bằng chứng/báo cáo. Ba frame và số box không thay thế tập ground truth hoặc nghiệm thu lớp học.

Nguồn đối chiếu: [YOLO26 end-to-end](https://docs.ultralytics.com/guides/end2end-detection/), [tracking](https://docs.ultralytics.com/modes/track/), [model evaluation](https://docs.ultralytics.com/guides/model-evaluation-insights/). Đã kiểm cả source Ultralytics 8.4.165 đang cài.
