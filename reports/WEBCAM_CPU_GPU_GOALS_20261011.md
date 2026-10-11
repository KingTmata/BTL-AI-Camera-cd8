# Mục tiêu webcam trên CPU và RTX 5060 Laptop — 11/10/2026

**Cập nhật hiện hành:** R04 mặc định, weights giữR04/baselinev05/R01. Streamlit đã có size512–1280, NMS tắt/0,5/0,6, một/ba lượt; mặc định CPU640/một lượt/NMS tắt. Các đoạn “chưa đổiapp” bên dưới là lịch sử. Preview thực dùng fragment0,3giây; preview_fps10 trong YAML chưa điều khiển đường hiển thị này. [Log và giới hạn triển khai](STREAMLIT_INFERENCE_CONTROLS_20261011.md).

## Hai môi trường cần nghiệm thu riêng

- **CPU máy hiện tại:** thông tin registry báo AMD Ryzen 7 8745HS với Radeon 780M. Các số đo CPU local trước đây khác số đo CPU 2 threads trên VM Kaggle; không dùng số VM làm kết luận cho máy này.
- **GPU mục tiêu:** RTX 5060 Laptop 8 GB. Chưa đo trực tiếp và chưa xác nhận runtime CUDA trên máy mục tiêu. Một Tesla T4 là tham chiếu GPU, không mô phỏng chính xác 5060.

Hai môi trường đều phải chạy được cùng ứng dụng webcam, không train lại chỉ để đổi thiết bị. Mỗi ảnh được xử lý dùng một forward. Có thể dùng hai cấu hình inference/runtime khác nhau, nhưng phải giữ nguồn checkpoint, tên lớp và protocol đánh giá rõ ràng. Chưa đổi ứng dụng hoặc trọng số mặc định trong bước ghi mục tiêu này.

## Điều kiện chọn cấu hình

1. Giữ ưu tiên người dùng: độ trễ số 1; mAP50/mAP50–95 số 2; F1 phải cân bằng. Không tự chọn 512 hoặc hạ confidence chỉ vì nhanh hơn nếu làm chất lượng giảm nhiều. Mức giảm F1 chấp nhận được chưa có ngưỡng số cụ thể.
2. Mục tiêu dưới 100 ms mỗi ảnh được xử lý. Đo và báo median/p95/p99/max và số lượt vượt 100 ms; không dùng mean để tuyên bố mọi frame đều đạt. Đây chưa phải cam kết rằng p95 đạt thì toàn bộ frame đạt.
3. Tách **FPS camera**, **FPS inference/tracking**, **FPS preview** và **latency từ frame đến hiển thị**. FPS ước tính T4 không thay cho phép đo webcam thực. Chưa có mục tiêu FPS CPU tối thiểu riêng được người dùng chốt; mục tiêu 30/60 FPS hiển thị phụ thuộc camera, giao diện và cấu hình được chọn.
4. Không tích hàng đợi frame gây trễ; kiểm hành vi giữ frame mới nhất, frame bị bỏ, thứ tự/timestamp và tracker dưới tốc độ xử lý thấp hơn camera. Không tự tắt tracker chỉ để đạt benchmark nhanh.
5. Kiểm Precision, Recall, F1 và hai mAP trên cùng validation, ở đúng size/conf/NMS/precision/backend triển khai. Nếu đổi FP16, ONNX, OpenVINO hay TensorRT, phải đo lại chất lượng và độ trễ; chưa có bằng chứng export nào tốt hơn trên máy đích.
6. Đề xuất phép kiểm chạy liên tục ít nhất 10 phút trên webcam thật cho mỗi cấu hình được chọn: không crash, bộ nhớ không tăng không kiểm soát, FPS/latency không suy giảm đáng kể, theo dõi nhiệt/công suất. Đây là kế hoạch chưa thực hiện, không phải kết quả đã đạt.

## CPU: các bước còn cần làm

- Xác nhận camera mở được; probe trước đó thất bại. Đo cả capture, preprocess, R04 inference, NMS, ByteTrack, vẽ và preview trên CPU Ryzen hiện tại.
- Thử số threads phù hợp bằng phép đo trên cùng clip/ảnh; không mặc định càng nhiều threads càng nhanh.
- Bắt đầu so R04 640, vì F1 ở512 thấp hơn khoảng7,1 điểm trong đánh giá trước. Chỉ giảm size hoặc đổi backend nếu có số đo và so chất lượng đầy đủ.
- Nếu PyTorch CPU chưa đạt, đánh giá ONNX Runtime CPU hoặc OpenVINO sau khi kiểm hỗ trợ và export/parity. Đây là phương án, chưa chọn hay cài đặt.

## GPU 5060: các bước còn cần làm

- Xác nhận laptop cụ thể, driver, CUDA/PyTorch tương thích, device thực và VRAM. Môi trường local hiện dùng torch CPU; nvidia-smi chưa được tìm thấy trong PATH. Thông tin CPU không chứng minh có hoặc không có GPU NVIDIA.
- Đo R04 640 FP32 trước, sau đó FP16 trên chính 5060, batch1 và một GPU. T4 FP32 nhanh/ổn định hơn FP16 trong benchmark vừa rồi, nhưng không suy ra 5060 cũng có thứ hạng đó.
- Ghi VRAM gồm context/runtime, RAM, mức công suất, chế độ cắm điện và nhiệt; không dùng số allocated/reserved của PyTorch thay toàn bộ VRAM tiến trình.
- Chỉ thử export TensorRT nếu cần và đã có parity chất lượng. Không bắt buộc đổi model hay dùng model lớn.

## Giao diện hiện tại cần kiểm riêng

`configs/app.yaml` vẫn dùng `weights/BEST_Project8.pt`, `device: cpu`, `imgsz: 640`, `nms: false`, `tracking.enabled: true`, `source.mode: live_latest`, `runtime.pending_frames: 1`, **`runtime.preview_fps: 10`**. Đây chưa phải cấu hình R04+NMS benchmark mới. Giới hạn preview10 nghĩa là không thể gọi 67,7 FPS xử lý T4 là 67,7 FPS hiển thị của cấu hình hiện tại.

Cần xác nhận cách các cấu hình này được ứng dụng sử dụng, đo preview/dropframe, rồi quyết định profile CPU và GPU riêng. Không thay các giá trị trong bước ghi tài liệu/dọn file này.

## Phụ thuộc chung còn thiếu

- Review nhãn toàn bộ, xử lý6 ảnh cấu trúc cần kiểm và leakage cảnh/video.
- Khôi phục9 checkpoint còn thiếu để đủ R01–R11 ởbốnsize; hiện2/11model,8/44cặp. Không tự tái train khi chưa được cho phép.
- Chốt confidence và profile sau đánh giá; kiểm độc lập cuối cùng với500test vẫn giữ riêng.
- Chỉ nghiệm thu hai profile sau số đo trên từng thiết bị thật. Nếu CPU không đạt chất lượng/tốc độ yêu cầu, ghi rõ giới hạn và phương án thay vì hạ tiêu chí âm thầm.

Nguồn: [tiến trình tổng hợp](PROJECT8_PROGRESS_20261011.md), [benchmark T4 và webcam ước tính](r04_single_t4_latency_20261011/RESULTS.md), [trạng thái đủ11model](webcam_all11_20261011/STATUS.md).
