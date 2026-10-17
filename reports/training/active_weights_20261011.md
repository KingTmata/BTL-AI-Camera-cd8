# Ba trọng số hiện hành — 11/10/2026

Theo yêu cầu giữ ba trọng số và đặt mặc định, thư mục `weights/` hiện chỉ còn ba file `.pt`:

| Vai trò | File | Lý do giữ |
|---|---|---|
| **Mặc định Streamlit: R04** | `weights/campaign_20261010/R04_YOLO26n_Project8_v05_finetuned_default.pt` | Fine-tune từ baseline v05; ứng viên webcam hiện tại, đã có fullval bốn size và benchmark CPU/một T4 |
| Baseline v05 | `weights/kaggle/Baseline_YOLO26n_Project8_v05_20261009.pt` | Baseline dự án đã học dữ liệu, không phải pretrained COCO; AP95 ở640 cao hơnR04 trong phép chấm GPU đã lưu |
| R01 | `weights/recovered_20261010/R01_YOLO26n_Project8_v05_recovered_epoch0.pt` | Checkpoint epoch0 đã khôi phục và chấm đủbốnsize, cần đối chứng tiếp |

Đây là ba ứng viên còn có file và được chấm trong cùng đợt bốn size, không phải top3 toànR01–R11. Chín checkpoint còn thiếu vẫn chưa có; không tái train.

Theo yêu cầu đặt tên dễ nhận biết, đã đổi tên cả ba file ngày11/10/2026, giữ nguyên nội dung/SHA256, cập nhật đường dẫn ở Streamlit, cấu hình app/train và evaluator. Dropdown hiển thị tên ngắn có vai trò thay vì đường dẫn dài. `epoch0` là tên checkpoint khôi phục (epoch đầu được đánh số0), không có nghĩa chưa train. NMS, size và số lượt suy luận là tùy chọn nên không gắn vào tên trọng số. [Ánh xạ tên cũ/mới và SHA256](weights_renamed_20261011.json). Các log và JSON phép đo trước đổi tên giữ đường dẫn lịch sử để bảo toàn bằng chứng; dùng ánh xạ này để tìm file hiện tại.

Đã xóa7file khỏi `weights/`: alias v04 `BEST_Project8.pt`, bảnv03 ngày08/10, bảnv04 ngày08/10, pretrained COCO `yolo26n.pt`, best ngày06/10, lastv05 ngày09/10 và checkpoint ngày05/10. Lastv05 chưa được xếp hạng bằng protocol mới; được bỏ để tuân thủ giới hạn bafile, không khẳng định đã chứng minh nó yếu hơn ởmọi chỉ số. Tổng59.120.607byte≈56,38MiB. [Danh sách và SHA256 trước khi xóa](cleanup_20261011/weights_completed.json).

Đã đổi mặc định ở `configs/app.yaml` và bộ chọn Checkpoint Streamlit sangR04. Dropdown chỉ đọc thư mục `weights/`, không thêm checkpoint lịch sử từ `runs/train`. Đường dẫn mặc định evaluator và `configs/train_n.yaml` cũng trỏR04 để không dùng pretrained vừa xóa. Đây là cập nhật đường dẫn; không chạytrain/evaluate hay đổiprecision/device/NMS trong bước này.

Streamlit hiện mặc định CPU640, một lượt và không thêmNMS; đã có bộ chọn size512–1280, NMS0,5/0,6 và một/ba lượt trên cùng trọng số. Ba lượt bắt buộc NMS để gộp. Đã nạpR04 và chạy ba lượt qua giao diện localhost; chưa có profileGPU hoặc nghiệmthuwebcam. [Log triển khai](../webcam/streamlit_inference_controls_20261011.md). Các báo cáo cũ ghi alias/v04 là lịch sử, không còn mô tả default mới.

Checkpoint lịch sử ngoài `weights/` không bị xóa trong bước này. Pretrained COCO80 không còn trongweights; tác vụ xuất gói baselineCOCO trong `src/classroom_ai/training/kaggle_export.py` sẽ cần khôi phục nó nếu sau này được yêu cầu. Không dùngR04 tám lớp giả làm baselineCOCO80.

Đã kiểm đúng ba file còn lại, SHA256 và nạp Ultralytics đủtám lớp; cú phápPython/cấu hình default hợp lệ. Khôngtrain/chấmtest. Bước tiếp theo: sochất lượng/latency đầyđủ, nghiệmthu webcamCPU vàRTX5060 thật theo [mục tiêu hai môi trường](../webcam/webcam_cpu_gpu_goals_20261011.md).
