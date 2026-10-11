# Chọn phương án theo 10 tiêu chí cho webcam640

**Bản cập nhật hiện hành:** [bốn kích thước và 10 mục](../webcam_sizes_20261010/RESULTS.md). Người dùng ưu tiên độ trễ, nhưng yêu cầu F1 cao và không giảm F1 quá nhiều để đổi lấy tốc độ. Đã có kết quả tài nguyên, robustness và Kaggle cho R04/baseline ở 512/640/768/960; camera thực chưa mở được. Phần dưới là ghi nhận trước khi bổ sung những phép đo này.

Quyết định10/10/2026: chọn **checkpoint R04 + một lượt model ở640 + class-aware NMS0.6**, conf0.25, làm phương án hiện tại trong các cấu hình640 đã đo. Giữ trọng số rawR04, NMS là cấu hình hậu xử lý; không phải wrapperTTA R04 cộng ba lượt. Chưa áp dụng vàoapp, chưatrain.

Không có chứng cứ phương án này thắng toàn bộR01–R11: các checkpoint khác chưa sẵn để chấm640. Các tiêu chí chưa đo là chưa biết, không gán điểm0 hoặc tự cho là đạt. Không cộng10chỉ số có đơn vị khác nhau thành một điểm tùy ý.

| # | Tiêu chí | R04 một lượt + NMS0.6 ở640 | Vai trò chọn |
|---|---|---|---|
|1|Precision|76,55%, micro tạiconf0.25/IoU0.5|Giảm báo nhầm|
|2|Recall|58,73%, cùngthreshold|Giảm bỏ sót; vẫn cần cải thiện|
|3|mAP50|71,58%|Chất lượng phát hiện tổng thể|
|4|mAP50–95|49,62%|Độ chính xác vị tríbox|
|5|F1|66,46%, cùngthreshold|Ưu tiên cân bằngprecision/recall|
|6|Latency|Predict+NMS p50/p95=48,87/60,69ms|Replay40ảnh, chưa đo toànframe webcam|
|7|FPS và độtrễ frame thực|Chưa đo|Không tínhFPS từ nghịch đảo latency ảnh|
|8|RAM/VRAM và tảiCPU/GPU|Chưa benchmark|Chưa kết luận tài nguyên khiapp chạy lâu|
|9|Cân bằng các lớp|AP50 thấp nhấtcell phone53,10%; book62,47%|Không đểmAP trung bình che lớp yếu|
|10|Robustness và ổn định|Chưa kiểm đủ thiếu sáng/nhòe/che khuất/cảnh mới/seed; có nhóm cảnh trùngsplit|Chưa xác nhận khái quát ngoài validation hiện tại|

Điều kiện bắt buộc:640, chỉ1forward, mục tiêu xử lýframe dưới100ms trênphầncứng đích. Trong hai cấu hìnhR04, NMS làmF1 tăng từ64,16% lên66,46%, AP50 tăng69,48% lên71,58%, giảm552FP nhưng mất10TP. Chi phíNMS trung vị0,42ms. Chọn vì lợi ích cân bằng đã đo, không tuyên bố thắng mọi tiêu chí; recallmicro giảm0,12điểm phần trăm.

Thứ tự ra quyết định khi có thêm checkpoint: loại phương án không đáp ứng1forward640 hoặc yêu cầu latency thực; soP/R/F1 ở cùngconf; ưu tiênF1, kiểm lớp yếu vàrobustness, rồi dùngmAP50–95 vàlatency làm đối chiếu thêm. Nếu robustness hoặcFPS chưa đo, lựa chọn chỉ là phương án hiện tại, chưa nghiệm thu cuối. Ngưỡngconf0.25 cố định cho phépso hiện tại; chưa tối ưu threshold bằngtest.

Tại1280 lịch sử,R07 dẫnAP/recall nhưngR02 dẫnprecision; không cómodel thắng tất cả chỉ số đó. Không suy thứ hạng640 từ1280. Nếu khôi phục được bảnR07/R02 thật, chấm cùngprotocol640 trước khi thay lựa chọn. Khôngtrain lại đểkhôi phục.

Nguồn đo: [RESULTS.md](RESULTS.md), [comparison.json](R04/comparison.json), [latency_nms_0.6.json](R04/latency_nms_0.6.json). Reviewnhãn chưa xong; chưa dùngtest500.
