# Chấm đủ R01–R11: trạng thái thực tế

**Đã có 8/44 kết quả theo model × kích thước; 2/11 model đủ bốn kích thước.** Không tính baseline vào số11. Trạng thái: `partial_missing_checkpoints`.

Kaggle GPU T4 đã bật và runner validation riêng đã chạy. R04 dùng lại bốn kết quả được xác nhận trước đó; R01 chấm mới trên đủ2.500val/8.359GT. Một forward, không train, không chấm500test. Không có đủ checkpoint để chạy chín model còn lại.

| Model | 512 | 640 | 768 | 960 |
|---|---|---|---|---|
| R01 | Đã chấm | Đã chấm | Đã chấm | Đã chấm |
| R02 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R03 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R04 | Đã chấm | Đã chấm | Đã chấm | Đã chấm |
| R05 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R06 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R07 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R08 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R09 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R10 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |
| R11 | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint | Thiếu checkpoint |

## Kết quả thật trên các checkpoint có sẵn

mAP/P/R/F1 từ GPUFP16, cùng validation. P/R/F1micro ởconfidence0,25, matchIoU0,5; NMS theo lớpIoU0,6. Runtime từ CPUFP32,4threads: cùng40ảnh đãdecode,3lần lặp, model/size xen kẽ ngẫu nhiên. Tính predict+NMS+vẽ; chưa có capture/tracker/UI. Không lấy FPS replay làm FPS webcam.

| Model / size | mAP50% | mAP50–95% | Precision% | Recall% | F1% | CPU p50ms | CPU p95ms | >100ms/120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R01_512 | 65.75 | 45.59 | 78.63 | 46.98 | 58.82 | 48.82 | 69.70 | 1 |
| R01_640 | 71.06 | 50.04 | 79.71 | 55.92 | 65.72 | 61.85 | 97.46 | 5 |
| R01_768 | 74.66 | 53.14 | 80.38 | 61.67 | 69.79 | 78.15 | 109.37 | 14 |
| R01_960 | 76.71 | 54.23 | 80.95 | 64.74 | 71.94 | 114.64 | 172.60 | 87 |
| R04_512 | 65.26 | 44.33 | 73.94 | 49.69 | 59.44 | 49.25 | 69.25 | 1 |
| R04_640 | 71.60 | 49.60 | 76.62 | 58.68 | 66.46 | 62.23 | 91.34 | 3 |
| R04_768 | 75.24 | 52.65 | 77.43 | 64.54 | 70.40 | 80.79 | 115.37 | 17 |
| R04_960 | 77.83 | 54.59 | 78.83 | 68.37 | 73.23 | 112.16 | 167.35 | 85 |

Runtime mới có hai model cùng nằm trong bộ nhớ; RSS toàn tiến trình khoảng660–670MiB, không phải memory riêng của từng model. Timing thay đổi giữa các đợt; ở640 R04 p95 mới khoảng91ms, khác đợt model đơn trước khoảng82ms. Vẫn có frame vượt100ms. Không khẳng định độ trễ webcam đã đạt.

R01/R04 chưa có ứng viên thắng mọi metric. Ở640 R04 cóAP50/F1 cao hơnR01, R01 cóAP50–95 cao hơnR04. Đây chỉ là so sánh hai model; **chưa chọn model cuối cùng khi chưa đủ11**.

## Điểm chặn

Thiếu: R02, R03, R05, R06, R07, R08, R09, R10, R11. Các checkpoint/ZIP đã bị xóa trong lượt dọn trước; log còn số đo và một phần cấu hình nhưng không thay thế được file trọng số. Đã tìm bản sao trên máy và kiểm preflight thật tạiKaggle.

Để chấm đúng11 model cũ cần khôi phục chín checkpoint. Tái train từ checkpoint dự án là phương án tạo model mới, không phải khôi phục đúng model cũ. Đã hỏi người dùng có cho phép phương án này không; chưa có câu trả lời thì khôngtrain, theo chỉ thị trước đó.

Mười tiêu chí được giữ trong báo cáo trước. Runtime R01/R04 đã đo thêm; robustness theo nguồn/ảnh xấu hiện chỉ cóR04, chưa đoR01 hoặc chín model thiếu. Full visual review chưa hoàn tất và camera0 chưa mở được.

[Kết quả trước và10tiêu chí](../webcam_sizes_20261010/RESULTS.md)

![Kaggle xác nhận 2/11 model trước khi dừng GPU](C:/Users/aawm8/Downloads/BTL-AI-Camera-cd8/reports/webcam_all11_20261011/kaggle_validation_summary.png)
