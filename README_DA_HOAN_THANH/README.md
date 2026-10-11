> **Dataset hiện hành08/10/2026:**v0.5 đủ15.000 ảnh (12.000train/2.500val/500test), tất cảfull_review_pending, chưatrainv05. Weight/metricv04 bên dưới làlịch sử. XemREADME gốc vàtasks/plan.md.

**Cập nhật v0.4 — 08/10/2026:** đã nhập ZIP full train 20/50 epoch; dùng `weights/BEST_Project8_v04_20261008.pt` (best epoch 19), alias `weights/BEST_Project8.pt`. Trainer 1280/NMS-free: 75,356% / 53,488%; cùng protocol COCO 640/NMS=True trên val v0.4: trước 47,593% / 31,047% → mới 70,706% / 50,694%. Trên val v0.3: 56,727% / 35,896% → 55,279% / 35,049%, nên giữ best cũ để đối chứng. Dataset mới 12.000 ảnh (9.540/1.754/706); ID 7 bottle gồm cup+bottle. Recognition đã đổi mapping; local Training/export v0.3 vẫn giữ riêng. [Báo cáo và AP từng lớp](reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/REPORT.md).

# Hồ sơ công việc đã hoàn thành

Đối soát **08/10/2026**. Thư mục này lưu các hồ sơ của những đợt nhập/gộp nguồn, kiểm kê, dọn, chuẩn bị dữ liệu và train đã kết thúc (13 tài liệu gốc trong lần sắp xếp 05/10, sau đó bổ sung báo cáo). “Hoàn thành” áp dụng cho đợt công việc cụ thể, **không đánh dấu cả tuần hoặc model đã nghiệm thu**. Các phần hạn chế/chưa làm trong từng báo cáo vẫn được giữ lại.

[Về danh mục gốc](../README.md) · [Kế hoạch còn việc và hướng dẫn đang dùng](../README/README.md)

## Hồ sơ nguồn và dữ liệu tuần 1–2

| Tài liệu | Phạm vi đã kết thúc | Giới hạn khi đọc lại |
|---|---|---|
| [COCO500](docs/COCO500_DATA_PROFILE.md) | Nhập/kiểm/thống kê 500 ảnh nguồn | Hồ sơ cũ; không phải bộ train hiện hành hay test phòng học độc lập |
| [Laptop thành viên](docs/LAPTOP_SUBMISSION_DATA_PROFILE.md) | Nhập/mapping/kiểm 1.300 ảnh nguồn | Nhãn và hạn chế tại thời điểm intake được giữ; sau đó gộp vào bộ chính |
| [Dataset project8_v0.2](docs/PROJECT8_V0_2_DATA_PROFILE.md) | Gộp hai nguồn thành 1.800 ảnh và lập hồ sơ | Bản draft cũ đã được thay bằng v0.3; không dùng lệnh importer đã dọn để train hiện tại |
| [COCO ba lớp bổ sung](reports/results/coco_three_classes_350_20261003/REPORT.md) | Tải/kiểm 566 ảnh bổ sung | Số tổng dữ liệu trong báo cáo thuộc snapshot cũ |
| [Sách và điện thoại](reports/results/book_phone_balance_20261003/REPORT.md) | Chọn/kiểm 600 ảnh ứng viên | Cân số ảnh mang nhãn không đồng nghĩa cân số box hoặc đủ tám lớp ở thời điểm chọn |
| [Audit ZIP](reports/results/zip_label_audit_20261003/REPORT.md) | Kiểm kê/dọn gói nguồn theo yêu cầu | Giữ số liệu nguồn và đường dẫn lịch sử đã bị dọn |
| [Audit ZIP sau tổ chức](reports/results/zip_label_audit_20261003/ORGANIZED_REPORT.md) | Ghi kết quả tổ chức/dọn tương ứng | Không cộng lại ảnh trong ZIP vào ảnh đã giải nén |
| [Gộp classroom](reports/results/project_progress_classroom_20261003/REPORT.md) | Gộp/mapping và lập snapshot tiến độ | Trạng thái pending trước review không phải hiện tại |
| [Dọn bộ chính/code](reports/results/primary_cleanup_20261004/REPORT.md) | Dọn bản sao/pending và kiểm bảo toàn 5.002 cặp | Kiểm kỹ thuật không thay đánh giá chất lượng model |
| [Kiểm kê sau dọn](reports/results/project_progress_20261004/REPORT.md) | Lập bảng 5.002 ảnh/26.566 box sau dọn | Snapshot trước xác nhận review/khóa và trước train |

## Chuẩn bị và các run tuần 3 đã kết thúc

| Tài liệu | Bằng chứng hoàn thành | Phần chưa được kết luận |
|---|---|---|
| [Readiness và release](reports/results/week3_readiness_20261004/REPORT.md) | Người dùng xác nhận review; khóa tại chỗ; validator đạt 5.002 ảnh và checksum | Đây là bước chuẩn bị, không phải baseline hoặc nghiệm thu G3 |
| [Run R1 — 1 epoch](reports/results/project8_v03_26n_cpu_20261004_181257_1eb3de94/REPORT.md) | state completed 1/1, exit 0; best/last/epoch0 hợp lệ tám lớp | Lượt kiểm khả năng chạy; chất lượng còn thấp |
| [Run R2 — 4 epoch từ R1 best](reports/results/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/REPORT.md) | state completed 4/4, exit 0; best/last/epoch0–epoch3 hợp lệ; bảo toàn dataset/nguồn | Reset optimizer/scheduler; tổng 5 epoch chưa thay đối chứng cùng protocol hoặc model cuối |
| [Kaggle — fine-tune 14 epoch, 06/10/2026](reports/results/kaggle/project8_26n_kaggle_20261006_083226/REPORT.md) | 14/14 epoch; đã nhập best và xác minh hash/model; baseline và tuned được chấm cùng protocol trên 798 ảnh val | Chưa đánh giá test độc lập, lỗi ảnh, tốc độ hay nghiệm thu triển khai |
| [Đối soát và bảng đủ 30 epoch](reports/results/training_30_epochs_20261006/REPORT.md) | 5 run có CSV thật; kiểm ZIP, checkpoint, nguồn trọng số và checksum dataset; bảng P/R/AP, loss và thời gian từng epoch | 30 epoch thực thi qua các run mới; không phải resume liên tục; K2 chỉ tải best và result ZIP |
| [Kaggle 08/10 — 20 epoch](reports/results/kaggle/project8_26n_kaggle_20261008_040051/REPORT.md) | completed 20/20, best epoch 19; ZIP/hash, model và CSV đã kiểm; best/last khác tensor | Validation, chưa kiểm test/FPS camera thật |
| [Bảng toàn bộ 50 epoch](reports/results/training_50_epochs_20261008/REPORT.md) | 50 dòng từ 6 run; đủ loss/P/R/AP/thời gian, CSV có learning rate | Không phải 50 run hoặc resume liên tục; imgsz khác giữa run |

Đường dẫn weights, JSON, CSV, log và hình trong báo cáo đã được đối chiếu lại từ vị trí mới. Các artifact đó tiếp tục ở runs/train và reports/results; chỉ nội dung tài liệu Markdown được chuyển. Các đường dẫn nguồn/media cũ đã bị dọn trước lần sắp xếp này vẫn là thông tin lịch sử.

Tuần 1–2 còn mục nghiệm thu và tuần 3 còn phân tích/đối chứng; xem [trạng thái theo tuần](../README/README.md#trạng-thái-theo-tuần). Hướng dẫn tiếp tục train giữ nguyên ở [Triển khai tuần ba tiếp tục](../TRIEN_KHAI_TUAN_3_TIEP_TUC.md).
