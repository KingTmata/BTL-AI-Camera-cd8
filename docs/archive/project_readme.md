> **Dataset hiện hành08/10/2026:**v0.5 đủ15.000 ảnh (12.000train/2.500val/500test), tất cảfull_review_pending, chưatrainv05. Weight/metricv04 bên dưới làlịch sử. XemREADME gốc vàtasks/plan.md.

**Cập nhật v0.4 — 08/10/2026:** đã nhập ZIP full train 20/50 epoch; dùng `weights/BEST_Project8_v04_20261008.pt` (best epoch 19), alias `weights/BEST_Project8.pt`. Trainer 1280/NMS-free: 75,356% / 53,488%; cùng protocol COCO 640/NMS=True trên val v0.4: trước 47,593% / 31,047% → mới 70,706% / 50,694%. Trên val v0.3: 56,727% / 35,896% → 55,279% / 35,049%, nên giữ best cũ để đối chứng. Dataset mới 12.000 ảnh (9.540/1.754/706); ID 7 bottle gồm cup+bottle. Recognition đã đổi mapping; local Training/export v0.3 vẫn giữ riêng. [Báo cáo và AP từng lớp](completed/reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/report.md).

# Kế hoạch và hướng dẫn đang dùng

Đối soát **08/10/2026**. Thư mục này chứa các tài liệu còn đầu việc chưa hoàn tất hoặc là hướng dẫn/tham chiếu cần dùng tiếp (30 tài liệu gốc trong lần sắp xếp 05/10). Không đưa một kế hoạch vào nhóm hoàn thành chỉ vì đã có mã, đã tạo file báo cáo hoặc đã train thử.

[Về danh mục gốc](readme_snapshot_20261015.md) · [Hồ sơ công việc đã kết thúc](completed/README.md) · [Triển khai tuần ba tiếp tục](plans/week3_handoff.md)

## Trạng thái theo tuần

| Mốc | Phần đã có bằng chứng | Phần cần tiếp tục | Phân loại |
|---|---|---|---|
| G1 / tuần 1 | Môi trường, pretrained, app ảnh/video/webcam, mapping và kiểm phần mềm | Kiểm webcam thật, video quay liên tục khác phiên và bằng chứng chạy trên máy thành viên | Giữ kế hoạch/bàn giao trong README |
| G2 / tuần 2 | Nhập/gộp/dọn dữ liệu; project8_v0.3 đủ 5.002 ảnh và 26.566 box; người dùng duyệt/khóa; công cụ validator/evaluator; baseline pretrained đã chấm validation cùng protocol | Video phát triển thật và sự kiện đã review; test đủ lớp/độc lập theo kế hoạch | Chỉ lưu trữ hồ sơ của từng đợt dữ liệu đã kết thúc; G2 chưa nghiệm thu toàn bộ |
| G3 / tuần 3 | 6 run, 50 epoch thực thi (1+4+1+10+14+20); K3 completed 20/20, best epoch 19; trọng số local đã kiểm hash, mapping, tensor và CPU smoke; evaluator chung K2→K3 mAP50 53,848%→56,727%, mAP50–95 33,847%→35,896% | Cần kiểm ảnh lỗi theo lớp, test độc lập/FPS; các hạng mục recognition/SSL trong kế hoạch chưa hoàn tất | Các run có báo cáo; G3 và model chưa nghiệm thu |
| Tuần 4–6 | Có kế hoạch/phân công và quy chuẩn bàn giao | Tracking, sự kiện, trạng thái laptop, giơ tay, SSL, benchmark, bộ nộp và kiểm máy khác theo phạm vi kế hoạch | Chưa hoàn thành |

Release v0.3 lịch sử có **3.883 / 798 / 321 ảnh**; v0.4 hiện hành có **9.540 / 1.754 / 706 ảnh**. Test thiếu ground truth backpack/cup và được giữ ngoài việc chọn model/tham số. Review của người dùng đã ghi nhận; mô tả “chưa review/khóa/chưa train” trong snapshot trước đó không phải trạng thái hiện tại.

Các run dùng checkpoint best của lượt trước để tạo **run mới**, reset optimizer/scheduler/warmup (`resume=False`). **50 là số epoch đã thực thi**, không phải một run resume liên tục. K2 học từ best epoch 9 của K1; K3 học từ K2 best và chọn best epoch 19. Xem [bảng đủ 50 epoch](completed/reports/results/training_50_epochs_20261008/report.md) và [báo cáo K3](completed/reports/results/kaggle/project8_26n_kaggle_20261008_040051/report.md); trainer validation NMS-free 1280 được tách khỏi evaluator chung 640/NMS=True.

Bằng chứng chính: [chuẩn bị/khóa dữ liệu](completed/reports/results/week3_readiness_20261004/report.md), [báo cáo R1](completed/reports/results/project8_v03_26n_cpu_20261004_181257_1eb3de94/report.md), [báo cáo R2](completed/reports/results/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/report.md), [sổ theo dõi](guides/week3_training_log_and_plan.md).

## Kế hoạch và bàn giao chưa đủ nghiệm thu

| Tài liệu | Lý do giữ trong nhóm đang dùng |
|---|---|
| [Kế hoạch tuần 1–2–3](plans/week1_2_3_plan.md) | Kế hoạch tổng còn nhiều tiêu chí G1–G3 và bộ nộp chưa đạt |
| [Kế hoạch G2](plans/week2_g2.md) | Còn baseline validation và video thật; xem trạng thái mới ở trên |
| [Bàn giao tuần 1](guides/week1_handoff.md) | Kiểm thiết bị thật và máy thành viên chưa đủ bằng chứng |
| [Báo cáo triển khai tuần 2](guides/week2_implementation_report.md) | Công cụ đã triển khai nhưng file còn checklist G2 chưa hoàn tất |
| [Workflow tuần 2](guides/week2_workflow.md) | Tiếp tục dùng khi kiểm/khóa phiên bản dữ liệu mới |
| [Bắt đầu tuần 3](guides/week3_start.md) | Hướng dẫn CPU và điều kiện G3 vẫn cần dùng; phần trạng thái cũ đã có ghi chú |
| [Sổ theo dõi train và kế hoạch quyết định](guides/week3_training_log_and_plan.md) | Lịch sử R1/R2 hoàn tất; phần quyết định lượt tiếp theo vẫn đang mở |
| [Kế hoạch tuần 4–5–6](plans/week4_5_6_plan.md) | Tracking, recognition, SSL, benchmark và bộ nộp chưa hoàn tất |

## Hướng dẫn và tham chiếu tiếp tục sử dụng

| Tài liệu | Mục đích |
|---|---|
| [README dự án](../guides/project_overview.md) | Cài môi trường và chạy app/CLI |
| [Protocol đánh giá](../research/evaluation_protocol.md) | Giữ cùng mapping, head và cách chấm giữa các model |
| [Handoff huấn luyện Kaggle](guides/kaggle_training_handoff.md) | Export train/val, chọn CPU/GPU notebook, lưu checkpoint và nhập kết quả |
| [Quy chuẩn gán nhãn](../datasets/labeling_guide.md) | Review hoặc tạo phiên bản nhãn mới |
| [Nguồn bên thứ ba](../research/third_party.md) | Ghi công và thông tin tái sử dụng |
| [Cấu trúc và luồng](guides/cau_truc_va_luong.md) | Hướng dẫn cấu trúc mã hiện có |
| [Kiểm thử](guides/kiem_thu.md) | Cách kiểm phần mềm và giới hạn của kiểm mô phỏng |
| [Train và đánh giá](guides/train_va_danh_gia.md) | CLI train/evaluator và điều kiện so sánh |
| [Hồ sơ dataset v0.3](guides/project8_v0_3_data_profile.md) | Nguồn, mapping, split và số liệu bộ dữ liệu đang dùng |
| [Bàn giao video](guides/video_annotation.md) | Quy chuẩn video và sự kiện còn cần thu/gán/review |
| [Data card dự án](../reference/data/data_card.md) | Giới hạn và số liệu dữ liệu tổng |

Các hướng dẫn thư mục cũng còn dùng khi triển khai tiếp. Chúng được chuyển theo cây đường dẫn gốc để tránh trùng nhiều file tên README:

| Thư mục được hướng dẫn | File tài liệu |
|---|---|
| data | [README dữ liệu](../reference/data.md) |
| data/templates | [Mẫu metadata](../reference/data_templates.md) |
| data/raw/week456_behavior | [Nguồn hành vi cho tuần 4–6](../reference/data_raw_week456_behavior.md) |
| demo | [Video demo và giới hạn smoke](../reference/demo.md) |
| reports | [Kết quả thực nghiệm](../reference/reports.md) |
| scripts | Công cụ thiết lập môi trường: scripts/setup_env.py |
| src | [Mã nguồn](../reference/src.md) |
| src/inference | [Inference](../reference/src_inference.md) |
| src/training | [Training](../reference/src_training.md) |
| src/ui | [Giao diện](../reference/src_ui.md) |
| Kiểm chứng hiện tại | [Biên bản dataset](../../reports/audits/verification_current_20261008.json) |
| weights | [Trọng số](../reference/weights.md) |

Các lệnh chạy từ gốc dự án; mã nguồn, cấu hình, media, JSON/CSV và weights vẫn ở các thư mục chức năng của chúng.

## Metadata giữ tại vị trí gốc

Đây là bảy file Markdown thuộc hồ sơ dataset/artifact, không phải kế hoạch tuần. Giữ nguyên bytes và đường dẫn để không làm sai checksum hay tách hồ sơ khỏi dữ liệu. Các bản cũ là provenance lịch sử; project8_v0.3 vẫn là release của Training local; Recognition/Kaggle v0.4 dùng bộ 12k.

- [project8_v0.3 DATA_CARD](../../data/dataset/project8_v0.3/DATA_CARD.md).
- [project8_v0.2 DATA_CARD](../../data/dataset/project8_v0.2/DATA_CARD.md).
- [COCO500 DATA_CARD](../../data/dataset/coco500_v0.1/DATA_CARD.md).
- [Laptop intake DATA_CARD](../../data/dataset/laptop_roboflow_v1_intake/DATA_CARD.md).
- [COCO bổ sung README](../../data/dataset/coco_three_classes_350_v0.1/README.md).
- [Sách/điện thoại README](../../data/dataset/student_behaviour_book_phone_600_v0.2/README.md).
- [Synthetic smoke DATA_CARD](../../runs/week2/synthetic_smoke/release/DATA_CARD.md): dữ liệu nhân tạo chỉ để kiểm phần mềm.

Ba liên kết nội bộ trong DATA_CARD của Laptop intake/v0.2 đã thiếu đích từ trước lần sắp xếp này. Không sửa các file metadata đó; hồ sơ nguồn tương ứng hiện truy cập được qua [danh mục lưu trữ](completed/README.md#hồ-sơ-nguồn-và-dữ-liệu-tuần-1–2).

File [Triển khai tuần ba tiếp tục](plans/week3_handoff.md) được giữ nguyên ở gốc theo yêu cầu. Liên kết protocol và sổ log đã chuyển trực tiếp đến tài liệu chính; bỏ hai file điều hướng thừa.

Môi trường ảo, cache, tài liệu thư viện/vendor và hướng dẫn của công cụ không thuộc phạm vi sắp xếp tài liệu dự án.
