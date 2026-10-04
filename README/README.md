# Kế hoạch và hướng dẫn đang dùng

Đối soát **05/10/2026**. Thư mục này chứa **30 tài liệu gốc** còn đầu việc chưa hoàn tất hoặc là hướng dẫn/tham chiếu cần dùng tiếp. Không đưa một kế hoạch vào nhóm hoàn thành chỉ vì đã có mã, đã tạo file báo cáo hoặc đã train thử.

[Về danh mục gốc](../README.md) · [Hồ sơ công việc đã kết thúc](../README_DA_HOAN_THANH/README.md) · [Triển khai tuần ba tiếp tục](../TRIEN_KHAI_TUAN_3_TIEP_TUC.md)

## Trạng thái theo tuần

| Mốc | Phần đã có bằng chứng | Phần cần tiếp tục | Phân loại |
|---|---|---|---|
| G1 / tuần 1 | Môi trường, pretrained, app ảnh/video/webcam, mapping và kiểm phần mềm | Kiểm webcam thật, video quay liên tục khác phiên và bằng chứng chạy trên máy thành viên | Giữ kế hoạch/bàn giao trong README |
| G2 / tuần 2 | Nhập/gộp/dọn dữ liệu; project8_v0.3 đủ 5.002 ảnh và 26.566 box; người dùng duyệt/khóa; công cụ validator/evaluator | Video phát triển thật và sự kiện đã review; baseline pretrained trên validation bằng evaluator chung; test đủ lớp/độc lập theo kế hoạch | Chỉ lưu trữ hồ sơ của từng đợt dữ liệu đã kết thúc; G2 chưa nghiệm thu toàn bộ |
| G3 / tuần 3 | R1 1/1 epoch và R2 4/4 epoch completed, exit 0; checkpoint kiểm hợp lệ tám lớp; tổng 5 epoch, chưa SSL | Baseline và fine-tuned cùng protocol, AP/ảnh lỗi từng lớp, quyết định ngân sách tiếp theo; các hạng mục recognition/SSL trong kế hoạch vẫn chưa hoàn tất | Báo cáo hai run vào nhóm hoàn thành; sổ theo dõi và kế hoạch vẫn đang dùng |
| Tuần 4–6 | Có kế hoạch/phân công và quy chuẩn bàn giao | Tracking, sự kiện, trạng thái laptop, giơ tay, SSL, benchmark, bộ nộp và kiểm máy khác theo phạm vi kế hoạch | Chưa hoàn thành |

Train/val/test hiện hành là **3.883 / 798 / 321 ảnh**. Test thiếu ground truth backpack/cup và được giữ ngoài việc chọn model/tham số. Review của người dùng đã ghi nhận; mô tả “chưa review/khóa/chưa train” trong snapshot trước đó không phải trạng thái hiện tại.

R2 dùng R1 best để tạo **run mới**, reset optimizer/scheduler/warmup. Vì vậy tổng 5 epoch không phải một run resume liên tục. Số validation của trainer chưa thay bảng đối chứng evaluator chung.

Bằng chứng chính: [chuẩn bị/khóa dữ liệu](../README_DA_HOAN_THANH/reports/results/week3_readiness_20261004/REPORT.md), [báo cáo R1](../README_DA_HOAN_THANH/reports/results/project8_v03_26n_cpu_20261004_181257_1eb3de94/REPORT.md), [báo cáo R2](../README_DA_HOAN_THANH/reports/results/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/REPORT.md), [sổ theo dõi](docs/WEEK3_TRAINING_LOG_AND_PLAN.md).

## Kế hoạch và bàn giao chưa đủ nghiệm thu

| Tài liệu | Lý do giữ trong nhóm đang dùng |
|---|---|
| [Kế hoạch tuần 1–2–3](WEEK1_2_3_PLAN.md) | Kế hoạch tổng còn nhiều tiêu chí G1–G3 và bộ nộp chưa đạt |
| [Kế hoạch G2](WEEK2_G2.md) | Còn baseline validation và video thật; xem trạng thái mới ở trên |
| [Bàn giao tuần 1](docs/WEEK1_HANDOFF.md) | Kiểm thiết bị thật và máy thành viên chưa đủ bằng chứng |
| [Báo cáo triển khai tuần 2](docs/WEEK2_IMPLEMENTATION_REPORT.md) | Công cụ đã triển khai nhưng file còn checklist G2 chưa hoàn tất |
| [Workflow tuần 2](docs/WEEK2_WORKFLOW.md) | Tiếp tục dùng khi kiểm/khóa phiên bản dữ liệu mới |
| [Bắt đầu tuần 3](docs/WEEK3_START.md) | Hướng dẫn CPU và điều kiện G3 vẫn cần dùng; phần trạng thái cũ đã có ghi chú |
| [Sổ theo dõi train và kế hoạch quyết định](docs/WEEK3_TRAINING_LOG_AND_PLAN.md) | Lịch sử R1/R2 hoàn tất; phần quyết định lượt tiếp theo vẫn đang mở |
| [Kế hoạch tuần 4–5–6](WEEK4_5_6_PLAN.md) | Tracking, recognition, SSL, benchmark và bộ nộp chưa hoàn tất |

## Hướng dẫn và tham chiếu tiếp tục sử dụng

| Tài liệu | Mục đích |
|---|---|
| [README dự án](README_DU_AN.md) | Cài môi trường và chạy app/CLI |
| [Protocol đánh giá](EVALUATION_PROTOCOL.md) | Giữ cùng mapping, head và cách chấm giữa các model |
| [Quy chuẩn gán nhãn](LABELING_GUIDE.md) | Review hoặc tạo phiên bản nhãn mới |
| [Nguồn bên thứ ba](THIRD_PARTY.md) | Ghi công và thông tin tái sử dụng |
| [Cấu trúc và luồng](docs/CAU_TRUC_VA_LUONG.md) | Hướng dẫn cấu trúc mã hiện có |
| [Kiểm thử](docs/KIEM_THU.md) | Cách kiểm phần mềm và giới hạn của kiểm mô phỏng |
| [Train và đánh giá](docs/TRAIN_VA_DANH_GIA.md) | CLI train/evaluator và điều kiện so sánh |
| [Hồ sơ dataset v0.3](docs/PROJECT8_V0_3_DATA_PROFILE.md) | Nguồn, mapping, split và số liệu bộ dữ liệu đang dùng |
| [Bàn giao video](docs/VIDEO_ANNOTATION.md) | Quy chuẩn video và sự kiện còn cần thu/gán/review |
| [Data card dự án](data/DATA_CARD.md) | Giới hạn và số liệu dữ liệu tổng |

Các hướng dẫn thư mục cũng còn dùng khi triển khai tiếp. Chúng được chuyển theo cây đường dẫn gốc để tránh trùng nhiều file tên README:

| Thư mục được hướng dẫn | File tài liệu |
|---|---|
| data | [README dữ liệu](data/README.md) |
| data/templates | [Mẫu metadata](data/templates/README.md) |
| data/raw/week456_behavior | [Nguồn hành vi cho tuần 4–6](data/raw/week456_behavior/README.md) |
| demo | [Video demo và giới hạn smoke](demo/README.md) |
| reports | [Kết quả thực nghiệm](reports/README.md) |
| scripts | [Các công cụ CLI](scripts/README.md) |
| src | [Mã nguồn](src/README.md) |
| src/inference | [Inference](src/inference/README.md) |
| src/training | [Training](src/training/README.md) |
| src/ui | [Giao diện](src/ui/README.md) |
| tests | [Kiểm thử mã](tests/README.md) |
| weights | [Trọng số](weights/README.md) |

Các lệnh chạy từ gốc dự án; mã nguồn, cấu hình, media, JSON/CSV và weights vẫn ở các thư mục chức năng của chúng.

## Metadata giữ tại vị trí gốc

Đây là bảy file Markdown thuộc hồ sơ dataset/artifact, không phải kế hoạch tuần. Giữ nguyên bytes và đường dẫn để không làm sai checksum hay tách hồ sơ khỏi dữ liệu. Các bản cũ là provenance lịch sử; chỉ project8_v0.3 là dataset train hiện hành.

- [project8_v0.3 DATA_CARD](../data/dataset/project8_v0.3/DATA_CARD.md).
- [project8_v0.2 DATA_CARD](../data/dataset/project8_v0.2/DATA_CARD.md).
- [COCO500 DATA_CARD](../data/dataset/coco500_v0.1/DATA_CARD.md).
- [Laptop intake DATA_CARD](../data/dataset/laptop_roboflow_v1_intake/DATA_CARD.md).
- [COCO bổ sung README](../data/dataset/coco_three_classes_350_v0.1/README.md).
- [Sách/điện thoại README](../data/dataset/student_behaviour_book_phone_600_v0.2/README.md).
- [Synthetic smoke DATA_CARD](../runs/week2/synthetic_smoke/release/DATA_CARD.md): dữ liệu nhân tạo chỉ để kiểm phần mềm.

Ba liên kết nội bộ trong DATA_CARD của Laptop intake/v0.2 đã thiếu đích từ trước lần sắp xếp này. Không sửa các file metadata đó; hồ sơ nguồn tương ứng hiện truy cập được qua [danh mục lưu trữ](../README_DA_HOAN_THANH/README.md#hồ-sơ-nguồn-và-dữ-liệu-tuần-1–2).

File [Triển khai tuần ba tiếp tục](../TRIEN_KHAI_TUAN_3_TIEP_TUC.md) được giữ nguyên ở gốc theo yêu cầu. Ba file điều hướng tại đường dẫn cũ EVALUATION_PROTOCOL.md, docs/WEEK3_TRAINING_LOG_AND_PLAN.md và REPORT.md của R2 giữ các liên kết từ file này hoạt động.

Môi trường ảo, cache, tài liệu thư viện/vendor và hướng dẫn của công cụ không thuộc phạm vi sắp xếp tài liệu dự án.
