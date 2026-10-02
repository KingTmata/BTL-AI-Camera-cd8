# Hồ sơ dataset nhóm — project8_v0.2

**Cập nhật 03/10/2026: DATASET CHÍNH DRAFT.** Có 1.800 ảnh và 5.862 bounding box; chưa khóa release, chưa có review người thứ hai đã duyệt, baseline hoặc fine-tune.

## Tên, phiên bản, nguồn và quyền

- Dataset nhóm: project8_v0.2, tạo 02/10/2026; nguồn COCO2017 (coco500_v0.1) và Roboflow Laptop v1 do thành viên đóng góp ngang hàng.
- COCO: 500 ảnh; metadata từng ảnh lưu URL nguồn, license và date_captured. License ảnh khác nhau, không áp một giấy phép chung cho toàn bộ ảnh.
- Laptop: 1.300 ảnh; nguồn https://universe.roboflow.com/new-workspace-xp2sh/laptop-tgbyh/dataset/1; ZIP khai báo CC BY 4.0, export 18/03/2023. Chưa xác minh riêng tác giả/quyền từng ảnh; ngày export không phải ngày chụp.
- Tên thành viên giao/chuẩn hóa, tác giả ảnh gốc và reviewer: chưa có thông tin đầy đủ, không tự điền.

## Quy mô và split hiện tại

| Tập | COCO | Laptop | Tổng |
|---|---:|---:|---:|
| train | 400 | 910 | 1.310 |
| validation | 100 | 260 | 360 |
| test | 0 | 130 | 130 |

**1.310 + 360 + 130 = 1.800 ảnh.** 1.670 ảnh train/val; 130 test nằm ở images/test và labels/test của bản chính, không mất và không tính vào chỉ tiêu ảnh phát triển. Nếu giữ mục tiêu 2.500 train/val thì còn thiếu 830 ảnh, chưa tính phần chưa hoàn thiện nhãn. 130 test hiện chỉ có nhãn book/laptop, chưa đủ chấm tám lớp hoặc chứng minh độc lập phòng học.

## Dữ liệu, nhãn và vị trí

Ảnh tĩnh, detection bounding box. Laptop: 416×416 RGB, stretch theo nguồn; COCO JPEG, 498 RGB + 2 grayscale, kích thước khác nhau đến cạnh dài 640. Nhãn YOLO TXT năm cột class_id cx cy width height, chuẩn hóa [0,1].

| ID nhóm | Lớp | Ảnh có nhãn | Box |
|---:|---|---:|---:|
| 0 | person | 425 | 1366 |
| 1 | table | 86 | 98 |
| 2 | chair | 75 | 200 |
| 3 | laptop | 818 | 1406 |
| 4 | cell phone | 39 | 46 |
| 5 | backpack | 28 | 42 |
| 6 | book | 530 | 2569 |
| 7 | cup | 58 | 135 |

Lớp table gồm các loại bàn; COCO dining table là đối chứng gần đúng. Book gồm sách/vở. Bộ Laptop nguồn chỉ có hai lớp book/laptop; mỗi thành viên có thể phụ trách một hoặc nhiều lớp và bổ sung nhãn trên cùng ảnh theo phân công.

Ảnh: data/dataset/project8_v0.2/images/{train,val,test}; nhãn: labels/{train,val,test}, TXT cùng stem. manifest.csv ghi đường dẫn ảnh/nhãn, nguồn, split và hash; data.yaml khai báo tám lớp. Giữ raw/annotation gốc để đối chiếu.

## Chất lượng, outlier và noise

Đã kiểm kỹ thuật: giải mã ảnh, cặp TXT, class ID, hình học box, box lặp, trùng bytes; nguồn Laptop được kiểm thêm trùng pixel với chính nó và COCO500. Không phát hiện lỗi kỹ thuật/trùng ở 1.300 ảnh Laptop. Không đồng nghĩa mọi nhãn đúng/đủ về ngữ nghĩa.

- COCO: 64 ảnh có ít nhất một cờ; danh sách review 148 ca (100 mẫu + 48 ca bổ sung), pending.
- Laptop: 13 ảnh có box rất nhỏ, 22 ảnh tối, 4 ảnh sáng, 2 ảnh ít tương phản; cờ có thể chồng nhau.
- Noise nhãn đã biết: các lớp khác có xuất hiện nhưng chưa được gán trong nguồn Laptop; mẫu trực quan có box book bao cụm sách/kệ cần review.
- Chưa định lượng nhiễu cảm biến/nén hoặc tỷ lệ nhãn sai toàn bộ; không ghi noise=0. Chưa xác nhận gần trùng và độc lập theo phiên; session nguồn chưa biết.

## Review, khóa và đánh giá

Review người thứ hai chưa hoàn tất; annotator/reviewer thật còn chờ. Snapshot checksum draft đã có, không phải release khóa. Giữ split nguồn; nhóm theo tên ảnh gốc không chứng minh độc lập cảnh/phiên. Chưa fine-tune, chưa có best.pt/last.pt của bộ này, chưa đo baseline mAP trên validation.

COCO train có thể trùng nguồn pretraining; kết quả COCO không chứng minh camera phòng học. Chấm pretrained và fine-tuned bằng evaluator project8 trên cùng validation có ground truth đầy đủ, theo EVALUATION_PROTOCOL.md.

## Video và bàn giao tiếp theo

Chưa có video_dev/video_test thật có nhãn sự kiện. Cần chuẩn bị 3–5 clip phát triển và 5–10 clip test từ phiên độc lập theo kế hoạch. Slideshow chỉ kiểm pipeline.

Hồ sơ chi tiết: [dataset chính](../docs/PROJECT8_V0_2_DATA_PROFILE.md), [COCO500](../docs/COCO500_DATA_PROFILE.md), [Laptop](../docs/LAPTOP_SUBMISSION_DATA_PROFILE.md). Tiến độ: [tuần 2](../WEEK2_G2.md), [bàn giao tuần 1](../docs/WEEK1_HANDOFF.md).
