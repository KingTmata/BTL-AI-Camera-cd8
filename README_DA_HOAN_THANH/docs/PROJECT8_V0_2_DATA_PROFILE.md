> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

# Dataset chính — project8_v0.2

> Hồ sơ phiên bản lịch sử. Bản sao media v0.2 đã dọn theo yêu cầu. Dataset chính hiện hành là [project8_v0.3](../../README/docs/PROJECT8_V0_3_DATA_PROFILE.md): 5.002 ảnh, không còn pending.

Cập nhật 03/10/2026. **Bản draft chính của dự án**, gồm hai nguồn ngang hàng: COCO500 và bộ Roboflow Laptop do thành viên đóng góp.

Tổng **1800 ảnh duy nhất**, **5862 box**; COCO 500 ảnh, thành viên 1.300 ảnh. Không tính COCO128 hoặc ảnh smoke/mẫu.

## Split hiện tại

| Tập | COCO | Thành viên | Tổng |
|---|---:|---:|---:|
| train | 400 | 910 | 1.310 |
| validation | 100 | 260 | 360 |
| test | 0 | 130 | 130 |

130 ảnh test vẫn đầy đủ trong images/test và labels/test; không được tính vào 1.670 ảnh train + validation. Có 215 box book và 142 box laptop trên test nguồn này. Tổng 1.310 + 360 + 130 = 1.800 ảnh.

Giữ split nguồn. Nhóm theo tên ảnh gốc trước hậu tố Roboflow; không thấy nhóm tên gốc xuyên split. Không có metadata phiên chụp nên chưa chứng minh độc lập cảnh/phiên hoặc loại gần trùng. Test hiện chỉ có nhãn book/laptop từ nguồn này, chưa là test đủ tám lớp hay test phòng học độc lập.

## Nhãn

Thứ tự: person, table, chair, laptop, cell phone, backpack, book, cup. Book nguồn ID0 → book ID6; Laptop nguồn ID1 → laptop ID3.

| Lớp | Ảnh có nhãn | Box |
|---|---:|---:|
| person | 425 | 1366 |
| table | 86 | 98 |
| chair | 75 | 200 |
| laptop | 818 | 1406 |
| cell phone | 39 | 46 |
| backpack | 28 | 42 |
| book | 530 | 2569 |
| cup | 58 | 135 |

Số trên là nhãn đã có, không phải số vật thật xuất hiện. Một người có thể phụ trách nhiều lớp; 1.300 ảnh nguồn thành viên hiện chỉ có nhãn book/laptop. Người phụ trách các lớp còn lại bổ sung trên những ảnh có vật tương ứng. Không tự đánh dấu chúng vắng mặt hoặc tự tạo nhãn.

## Kiểm tự động

1.800 ảnh giải mã được, cặp TXT đầy đủ, ID nhóm hợp lệ, box hữu hạn/dương/trong ảnh, không box trùng trong TXT, không trùng bytes/pixel trong nguồn mới hoặc với COCO500. Ảnh/nhãn khi gộp được đối chiếu checksum với bản nguồn.

Kiểm định dạng thành công không xác nhận đầy đủ/đúng ngữ nghĩa mọi nhãn. Review người thứ hai chưa hoàn tất; checksum là snapshot draft, chưa khóa release. Baseline tám lớp cần ground truth đầy đủ trên tập đánh giá.

## Hồ sơ

`data/dataset/project8_v0.2/`: images/, labels/, data.yaml, manifest.csv, splits/, validation.json, statistics.json, review.csv, DATA_CARD.md, checksums.json. Raw nguồn và COCO500 giữ nguyên để truy thí nghiệm.

Đặc tính chi tiết: [COCO500](COCO500_DATA_PROFILE.md), [bộ Laptop thành viên](LAPTOP_SUBMISSION_DATA_PROFILE.md).

Script gộp v0.2 và các bản sao nguồn đã được bỏ sau khi chốt kho chính. Số liệu trong hồ sơ này là snapshot lịch sử, không phải kiểm kê hiện hành.
