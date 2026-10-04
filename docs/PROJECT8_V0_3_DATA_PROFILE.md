# Dataset chính — project8_v0.3 sau dọn dữ liệu

Cập nhật 04/10/2026 theo yêu cầu chỉ giữ 5.002 ảnh.

Kho chính hiện có **5.002 ảnh / 5.002 TXT / 26.566 bounding box**. Train/val/test: **3.883 / 798 / 321**. Không còn ảnh pending.
Tám lớp: person, table, chair, laptop, cell phone, backpack, book, cup. Nhãn vẫn là draft chưa review đầy đủ tám lớp; chưa khóa release, chưa train.

| Nguồn | Số ảnh |
|---|---:|
| project8_v0.2 (nguồn lịch sử) | 1.800 |
| COCO bổ sung | 566 |
| Sách/điện thoại | 600 |
| Classroom sau lọc | 2.036 |
| Tổng | 5.002 |

| Lớp | Ảnh có nhãn | Box |
|---|---:|---:|
| person | 1346 | 8169 |
| table | 1017 | 2479 |
| chair | 1286 | 2173 |
| laptop | 887 | 1525 |
| cell phone | 711 | 2012 |
| backpack | 378 | 525 |
| book | 1214 | 8651 |
| cup | 408 | 1032 |

## Mapping và dữ liệu đã xóa

Student 0→person 0; chair 1→chair 2; table 2 và with-student 3→table 1 theo quyết định người dùng. Không sửa tọa độ nhãn nguồn.
Đã xóa 107 ảnh pending và TXT đi kèm: 72 ảnh nhãn rỗng, 35 bản ảnh xuyên split. Lớp bàn giữ nguyên 1.017 ảnh / 2.479 box; 35 box bị xóa thuộc chair.
Đã xóa 6566 bản sao ảnh và 189 ảnh tham khảo/kết quả thử; 3 ZIP nguồn/tham khảo đã xóa. Metadata và hash nguồn giữ để đối chiếu. Việc dọn thực hiện từng nhóm nhỏ; xem audit để biết phần đã xóa thực tế.
Hai ZIP hành vi gốc giữ riêng cho tuần 4–6 theo phạm vi đã chọn trước đó. Chúng chứa 4.844 lượt ảnh, không nằm trong 5.002 ảnh đã giải nén hiện hành; 600 ảnh chọn từ nguồn hành vi đã nằm trong kho chính.

## Hồ sơ và kiểm tra

inventory_all.csv, manifest.csv và review.csv đều kê 5.002 ảnh; data/manifest.csv là bản điều hướng cho workflow mặc định. Không còn pending_review.csv.
Kiểm cặp ảnh/TXT, giải mã, ID/hình học, hash media, nhóm split và checksum đạt. Hash ảnh/nhãn của 5.002 cặp giữ lại không đổi so với trước khi dọn. Chưa xác nhận độc lập video/cảnh hoặc loại hết gần trùng.
Giao diện đọc 50 ảnh train từ manifest chính và dùng profile nhãn project8; không cần ảnh mẫu COCO128 riêng.
Audit việc xóa: reports/results/primary_cleanup_20261004/cleanup_plan.json và deleted_pending_metadata.csv. Báo cáo trước khi dọn là snapshot lịch sử, không dùng làm số liệu hiện hành.
