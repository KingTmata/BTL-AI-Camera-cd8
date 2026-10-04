> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

> **Báo cáo lịch sử ngày 03/10/2026.** Dữ liệu đã được gộp và dọn sau thời điểm này; ảnh nguồn/bản sao/ảnh pending được nhắc bên dưới có thể đã xóa. Xem [báo cáo hiện hành 04/10/2026](../project_progress_20261004/REPORT.md) và [audit dọn](../primary_cleanup_20261004/REPORT.md).

# Tập ứng viên sách và điện thoại — 600 ảnh

Đã chọn 600 ảnh có đồng thời book và cell phone, max một ảnh mỗi nhóm tên gốc, seed 42; giữ split nguồn. Hai lớp có cùng số ảnh mang nhãn, không phải cùng số bounding box.

Từ 2.596 ảnh: 1.523 ảnh có cả hai nhãn, 842 ảnh chỉ book và 231 ảnh chỉ cell phone. Nhóm có cả hai gồm 959 tên gốc; chọn 600 nhóm và một ảnh mỗi nhóm, còn 923 ảnh có cả hai không giữ trong tập hoạt động. Tổng loại khỏi tập hoạt động 1.996 ảnh, nhưng ZIP nguồn vẫn giữ đầy đủ cho tuần 4–6.

Ảnh còn thiếu nhãn các lớp mục tiêu khác đang xuất hiện; không xem sáu lớp đó vắng mặt. Chưa review, khóa release hoặc train. Nhóm tên gốc không chứng minh độc lập cảnh hoặc loại hết ảnh gần trùng.

Ảnh/TXT qua kiểm kỹ thuật và checksum sao chép. Dataset chính và COCO mới giữ nguyên.

## Số liệu sau khi dọn bản ứng viên cũ

Đã xóa bản ứng viên 2.596 ảnh sau khi kiểm bản 600 ảnh khớp checksum và validator đạt. ZIP hành vi gốc vẫn giữ nguyên.

Tổng detection hoạt động: 2,966 file ảnh; train 2,096, val 639, test 231. Ảnh + TXT: 252.19 MB.

| Lớp | Ảnh có nhãn toàn bộ | Ảnh có nhãn train + val |
|---|---:|---:|
| table | 322 | 322 |
| backpack | 378 | 378 |
| cup | 408 | 408 |
| chair | 425 | 425 |
| cell phone | 711 | 610 |
| person | 866 | 866 |
| laptop | 887 | 802 |
| book | 1214 | 1068 |

Hai lớp trong tập ứng viên có số ảnh bằng nhau, nhưng số box khác nhau. Cả tám lớp của toàn dự án chưa cân bằng; bộ chính cũ còn nhiều book/laptop. Cần bổ sung nhãn thiếu và review trước train chính thức.
