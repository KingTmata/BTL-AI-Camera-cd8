> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

> **Báo cáo lịch sử ngày 03/10/2026.** Dữ liệu đã được gộp và dọn sau thời điểm này; ảnh nguồn/bản sao/ảnh pending được nhắc bên dưới có thể đã xóa. Xem [báo cáo hiện hành 04/10/2026](../project_progress_20261004/REPORT.md) và [audit dọn](../primary_cleanup_20261004/REPORT.md).

> Số liệu lịch sử trước lọc sách/điện thoại. Tập ứng viên hiện là 600 ảnh; tổng detection hiện là 2.966 file. Xem reports/results/book_phone_balance_20261003/REPORT.md.

# Dữ liệu sau khi bỏ archive.zip — 03/10/2026

Đã xóa archive.zip (8.884 ảnh trong gói) và thư mục classroom_archive_unknown theo yêu cầu. Giải phóng 1.070.997.564 byte, khoảng 1,071 GB thập phân. Bộ này chưa được giải nén hoặc nhập dataset chính, nên không có ảnh/nhãn của nó trong dataset chính hoặc tập ứng viên để xóa thêm. Danh sách audit đã bỏ toàn bộ 8.884 dòng thuộc bộ này.

## Dữ liệu còn giữ

| Phần | Số file ảnh | Dung lượng |
|---|---:|---:|
| Student Behaviour Detection v6 ZIP | 4.065 | 271,85 MB nén |
| Student Classroom Activity v6 ZIP | 779 | 9,81 MB nén |
| Dataset chính project8_v0.2 | 1.800 | 113,65 MB ảnh + TXT |
| Ứng viên book/cell phone | 2.596 | 182,40 MB ảnh + TXT |

Hai ZIP còn giữ tổng 281,66 MB và 4.844 file ảnh nguồn cho phần tuần 4–6. Dataset chính + ứng viên có 4.396 file ảnh, không phải 4.396 ảnh đã hoàn thiện đủ tám lớp; ứng viên trích từ ZIP nên không cộng nó lần nữa vào số ảnh nguồn.

## Chất lượng và nhãn

- Dataset chính giữ nguyên, draft chờ hoàn thiện nhãn/review.
- Ứng viên: 1.938 train / 369 val / 289 test; 16.985 box book, 5.070 box cell phone. Đã loại nhãn hành vi trong bản ứng viên, nhãn gốc giữ trong ZIP.
- 2.596 ảnh ứng viên pass kiểm kỹ thuật; không trùng bytes với dataset chính hoặc trong tập ứng viên. Chưa kiểm gần trùng/độc lập cảnh; có augmentation và frame video.
- Ứng viên còn thiếu nhãn các vật thuộc lớp mục tiêu khác xuất hiện trong ảnh; chưa khóa release hoặc train.
- Phone của Student Classroom Activity là nhãn hành vi bao người, không đổi thành cell phone.

## Đối chiếu

- data/raw/week456_behavior/README.md
- data/dataset/student_behaviour_project8_candidates_v0.1/manifest.csv
- data/dataset/student_behaviour_project8_candidates_v0.1/validation.json
- reports/results/zip_label_audit_20261003/organized_summary.json
- reports/results/zip_label_audit_20261003/image_decisions.csv
