> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

# Báo cáo dọn dữ liệu và code — 04/10/2026

Giữ **5.002 ảnh chính / 5.002 TXT / 26.566 bounding box**, gồm 3.883 train, 798 val, 321 test. Không còn ảnh pending. Media, nhãn và split của bộ giữ lại không thay đổi trong quá trình dọn; checksum: `f762c7ad5f775c35f1717a123a7c1312095d0d6552618cd5c196a4035722cb46`.

| Phần đã xóa | Số lượng |
|---|---:|
| Bản sao ảnh, đã đối chiếu nội dung | 6.566 |
| Ảnh pending theo yêu cầu | 107 |
| Ảnh tham khảo hoặc kết quả thử | 189 |
| TXT nguồn/tham khảo cũ | 6.202 |
| TXT đi cùng ảnh pending | 107 |
| ZIP nguồn đã nhập/tham khảo | 3 |
| Manifest pending | 1 |
| ZIP sao lưu demo tuần 1, không có ảnh | 1 |

Đợt dọn dữ liệu xóa 13.175 file, giải phóng 881.483.721 byte. ZIP sao lưu code/weights tuần 1 được xóa riêng, thêm 4.982.427 byte. Các con số này chưa cộng kích thước code đã bỏ. Xóa theo nhóm nguồn, mỗi lượt tối đa 100 ảnh; kiểm bộ chính sau từng lượt.

Hai ZIP hành vi gốc phục vụ tuần 4–6 giữ nguyên, chứa 4.065 và 779 lượt ảnh. Đây là ảnh trong archive, tách khỏi số 5.002 ảnh đã giải nén. 600 ảnh được chọn từ các nguồn hành vi đã nằm trong bộ chính nên không cộng ZIP vào tổng ảnh độc lập.

Đã bỏ 11 file Python của pipeline nhập/chọn/gộp dữ liệu cũ và CLI tuần 1 trùng chức năng (9 file có sẵn, 2 file tạo trong đợt nhập classroom). Bỏ manifest mẫu COCO128 cũ; giao diện lấy tối đa 50 ảnh train từ manifest chính và đọc nhãn theo taxonomy project8. Giữ CLI/UI nhận dạng, validator/evaluator, công cụ tuần 2, setup và script báo cáo.

Kiểm thử sau sửa code: **27/27 pass, không skip**, bằng `.venv/Scripts/python.exe -B -m unittest discover -s tests -v`. `git diff --check` pass. Validator dữ liệu: 5.002 ảnh, không lỗi; kiểm lại checksum khi lập báo cáo hiện hành.

Nhãn vẫn là draft, chưa được review đầy đủ; chưa fine-tune hoặc nghiệm thu G2/G3. Test chưa có nhãn backpack/cup. Xóa bản sao không thay thế kiểm gần trùng, độc lập phiên hoặc duyệt nhãn.

Bằng chứng chi tiết: [kết quả dọn](../../../../reports/results/primary_cleanup_20261004/cleanup_result.json), [đối chiếu bản sao](../../../../reports/results/primary_cleanup_20261004/verified_duplicate_cleanup_plan.json), [đối chiếu nhãn](../../../../reports/results/primary_cleanup_20261004/label_equivalence_proof.json), [đối chiếu ZIP](../../../../reports/results/primary_cleanup_20261004/archive_content_proof.json), [ZIP sao lưu code](../../../../reports/results/primary_cleanup_20261004/code_backup_proof.json), [metadata 107 ảnh đã xóa](../../../../reports/results/primary_cleanup_20261004/deleted_pending_metadata.csv).

Bảng số liệu hiện hành: [báo cáo tuần 1–2–3](../project_progress_20261004/REPORT.md).
