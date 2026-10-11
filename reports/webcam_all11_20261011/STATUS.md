# Yêu cầu chấm đủ R01–R11

Phạm vi: 11 checkpoint cũ, mỗi checkpoint chấm 512, 640, 768 và 960 trên cùng 2.500 ảnh validation; tổng **44 phép chấm**. Một lượt suy luận, có báo native và NMS theo lớp 0,6. Giữ các tiêu chí độ trễ, mAP và F1 cho lựa chọn webcam. Không tính baseline vào số 11 model.

| Checkpoint | Trạng thái |
|---|---|
| R01 | Khôi phục đúng checkpoint cũ; đã chấm xong cả4 kích thước trên GPU T4. |
| R02 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R03 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R04 | Giữ đúng checkpoint cũ; đủ 4 phép chấm Kaggle đã lưu local. |
| R05 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R06 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R07 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R08 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R09 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R10 | Chưa tìm được file trọng số sau lượt dọn artifact. |
| R11 | Chưa tìm được file trọng số sau lượt dọn artifact. |

Đã kiểm Downloads, workspace kể cả file ẩn, Documents, Desktop và các vị trí worktree/OneDrive tìm được. Log cleanup xác nhận các PT/ZIP campaign đã bị xóa và chỉ giữ R04. Version Kaggle đã xác nhận trước đó chỉ giữ output đến R01; chưa có nguồn khôi phục chín checkpoint còn thiếu.

Đã chuẩn bị `input_manifest.json` và runner validation riêng; runner không chứa lệnh train, chỉ ghi hoàn tất khi đủ 44 kết quả. Các checkpoint thiếu được ghi rõ, không lấy chỉ số1280 cũ thay cho số đo mới.

Preflight thực trên Kaggle đã xác nhận chỉ có R01/R04 và chín checkpoint thiếu. Kết quả R04 bốn size được dùng lại từ completionJSON đã xác nhận; R01 được chấm mới, cùng dataset/protocol, loader workers0. Đã đo xong runtime CPU R01/R04 trên cùng ảnh theo thứ tự xen kẽ. **Hiện2/11 model,8/44 phép chấm; trạng tháipartial_missing_checkpoints.** Chưa chốt model cuối cùng khi chưa đủ11. Kết quả/cache parity đã kiểm; khôngtrain hoặcchấmtest. Xem [báo cáo cập nhật](RESULTS.md).

Người dùng từng chốt chưa train. Đã hỏi liệu có cho phép tái train chín cấu hình từ checkpoint dự án hiện có để tạo model mới. **Chưa có trả lời thì chưa train.** Tái train không phải khôi phục đúng các checkpoint cũ, phải ghi thành experiment mới và giữ nguồn gốc riêng.

Phiên GPU đã dừng sau khi lưu đủ kết quả hiện có trên máy, tránh tốn quota khi còn thiếu checkpoint. Trình duyệt Kaggle vẫn mở. Quick Save mới vẫn chưa được xác nhận thành version9; không coi artifact online là đã lưu bền vững. CompletionJSON, bảng metric, runtime và ảnh xác nhận đã giữ trong thư mục báo cáo này.
