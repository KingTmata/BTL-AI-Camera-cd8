# Quy chuẩn gán nhãn — project8, cập nhật 02/10/2026

**Mục đích:** để hai thành viên cùng gán 20 ảnh rồi so lỗi trước khi thu và gán dataset chính. Bộ COCO128 hiện chỉ là ảnh tham khảo kỹ thuật; chưa có ảnh tự thu đã được review.

## Tám lớp cố định

| ID dự án | Tên nhãn | ID COCO của model pretrained | Ghi chú |
|---:|---|---:|---|
| 0 | `person` | 0 | Người thật nhìn thấy trong cảnh. |
| 1 | `table` | 60 dining table | Bàn ăn, bàn học, bàn làm việc; không gán tủ/kệ. Pretrained chỉ có dining table. |
| 2 | `chair` | 56 | Ghế; không gán sofa/giường. |
| 3 | `laptop` | 63 | Máy tính xách tay mở hoặc gập khi nhận dạng chắc chắn. |
| 4 | `cell phone` | 67 | Điện thoại; không bao cả bàn tay. |
| 5 | `backpack` | 24 | Balo; không gán túi xách hoặc vali. |
| 6 | `book` | 73 | Sách và vở, mở hoặc đóng; không gán tờ giấy rời. |
| 7 | `cup` | 41 | Cốc/ly uống nước, kể cả có quai; không gán chai. |

ID trong file nhãn **dataset tám lớp** phải dùng cột `ID dự án`. Nhãn gốc COCO128 dùng cột `ID COCO`; không chép nguyên raw ID vào dataset nhóm. Nhãn bốn lớp cũ có bottle phải được gán lại; đổi cấu hình không tự chuyển nhãn. Mapping được quản lý tại `src/classes.py`.

## Quy tắc hộp

1. Bao phần **nhìn thấy và nhận dạng được** của đối tượng; cắt hộp tại mép ảnh. Không đoán phần hoàn toàn khuất.
2. Gán mọi đối tượng thuộc tám lớp nhìn thấy được trong ảnh. Không xóa điện thoại nhỏ chỉ vì khó thấy; đưa ảnh mơ hồ ra danh sách cần trao đổi.
3. Người bị bàn che: hộp bao phần nhìn thấy; đáy hộp không được hiểu là chân. Laptop mở: bao màn hình và thân/phím. Vật trên màn hình/poster không tính là vật thật trong phòng theo phạm vi nhóm.
4. Ảnh không có tám lớp là ảnh âm tính và có file nhãn `.txt` rỗng sau review người thứ hai. Ảnh **chưa gán nhãn** phải được đánh dấu riêng, không coi là âm tính.
5. Nhãn YOLO một dòng: `class_id x_center y_center width height`, tọa độ chuẩn hóa 0–1; `width` và `height` >0, hộp không vượt khung ảnh.

## Thử chung 20 ảnh

Tuần 1 dùng bộ review 40 ảnh tám lớp theo [bàn giao hiện hành](docs/WEEK1_HANDOFF.md); danh sách 20 ảnh bốn lớp cũ chưa có kết quả đã được bỏ. Chọn mẫu có đủ tám lớp để hai người gán độc lập, rồi so thiếu/thừa vật, sai lớp, hộp và vật bị che. Ghi người gán, reviewer và quyết định trong `data/review.csv`; chỉ đặt `approved` khi thực sự xem và thống nhất. Review mục tiêu 20%, tối thiểu 10% mỗi split, có mẫu mọi lớp và toàn bộ ca nghi ngờ/âm tính.

Quy tắc chính thức cho dataset tự thu phải được cập nhật nếu hai người phát hiện trường hợp mơ hồ; lưu ngày và lý do sửa.
