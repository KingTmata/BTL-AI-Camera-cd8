# Quy chuẩn gán nhãn thử — phiên bản tuần 1

**Mục đích:** để hai thành viên cùng gán 20 ảnh rồi so lỗi trước khi thu và gán dataset chính. Bộ COCO128 hiện chỉ là ảnh tham khảo kỹ thuật; chưa có ảnh tự thu đã được review.

## Bốn lớp cố định

| ID dự án | Tên nhãn | ID COCO của model pretrained | Ghi chú |
|---:|---|---:|---|
| 0 | `person` | 0 | Người thật nhìn thấy trong cảnh. |
| 1 | `bottle` | 39 | Chai, kể cả nắp/cổ/thân thấy được; không gán ly/cốc. |
| 2 | `cell phone` | 67 | Điện thoại, không kéo hộp bao cả bàn tay. |
| 3 | `laptop` | 63 | Máy tính xách tay mở hoặc gập khi nhận dạng chắc chắn. |

ID trong file nhãn **dataset bốn lớp** phải dùng cột `ID dự án`. Nhãn gốc COCO128 dùng cột `ID COCO`; không chép nguyên ID 39/67/63 vào dataset của nhóm.

## Quy tắc hộp

1. Bao phần **nhìn thấy và nhận dạng được** của đối tượng; cắt hộp tại mép ảnh. Không đoán phần hoàn toàn khuất.
2. Gán mọi đối tượng thuộc bốn lớp nhìn thấy được trong ảnh. Không xóa điện thoại nhỏ chỉ vì khó thấy; đưa ảnh mơ hồ ra danh sách cần trao đổi.
3. Người bị bàn che: hộp bao phần nhìn thấy; đáy hộp không được hiểu là chân. Laptop mở: bao màn hình và thân/phím. Vật trên màn hình/poster không tính là vật thật trong phòng theo phạm vi nhóm.
4. Ảnh không có bốn lớp là ảnh âm tính và có file nhãn `.txt` rỗng. Ảnh **chưa gán nhãn** phải được đánh dấu riêng, không coi là âm tính.
5. Nhãn YOLO một dòng: `class_id x_center y_center width height`, tọa độ chuẩn hóa 0–1; `width` và `height` >0, hộp không vượt khung ảnh.

## Thử chung 20 ảnh

Danh sách ở [data/week1_review_20.csv](data/week1_review_20.csv). Mỗi người gán độc lập bốn lớp, sau đó so từng ảnh: thiếu/thừa vật, sai lớp, hộp quá rộng/hẹp, vật bị che và ảnh âm tính. Ghi người gán, điểm bất đồng và quyết định cuối trong CSV. Chỉ đổi `review_status` thành `reviewed` sau khi **hai người thực sự xem và thống nhất**. COCO128 có rất ít ảnh laptop/điện thoại, nên cần bổ sung các lớp này khi thu dữ liệu phòng học.

Quy tắc chính thức cho dataset tự thu phải được cập nhật nếu hai người phát hiện trường hợp mơ hồ; lưu ngày và lý do sửa.
