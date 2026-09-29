# Dữ liệu

`dataset/images/{train,val,test}` chứa ảnh; `dataset/labels/{train,val,test}` chứa nhãn YOLO cùng tên. `raw/` giữ dữ liệu gốc được phép sử dụng. `video_dev/` và `video_test/` là các phiên độc lập. `manifest.csv` cần ghi nguồn, quyền, session và split cho từng ảnh.

Thư mục này hiện chưa có dữ liệu thật. Không đưa ảnh/clip cá nhân lên Git nếu chưa được phép.

Tuần 1 đã tải COCO128 vào `reference/coco128/` (không commit), tạo danh sách 50 ảnh ở `week1_coco128_manifest.csv` và 20 ảnh cần hai người review ở `week1_review_20.csv`. Tất cả dòng review vẫn là `pending`. Đây là ảnh tham khảo, chưa phải ảnh tự thu trong phòng học. Gói COCO128 tải về có 128 ảnh và 128 nhãn nhưng chỉ 126 cặp trùng tên; hai nhãn `000000000656`, `000000000659` không có ảnh tương ứng và hai ảnh `000000000250`, `000000000508` không có nhãn tương ứng. Danh sách 50 ảnh chỉ lấy các cặp đầy đủ.
