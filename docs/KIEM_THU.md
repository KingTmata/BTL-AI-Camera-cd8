# Chạy kiểm thử và xem kết quả

Mở PowerShell ở gốc repo. Lệnh bên dưới dùng `.venv` đã cài; không cần activate hay có Python trong PATH.

## 1. Mở giao diện xem ảnh

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/app.py
```

Mở `http://127.0.0.1:8501`. Giữ terminal đang chạy; Ctrl+C để dừng server. Nếu cổng bận, thêm `--server.port 8502` và mở đúng cổng. Hai lệnh khởi động sẽ tạo hai server, không cần mở lại terminal mỗi lần đổi ảnh.

1. Chọn “Bộ mẫu COCO128”, chọn ảnh bằng danh sách hoặc mở “Duyệt 50 ảnh thu nhỏ” rồi bấm **Xem** dưới ảnh.
2. Bấm **Phân tích ảnh**. Xem ảnh gốc và dự đoán cạnh nhau. Phóng to bằng nút góc ảnh.
3. Chọn một dòng trong bảng đối tượng để xem vùng ảnh cắt, confidence và ID. Số thứ tự trong bảng khớp số trên hộp.
4. Mở “Đối chiếu nhãn gốc COCO128” để xem hộp tham khảo. Đây là đối chiếu bằng mắt, chưa chấm mAP.
5. Thay confidence rồi chạy lại. Không coi confidence là xác suất đảm bảo hoặc accuracy của model.
6. “Lớp hiển thị” lọc hộp hiện có. Bỏ hết lớp sẽ hiện trạng thái rỗng, không chạy YOLO với tất cả 80 lớp.
7. Chọn “Ảnh của bạn” để mở JPG/PNG/WebP; chọn “Khung hình video” để kéo thanh Frame. Đặt video riêng trong `data/video_dev/` rồi tải lại trang.
8. Tải PNG/JSON/CSV khi muốn giữ kết quả. File upload chỉ giữ trong phiên; web không tự thêm ảnh vào train hoặc tự ghi review.

Ảnh mẫu `000000000283` đã được dùng để kiểm nhận chai. Ảnh `000000000328` là ca baseline từng bỏ sót điện thoại: dùng để nhìn lỗi, không sửa nhãn thành “không có điện thoại” chỉ vì model không thấy.

## 2. Kiểm mã tự động

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -m pip check
```

`test_inspection.py` kiểm nhầm ID 80→4, tọa độ hộp, nhãn hỏng, crop ra ngoài ảnh, chuyển RGB/BGR. `test_app.py` mở app bằng Streamlit AppTest, chạy YOLO thật trên mẫu chai, kiểm lọc rỗng và việc ẩn kết quả cũ khi đổi confidence/nguồn. Bài integration tự skip nếu không có weights/COCO128; skip không được tính là đã test giao diện. AppTest không thay việc bấm thử bảng/crop trong trình duyệt.

Kiểm bằng tay sau sửa UI: chọn ảnh khác → chạy → chọn đối tượng → tải file → đổi confidence → chạy lại → mở nguồn upload khi chưa chọn file → đổi sang video và frame khác. Không được nhìn thấy dự đoán của ảnh/frame trước.

## 3. Test ảnh bằng CLI

```powershell
& .\.venv\Scripts\python.exe -m src.inference.demo --source data/reference/coco128/images/train2017/000000000283.jpg --no-window --save --output-dir runs/manual/image_bottle
```

Kỳ vọng: `summary.json` và `000000000283_detected.jpg` trong thư mục output. Model cố định `conf=0.25`, CPU, `imgsz=640`, lọc bốn lớp. Lệnh cũ `-m src.week1_demo` vẫn dùng được. Output có thể thay đổi nếu đổi weights/phiên bản; không biến confidence cụ thể thành tiêu chí chất lượng.

## 4. Test video và đóng/mở lại

```powershell
& .\.venv\Scripts\python.exe -m src.inference.demo --source demo/smoke/coco128_slideshow_1.mp4 --no-window --save --reopen-at 6 --max-frames 12 --output-dir runs/manual/video_reopen
```

Kỳ vọng trên file mẫu cục bộ: 12 frame được xử lý, `reopen_count=1`, `summary.json` và `detected.mp4`. Nguồn được mở lại từ đầu ở frame xử lý thứ 6, vì vậy đây là test lifecycle, không phải đo toàn bộ clip theo timeline. File slideshow chỉ có trên máy đã tạo; người clone repo dùng video của mình và điều chỉnh max-frames cho phù hợp.

## 5. Test webcam khi có thiết bị

```powershell
& .\.venv\Scripts\python.exe -m src.inference.demo --source 0
```

Đưa người/chai/điện thoại/laptop vào cảnh; `r` đóng rồi mở lại camera, `q` dừng. Chạy lại lệnh để xác nhận mở lần nữa. Nếu camera không phải 0, thử chỉ số đúng của thiết bị. Hiện nhóm chưa có webcam nên chưa thể nghiệm thu mục này. Web đang hỗ trợ xem ảnh/frame; CLI mới là lệnh kiểm camera live.

## 6. Đọc kết quả đúng nghĩa

| Giá trị | Hiểu đúng |
|---|---|
| Confidence của hộp | Điểm dự đoán từng hộp; không phải % accuracy của toàn dự án |
| `detections_by_class` trong CLI video | Tổng số hộp quan sát qua frame; cùng một người qua 10 frame có thể góp 10 hộp, không phải 10 người riêng |
| `processing_fps` | Số frame / thời gian đọc-xử lý-vẽ-lưu/chờ UI trong lần chạy; không gồm load model |
| `processing_ms` trên web | Predict sau khi nạp model, có thể có warmup lần đầu; không phải độ trễ camera đến màn hình |
| Test mã pass | Mapping và các ca được kiểm hoạt động đúng; chưa chứng minh chất lượng mô hình |

Muốn đo FPS demo: dùng clip thật, cấu hình cố định, chạy đủ dài sau warmup và ghi thông số máy. Các slideshow 10–12 frame hiện có chỉ là smoke test.

## 7. Lỗi thường gặp

- **No module named streamlit/ultralytics:** gọi sai Python; dùng đúng `.venv\Scripts\python.exe`.
- **Không thấy ảnh mẫu:** tải lại COCO128 theo README. Không cần tải để dùng ảnh riêng.
- **Không có hộp:** xem confidence, lớp hiển thị, kích thước vật và checkpoint. Có thể là model bỏ sót thật.
- **Sai profile checkpoint:** checkpoint phải là COCO80 hoặc đúng thứ tự project4; kiểm `model.names`.
- **Đổi ảnh vẫn thấy kết quả cũ:** phải là lỗi; ghi nguồn, confidence và bước tái hiện để sửa.
- **Train/val/test trong data khác tests/:** đọc [cấu trúc](CAU_TRUC_VA_LUONG.md), không đưa ảnh test vào code test.
