# Bàn giao phần mềm tuần 1 — 03/10/2026

> Cập nhật 04/10/2026: các ảnh tham khảo/review COCO128 và gói ZIP demo cũ đã dọn theo yêu cầu. App hiện dùng ảnh train từ dataset chính; CLI dùng src.inference.demo. Các kết quả phần mềm dưới đây là bằng chứng lịch sử; G1 vẫn chờ webcam thật và review người.

Đã hoàn thiện phần mềm webcam cục bộ. **G1 vẫn chờ kiểm webcam thật và review ảnh của người trong nhóm.** Không dùng kiểm thử mô phỏng hoặc kết quả COCO128 để thay bằng chứng đó.

## Chạy trên từng máy Windows

1. Giải nén gói demo vào một thư mục riêng. Cài Python 3.11 từ python.org, giữ Python Launcher khi cài.
2. Mở `setup-demo.cmd` một lần. Cần internet để cài dependencies; script tạo `.venv` riêng và kiểm model. Không sao chép `.venv` từ máy khác.
3. Mở `run-demo.cmd`. Trình duyệt mở ứng dụng tại `http://127.0.0.1:8501`.
4. Chọn **Webcam trực tiếp**, chỉ số **0**, rồi **Bật webcam**. Camera USB có thể dùng chỉ số 1 hoặc 2.
5. **Dừng webcam** giải phóng thiết bị và bỏ hình cũ. **Mở lại webcam** đóng rồi mở một phiên mới. Đổi model/confidence/chỉ số camera sẽ dừng phiên; bấm Bật để áp dụng cấu hình mới.

Ứng dụng chạy trên chính máy của mỗi thành viên và truy cập webcam của máy đó. Gửi link localhost của một người cho người khác không mở webcam của người nhận. Hình chỉ xử lý trong RAM, không tự ghi video/ảnh. Đổi sang nguồn ảnh/video sẽ dừng webcam; đóng hoặc tải lại trang thì worker cũ tự dừng sau khoảng 15 giây không nhận heartbeat (có thể lâu hơn nếu driver đang chặn đọc hoặc đang suy luận).

Nếu lỗi: đóng Camera/Teams/Zoom hoặc phiên ứng dụng đang dùng camera; mở Settings → Privacy & security → Camera và cho ứng dụng desktop truy cập camera; kiểm nắp che/cáp USB; thử chỉ số khác. Khi port 8501 đang được dùng, dùng port khác:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/app.py --server.port 8511 --server.headless false
```

## macOS/Linux

Cần Python 3.11 và webcam/driver hoạt động. Mở terminal ở thư mục giải nén:

```sh
python3.11 scripts/setup_demo.py
.venv/bin/python -m streamlit run src/app.py --server.headless false
```

Cho Terminal/Python truy cập Camera trên macOS; Linux cần quyền đọc thiết bị video. Backend được chọn theo hệ điều hành. Bootstrap macOS/Linux mới là đường hỗ trợ trong mã, **chưa được kiểm trên máy thật của hai hệ này**; không cam kết tương thích mọi máy trước khi nhóm thử.

## Bài kiểm webcam cho mỗi thành viên

- Bật → có hình và dự đoán; đưa vật thuộc tám lớp vào cảnh.
- Dừng → hình cũ biến mất; camera không còn bị phiên đó giữ.
- Bật lại → có hình; Mở lại → có hình sau khi mở phiên mới.
- Chuyển nguồn sang ảnh → webcam dừng.
- Chạy liên tục 10 phút → không crash hoặc trễ tăng dần.
- Sau khi dừng, tải **kết quả kiểm webcam**; điền `data/templates/week1_machines.csv` với người, máy, kết quả và đường dẫn JSON thật.

FPS là số frame xử lý / thời gian phiên, có gồm thời gian mở camera. p95 tính từ khi Python nhận frame đến khi dự đoán sẵn sàng, trên tối đa 300 frame gần nhất; không phải trễ từ cảm biến đến màn hình. Mục tiêu FPS/p95 phải ghi số đo thật; không bảo đảm đạt trên mọi CPU.

## Review ảnh tuần 1

Trên workspace này đã tạo `runs/week1/project8_review40/index.html`: 40 ảnh COCO128 tham khảo, có đủ tám lớp trong cả bộ, mỗi ảnh có nhãn nguồn và dự đoán YOLO26n pretrained để đối chiếu. Mở HTML trên trình duyệt; người trong nhóm xem từng ảnh và ghi tên, vật bỏ sót, báo sai, lỗi hộp/lớp vào `review.csv` cùng thư mục. Chỉ đổi trạng thái sau khi thực sự xem. Nhãn COCO có thể thiếu/khác phạm vi table/book của nhóm; ghi nhận khi phát hiện.

Ảnh review COCO128 và script tạo review cũ đã dọn theo yêu cầu ngày 04/10/2026. Để review hiện hành, mở giao diện **Ảnh dữ liệu chính**, chọn ảnh train và ghi review trong hồ sơ dataset chính. Không tạo bản sao ảnh để làm thư viện riêng.

Script từ chối ghi đè output đã có. Tất cả trạng thái ban đầu là `pending`; dự đoán ở confidence 0.25 không tự tạo số TP/FP/FN hoặc mAP. COCO128 là tham khảo kỹ thuật, không phải validation độc lập. Bộ demo ZIP không kèm dữ liệu ảnh; bộ review nằm trong workspace hiện tại.

## Bằng chứng đã có

| Hạng mục | Kết quả |
|---|---|
| Kiểm môi trường/checkpoint | `scripts/setup_demo.py --check` thành công; `pip check` không có xung đột |
| Toàn bộ kiểm thử | 30/30 pass, không skip; gồm 5 test camera và 1 test UI webcam dùng thiết bị mô phỏng + YOLO thật |
| CLI video | 12 frame, mở lại 1 lần; `runs/week1/webcam_update_video_regression/summary.json` |
| Trình duyệt thật | Hiển thị nút webcam và thông báo không có thiết bị đọc được; không có lỗi/warning console ở lần kiểm |
| Camera thật máy hiện tại | Chỉ số 0 không đọc được; `runs/week1/webcam_availability.json`. Chưa đạt kiểm webcam live |
| Bộ review tám lớp | 40 ảnh và dự đoán thật đã tạo; review người đang pending |
| Máy thành viên | Chưa có log kiểm trực tiếp; dùng mẫu `week1_machines.csv` |

Model baseline, ảnh/video và camera tuần 1 không cập nhật weights. Phần tuần 2 và fine-tune tuần 3 không bị thay đổi bởi lần bàn giao này. Các thay đổi Git có sẵn được giữ nguyên; chưa commit/push.

Nguồn API: [Streamlit fragment](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment), [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html).
