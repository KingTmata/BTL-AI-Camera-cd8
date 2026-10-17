# Log triển khai video OpenCV và TUI

**Ngày ghi:** 15/10/2026 — Asia/Saigon.  
**Phạm vi:** công việc thực hiện trong chat này, từ khảo sát luồng video đến CLI, bộ xử lý video và giao diện terminal tương tác.

Log tổng hợp việc đã làm và bằng chứng hiện có; không thay thế [mục tiêu nghiên cứu chính thức](../../tasks/plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026). Số đo là kết quả của từng phép thử cụ thể, không phải cam kết cho mọi thiết bị hoặc video.

## 1. Khảo sát ban đầu

- Đọc ghi nhớ dự án và mục tiêu chính thức trước khi triển khai.
- Kiểm tra mã đọc nguồn, detector, worker webcam, giao diện Streamlit, cấu hình và các báo cáo hiệu năng đã có.
- Xác định luồng video cũ chỉ mở file rồi chọn từng frame để phân tích như ảnh; chưa phát và nhận diện video liên tục.
- Xác định webcam có worker nền, nhưng hiển thị qua JPEG/base64 và `st.image`; chưa có bằng chứng khẳng định lỗi dưới 1 FPS hoàn toàn do Streamlit.
- Phân biệt FPS nhận diện, FPS hiển thị, thời gian khởi động, thời gian xử lý toàn video và thời gian có kết quả đầu.

## 2. Các lựa chọn đã thống nhất với người dùng

| Nội dung | Lựa chọn |
|---|---|
| Giao diện phát video đầu tiên | OpenCV + terminal, độc lập Streamlit |
| Chế độ Full | Nhận diện mọi frame, xuất xong toàn bộ rồi mới phát |
| Chế độ Fast | Lấy mẫu frame, xử lý và phát cuốn chiếu |
| FPS video | Giữ theo nguồn; 30 FPS khi nguồn là 30 FPS |
| Mục tiêu dưới 3 giây | Kết quả đầu đã hiển thị của Fast khi model sẵn sàng; báo cold start riêng |
| Quy mô | Bắt đầu video 1–3 phút; chuẩn bị kiến trúc cho 30 phút–6 giờ |
| Thiết bị | Đo máy hiện tại; RTX 5060 Laptop cần phép đo riêng |
| Số đo 120 ms/ảnh | Chỉ là số đo một thiết bị, không dùng làm chuẩn chung |
| Webcam | Để đợt sau; không sửa luồng webcam trong công việc này |
| Giao diện điều khiển bổ sung | TUI tối giản, nút bấm ngay trong terminal |

Đã lưu [kế hoạch triển khai riêng](../../tasks/video/video_processing_plan_20261015.md), cập nhật liên kết/ghi nhớ và giữ nguồn mục tiêu chính thức ở vị trí cũ.

## 3. CLI và bộ xử lý video đã viết

Tạo package `src/video/`, chạy bằng `python -m classroom_ai.video`.

| File | Công việc |
|---|---|
| `src/classroom_ai/video/__init__.py`, `__main__.py` | Điểm vào, tham số CLI, mở TUI khi được yêu cầu |
| `src/classroom_ai/video/core.py` | Frame/index/timestamp, lấy mẫu theo thời gian, bộ đệm giới hạn, mailbox và đồng hồ phát |
| `src/classroom_ai/video/media.py` | Kiểm nguồn bằng ffprobe, giải mã OpenCV, xuất MP4 bằng FFmpeg |
| `src/classroom_ai/video/pipeline.py` | Điều phối Full/Fast, nạp model/warm-up, tái sử dụng detector và khóa Training, ghi kết quả/số đo |
| `src/classroom_ai/interfaces/native_video.py` | Cửa sổ OpenCV trên main thread, hiển thị đúng frame, pause/stop và xử lý đóng cửa sổ |

### Chế độ Full

- Đọc và nhận diện từng frame, vẽ kết quả rồi ghi dần ra đĩa.
- Dùng FFmpeg xuất H.264/MP4, giữ FPS/số frame và track âm thanh đầu, mã hóa AAC.
- Kiểm file đầu ra trước khi đổi tên từ `annotated.partial.mp4` thành `annotated.mp4`.
- Xong bước xuất/kiểm mới mở phát video kết quả. Cửa sổ OpenCV xem trước không phát âm thanh; file MP4 có âm thanh.
- Không tuyên bố hoàn thành toàn video dưới ba giây.

### Chế độ Fast

- Mặc định lấy mẫu 5 frame/giây theo timestamp, không mặc định mọi nguồn đều 30 FPS.
- Một tác vụ nhận diện đang chạy và tối đa một tác vụ chờ; tác vụ chờ cũ có thể bị thay và được đếm rõ.
- Bộ đệm phát mặc định 250 ms, giới hạn tối đa 12 frame/128 MiB ảnh thô.
- Chỉ vẽ box lên đúng frame đã nhận diện; thông tin gần nhất ghi timestamp/tuổi kết quả. Chưa bổ sung tracking.
- Video vẫn chạy theo thời gian nguồn; số frame nhận diện và frame hiển thị được báo riêng.

### Đo tiến độ và chi phí

- Báo các bước: kiểm nguồn → runtime/thiết bị → nạp model → warm-up → xử lý → hoàn tất xuất/kiểm → phát.
- Có thời gian đã dùng, phần trăm và ETA khi tính được, RAM/VRAM khi khả dụng.
- Có thông báo `MODEL READY` với chi phí nạp/warm-up và RAM; không tạo phần trăm giả khi load.
- Báo riêng thời gian có kết quả trong bộ nhớ và thời gian đã trình bày lên giao diện; phân biệt cold/warm.
- Ghi `metadata.json`, `detections.jsonl`, `report.json`; Full có thêm video và log encoder.
- Giới hạn lịch sử mẫu đo/kết quả trong RAM, ghi dự đoán dần ra đĩa.
- Kiểm CFR theo timestamp, báo nguồn VFR/gián đoạn/hỏng; không tự chuyển VFR âm thầm.

## 4. Tối ưu và sửa lỗi trong quá trình kiểm chứng

- Phép thử native ban đầu cho thấy `waitKey(1)` gây khoản chờ khoảng 15–16 ms mỗi lần trên máy này; đã chuyển sang `pollKey()` để xử lý sự kiện không kèm khoản chờ đó.
- Giữ đồng hồ phát theo timestamp; tách pause/buffering, bổ sung xử lý pause lồng trong buffering.
- Báo FPS tính cả wall time và FPS trừ pause tường minh; không loại decoder buffering để làm đẹp số liệu.
- Không mang kết quả box của frame cũ sang frame mới.
- Xử lý trường hợp cửa sổ đã bị window manager đóng, để poll/cleanup không làm hỏng đường kết thúc phiên.
- Tái sử dụng cơ chế bảo vệ Training/inference hiện có; không bỏ khóa để tăng FPS.

Các thay đổi trên thuộc module video mới; không kết luận chúng đã sửa nguyên nhân webcam dưới 1 FPS của Streamlit.

## 5. TUI tối giản đã bổ sung

Tạo `src/classroom_ai/interfaces/terminal.py` và `src/classroom_ai/interfaces/terminal_process.py`.

- Dùng **Textual 8.2.8**, cài trong `.venv` dự án và khai báo ở `requirements/video_tui.txt`.
- Có ô đường dẫn, cây thư mục chọn video, Fast/Full, Chạy/Dừng, thanh tiến độ và khung log.
- Hỗ trợ chuột, Tab/Enter; Ctrl+R chạy, Ctrl+D dừng, Ctrl+Q/Ctrl+C thoát.
- Tắt animation/con trỏ nhấp nháy, cập nhật log một lần/giây, giữ tối đa 200 dòng hiển thị.
- TUI khởi chạy backend bằng subprocess/argv, không shell; process giao diện không import torch, cv2 hoặc Streamlit.
- Chỉ trao đổi cấu hình, cờ dừng và văn bản; không truyền frame/model vào TUI.
- Backend kiểm cờ dừng tối đa mỗi 200 ms, tránh kiểm file theo từng nhịp render hoặc từng frame preflight.
- Khi thoát: yêu cầu dừng trước; sau 10 giây mới buộc dừng cây process thuộc lượt chạy. Force-stop có thể không tạo được báo cáo cuối.
- Cập nhật `run_video.cmd`: không tham số thì mở TUI; có tham số thì tiếp tục chạy CLI như trước.

## 6. Kiểm thử và số đo đã có

### Kiểm thử chức năng

- Thêm `tests/unit/test_video_core.py`, `tests/integration/test_video_pipeline.py`, `tests/integration/test_video_tui.py`.
- Nhóm cuối cùng: **25/25 test video/TUI đạt**.
- Suite toàn dự án: **96/97 đạt**. Test augmentation cũ lỗi do thiếu `albumentations`; không xóa/skip test hoặc sửa logic augmentation để đạt xanh.
- Kiểm `compileall` và `pip check` đạt.
- Test bao phủ lấy mẫu/timestamp, giới hạn bộ đệm, mailbox, pause/buffering, frame mapping, CFR/VFR, FPS phân số, âm thanh, lỗi detector/encoder, stop/đóng cửa sổ và thao tác TUI.
- Có integration dùng FFmpeg thật và backend subprocess thật, gồm dừng ngay khi preflight.
- Khi test Streamlit/Textual bị giới hạn socket localhost của sandbox, đã chạy lại với quyền phù hợp; dùng TEMP/TMP trong workspace và UTF-8 cho môi trường Windows.

### Số đo video native

Nguồn **tổng hợp 640×360, 3 phút, 30 FPS**; R04, một lượt, imgsz 1280, CPU:

- Đọc 5.400 frame, nhận diện 900 frame lấy mẫu, hiển thị 5.395 frame.
- Bỏ 5 frame hiển thị, khoảng **0,0926%**.
- Khoảng **29,97 FPS trừ pause tường minh**; **28,02 FPS tính cả pause**. Phiên có 12,572 giây pause được ghi nhận.
- Kết quả đầu đã trình bày: khoảng **131 ms warm**, **10,7 giây cold**.
- Callback nhận diện trung bình khoảng **105,91 ms**, p95 khoảng **122,18 ms**.
- Đây là số đo trên nguồn tổng hợp/cấu hình đó, không phải 30 FPS nhận diện hoặc đánh giá độ chính xác lớp học.

Báo cáo và điều kiện chi tiết: [video native](native_20261015/results.md), [số đo](native_20261015/benchmarks.json).

### Số đo TUI

- Pilot headless 100×30, idle khoảng tám giây: process frontend khoảng **41,7 MiB RSS**, CPU khoảng **1,37% một core** trong cửa sổ đo ngắn.
- TUI điều khiển backend phát nguồn tổng hợp **5 giây, 30 FPS, không YOLO**: khoảng **29,994 FPS**, hiển thị đủ **150/150 frame**, không bỏ frame trong lượt này.
- Xác nhận process frontend không import torch/cv2/Streamlit.
- Không dùng phép thử ngắn/headless để cam kết overhead bằng không trên mọi terminal hoặc thiết bị.

Báo cáo: [TUI](tui_20261015/results.md), [số đo](tui_20261015/checks.json), [ảnh renderer](tui_20261015/preview.svg).

## 7. Tài liệu và hướng dẫn đã cập nhật

- [Kế hoạch video](../../tasks/video/video_processing_plan_20261015.md).
- [Hướng dẫn chạy và giải thích kiến trúc](../../docs/guides/video_processing.md).
- [Ghi nhớ dự án](../../tasks/project_memory.md) và liên kết từ README gốc.
- Giải thích cho người dùng: terminal là môi trường chứa giao diện chữ; TUI là giao diện tương tác; backend thực hiện công việc; OpenCV đọc/vẽ/phát frame, YOLO nhận diện, FFmpeg xuất video.
- Giải thích ảnh tiến độ người dùng gửi: Full đang xử lý khoảng 230/2.366 frame (9,7%), ETA khoảng 200 giây, RAM khoảng 501,6 MiB. Đây chỉ là ảnh trạng thái đang chạy, chưa phải bằng chứng lượt video đó đã hoàn tất.

## 8. Cách mở hiện tại

Trong thư mục dự án, bấm đúp `run_video.cmd`, hoặc:

```powershell
.\run_video.cmd
```

Chọn video → chọn Nhanh/Đầy đủ → Chạy. Backend và cửa sổ OpenCV tự khởi động; không cần bật Streamlit. Máy mới cài thêm TUI bằng `pip install -r requirements/video_tui.txt` trong `.venv` dự án.

CLI trực tiếp vẫn dùng được:

```powershell
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode full
```

## 9. Chưa làm hoặc chưa nghiệm thu

- Chưa sửa/kiểm lại luồng webcam của Streamlit trong phạm vi này.
- Chưa benchmark RTX 5060 Laptop; không suy từ CPU/T4 ra tốc độ GPU đó.
- Chưa nghiệm thu video lớp học thực tế 1–3 phút, sync âm thanh bằng nghe/xem thủ công, soak 30 phút hoặc sáu giờ.
- Chưa triển khai tracking ID, phân tích chuỗi hành động hoặc đánh giá mức tương tác sinh viên.
- Chưa có cơ sở khẳng định hoàn thành toàn bộ video dài dưới ba giây, hoặc TUI không ảnh hưởng hiệu năng trên mọi máy.
- Các thay đổi trong phiên chưa được commit/push bởi trợ lý; workspace có sẵn những thay đổi khác và chúng được giữ lại.

**Lần ghi log này chỉ thêm tài liệu tổng hợp và liên kết ghi nhớ; không thay đổi mã xử lý, không chạy lại benchmark hay sửa các số đo đã có.**

## 10. Bổ sung sau lần ghi log đầu: tua full, E3 và confidence thấp

Theo yêu cầu tiếp theo của người dùng ngày 15/10/2026:

- Đổi mặc định riêng CLI/TUI video sang `weights/checkpoint07_matrix_20261012/E3_YOLO26n_Project8_v05_from_CP07_v04_best19_AdamW_lr0p0001_bestEpoch10_20261012.pt`, confidence **0,05**, imgsz vẫn 1280. “Min” được hiểu là mức thấp nhất của slider UI hiện có, không phải giới hạn của YOLO. Ngưỡng thấp có thể làm tăng báo nhầm. Không đổi mặc định Streamlit/Training; override tường minh vẫn giữ.
- Backend báo checkpoint/imgsz/confidence ngay khi bắt đầu; metadata giữ hash checkpoint. Các video và metadata cũ không bị đổi sang E3.
- Bổ sung `src/classroom_ai/video/playback.py`: phát/tua MP4 full đã hoàn tất bằng OpenCV, không chạy lại YOLO hoặc xuất lại. Thanh Frame/A/D lùi/tiến 5 giây, kể cả khi pause; hết video giữ frame cuối, Space phát lại, Q/Esc đóng. Fast chưa tua.
- Sửa chờ frame để thao tác tua có thể ngắt vòng chờ khi đang pause. Slider cập nhật tự động không tạo yêu cầu tua ngược. Phiên có tua ghi số lần/chi phí tua, không dùng FPS trung bình của phiên tương tác làm benchmark tuyến tính.
- Cập nhật README video, kế hoạch bổ sung và ghi nhớ. Giải thích chất lượng gốc: đọc/vẽ/xuất ở kích thước nguồn và FPS nguồn, preview thu nhỏ riêng, YOLO resize theo imgsz riêng. H.264 CRF 20/yuv420p và AAC hiện tại có nén mất dữ liệu; không tuyên bố giữ từng pixel, HDR/10-bit hay lossless. Kích thước lẻ được pad tối đa một pixel. File nguồn giữ nguyên.

### Kiểm chứng bổ sung

- Test mặc định E3/conf 0,05; slider và phím tua; tua tới trước/sau khi pause bằng FFmpeg thật và detector xác định, kiểm đúng thứ tự frame và không tăng số lần inference/export; kiểm tua ngắt vòng chờ khi pause.
- Smoke YOLO thật, CPU, E3/1280/conf 0,05, full không hiển thị: `runs/video_e3_smoke/20261015_213811_6073445eb2`. Hoàn tất 10/10 frame nhận diện/xuất trên nguồn tổng hợp 640×480, 5 FPS, 2 giây; tổng 12,66 giây gồm cold start. Hash E3: `9474d7f2b4c869e2101cf9f4f64b0f7f0411b859a8d31d3b2425aba48f479a13`. p50 model 128,4 ms, p95 138,8 ms. Đây là smoke ngắn, không phải benchmark chất lượng/30 FPS/RTX 5060.
- Các benchmark R04 trước đó vẫn là lịch sử của R04/0,25; không dùng làm số đo E3/0,05. Tua qua kiểm thử tự động, chưa nghiệm thu thao tác chuột và cảm nhận mượt trên video lớp học dài.
- Kiểm tra cuối sau sửa: **30/30 test video/TUI đạt** trong bộ toàn dự án; **101/102 test đạt**, một lỗi cũ do môi trường thiếu `albumentations` trong `test_e3_improvement_eval`, không thuộc video. Không bỏ/skip test để che lỗi. `compileall src/video` và `git diff --check` đạt (Git chỉ cảnh báo đổi LF/CRLF ở các file có thay đổi sẵn).

### Giải thích ảnh log backend người dùng gửi

Đã đối chiếu với `runs/video_tui/20261015_212254_9a55fad323/20261015_212254_e314cf7958/metadata.json` và `report.json`: lượt **full R04**, 1280, confidence 0,25, PyTorch CPU; chưa phải E3. Nguồn 1914×844, 30 FPS, 2.366 frame (~78,87 giây). Đã hoàn tất, nhận diện/xuất/hiển thị đủ frame, không bỏ frame hiển thị.

- FPS 25,27 tính cả 14,766 giây người dùng pause; bỏ thời gian pause là **30,007 FPS**. Đây là tốc độ phát video đã xử lý, không phải YOLO chạy 30 FPS.
- YOLO `model_processing`: p50 82,5 ms, p95 97,6 ms; tổng 198,06 giây. Callback gồm chi phí bao quanh: tổng 203,20 giây, p50 84,6 ms, p95 100,1 ms. RAM peak lấy mẫu 553,9 MiB.
- `first YOLO result available warm = 233,63 ms` chỉ là kết quả nhận diện đầu trong RAM. `first displayed ... = None` vì full không ghi mốc hiển thị đầu theo phép đo fast; không phải thông báo lỗi. Xuất xong tính từ khởi chạy khoảng 256,72 giây, sau đó kiểm nguồn kết quả trước khi phát.
- `source_preflight` ~15,09 giây là quét timestamp nguồn; `export_finalize_validate` ~14,13 giây là đóng encoder và kiểm file. `decode`, `draw`, `encode_write`, `draw_present` lần lượt là đọc frame, vẽ box, gửi encoder, và trình bày cửa sổ. `display_interval` p50 33,4 ms phù hợp nhịp 30 FPS.
- `total` là thời gian tích lũy công đoạn; `p50` trung vị, `p95` mức mà 95% mẫu không vượt quá (cửa sổ tối đa 2.000 mẫu gần nhất). Các công đoạn có thể lồng/đồng thời nên không cộng tất cả total để suy thời gian cả phiên.

Đây là bản tổng kết do backend tạo từ các số đo YOLO/OpenCV/FFmpeg, không phải một lỗi riêng của OpenCV.

## 11. Tracking full/person-only và màu box theo ID

Đã triển khai kế hoạch người dùng duyệt: ByteTrack session-local, hai hit liên tiếp, TTL3s, raw YOLO nguyên vẹn, EMA0,65 và nội suy<=0,3s có hai đầu cùng ID/cảnh. Full hai lượt không infer lại; cảnh cắt reset nhưng không tái dùng ID. Frame nội suy nét đứt và nhãn riêng, không gán confidence giả. Màu HSV giả ngẫu nhiên seed20261015 theo ID ổn định giữa frame, theo bổ sung của người dùng.

CLI/TUI có bật/tắt, mặc định off; chỉ Full. Thêm dependency tùy chọn lap0.5.12 trong `.venv`, không auto-install lúc chạy. Có progress hai bước, JSONL raw/rendered, cấu hình/ngưỡng/màu và chi phí riêng. Source/Streamlit/Training/fast không đổi hành vi; demo chọn LIGHT_CONTRAST seed123/conf0,05/imgsz1280 tường minh.

Kế hoạch đã lưu [video_tracking_plan_20261015.md](../../tasks/video/video_tracking_plan_20261015.md). Bằng chứng ba lượt YOLO thật cùng video541frame, test, rung tổng hợp và giới hạn: [video_tracking_20261015/RESULTS.md](tracking_20261015/results.md).47/47 test video/TUI qua,118/119 toàn dự án qua, một lỗi cũ thiếu albumentations. Không dùng208box nội suy/95khoảng hoặc nhiều box hơn để tuyên bố tăng Recall/giữ đúng ID trên video chưa cóGT. Chưa xác nhận GPU/long-run.

Kiểm riêng full với detector tổng hợp và đường tracker/render/encoder/HighGUI thật trên3phút30FPS: đủ5400frame xuất,5399frame hiện/1drop (0,01852%),30,0028FPS, không pause, playback hoàn tất. Đạt tiêu chí đường phát trong lượt này; không phải tốc độ YOLO hoặc nghiệm thu độ chính xác lớp học. Raw report ở báo cáo tracking.
