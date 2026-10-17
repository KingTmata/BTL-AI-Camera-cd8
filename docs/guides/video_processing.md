# Video local: OpenCV + terminal

Chạy từ thư mục gốc repository. Cần môi trường `.venv` hiện có và **ffmpeg, ffprobe trong PATH**. CLI video dùng môi trường hiện có; TUI nút bấm dùng thêm Textual tùy chọn. Không cài `opencv-python-headless`: bản này cần HighGUI để mở cửa sổ.

## Giao diện bấm trong terminal

**Bấm đúp `run_video.cmd`**, hoặc chạy `.\run_video.cmd` không tham số. Giao diện tối giản có ô đường dẫn, nút **Chọn…**, chế độ **Nhanh/Đầy đủ**, **Chạy/Dừng**, thanh tiến độ và khung log. Dùng chuột hoặc Tab/Enter; Ctrl+R chạy, Ctrl+D dừng, Ctrl+Q thoát. Nút Chọn mở cây thư mục ngay trong terminal; có thể dán đường dẫn thư mục khác, Enter rồi chọn file video.

Đã cài Textual vào `.venv` trên máy triển khai. Máy mới cài phần tùy chọn bằng:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/video_tui.txt
.\run_video.cmd
# Hoặc chọn sẵn thiết bị/cấu hình khi mở TUI:
.\run_video.cmd --tui --device cuda:0 --imgsz 1280
```

Mặc định video dùng E3/1280/confidence 0,05/device auto (checkpoint đầy đủ bên dưới). TUI không tự bắt đầu xử lý khi mở. Cửa sổ video OpenCV xuất hiện khi bấm Chạy; Space trong cửa sổ video để pause/resume. Tiến độ và log trong terminal cập nhật một lần/giây, giữ tối đa 200 dòng hiển thị. Tắt animation và con trỏ nhấp nháy. Trạng thái nạp model có chữ/thời gian, không phần trăm giả.

Mỗi lượt có thư mục mới dưới `runs/video_tui`, gồm `console.log` và thư mục kết quả backend. Dừng gửi cờ cho backend kiểm tra tối đa mỗi 200 ms, đợi tác vụ native đang chạy hoàn tất. Thoát TUI yêu cầu dừng trước; sau 10 giây mới buộc dừng cây process do TUI tạo để tránh bỏ process chạy nền. Ép thoát có thể để lại file partial và không có báo cáo cuối; không coi đó là completed.

### Các phần liên quan thế nào?

```text
Terminal: Windows Terminal / cửa sổ console
  └─ TUI Textual: ô nhập, nút bấm, tiến độ
       ├─ gửi cấu hình và lệnh dừng
       └─ nhận log văn bản, không nhận hình ảnh
            ↕
       Backend Python ở process riêng
         ├─ OpenCV: giải mã video thành frame trong RAM
         ├─ YOLO: nhận diện đối tượng trên frame được chọn
         ├─ OpenCV: vẽ box và mở cửa sổ phát video
         └─ FFmpeg: xuất MP4/âm thanh trong chế độ full
```

**Terminal** là nơi hiển thị ứng dụng chữ và nhận thao tác. **PowerShell/cmd** là shell dùng để khởi chạy lệnh. **TUI** (Text User Interface) là giao diện tương tác nằm trong terminal. **Backend** ở đây là process Python local thực hiện công việc, không cần web server. **OpenCV** là thư viện xử lý ảnh/video mà backend sử dụng; YOLO thực hiện nhận diện, còn FFmpeg thực hiện xuất video.

Process TUI không import torch, cv2 hoặc Streamlit; frontend không giữ model hay frame. Vì vậy nhịp cập nhật chữ 1 Hz độc lập với nhịp phát video 30 FPS. TUI vẫn có chi phí CPU/RAM riêng. Bằng chứng thử ngắn và giới hạn đo nằm ở [báo cáo TUI](../../reports/video/tui_20261015/results.md); không cam kết không có overhead trên mọi thiết bị.

## Chạy

```powershell
# Nhanh: phát theo tốc độ gốc, lấy mẫu 5 frame/giây cho YOLO.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast

# Đầy đủ: nhận diện mọi frame, xuất xong mới phát kết quả.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode full

# Demo trên laptop GPU: chỉ rõ GPU, không âm thầm chuyển CPU nếu lỗi.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast --device cuda:0 --imgsz 1280

# Đo riêng đường phát, không nạp YOLO.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast --baseline

# Đo nhận diện/xuất video không có cửa sổ.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode full --device cpu --no-display

# Chọn checkpoint và sampling tường minh.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast --weights "weights\BEST_Project8.pt" --sample-fps 5 --buffer-ms 250
```

Tương đương: `.venv\Scripts\python.exe -m src.video ...`. `--help` liệt kê tham số. `Space` tạm dừng/tiếp tục **phát**; `Q`, `Esc` hoặc đóng cửa sổ để dừng. Đang xử lý toàn bộ có thể dừng bằng Q/Esc hoặc Ctrl+C, không tạm dừng inference. Khi phát kết quả **full**, kéo thanh `Frame` hoặc dùng **A/D** để lùi/tiến 5 giây, kể cả khi pause; tua chỉ giải mã MP4 đã xuất, không chạy lại YOLO. Hết video giữ frame cuối: Space phát lại từ đầu, Q đóng và hoàn tất phiên. Chế độ fast chưa hỗ trợ tua. Cửa sổ xem trước **không phát âm thanh**; MP4 chế độ full giữ track âm thanh đầu tiên, mã hóa AAC.

## Bám người và giảm nháy trong Full

Tính năng tùy chọn; cài thêm `pip install -r requirements/video_tracking.txt` trong `.venv`. TUI chọn **Đầy đủ** rồi bật **Bám người**. Hoặc mở sẵn cấu hình demo:

```powershell
.\run_video.cmd --tui --mode full --weights "weights/seed_repeats_20261014/E3_LIGHT_CONTRAST_seed123_E2_best.pt" --confidence 0.05 --imgsz 1280 --tracking bytetrack
```

ByteTrack chỉ theo dõi person, đợi hai frame liên tiếp trước khi hiện P{id}. Mỗi ID có màu giả ngẫu nhiên riêng, giữ nguyên qua các frame và khoảng nội suy. Đây là ID trong video, không phải danh tính sinh viên. Bảy lớp khác vẫn vẽ detection gốc.

Lượt một chạy YOLO/tracking và ghi JSONL; lượt hai đọc lại nguồn, làm mượt/nội suy và xuất MP4, không chạy lại YOLO. Có detection thì nét liền; chỉ lấp khoảng <=0,3s có hai đầu xác nhận cùng ID/cảnh, nét đứt và nhãn `noi suy`. Mất dấu lâu thì ẩn, dù còn nhớ track khoảng3s để thử ghép lại. Không giữ box đứng yên3s hoặc suy ra YOLO đã phát hiện trên frame nội suy.

CLI thêm `--tracking off|bytetrack` (mặc định off), `--track-max-lost-seconds 3`, `--track-gap-max-ms 300` (0 tắt), `--track-smooth-alpha 0.65` (1 tắt EMA). Chỉ full; fast/baseline không hỗ trợ tracking. Có thể chỉ định checkpoint khác như bình thường, không tự thay mặc định model.

Ngoài detection gốc có `tracks_raw.jsonl` và `tracks.jsonl`; trạng thái observed/interpolated, anchors, cấu hình/ngưỡng/màu và chi phí đều ghi riêng. Tiến độ chia nhận diện/tracking và dựng video. Chưa ReID/GMC, chưa bảo đảm ID ổn định khi che lâu/camera di chuyển mạnh; heuristic cắt cảnh không hoàn hảo. Xem [kế hoạch và tiêu chí](../../tasks/video/video_tracking_plan_20261015.md).

## Hai chế độ

- `full`: mỗi frame được nhận diện một lần, ghi dần qua FFmpeg; kiểm lại số frame, FPS và âm thanh trước khi đổi `annotated.partial.mp4` thành `annotated.mp4`. Giữ tốc độ nguồn (25, 30000/1001, 30...), không có cam kết hoàn thành toàn bộ dưới ba giây.
- `fast`: phát theo thời gian nguồn, YOLO lấy mẫu theo timestamp. Một tác vụ chạy và một tác vụ chờ; frame chờ cũ bị thay khi xử lý chậm. Chỉ vẽ box lên đúng frame đã nhận diện. Footer giữ số lượng/timestamp/tuổi kết quả gần nhất, không coi frame chưa nhận diện là đã được YOLO kiểm tra. Vì không tracking, box không xuất hiện liên tục giữa các frame lấy mẫu.

Bộ đệm mặc định 250 ms, tối đa 12 frame/128 MiB ảnh thô. Ngưỡng thực dùng giá trị nhỏ hơn khi độ phân giải/FPS cao. Decoder ngừng đọc khi đầy; pause không tiếp tục tích hàng đợi. Có thể có một lần đọc đang chạy khi bấm pause. Hàng đợi không đại diện toàn bộ RAM: còn model, bộ nhớ decoder/encoder và frame đang xử lý.

Mặc định video dùng `weights/checkpoint07_matrix_20261012/E3_YOLO26n_Project8_v05_from_CP07_v04_best19_AdamW_lr0p0001_bestEpoch10_20261012.pt`, `imgsz=1280`, confidence **0,05**, một lượt, không thêm NMS. Đây là mức thấp nhất của slider UI hiện có, không phải giới hạn toán học của YOLO; có thể tăng nhận diện yếu và báo nhầm. `--weights`/`--confidence` ghi đè tường minh. Dòng khởi chạy báo tên checkpoint, imgsz và confidence; metadata giữ đường dẫn/hash thực. Các lượt cũ giữ cấu hình cũ; không thay mặc định Streamlit/Training. `--nms-iou` bật NMS bổ sung. `--device auto` kiểm CUDA thực tế, nếu không dùng được sẽ ghi lý do chọn CPU trong báo cáo và bản tổng kết. Không dùng số đo 120 ms từ một máy làm chuẩn máy khác.

## Độ phân giải và chất lượng

Decoder đọc frame ở kích thước nguồn; full vẽ box trên frame đó và xuất cùng FPS/số frame. Kích thước lẻ được pad tối đa một pixel để tương thích H.264/yuv420p. Cửa sổ xem trước có thể thu nhỏ tới 1280×800 để vừa màn hình; việc này không thu nhỏ video xuất.

YOLO đổi kích thước bản ảnh đầu vào theo `imgsz` và padding, sau đó tọa độ được quy về ảnh gốc. Vì vậy video 1920×1080 vẫn xuất 1920×1080 khi suy luận ở 1280. Muốn suy luận gần kích thước nguồn Full HD, chọn rõ `--imgsz 1920`; tốc độ/RAM/VRAM cần đo lại, không bảo đảm nhanh hoặc chính xác hơn trên mọi nguồn. Frame ở trong RAM, không ghi từng ảnh JPEG trung gian.

Giữ độ phân giải/FPS không đồng nghĩa giữ nguyên từng pixel. Hiện xuất **H.264, CRF 20, yuv420p** và AAC: có nén mất dữ liệu, chưa có tùy chọn lossless/HDR/10-bit. File gốc được giữ nguyên. Chưa tự thay codec/chất lượng nén vì sẽ thay đổi dung lượng, thời gian mã hóa và khả năng phát tương thích.

## Tiến độ và chi phí

Terminal báo bước hiện tại, thời gian bước/tổng, tiến độ xử lý full, ETA, RAM và VRAM khi khả dụng; cập nhật nhiều nhất một nhịp/giây trong vòng lặp. Nạp model/warm-up không có phần trăm giả. Sau khi sẵn sàng, nhịp tiến độ tiếp theo báo `MODEL READY` kèm thời gian load/warm-up và RAM; báo cáo giữ `model_ready_rss_bytes`. Cuối phiên in chi phí từng công đoạn và đường dẫn báo cáo.

Kết quả nằm trong một thư mục mới dưới `runs/video/` hoặc `--output-dir`:

| File | Nội dung |
|---|---|
| `metadata.json` | Nguồn/FPS/phân giải, cấu hình, hash checkpoint, runtime, thiết bị thực |
| `detections.jsonl` | Frame, timestamp, box, confidence, chi phí mỗi lần nhận diện |
| `report.json` | Trạng thái, lỗi, counters, latency cold/warm, chi phí, bộ đệm, RAM/VRAM |
| `annotated.mp4` | Video full đã hoàn tất và kiểm tra |
| `encoder.log` | Chi tiết FFmpeg trong chế độ full |

`failed`/`stopped` không được báo là hoàn thành. Giữ file `.partial.mp4` khi bị ngắt để phục vụ chẩn đoán; không tự coi đó là video hoàn chỉnh. Mỗi lượt có tên riêng, không ghi đè nguồn hoặc lượt cũ.

Các mốc khác nhau:

- `first_result_available_*`: có kết quả trong bộ nhớ, chưa chắc đã hiển thị.
- `first_result_display_warm_ms`: kết quả đầu đã qua `imshow` và xử lý sự kiện HighGUI, tính từ sau warm-up. Đây là phép đo gần điểm trình bày, không phải đo photon trên màn hình.
- `first_result_display_cold_ms`: cùng sự kiện nhưng gồm toàn bộ khởi động, import runtime, kiểm nguồn và nạp model.
- `--no-display` và baseline không được báo đạt latency hiển thị; các trường tương ứng là `null`.
- FPS callback YOLO khác FPS hiển thị. Percentile từng công đoạn dùng tối đa 2.000 mẫu gần nhất; count/total/max tính cả phiên. RAM peak là đỉnh **lấy mẫu**, không phải bảo đảm đỉnh tuyệt đối. Tối đa 3.600 mẫu tài nguyên được giữ trong RAM.
- `display_fps_wall` gồm thời gian người dùng pause; `display_fps_excluding_user_pause` chỉ trừ thời gian pause tường minh, vẫn tính decoder buffering. Cả hai được giữ để không che giật/đệm. `user_pause_seconds` ghi riêng.
- Khi tua/phát lại full, hai trường FPS trung bình được để `null` và có ghi chú phiên tương tác; vẫn ghi chi phí từng lần hiển thị/tua. Không dùng phiên tua làm benchmark phát tuyến tính. Full không ghi mốc first displayed của fast; `None` ở mốc đó không phải lỗi phát.

Kiểm CFR quét timestamp cả nguồn bằng ffprobe, streaming và giới hạn bộ nhớ. Chi phí kiểm nguồn được báo riêng và tính vào cold start; video dài có thể mất thời gian đáng kể trước khi nạp model. Nguồn VFR/gián đoạn/hỏng bị từ chối rõ ràng. Bản đầu không tự chuyển đổi VFR. Chiều ảnh lẻ được pad thêm tối đa một pixel khi xuất H.264/yuv420p.

## Kiểm chứng

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_video*.py" -v
```

Thử cùng video/checkpoint/settings: baseline hiển thị, full không hiển thị, fast có hiển thị. Chạy riêng từng phiên để tránh tranh tài nguyên. Nguồn test tổng hợp chỉ kiểm pipeline/thời gian, không đánh giá chất lượng nhận diện lớp học. Không suy ra tốc độ RTX 5060 Laptop từ CPU/T4.

Chưa xác nhận mục tiêu 28,5+ FPS trên mọi máy, độ ổn định 30 phút/sáu giờ hoặc hiệu năng RTX 5060 Laptop nếu chưa có báo cáo chạy thực tế tương ứng. Không thay đổi webcam/Training UI. Kế hoạch chi tiết: [video_processing_plan_20261015.md](../../tasks/video/video_processing_plan_20261015.md).

Nguồn API: [OpenCV HighGUI](https://docs.opencv.org/4.10.0/d7/dfc/group__highgui.html), [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/), [FFmpeg](https://ffmpeg.org/ffmpeg.html), [ffprobe](https://ffmpeg.org/ffprobe.html).
