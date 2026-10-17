# Kế hoạch xử lý video bằng OpenCV + terminal — 15/10/2026

Nguồn mục tiêu nghiên cứu vẫn là [mục chính thức trong plan.md](../plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026). Tài liệu này chỉ đặc tả triển khai video đã được người dùng duyệt trong chat, không thay thế định hướng nghiên cứu.

## 1. Mục tiêu và ranh giới

Chương trình độc lập Streamlit: Python xử lý, OpenCV hiển thị, terminal nhận tham số và báo tiến độ/chi phí. Video thực tế giai đoạn đầu 1–3 phút; kiến trúc streaming chuẩn bị cho 30 phút–6 giờ. Không nạp toàn bộ ảnh/kết quả/timestamp vào RAM.

| Chế độ | Nhận diện | Phát | Đầu ra |
|---|---|---|---|
| full | Mỗi frame đúng một lần | Xuất/kiểm xong toàn bộ mới phát theo FPS nguồn | MP4/H.264, track âm thanh đầu AAC, JSONL, báo cáo |
| fast | Lấy mẫu theo timestamp, mặc định 5 FPS | Cuốn chiếu theo tốc độ nguồn; đệm ngắn | JSONL/báo cáo; chưa xuất video |

Nguồn 30 FPS giữ 30; 25 và 30000/1001 giữ nguyên. Mốc dưới ba giây chỉ là mục tiêu kết quả đầu đã hiển thị của fast khi model sẵn sàng; đo riêng cold start bao gồm import, kiểm nguồn, nạp model và warm-up. Không dùng 120 ms/ảnh trên một thiết bị làm tham chiếu chung.

Không bao gồm sửa webcam, chuyển Training UI, xây web, training, tracking ID hay nhận dạng/đánh giá hành vi. Không mặc định triển khai ByteTrack hay các công nghệ chưa được chốt trong mục tiêu chính thức.

## 2. Kiến trúc và hợp đồng

Module `src.video` gồm: core (frame/hàng đợi/đồng hồ/metric), media (ffprobe/decoder/FFmpeg), pipeline (model/điều phối/artifact), player (HighGUI trên main thread) và CLI. Tái sử dụng `ImageInspector`, mapping nhãn và `RunManager.inference_slot`; không sửa lõi webcam đang có.

- Frame: `frame_index`, `timestamp_seconds`, ảnh BGR; kết quả giữ nguyên index/timestamp, box/class/confidence và thời gian xử lý.
- Model được một worker sở hữu và tái sử dụng; nạp/warm-up trong worker, main thread vẫn xử lý sự kiện giao diện. Không tạo model theo từng frame.
- Bộ đệm phát: 250 ms mặc định; tối đa 12 frame và 128 MiB ảnh thô. Ngưỡng thực hạ theo giới hạn dung lượng; không giảm kích thước suy luận ngầm.
- Fast: một inference chạy, tối đa một pending; thay pending cũ nếu chậm, đếm rõ. Frame đầu được bảo vệ để có kết quả trước khi bắt đầu phát. Cache kết quả tối đa 64 mục; lịch sử ghi JSONL.
- Chỉ overlay đúng frame; kết quả gần nhất được ghi timestamp/tuổi và số lượng ở footer. Không mang box cũ qua frame mới hoặc nhận đó là tracking.
- Full: không bỏ frame inference/encoder; FFmpeg ghi tuần tự, file tạm, xác minh số frame/FPS/thời lượng/audio rồi mới publish. Bản xem trước không phát âm thanh; file MP4 có âm thanh.
- Phát theo timestamp và đồng hồ monotonic; pause/buffering là trạng thái riêng. Nếu UI trễ quá một frame thì bỏ frame hiển thị đã hết hạn và đếm, không biến thành slow motion ngầm.
- Space pause/resume phát, Q/Esc/đóng cửa sổ dừng; Ctrl+C dừng xử lý. Không tua ở v1. Decoder không đọc tiếp khi pause, trừ read đang thực hiện.
- Source local CFR; strict ffprobe timestamp scan giới hạn bộ nhớ, có tiến độ khi biết số frame. VFR/timestamp gián đoạn/hỏng báo lỗi. Preflight toàn nguồn có chi phí riêng, không giấu khỏi cold start.
- Giải phóng decoder/encoder/worker khi EOF, lỗi hoặc stop. Không ghi đè nguồn/kết quả. Giữ bảo vệ tranh chấp với Training.

## 3. Giao diện sử dụng và tiến độ

`python -m classroom_ai.video --source FILE --mode full|fast`; wrapper Windows `run_video.cmd`.

| Option | Mặc định/quy tắc |
|---|---|
| `--source`, `--mode` | Bắt buộc, video local và full/fast |
| `--weights` | R04 hiện hành, ghi hash thực |
| `--device` | auto; CUDA phải kiểm phép tính thành công, fallback CPU có ghi lý do; explicit GPU lỗi thì dừng |
| `--imgsz` | 1280, độc lập kích thước cửa sổ |
| `--confidence`, `--nms-iou` | 0,25; không thêm NMS mặc định |
| `--sample-fps`, `--buffer-ms` | 5 và 250 |
| `--output-dir` | Thư mục cha, mỗi lượt tạo thư mục mới |
| `--no-display` | Đo không cửa sổ, không tuyên bố đạt latency hiển thị |
| `--baseline` | Fast chỉ phát video, không nạp YOLO |

Theo yêu cầu bổ sung: báo kiểm nguồn → runtime/device → nạp model → warm-up → xử lý → hoàn tất encoder/kiểm file → phát. Có thời gian mỗi bước/tổng, phần trăm và ETA khi biết lượng công việc, RAM/VRAM. Không dùng phần trăm giả cho nạp model. Vòng lặp tiến độ tối đa một lần/giây, cuối lượt tổng kết chi phí và nơi lưu.

Artifact: `metadata.json`, `detections.jsonl`, `report.json`; full có `annotated.mp4`, `encoder.log`. Trạng thái completed/stopped/failed; file tạm không được gọi là hoàn thành. JSONL ghi/flush tuần tự; mỗi stage giữ tối đa 2.000 mẫu percentile, counters/total/max cả phiên; resource ring 3.600 mẫu. Báo RAM peak lấy mẫu và CPU/thread/VRAM nếu có.

## 4. Công việc triển khai

- [x] Tạo module riêng/CLI và tài liệu, giữ thay đổi không liên quan.
- [x] Timestamp CFR, sampling 25/29,97/30, giới hạn bộ đệm và mailbox.
- [x] Native playback, pause/stop, baseline không YOLO, headless scheduling.
- [x] Tái sử dụng detector, checkpoint/device, warm-up và khóa Training.
- [x] Full export giữ frame/FPS/âm thanh, xác minh trước publish.
- [x] Fast có sampling, counters thay pending, overlay đúng frame, bounded history.
- [x] Báo tiến độ load và chi phí, trạng thái lỗi/hủy, cold/warm riêng.
- [ ] Nghiệm thu trên video lớp học 1–3 phút thực tế, bao gồm quan sát hình/âm thanh.
- [ ] Benchmark RTX 5060 Laptop; ghi driver/CUDA/PyTorch, nguồn điện, checkpoint và settings.
- [ ] Chạy ổn định 30 phút và sáu giờ thực tế trước khi xác nhận tương ứng.

Checklist cuối phải phản ánh bằng chứng; không đánh dấu GPU/long-run hoàn tất chỉ vì unit test qua.

## 5. Kiểm thử và nghiệm thu

Unit: lấy mẫu theo thời gian, bounded bytes/frames, mailbox thay đúng pending, pause/rebuffer clock, cache kết quả không dùng box cũ/tương lai. Integration: FFmpeg nguồn có số frame/timestamp biết trước; 25/30000/1001/30 FPS, có/không âm thanh, VFR, đường dẫn dấu/khoảng trắng, detector chậm/lỗi, full finalize và headless không báo latency hiển thị.

Đầy đủ: frame output bằng input; thời lượng lệch không quá một khoảng frame; kiểm audio stream và phát thủ công để xác nhận sync. Nhanh: công khai tỷ lệ frame nhận diện, pending thay thế và display drops. Thử nguồn lỗi, encoder lỗi/hết đĩa, GPU lỗi, đóng cửa sổ/Ctrl+C, pause/resume và Training guard.

Hiệu năng nguồn 30 FPS: mục tiêu display trung bình >=28,5 FPS, drops <=1% trên ba phút; fast first displayed warm <3 giây, báo cả lần không đạt. Đo riêng decode, chờ queue/lock, predict, draw/present, encode; p50/p95, RAM/VRAM, tuổi kết quả. So baseline, full headless và fast native **không chạy đồng thời**. Không lấy FPS model làm FPS hiển thị.

So bộ nhớ sau warm-up trong phiên ổn định 30 phút: mục tiêu tăng <=10% cùng tải. Chỉ xác nhận sáu giờ sau lượt chạy sáu giờ thực tế. Synthetic media kiểm pipeline/khối lượng, không chứng minh độ chính xác lớp học, thermal/power hay ổn định lâu dài. Không hạ imgsz/checkpoint/tiêu chí ngầm để báo đạt.

Hướng dẫn chạy và giới hạn: [VIDEO_NATIVE.md](../../docs/guides/video_processing.md).

## Bổ sung được yêu cầu sau bản đầu — 15/10/2026

- Thay mặc định video R04/0,25 ở bảng kế hoạch lịch sử bằng E3 `E3_YOLO26n_Project8_v05_from_CP07_v04_best19_AdamW_lr0p0001_bestEpoch10_20261012.pt`, confidence 0,05; imgsz vẫn 1280. Tên/hash được báo rõ, cấu hình tường minh vẫn được ưu tiên, Streamlit/Training không đổi.
- Mở rộng giới hạn “không tua v1” cho **phát MP4 full đã hoàn tất**: thanh Frame, A/D ±5 giây, tua khi pause, Space phát lại sau khi hết, Q đóng. Không chạy lại inference/export; fast vẫn không tua. Phiên có tua không được dùng làm benchmark FPS tuyến tính.
- Giữ kích thước và tốc độ video nguồn độc lập imgsz và kích thước preview; H.264 CRF 20/yuv420p hiện có là nén mất dữ liệu, không tuyên bố chất lượng pixel nguyên gốc.

Bằng chứng triển khai và giới hạn nghiệm thu: [RESULTS.md](../../reports/video/native_20261015/results.md). Đã có 19 test video đạt và lượt tổng hợp ba phút với YOLO thật trên CPU; đây không thay thế video lớp học, GPU hoặc kiểm ổn định dài.
