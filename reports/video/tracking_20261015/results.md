# Kiểm chứng tracking full/person-only — 15/10/2026

Đã triển khai ByteTrack, xác nhận hai frame, EMA, nội suy có hai đầu tối đa0,3s, ID/màu riêng ổn định và xuất full hai lượt. Không ReID, không tracking fast/đồ vật. Kế hoạch/hợp đồng: [video_tracking_plan_20261015.md](../../../tasks/video/video_tracking_plan_20261015.md); cách bật: [VIDEO_NATIVE.md](../../../docs/guides/video_processing.md).

## Ba lượt YOLO thật

Nguồn lớp học local21,64s, 898×506,25FPS,541frame, không audio. LIGHT_CONTRAST seed123 SHA256 `5487978ad53e7c5fc9d5cbe0c043edab8497f4e83dc248a7c0f785775654e816`, CPU, imgsz1280/conf0,05, một inference/frame ngoài warm-up. Chạy tuần tự không giao diện. Runtime: Python3.11.16, torch2.14.0+cpu, Ultralytics8.4.165, OpenCV5.0.0.93, lap0.5.12.

| Cấu hình | Tổng giây | Infer / encode | RAM peak lấy mẫu | Tracking p95 | Smoothing p95 | Nội suy p95 |
|---|---:|---:|---:|---:|---:|---:|
| Detection hiện tại | 73,02 | 541 / 541 | 382,6MiB | — | — | — |
| Tracking, alpha1/gap0 | 67,18 | 541 / 541 | 474,4MiB | 1,57ms | 0,35ms | 0,005ms |
| Tracking, alpha0,65/gap300ms | 64,24 | 541 / 541 | 481,2MiB | 1,40ms | 0,41ms | 0,24ms |

Không coi thời gian tổng thấp hơn là bằng chứng tracking làm YOLO nhanh hơn: chạy theo thứ tự trong cùng process, import/cache/warm-up và tải CPU khác nhau; inference callback p95 lần lượt134,96/136,28/122,31ms. Tổng chi phí association của bản đầy đủ0,489s; smoothing0,092s; interpolation0,034s; giải mã lượt hai0,708s. Bộ nhớ là toàn tiến trình, các model trước được Python/PyTorch giải phóng/cache theo runtime; không phải footprint riêng của tracker.

Cả hai bản tracking tạo44track, xác nhận21track,127lần tìm lại track đã xác nhận. Bản đầy đủ nội suy95khoảng,208box trên160frame.32lần tìm lại còn lại không có khoảng nội suy hợp lệ trong quy tắc đã chọn; không có ý nghĩa là tìm lại đúng người theo ground truth. Peak cửa sổ metadata9frame; bản gap0 giữ2record (gồm record biên). Không giữ ảnh tương lai.

Các file trong `runs/video_tracking_benchmark_20261015`:

- Detection: `detection/20261015_231428_0b00aa2ebf`.
- Tracking không EMA/nội suy: `tracking/20261015_231541_b4321548c1`.
- Tracking đầy đủ: `tracking_smooth_gap/20261015_231648_7c35de1a7a`.

Mỗi lượt có metadata/report/detection JSONL/video; tracking có thêm hai track JSONL. Bản sao báo cáo đầy đủ và cấu hình ở [benchmarks.json](benchmarks.json); ảnh đối chiếu tại frame250 ở [preview_frame250.jpg](preview_frame250.jpg). Video không đưa vào Git.

## Hình ảnh và giới hạn chất lượng

Đã trích và xem frame250 của ba MP4. Box người có ID, màu riêng khác nhau và ổn định theo ID; đồ vật vẫn dùng cách vẽ cũ. Box person và đồ vật sai/bỏ sót từ detector vẫn có thể tồn tại: ví dụ đồ nền bị gọi backpack và người ở biên chưa được thấy. Tracking không sửa detector hoặc bảo đảm toàn bộ box đúng người. Nét đứt/nhãn `noi suy` phân biệt box nội suy, JSON không gán confidence YOLO giả.

Đã xem thêm [frame4 có nội suy](preview_interpolated.jpg): P4 nét đứt xanh, hai neo frame3/6; các ID khác giữ màu vàng/tím/xanh lá. Cũng thấy box chồng lấn và có thể khoanh chưa đúng người từ detector confidence thấp. Nội suy có thể nối tiếp một track dự đoán sai; đây là giới hạn chất lượng còn lại, không được che bằng số khoảng lấp thành công.

Nguồn chưa có nhãn chuỗi kiểm duyệt nên **chưa báo personRecall/IDF1/HOTA/IDswitches chính xác**, chưa kết luận cải thiện lớp học hoặc đạt78,61% trên video này. Số box/track/gap không thay cho ground truth.

## Kiểm thử và phép thử rung

- **47/47 test video/TUI đạt** trong lượt toàn bộ dự án: ByteTrack thật, weak score0,05, hai hit đầu video, lost/refound/expiry, không tái dùng ID/không ảnh hưởng giữa phiên, mất dấu budget0 không phá track liên tục, cut, EMA, nội suy/cấm ngoại suy/neo/cảnh/ID khác, giới hạn metadata, màu100ID tái lập, hai lượt không infer lại, lỗi render không final sai, nguồn một frame không người25/29,97/30FPS có/không audio, CLI/TUI và đường off/fast cũ.
- Toàn dự án **118/119 test đạt**. Một lỗi môi trường cũ: thiếu `albumentations` trong `test_e3_improvement_eval`, không skip/xóa test hoặc sửa augmentation để làm xanh.
- Phép thử đứng yên300frame, seed123, nhiễu Gaussian sigma2pixel: sau20frame warm-up, std x từ1,9561px xuống1,3768px, giảm **29,62%** (mục tiêu>=20%); bias trung bình0,00264px. Đây là nhiễu tổng hợp, không phải đo rung video lớp học. Dữ liệu: [jitter.json](jitter.json).
- Các test integration dùng FFmpeg thật, audio stream có kiểm; chưa thay cho nghe/xem thủ công để nghiệm thu A/V sync trên nhiều codec.
- `pip check`, `compileall` và `git diff --check` đạt; Git có cảnh báo LF/CRLF ở thay đổi sẵn không thuộc tracking.

## Tái lập

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/video_tracking.txt
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_video*.py" -v
.venv\Scripts\python.exe -m scripts.benchmarks.video_tracking --source "stock-footage-mature-teacher-monitoring-students-as-they-write-an-exam-in-a-classroom-at-secondary-school-female.webm" --weights "weights/seed_repeats_20261014/E3_LIGHT_CONTRAST_seed123_E2_best.pt" --output-dir runs/video_tracking_comparison_new
```

## Đường phát 30 FPS trong ba phút

Đã chạy full từ đầu đến cuối trên test pattern640×360/30FPS/5.400frame với **detector tổng hợp**, ByteTrack, EMA/nội suy, FFmpeg và HighGUI thật. Không có model YOLO trong lượt này; fixture trọng số được đặt tên `SYNTHETIC_DETECTOR_NOT_YOLO.fixture`. Lượt đầu5400frame, lượt dựng5400frame, MP4 đủ5400frame; có180khoảng nội suy/360frame.

- Phát đủ timeline180s, không pause, `playback_completed=true`.
- FPS hiển thị30,0028;5.399frame được trình bày,1frame bị bỏ, tỷ lệ0,01852%: **đạt mục tiêu >=28,5FPS và <=1% drops trong phép thử này**.
- Khoảng hiển thị p50≈33,3ms, p95≈35,4ms; draw/present p95≈2,8ms.
- Metadata nhìn trước đạt11record, không giữ ảnh tương lai. RSS peak lấy mẫu≈283,3MiB, cuối≈243,7MiB; chưa dùng làm bằng chứng soak30phút.
- Toàn phiên219,10s gồm chuẩn bị, xử lý/xuất và180s phát; không dùng tổng phiên làm FPS nhận diện.

Báo cáo nguyên ở [playback_3min.json](playback_3min.json); nguồn và đường dẫn artifact nằm trong đó. Tái lập bằng `python -m scripts.benchmarks.video_tracking_playback --source .temp/video_tests/synthetic_3min_30fps.mp4 --output-dir runs/video_tracking_playback_new`. Cần nguồn CFR30FPS180s tương ứng. Không dùng phép thử detector tổng hợp để tuyên bố tốc độ YOLO, Recall hay chất lượng tracking lớp học.

## Chưa nghiệm thu

Chưa có đánh giá GT lớp học, camera chuyển động mạnh/che lâu, GPU RTX5060, soak30phút/sáu giờ. A/V sync nhiều codec còn cần nghe/xem thủ công. ByteTrack không có ReID; ID có thể đổi/ghép sai khi che khuất/cắt cảnh mà heuristic không nhận ra. Box sai từ detector có thể được tracker giữ/nội suy tiếp; phải đo riêng và cải thiện detector.
