# Tracking người và giảm nháy box video full — 15/10/2026

Kế hoạch đã được người dùng yêu cầu triển khai. Nguồn mục tiêu nghiên cứu vẫn là [plan.md](../plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026); tài liệu này chỉ quy định đợt video tracking. Bằng chứng nghiệm thu được ghi riêng, không dùng trạng thái đã triển khai để thay cho chất lượng đã đo.

## 1. Mục tiêu và phạm vi đã chốt

- Áp dụng **full trước**, chỉ theo dõi **person**, dùng **ByteTrack**, chưa ReID.
- Người mới cần hai frame liên tiếp ghép cùng track trước khi hiện box và ID, kể cả frame đầu video. Sau khi đã xác nhận, tìm lại trong hạn giữ track không cần xác nhận hai frame lại.
- Chỉ vẽ detection hiện tại; cho phép **nội suy tối đa 0,3 giây** có hai đầu xác nhận cùng ID/cảnh. Không ngoại suy sau detection cuối.
- Mất dấu lâu hơn thì ẩn box; giữ track tối đa khoảng ba giây để thử nối ID. Không hứa ID vĩnh viễn hoặc luôn có box.
- Bảy lớp đồ vật vẫn được nhận diện/vẽ theo detection gốc. Fast không thay hành vi.
- Demo chọn LIGHT_CONTRAST seed123, conf0,05, imgsz1280 tường minh; không tự đổi checkpoint mặc định toàn ứng dụng.
- Bổ sung khi triển khai theo yêu cầu người dùng: mỗi ID person có màu ngẫu nhiên tái lập riêng, giữ suốt track và cả khi nội suy; không đổi màu mỗi frame.
- Không triển khai danh tính sinh viên, webcam, nhận dạng hành vi, đánh giá tương tác, optical flow hoặc tracking đồ vật trong đợt này.

## 2. Quy tắc tracking, smoothing và nội suy

### ByteTrack

Một tracker thuộc một phiên; tái sử dụng detection của đường YOLO hiện có, không tạo model/tracker mỗi frame, không sửa thư viện Ultralytics. Giữ khóa tránh tranh chấp Training ở đường inference.

| Tham số | Giá trị khởi đầu |
|---|---|
| Confidence YOLO | 0,05 |
| `track_high_thresh` | confidence YOLO |
| `new_track_thresh` | confidence YOLO |
| `track_low_thresh` | 0,8 × confidence YOLO |
| `match_thresh` | 0,8 |
| `fuse_score` | False |
| `track_buffer` | ceil(FPS nguồn × thời hạn mất dấu) |
| Xác nhận | Hai detection liên tiếp |

Mọi detection YOLO trả về có thể tham gia ghép chính. Nhánh low-score không có thêm ứng viên dưới confidence YOLO; không tuyên bố khôi phục ứng viên đã bị lọc. Tắt score fusion để confidence yếu không lấn át ghép hình học. Ngưỡng là cấu hình thử nghiệm, chưa được chứng minh tối ưu.

Vòng đời: chờ xác nhận → xác nhận → tạm mất → tìm lại hoặc kết thúc. ID không được tái dùng trong phiên, kể cả reset cảnh; các phiên không tranh chấp bộ cấp ID. Hết hạn mất dấu phải loại trước association để không ghép lại một track đã hết hạn. Trạng thái cũ được dọn, không giữ toàn bộ lịch sử video trong RAM.

### Làm mượt

- EMA trên tâm, chiều rộng/cao: 0,65 mới + 0,35 trước.
- Chỉ nhận detection hiện tại; sau khoảng mất dấu khởi tạo lại EMA.
- Nếu IoU box đã làm mượt với detection hiện tại <0,5, dùng detection hiện tại.
- Clip vào ảnh, loại box không hợp lệ; giữ nguyên detection gốc.
- Không đưa box hiển thị quay lại tracker.

### Nội suy

Hai đầu phải cùng ID/cảnh và đều đã xác nhận. Khoảng cách timestamp hai neo <=0,3 giây, tỷ lệ rộng/cao từng chiều trong [0,5;2], dịch tâm <=0,75 đường chéo trung bình hai box. Nội suy tuyến tính theo timestamp trên tâm/rộng/cao của hai box hiển thị.

Không nội suy trước xác nhận, sau detection cuối, qua ID khác/cảnh khác, hoặc lấp một khoảng quá dài bằng nhiều khoảng nhỏ giả. Frame nội suy có confidence=null, hai frame neo và hai timestamp neo. Chỉ detection gốc được coi là YOLO xác nhận.

Box observed nét liền, nhãn P{id}; interpolated nét đứt, nhãn P{id} và “nội suy”. OpenCV font mặc định hiển thị nhãn ASCII `noi suy`; JSONL giữ trạng thái rõ ràng. Màu HSV giả ngẫu nhiên theo ID và seed20261015, không phụ thuộc frame; metadata ghi quy tắc tái lập.

### Đổi cảnh

Thu nhỏ ảnh về160×90. Histogram HSV 32×32 bins, chuẩn hóa L1. Reset nếu đồng thời khoảng cách Bhattacharyya >0,6 và sai khác ảnh xám tuyệt đối trung bình/255 >0,25. Kết thúc track cũ, xóa EMA và cấm nội suy qua cảnh; ID phiên tiếp tục tăng.

Đây là heuristic cảnh cắt rõ, không bảo đảm mọi kiểu chuyển cảnh; ưu tiên camera cố định/chuyển nhẹ. Chưa có GMC/ReID cho camera di chuyển mạnh hoặc che lâu.

## 3. Kiến trúc và hợp đồng dữ liệu

```text
Lượt 1: đọc nguồn → YOLO → ByteTrack → detection/track JSONL
Lượt 2: đọc lại nguồn + metadata tuần tự → EMA/nội suy → vẽ → FFmpeg
                                                        ↓
                                            kiểm MP4 → phát/tua
```

YOLO chỉ suy luận một lần mỗi frame nguồn, ngoài warm-up độc lập không tính frame. Lượt hai không infer. Không tracking vẫn đi đường full một lượt hiện có.

Adapter nhận frame index/timestamp/detection; trả track và liên kết detection nguồn, không sở hữu GUI/encoder/model. Main thread vẫn bơm sự kiện HighGUI; worker xử lý hai lượt và chịu điều phối dừng/lỗi hiện có.

Lượt dựng chỉ giữ metadata nhìn trước0,3 giây, cộng tối đa một record vượt biên để biết đã đọc đủ; không giữ ảnh tương lai. Dọn lịch sử track theo cửa sổ cần thiết. Hai lượt kiểm đúng index/timestamp/count, không dùng zip âm thầm cắt phần thừa.

| Đầu ra | Hợp đồng |
|---|---|
| detections.jsonl | YOLO nguyên gốc đủ8lớp, không bịa person từ interpolation |
| tracks_raw.jsonl | Mỗi frame, segment, track ID, confirmed, detection_index, bbox/score gốc |
| tracks.jsonl | Mỗi frame, box dùng xuất, raw bbox khi observed, state observed/interpolated, anchors |
| metadata.json | Cấu hình, version, checkpoint/hash, quy tắc tracking/màu/cảnh |
| report.json | Trạng thái, chi phí, counters/đỉnh state/buffer, RAM/VRAM |
| annotated.mp4 | Đủ frame/FPS/âm thanh theo hợp đồng full hiện tại |

Frame có tracks=[] vẫn phải có record. Giữ raw/tracks và trạng thái stopped/failed khi dang dở; file partial không được đánh dấu hoàn tất. Không ghi đè nguồn/lượt cũ.

## 4. CLI/TUI và tiến độ

| Tham số | Mặc định |
|---|---|
| --tracking off/bytetrack | off |
| --track-max-lost-seconds | 3 |
| --track-gap-max-ms | 300; 0 tắt nội suy |
| --track-smooth-alpha | 0,65; 1 tắt EMA |

Tracking chỉ full, từ chối fast/baseline trước khi xử lý. Thời gian hữu hạn không âm, alpha trong(0;1]. TUI thêm “Bám người”, mặc định tắt, chỉ chọn được ở Full. Sang Fast bỏ tracking khỏi đối số; đang chạy khóa các lựa chọn.

Phụ thuộc tùy chọn `requirements/video_tracking.txt`, chỉ cài trong môi trường dự án. Không auto-install trong vòng chạy; thiếu lap báo cách cài rõ ràng.

Demo:

```powershell
.\run_video.cmd --tui --mode full --weights "weights/seed_repeats_20261014/E3_LIGHT_CONTRAST_seed123_E2_best.pt" --confidence 0.05 --imgsz 1280 --tracking bytetrack
```

Tiến độ riêng `Detecting and tracking i/N` và `Rendering tracked video i/N`, tối đa1Hz, ETA riêng từng bước. Model load/warm-up không phần trăm giả. Chi phí: tracking setup, scene check, association, smoothing, interpolation, decode lượt hai, draw, encode. Count: tạo/xác nhận/tìm lại/kết thúc track, gap/box/frame nội suy, gap bị từ chối, reset cảnh. Giữ FPS phát khác FPS YOLO; tua không chạy lại model/tracker.

## 5. Thứ tự và kiểm chứng

1. Ghi kế hoạch/hợp đồng và test trước; giữ mục tiêu chính thức.
2. Adapter person-only, xác nhận2frame, TTL và reset/cấp ID; kiểm ByteTrack thật với chuỗi xác định.
3. Lượt dựng streaming và EMA; kiểm chỉ một inference/frame, đủ output/timeline.
4. Nội suy và nét vẽ/màu; kiểm giới hạn, anchors và không ngoại suy.
5. CLI/TUI/progress/error cleanup, giữ đường off và fast.
6. Đối chứng ba cấu hình cùng video/checkpoint/conf: detection; tracking alpha1/gap0; tracking alpha0,65/gap300.
7. Lưu artifact, số đo và giới hạn công khai.

### Kiểm thử chức năng

Người đứng yên/di chuyển/đổi hướng/hai người, weak score gần0,05; một hit không hiện/hai hit hiện; đầu video không ngoại lệ; gap ngắn/dài/hết3s; nhiều phiên không ảnh hưởng ID. Giới hạn nội suy dưới/bằng/trên0,3s với25/29,97/30FPS. Không bridge ID/cảnh/đầu/cuối; jump/size gate. Video không người/một frame/đường dẫn dấu, đồ vật không đổi. Stop/lỗi inference/render/encoder không tạo final sai. Pause/tua không thêm inference. Off giữ các test cũ.

### Tiêu chí tự động và đo thật

- Chuỗi tổng hợp không mơ hồ không đổi/trùng ID ngoài dự kiến; nội suy tuyến tính sai số<=1pixel.
- Frame không có detection nhưng có box phải interpolated và có hai neo hợp lệ.
- Test rung đứng yên có nhiễu xác định giảm độ lệch chuẩn ít nhất20%, đồng thời kiểm độ lệch box.
- Bộ nhớ state/metadata không tăng theo độ dài video; report ghi đỉnh.
- Giữ số frame/FPS/audio, nguồn25/30000/1001/30 và output tạm→validate→rename.
- Mục tiêu playback30FPS giữ >=28,5FPS, drop<=1% trong3phút; đây không phải inferenceFPS. Chưa có lượt đo tương ứng thì ghi chưa nghiệm thu.
- Video lớp học21,64s hiện có chỉ kiểm thực thi/quan sát; chưa GT không báo Recall/IDF1/HOTA hay cải thiện lớp học. Khi có GT, báo personP/R, IDs switches/IDF1, gap/flicker, localization/jitter và chất lượng interpolation riêng.
- Chưa xác nhận ổn định30phút/6giờ/RTX5060 nếu chưa có lượt đo thực.

Không hạ tiêu chí, loại trường hợp khó sau khi xem kết quả hoặc kéo dài box ma để làm đẹp hình. Nếu chưa đạt, giữ khả năng tắt và ghi failure. Tracking không thay thế cải thiện detector/dataset.

Nguồn API: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/), [ByteTrack](https://docs.ultralytics.com/reference/trackers/byte_tracker/); đối chiếu source8.4.165 đang cài.
