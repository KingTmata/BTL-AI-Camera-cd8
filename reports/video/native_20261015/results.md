# Kiểm chứng video OpenCV + terminal — 15/10/2026

Đã triển khai `python -m classroom_ai.video`, wrapper `run_video.cmd`, chế độ full/fast, progress từng bước và báo cáo chi phí. Streamlit/webcam hiện có không bị thay đổi trong đợt này. Chi tiết: [hướng dẫn](../../../docs/guides/video_processing.md), [kế hoạch](../../../tasks/video/video_processing_plan_20261015.md).

## Bằng chứng chạy thực

Thiết bị hiện tại: PyTorch **2.14.0+cpu**, OpenCV **5.0.0**, Ultralytics **8.4.165**. Không có CUDA dùng được trong môi trường này. Model R04, một lượt, imgsz **1280**, confidence 0,25, không thêm NMS. Các video benchmark là **test pattern tổng hợp 640×360**, không phải video lớp học và không đánh giá chất lượng nhận diện.

| Phiên | Thời lượng nguồn | Hiển thị | Bỏ frame hiển thị | Ghi chú |
|---|---:|---:|---:|---|
| Baseline trước sửa `waitKey` | 180 giây, 5.400 frame | 19,32 FPS | 1.922 frame | Khoảng một phần phiên trùng lúc chạy test cũ; không phải đối chứng A/B cô lập |
| Baseline sau đổi `pollKey` | 5 giây, 150 frame | 30,00 FPS | 0 | Sanity check ngắn, chưa thay thế benchmark dài |
| Fast có YOLO thật ở 1280 | 180 giây, 5.400 frame | 28,02 FPS tính cả pause; **29,97 FPS trừ pause tường minh** | **5 frame, 0,0926%** | Hệ thống ghi tổng pause bằng Space **12,572 giây**; wall playback 192,572 giây |

Ở phiên fast ba phút:

- Đọc đủ 5.400 frame, YOLO nhận diện **900 frame** (5 mẫu/giây, 16,67% nguồn), hiển thị 5.395. Không thay pending trong phiên này. Không gọi đây là 30 FPS nhận diện.
- Kết quả đầu được trình bày sau **131,34 ms warm**, **10.714,67 ms cold**; cold gồm kiểm nguồn/import runtime/nạp model/warm-up. Mốc warm dưới ba giây đạt **trong phiên tổng hợp này**, không phải bảo đảm mọi đầu vào/máy.
- Callback nhận diện: trung bình **105,91 ms**, p95 **122,18 ms**, quy đổi throughput callback khoảng **9,44 FPS**. Đây là số đo của thiết bị/cấu hình này.
- Vẽ/trình bày: p95 **2,01 ms**; khoảng hiển thị p95 **34,88 ms** (cửa sổ 2.000 mẫu cuối). Bản trước dùng `waitKey` có p95 vẽ/trình bày **16,6 ms**. Thay `pollKey` là tối ưu vòng sự kiện native, không chứng minh nguyên nhân chậm của Streamlit.
- Bộ đệm đạt tối đa **12 frame / 8.294.400 byte** ảnh thô. RAM lấy mẫu đỉnh **427.012.096 byte**; còn model/decoder và các cấp phát khác ngoài hàng đợi. Không có số VRAM GPU cho phiên CPU.
- Import/check runtime **7.773 ms**, nạp model **481 ms**, warm-up **259 ms**, kiểm nguồn **1.281 ms**. Không giấu chi phí khởi động bằng việc chỉ báo warm latency.

Hai FPS được giữ tách biệt: wall gồm pause; loại trừ pause chỉ trừ thời gian phím Space, **không loại decoder buffering**. Giá trị 29,97 được suy từ counters, wall FPS và `user_pause_seconds` của raw report; mã cuối đã xuất trực tiếp cả hai trường. Lượt ba phút chạy trước bản chỉnh metric pause lồng trong buffering, không có rebuffer trong lượt này.

Đã smoke full bằng YOLO thật trên slideshow COCO128 kỹ thuật: xuất đủ 10 frame, kiểm lại MP4 và xem ảnh trích từ kết quả có box/nhãn hiển thị. Không dùng ảnh xem trước để kết luận độ chính xác. Slideshow không phải kiểm nghiệm video lớp học. Integration dùng detector xác định và FFmpeg thật để kiểm 25/30000÷1001/30 FPS, frame order và nguồn có/không âm thanh.

Số đo chọn lọc và đường dẫn raw report: [benchmarks.json](benchmarks.json). Raw report/JSONL/video thử lưu dưới `runs/` (không đưa video thử vào Git).

## Kiểm thử

- **19/19 test video đạt**: sampling timestamp, byte/frame bound, mailbox, pause/rebuffer lồng nhau, overlay đúng frame, CFR/VFR, FPS phân số, full export/audio, lỗi detector/encoder, stop khi load, đóng cửa sổ native đã bị hủy, headless không báo latency hiển thị và baseline không nạp model.
- Bộ test toàn dự án: **90/91 đạt**, một lỗi **thiếu `albumentations`** ở test augmentation cũ `test_all_serialized_augmentation_groups_preserve_image_coordinates`. Không xóa/skip test hay sửa augmentation để đạt xanh.
- Test Streamlit cần socket localhost: sandbox kẹt trong `asyncio`/`socketpair` trước khi chạy script; đã kiểm lại ngoài hạn chế mạng sandbox. Đặt TEMP/TMP vào `.temp/testtmp` và bật UTF-8 để khắc phục đường dẫn tạm/mã hóa của môi trường Windows, không chỉnh test cũ.
- `compileall` thành công. Review giữ nguyên mapping/model mặc định của giao diện cũ, khóa bảo vệ Training và những thay đổi đang có trong workspace.

Lệnh tập trung:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_video*.py" -v
```

Tái tạo nguồn tổng hợp:

```powershell
ffmpeg -v error -nostdin -n -f lavfi -i testsrc2=size=640x360:rate=30:duration=180 -c:v libx264 -preset ultrafast -crf 30 -pix_fmt yuv420p .temp/video_tests/synthetic_3min_30fps.mp4
.\run_video.cmd --source .temp/video_tests/synthetic_3min_30fps.mp4 --mode fast --device cpu --imgsz 1280
```

## Chưa nghiệm thu

Video lớp học thực tế 1–3 phút, nghe/kiểm sync âm thanh thủ công, RTX 5060 Laptop, soak 30 phút và sáu giờ. Unit/integration, thử tổng hợp ba phút và cấu trúc bounded không chứng minh các mục đó đã đạt. Dừng/khởi động lại, pause và lỗi đã có test điều phối; thao tác GUI thực tế trên thiết bị demo vẫn cần kiểm thêm.
