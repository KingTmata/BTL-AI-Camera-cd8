# R04 640 — latency và FPS trên một Tesla T4, 11/10/2026

Đã đo xong 6.000 lần: 200 ảnh validation từ 15 nguồn × 10 vòng × 3 chế độ. Các lần lặp không phải 2.000 ảnh độc lập. Một GPU T4 số 0 được sử dụng, không DDP, không train, không chấm test và không đổi cấu hình ứng dụng. Kaggle cấp hai GPU nhưng CUDA chỉ thấy GPU 0 trong runner; ảnh chụp cũng cho thấy GPU còn lại không dùng bộ nhớ.

| Chế độ | p50 ms | p95 ms | p99 ms | Max ms | >100 ms | FPS xử lý ước tính | 1000/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu_fp32 | 71.09 | 79.13 | 84.96 | 108.79 | 7/2000 | 14.48 | 12.64 |
| t4_fp32 | 14.00 | 20.78 | 25.49 | 65.00 | 0/2000 | 67.70 | 48.12 |
| t4_fp16 | 14.97 | 21.39 | 26.36 | 2870.27 | 4/2000 | 47.08 | 46.76 |

FPS xử lý = 1000 / thời gian trung bình, cho xử lý tuần tự các ảnh đã nạp sẵn. 1000/p95 chỉ là mức quy đổi từ percentile độ trễ, không phải phép đo percentile FPS và không phải tốc độ được bảo đảm. Kết quả FP32 khoảng 67,7 FPS; webcam 30 FPS sẽ tối đa khoảng 30 FPS nếu không có nút thắt khác. Camera 60 FPS có thể đạt gần 60 FPS ở mức trung bình, nhưng chưa được xác nhận bằng webcam thực.

Phép đo gồm model.predict, một forward, tiền xử lý, chuyển dữ liệu CPU/GPU, extra class-aware NMS 0,6 và Results.plot. 640, batch 1, conf 0,25, không augment. CUDA synchronize trước và sau mỗi lần đo. 20 warmup/chế độ; jobs ba chế độ xen kẽ ngẫu nhiên từng vòng. CPU cùng VM dùng 2 threads, OpenCV 1 thread. Không gồm đọc/giải mã ảnh, capture camera, tracking, hàng đợi, giao diện hoặc truyền mạng. Không quy đổi trực tiếp sang RTX 5060 Laptop 8 GB.

FP16 có bốn lượt 2,8 giây ở vòng đầu trên bộ ảnh mở rộng; tất cả được giữ trong thống kê tổng. Chưa xác định nguyên nhân nên không tự loại outlier. Vòng 2–10 gồm 1.800 lượt/chế độ: FP16 mean 15,63 ms, p95 21,25 ms, khoảng 63,97 FPS, 0/1800 vượt 100 ms; FP32 mean 14,69 ms, p95 20,66 ms, khoảng 68,09 FPS, 0/1800 vượt 100 ms. Hai model GPU cùng resident; đây là benchmark so tương đối xen kẽ, không phải một phiên webcam liên tục chỉ một model.

CUDA peak allocated khoảng 69,91 MiB, peak reserved 96 MiB cho cả hai model GPU, chưa gồm CUDA context và các cấp phát ngoài PyTorch. Không áp giới hạn 8 GB; dung lượng này thấp hơn nhiều 8 GB nhưng không chứng minh tốc độ hai kiến trúc tương đương.

R04 compact SHA256: `2fedf0ef4cab22911ea9786b86cbb0ec37a0802012a8cb0982ec76d3c73c2105`. Trọng số compact trước đó đã kiểm tensor bằng bản R04 giữ lại. Manifest và từng ảnh chọn được kiểm SHA256 trước đo. Runner pin Ultralytics 8.4.165; GPU torch 2.11.0+cu128. Điểm F1/mAP trước đây không được chấm lại trong benchmark latency này.

Theo tiêu chí tốc độ/ổn định của phép đo này, ưu tiên thử triển khai R04 640 FP32 trên GPU. Chưa gọi đây là model tốt nhất R01–R11: chín checkpoint còn thiếu. Mục tiêu webcam dưới 100 ms cần kiểm capture/tracker/UI trên laptop thật.

Kết quả lưu local: `extended.json`, `extended_samples.json`, `steady_rounds_2_to_10.json`, `comparison.json`, `confirmation.json`, `kaggle_summary.png`. Phiên GPU đã dừng sau khi thu kết quả. Kaggle vẫn hiển thị Committing; chưa xác nhận bản kết quả mới được lưu thành một version online.

## Ước tính webcam thực — chưa đo capture/tracker/UI

Tham chiếu duy nhất là một Tesla T4, R04 640 FP32: mean predict+NMS+vẽ 14,77 ms, p95 20,78 ms. Để ước lượng FPS đầu ra, giả định thêm 5/10/20 ms mỗi ảnh cho phần capture/copy, tracking, giao diện và điều phối chưa nằm trong số đo. Ba con số này là kịch bản giả định, không phải overhead đã đo.

| Overhead thêm giả định | Mean xử lý tổng | FPS xử lý ước tính | Camera 30 FPS | Camera 60 FPS |
|---|---:|---:|---:|---:|
| 5 ms | 19.77 ms | 50.6 | 30.0 FPS | 50.6 FPS |
| 10 ms | 24.77 ms | 40.4 | 30.0 FPS | 40.4 FPS |
| 20 ms | 34.77 ms | 28.8 | 28.8 FPS | 28.8 FPS |

Công thức cho pipeline tuần tự: FPS xử lý = 1000 / (mean đo được + overhead giả định); FPS đầu ra = min(FPS camera, FPS xử lý, giới hạn hiển thị). Bảng giả định hiển thị không chậm hơn và không tích hàng đợi frame. Đây là trần ước tính theo kịch bản; scheduling, camera thực và biến động thời gian có thể làm FPS thấp hơn. Không khẳng định ứng dụng hiện tại đã có cơ chế bỏ frame cũ.

Nếu overhead thực khoảng 10 ms, kịch bản cho mean xử lý 24,77 ms, camera 30 FPS có dư thời gian để xử lý gần 30 FPS; camera 60 FPS bị giới hạn khoảng 40,4 FPS. Từ ba kịch bản này, dự kiến tham khảo khoảng 29–30 FPS với camera 30 FPS, hoặc khoảng 29–51 FPS với camera 60 FPS. Khoảng này có điều kiện theo giả định overhead 5–20 ms; không phải khoảng tin cậy, kết quả nghiệm thu hay bảo đảm phần cứng.

Không cộng cơ học thời gian chờ camera 33,3 ms vào từng phép suy luận để tính FPS. Capture có thể chồng lấp suy luận hoặc chặn luồng; cần đo pipeline thật. Độ trễ đầu-cuối từ cảnh đến màn hình còn phụ thuộc exposure, phase lấy frame, buffering, hàng đợi và trình bày; không suy ra từ bảng FPS này. Các mức p95 tổng nếu overhead cố định là 25,78/30,78/40,78 ms, chỉ là phép cộng giả định, không phải p95 webcam thực.

Kết quả này chưa áp dụng định lượng cho RTX 5060 Laptop 8 GB. Cần chạy cùng protocol trên laptop, rồi đo timestamp capture → predict → tracker → UI và FPS đầu ra để thay ước tính bằng số đo.
