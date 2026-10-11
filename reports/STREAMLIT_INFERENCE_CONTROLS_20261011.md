# Streamlit: kích thước, NMS và một/ba lượt — 11/10/2026

## Yêu cầu và kết quả triển khai

Giữ cùng checkpoint, thêm các cấu hình suy luận để người dùng thử chất lượng và tốc độ. Không train lại, không tạo file trọng số mới. Ba file hiện hành vẫn là R04, baseline v05 và R01; mặc định R04.

| Điều khiển | Lựa chọn | Mặc định phiên mới |
|---|---|---|
| Kích thước suy luận | 512, 640, 768, 960, 1280 | 640 |
| Số lượt suy luận mỗi ảnh | 1 hoặc 3 | 1 |
| NMS theo lớp, một lượt | Tắt, IoU 0,5, IoU 0,6 | Tắt |
| NMS theo lớp, ba lượt | IoU 0,5 hoặc 0,6, bắt buộc gộp | 0,5 khi chuyển từ mặc định |
| Confidence | Điều khiển hiện có | 0,25 |
| Thiết bị | CPU trong ứng dụng hiện tại | CPU |

Không có chế độ hai lượt hoặc hơn ba lượt: hai lượt chưa có phép chấm đối chứng để đề xuất; hơn ba không nằm trong phạm vi webcam người dùng yêu cầu. Các lựa chọn là ứng viên đã có cơ sở thử nghiệm, không chứng minh mọi tổ hợp đều tốt nhất.

## Ý nghĩa kỹ thuật

Một lượt gọi model một lần ở kích thước đã chọn. YOLO26 dùng đầu ra end-to-end; “NMS tắt” là không thêm bước lọc NMS theo lớp của dự án. IoU 0,5 lọc mạnh hơn 0,6: khi khung cùng lớp chồng lấn vượt ngưỡng, giữ khung có confidence cao hơn. Khung thuộc lớp khác không bị loại chỉ vì trùng tọa độ.

Ba lượt dùng cùng trọng số với ba góc nhìn trên tensor đã tiền xử lý: tỷ lệ 1,0; tỷ lệ 0,83 kèm lật ngang; tỷ lệ 0,67. Tọa độ được đưa về hệ ban đầu, rồi gộp bằng NMS theo lớp. Đây là ba forward, không phải ba checkpoint và không phải nạp lại trọng số mỗi khung hình.

Model ba lượt được tạo khi dùng lần đầu và giữ trong cache cùng model một lượt; có thêm bản model trong RAM nhưng không nhân bản file `.pt`. Lock bảo vệ suy luận và cập nhật confidence/NMS giữa các phiên dùng chung resource. Chuyển về một lượt dùng lại model một lượt, không giữ cấu hình ba lượt cho nhánh đó.

NMS có thêm chi phí hậu xử lý tùy số khung, không chạy thêm forward. Ba lượt tốn thêm tính toán và có thể giảm FPS đáng kể; tỷ lệ 0,83 và 0,67 khiến tổng chi phí không được suy ra chính xác bằng nhân ba. Chưa có benchmark webcam ba lượt để kết luận mức chậm cụ thể.

## Hướng dẫn sử dụng

1. Mở Recognition, chọn checkpoint. Để thử nhanh, bắt đầu R04 / 640 / một lượt; chọn NMS 0,6 để gần cấu hình benchmark đã lưu.
2. Muốn xem cấu hình chất lượng cao trong lịch sử: R04 / 1280 / một lượt / NMS 0,6, hoặc R04 / 1280 / ba lượt / NMS 0,5. Không tự chọn cấu hình này cho webcam khi chưa đo tốc độ thực.
3. Đổi ảnh, model, confidence, kích thước, NMS hoặc số lượt thì bấm **Phân tích ảnh** lại. Kết quả ảnh cũ không được hiển thị như kết quả của cấu hình mới.
4. Đổi cấu hình khi webcam đang chạy sẽ dừng phiên cũ. Bấm **Bật webcam** để dùng cấu hình mới. Closure suy luận nhận đúng kích thước/NMS/số lượt của phiên mới.
5. Tải JSON để giữ cấu hình cùng kết quả: `imgsz`, `device`, `confidence_threshold`, `extra_class_nms_iou`, `model_passes`, đường dẫn trọng số và thời gian xử lý. Trường `nms: false` chỉ cờ NMS native; NMS bổ sung nằm ở `extra_class_nms_iou`.

Hướng dẫn tương tự đã có trong expander **Hướng dẫn kích thước và NMS** ở sidebar. Thời gian ảnh đo suy luận và NMS; lần đầu có thể gồm warmup và nạp model ba lượt. Không bao gồm pipeline webcam hoàn chỉnh.

## Không nhầm ba mốc chất lượng

| Cấu hình R04 trong báo cáo đã có | mAP50 | mAP50–95 |
|---|---:|---:|
| 640, một lượt, GPU FP16, NMS 0,6 | 71,6033% | 49,5966% |
| 1280, một lượt, GPU FP16, NMS 0,6 | 78,9661% | 54,9561% |
| 1280, ba lượt, CPU FP32, NMS 0,5 | 80,5052% | 56,5696% |

Các mốc khác kích thước và có khác backend/precision; không phải bật NMS ở640 sẽ tăng từ71 lên78, hoặc chọn ba lượt ở640 sẽ đạt80,5. mAP là số đo trên bộ validation, không phải confidence của một đối tượng. Giao diện không tự chấm lại toàn bộ validation khi đổi lựa chọn.

## Kiểm tra đã thực hiện

- Bộ kiểm thử `python -X utf8 -B -m unittest discover -s tests`: **47 tests đạt**. Chạy với UTF-8 vì một test packaging đọc notebook UTF-8 bằng locale mặc định Windows sẽ lỗi GBK; các lỗi quyền file tạm trong sandbox đã được xử lý bằng chạy kiểm thử được cấp quyền.
- Test mới `tests/test_inspector_controls.py`: kích thước truyền đúng, mặc định một forward không NMS bổ sung, NMS chỉ loại khung trùng cùng lớp, tham số sai bị từ chối trước suy luận. Các test TTA/geometry có sẵn vẫn đạt.
- Streamlit AppTest: phiên mặc định không lỗi; chọn ba lượt chỉ có hai lựa chọn NMS và mặc định0,5; chuyển NMS0,6 và size1280 không lỗi; chuyển về một lượt khôi phục bộ chọn có tùy chọn tắt.
- Smoke với checkpoint thật: ba lượt chạy được, sau đó một lượt chạy được trên cùng inspector.
- Trình duyệt localhost: xác nhận mặc định R04/640/một lượt/NMS tắt; chuyển ba lượt tự dùng NMS0,5; phân tích ảnh mẫu thành công, hiển thị1person. JSON tải xuống xác nhận CPU,640,3lượt,NMS0,5. Thời gian633,99ms của lần chạy ảnh mẫu này chỉ xác nhận chức năng, không dùng để tính FPS webcam hay so sánh benchmark.

Mã thay đổi: `src/inference/detector.py`, `src/ui/app.py`; dùng lại `kaggle_tools/inference_variants.py` cho thuật toán ba góc nhìn và merge đã kiểm tra.

Bằng chứng lưu tại [ảnh giao diện](streamlit_controls_20261011.png) và [JSON ảnh mẫu ba lượt](streamlit_controls_sample_20261011.json). Sau thử nghiệm giao diện được trả về một lượt. Streamlit phục vụ tại localhost:8501.

## Giới hạn và việc còn thiếu

- Chưa nghiệm thu webcam thực sau thay đổi; chưa đo p50/p95/p99, FPS hiển thị, drop frame hoặc độ ổn định ba lượt trên camera.
- Ứng dụng vẫn chạy CPU; chưa thêm CUDA/FP16/TensorRT hoặc benchmark trực tiếp RTX5060Laptop8GB. T4 không giả lập chính xác5060.
- Preview webcam trong Streamlit hiện dùng fragment `run_every=0.3`, nên chu kỳ làm mới giao diện khoảng3,3lần/giây trước chi phí render. `runtime.preview_fps: 10` trong YAML không điều khiển fragment này. FPS suy luận T4 không phải FPS preview hiện tại; cần tối ưu và đo đường hiển thị riêng.
- Chưa có đủ checkpoint R01–R11: không dùng việc thêm lựa chọn giao diện để tuyên bố đã chấm đủ11lượt.
- Chưa chấm lại toàn validation cho mọi tổ hợp CPU/size/NMS/số lượt; không chạm tập test, không chạy train.

Mục tiêu webcam vẫn ưu tiên độ trễ, sau đó hai mAP và F1 cân bằng. Một lượt là mặc định; ba lượt là tùy chọn thử chất lượng theo yêu cầu mới của người dùng.
