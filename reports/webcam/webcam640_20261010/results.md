# Chấm một lượt ở 640 — 10/10/2026

Đã chấm đủ R04 trên 2.500 validation / 8.359 ground-truth boxes. Chưa thể chọn checkpoint thắng trong R01–R11 ở640: chưa khôi phục được R02/R07 và các checkpoint campaign khác. Không chạy train, không thay trọng số/cấu hình ứng dụng, không chấm500 test.

## Điều kiện

Ultralytics8.4.165, CPU FP32,4 threads, imgsz640, batch8, rect=True, conf0.001, max_det300, nms=False, augment=False. Mỗi ảnh một lượt model. NMS0.6 là class-aware hậu xử lý từ cùng cache prediction, không thêm forward. Native replay khớp AP/P/R của validator trong1e-10. imgsz640 là kích thước mục tiêu với rectangular padding; không bắt buộc mọi ảnh thành hình vuông640×640.

Checkpoint `weights/campaign_20261010/BEST_Project8_v05.pt`, SHA256 `087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758`. Manifest `22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd`. Hashes đủ2.500 ảnh/nhãn được kiểm lại trước chấm. Full visual review vẫn chưa hoàn tất; split còn nhóm cảnh dùng chung nên đây là validation hiện tại, chưa phải nghiệm thu độc lập.

## Kết quả R04 ở640

| Hậu xử lý | mAP50 | mAP50–95 | Precision evaluator | Recall evaluator |
|---|---:|---:|---:|---:|
| Native |69,479293%|49,239841%|77,908181%|61,549256%|
| NMS0.6 |71,576838%|49,617590%|77,503701%|65,119777%|

P/R evaluator là số tại threshold được evaluator chọn, không phải P/R của webcam với conf0.25. Đối chiếu cùng threshold conf0.25, IoU0.5, micro trên validation:

| Hậu xử lý | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Native |4919|2056|3440|70,523297%|58,846752%|64,158080%|
| NMS0.6 |4909|1504|3450|76,547638%|58,727120%|66,463580%|

NMS giảm552 false positives nhưng mất10 true positives. F1 tăng2,305500 điểm phần trăm; vì vậy R04+NMS0.6 là phương án cân bằng hơn trong hai cấu hình R04 đã chấm ở640. Đây chưa phải model tốt nhất của11 lượt. Conf0.25 được giữ cố định cho đối chiếu, chưa tune threshold theo kết quả.

AP50 từng lớp native / NMS0.6: person75,19/77,57%; table71,03/72,89%; chair62,27/64,32%; laptop76,97/78,80%; cell phone50,72/53,10%; backpack81,87/82,91%; book58,79/62,47%; bottle78,98/80,55%. Điện thoại và sách còn yếu ở640.

## Thời gian batch1 trên CPU

Replay40 ảnh validation trải đều, warmup5, conf0.25; ảnh đã decode trước khi tính giờ. Không bao gồm camera capture, tracker, drawing và UI.

| Đợt đo | p50 | p95 |
|---|---:|---:|
| Native: tổng predict |51,81ms|71,40ms|
| Native: inference |48,44ms|67,35ms|
| Đợt riêng: predict + NMS0.6 |48,87ms|60,69ms|
| Chỉ phần NMS của đợt riêng |0,42ms|1,44ms|

Hai đợt chạy nối tiếp có dao động; không kết luận thêm NMS khiến model nhanh hơn. p95 trong các mẫu này dưới100ms nhưng chưa xác nhận latency toàn frame/FPS webcam hoặc bảo đảm mọi ảnh dưới100ms. Không suy tốc độ webcam từ batch8 validator.

## Đối chiếu lịch sử1280 và khôi phục

Log cùng GPU T4 FP16 batch8, một lượt1280:

| Checkpoint | mAP50 native | mAP50–95 native | Precision | Recall |
|---|---:|---:|---:|---:|
| R02 |76,400744%|54,395666%|82,373006%|67,334471%|
| R04 |76,893780%|54,623630%|80,900112%|68,460746%|
| R07 |76,969425%|55,223987%|79,051814%|69,547600%|

R07 dẫn AP/recall native1280; R02 dẫn precision. Không có ứng viên thắng cả bốn chỉ số trong bảng này. Không chuyển thứ hạng1280 sang640; sốGPU FP16 cũ vàCPU FP32 mới cũng không phải đối chứng chỉ thay kích thước. Mốc80,505213% trước đây là R04 ba view1280+NMS0.5, không phải điểm webcam640.

Kiểm tra khôi phục: local campaign chỉ giữR04; script `.temp/cleanup_campaign_artifacts_20261010.ps1` dùngRemove-Item để dọn các PT/ZIP khác. Các filebest.zip/last.zip trongDownloads đã đối chiếu, không khớpR07. Notebook Kaggle `awm3636/project8-v05-fast-archive-from-best-v04` có8 phiên bản, phiên bản8 lưu output baseline/R01; chưa thấyR02/R07, không có phiên bản9 được xác nhận. Editor draft hiển thịoff, không có output đang dùng để lấyR07. Không bậtRunAll hoặc train lại để tạo model thay cho checkpoint đã mất.

SHA cần cho bản lưu thật: R02 `c0b19c81af77842a59a596a057548641ca920175600edd61dd6bac620afaa0c1`; R07 `ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785`. So sánh R04/R07 ở640 còn thiếu artifactR07. Tạm đề xuất R04 một lượt+NMS0.6 trong các cấu hình đã đo; không đổi alias hoặc tuyên bố đã chọn model thắng toàn bộ.

Kết quả máy đọc: `R04/comparison.json`, `R04/latency_nms_0.6.json`; cache `R04/validation_cache.pt` giữ để đối soát. Review nhãn bước đầu ở `../label_review_20261010`:27 ảnh tạm loại,6 ảnh lỗi biên cần xem thêm, candidate train chưa kích hoạt.
