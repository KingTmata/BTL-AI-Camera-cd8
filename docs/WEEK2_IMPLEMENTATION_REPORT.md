# Báo cáo triển khai tuần 2 — cập nhật 04/10/2026

## Kết quả và phạm vi

**Cập nhật 04/10/2026:** dataset chính draft `project8_v0.3` có **5.002 ảnh / 5.002 TXT / 26.566 box**: 3.883 train, 798 validation, 321 test. Nguồn: 1.800 ảnh v0.2, 566 COCO bổ sung, 600 sách/điện thoại và 2.036 classroom. Có 4.681 file ảnh phát triển; không coi đây là số cảnh/ảnh gốc độc lập vì nguồn có augmentation và frame video. `table` và `with-student` cùng map về table theo quyết định người dùng; lớp bàn có 1.017 ảnh / 2.479 box. Nhãn vẫn còn một phần/chưa review đầy đủ. **G2 chưa nghiệm thu:** còn nhãn/review, quyền/bối cảnh, gần trùng/phiên, khóa release, video thật và baseline validation. Chưa fine-tune.

Kiểm manifest chính: **5.002 ảnh dữ liệu chính**, không còn pending. Bằng chứng hiện hành: `data/dataset/project8_v0.3/manifest.csv`, `statistics.json`, `validation.json`, `checksums.json` và [bảng báo cáo mới](../reports/results/project_progress_20261004/REPORT.md). Đã xóa 107 cặp pending, 6.566 bản sao ảnh, 189 ảnh tham khảo/kết quả thử và các ZIP sao lưu. Hai ZIP hành vi gốc tuần 4–6 giữ nguyên. Các số liệu/run smoke v0.2 và COCO128 bên dưới là lịch sử, không dùng làm kiểm kê hiện hành.

Code hiện hành còn CLI/UI nhận dạng, setup demo, validator/evaluator, workflow tuần 2 và script báo cáo. Đã bỏ 11 file Python phục vụ nhập/gộp/chọn dữ liệu cũ và CLI tuần 1 trùng chức năng. Giao diện lấy 50 ảnh train từ manifest chính, dùng profile nhãn project8, không cần bản sao ảnh mẫu. Kiểm thử hiện hành: **27/27 pass**, không skip; các bảng 20 test trong phần triển khai gốc dưới đây là lịch sử.

## Quyết định đã áp dụng

- Tám lớp theo đúng thứ tự: person, table, chair, laptop, cell phone, backpack, book, cup.
- Table gồm các loại bàn; baseline COCO dùng dining table làm đối chứng gần đúng. Book gồm sách/vở; không tạo lớp notebook riêng.
- Giữ checkpoint pretrained80, lọc/mapping tám lớp khi inference. Không cắt file weights thành model tám lớp. Fine-tune sẽ dùng YAML tám lớp của release khi ảnh/nhãn đã sẵn sàng.
- Quy mô mục tiêu trong WEEK2_G2.md: ≥2.500 ảnh phát triển hợp lệ, chia theo phiên/nhóm. Hiện có 4.681 file train/val và 321 test; chưa xác nhận số cảnh độc lập. Tiếp tục hoàn thiện nhãn và test đủ lớp; test hiện thiếu backpack/cup.
- Không xây công cụ tìm gần trùng riêng. Thành viên xác nhận việc kiểm gần trùng sau khi gộp dữ liệu; giữ kiểm trùng tuyệt đối bằng SHA-256.

## Công cụ đã hoàn thành

| Thành phần | File | Hành vi |
|---|---|---|
| Hợp đồng lớp | src/classes.py | Tám tên/ID nhóm và mapping COCO80, dùng chung UI/CLI/evaluator |
| Inventory | scripts/data/week2.py inventory | Kê toàn bộ ảnh nguồn, tạo manifest pending; không tạo nhãn/approval |
| Validator | src/dataset.py; week2.py check | Kiểm ảnh, cặp nhãn, ID, hình học, box lặp, ảnh trùng tuyệt đối, hash, rò nhóm và thống kê |
| Chia nhóm | week2.py split | Hợp phiên/split_group/duplicate_group, giữ split nguồn, seed 42; không chia frame cùng phiên qua hai tập |
| Khóa release | week2.py lock | Kiểm quyền/review/đủ lớp, sao chép ảnh và nhãn, manifest, list split, YAML, data card thống kê và checksums |
| Kiểm phiên bản | week2.py verify | Phát hiện file thêm/xóa/sửa; từ chối release INCOMPLETE |
| Evaluator | src/evaluation.py | GT/predictions về project8, COCO bbox AP, metadata/hashes, ảnh âm tính và empty predictions |
| Kiểm toàn luồng | scripts/data/smoke_week2.py | Dữ liệu nhân tạo + pretrained thật; hoàn toàn tách khỏi baseline nhóm |

Validator không đoán nhãn thiếu/đúng vật; check kiểm 100% ảnh **trong manifest**, vì vậy phải inventory đủ thư mục bàn giao. Release yêu cầu annotation approved, quyền có thông tin, xác nhận near_duplicate_status, reviewer khác annotator, ít nhất 10% mỗi split và có review mọi lớp; tất cả ảnh âm tính/needs_review phải được review. Các trạng thái là xác nhận của người điền, phần mềm không chứng minh danh tính hay việc người đó thật sự xem ảnh.

Chia theo nhóm chỉ cân số ảnh, không hứa đúng 80/20 tuyệt đối hoặc cân bằng lớp. Lock từ chối train/val thiếu lớp; nhóm cần điều chỉnh cả nhóm hoặc nhận thêm dữ liệu. Không ghi đè release hay output đánh giá cũ. Source/split/list/hash thuộc release đi cùng dữ liệu; chỉ file mẫu và công cụ được commit.

## Evaluator và phép kiểm

Dùng pycocotools 2.0.11 đã cài vào .venv và pin trong requirements-evaluation.lock.txt. Bảng chính dùng imgsz=640, conf=0.001, nms=True, iou=0.70, max_det=300; COCO bbox IoU 0.50:0.05:0.95, area all, maxDets [1,10,100]. Ghi actual_end2end từ backend để xác nhận head thực tế. Đọc chi tiết trong EVALUATION_PROTOCOL.md.

**20/20 kiểm thử tự động pass**, gồm:

- 8 ca dataset: nhãn sai/NaN/hộp ngoài ảnh, ảnh hỏng, thiếu nhãn, ảnh trùng, đường dẫn ngoài repo, orphan labels, inventory pending, nhóm liên thông/seed, xung đột split, negative/review và checksum sửa/thêm/xóa.
- 6 ca evaluator: hộp đúng tám lớp AP≈1, sai lớp/empty predictions AP=0, FP trên ảnh âm tính giảm AP, hộp dư không thêm TP, mapping COCO80/project8 và lớp không GT trả null.
- 5 ca inspection: mapping/profile, nhãn gốc, crop và chuyển màu.
- 1 ca giao diện: YOLO thật, hiển thị tám lớp, lọc rỗng và ẩn kết quả cũ khi đổi nguồn/confidence.

Đã chạy CLI check→split→lock→verify→evaluate trên 8 ảnh nhân tạo, tạo train 6/val 2 theo nhóm. Thử pretrained thật ở cả NMS và NMS-free; backend thực lần lượt end2end=False/True. AP=0 trên ảnh màu/hộp giả là kết quả kiểm phần mềm, **không phải baseline chất lượng**. Output ở `runs/week2/synthetic_smoke/`; SMOKE_RESULT.json ghi rõ mục đích. `pip check`: No broken requirements found.

Đã xử lý ảnh hỏng bằng Pillow gốc để tránh wrapper Ultralytics tự tải HEIF khi bất kỳ ảnh nào không giải mã được. Chấp nhận JPG/JPEG/PNG/WebP/BMP; HEIC phải chuyển/chuẩn hóa trước khi gán nhãn.

## Hồ sơ và README

Đã cập nhật data/DATA_CARD.md bằng số liệu draft thực tế, EVALUATION_PROTOCOL.md, docs/WEEK2_WORKFLOW.md, docs/VIDEO_ANNOTATION.md và mẫu events.csv. Cập nhật LABELING_GUIDE, manifest schema, YAML, UI/CLI, README, hướng dẫn train/kiểm thử và mô tả thư mục. Sửa link tới tên kế hoạch cũ và WEEK1_G1 đã xóa, cùng tham chiếu scripts/setup.ps1 không tồn tại. Hồ sơ tuần 1 bốn lớp được đánh dấu lịch sử, không coi là review project8.

## Điều còn thiếu để G2 đạt

| Checklist G2 | Trạng thái thực tế |
|---|---|
| Ảnh đủ quy mô, nguồn/quyền và bối cảnh | 5.002 file ảnh chính; 4.681 phát triển, mục tiêu 2.500 ảnh hợp lệ; quyền/bối cảnh/độc lập cảnh còn cần xác minh |
| Lọc ảnh hỏng/trùng/gần trùng sau gộp | Đã kiểm giải mã/trùng bytes và pixel nguồn mới; xác nhận gần trùng/phiên còn chờ |
| Nhãn YOLO đầy đủ tám lớp | Có nhãn cả tám lớp trong bộ chính; nhãn nguồn còn một phần, cần bổ sung/review; test thiếu backpack/cup |
| Kiểm 100% và review người thứ hai | Đã kiểm kỹ thuật 5.002 ảnh; review người thứ hai chưa duyệt |
| Split train/val khóa, đủ lớp, checksum | Draft có split và checksum; chưa khóa release đã duyệt |
| YAML/manifest/data card thật | Có YAML/manifest/data card draft project8_v0.3 thực tế |
| 3–5 video_dev và sự kiện, kế hoạch test độc lập | Mẫu/quy ước sẵn; chưa có clip thật |
| Baseline validation pretrained | Evaluator sẵn; chưa chạy được trên val nhóm |

Từ draft hiện tại, làm theo docs/WEEK2_WORKFLOW.md để inventory, hoàn thiện metadata, kiểm, chia, review, lock và chạy baseline. Không tự đánh dấu review thay người khác. Video vẫn cần nhóm quay/chọn, gán sự kiện và review; quy ước sự kiện chưa phải evaluator tracking của tuần 4–5.

## Skill và hạn chế công cụ

Đã đọc using-agent-skills để chọn workflow, git-workflow-and-versioning để giữ các thay đổi Git có sẵn, incremental-implementation để triển khai/kiểm từng phần, yolo và yolo-datasets để tuân thủ format nhãn/split và kiểm phiên bản cài. Đã đọc graphify từ lượt audit; repo hiện không có graphify-out/graph.json. Sau sửa mã đã thử `graphify update .`, nhưng CLI không có trên PATH nên chưa cập nhật được graph. Việc kiểm dữ liệu/evaluator không phụ thuộc graph này.

Chưa commit/push, chưa train, chưa tạo dữ liệu nhóm giả để hoàn tất checklist. Thay đổi staged của người dùng được giữ nguyên. G2 còn hoàn thiện nhãn/metadata, xác nhận split, review, video thật và baseline.
