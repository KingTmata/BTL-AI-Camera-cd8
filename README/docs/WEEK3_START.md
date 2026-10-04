> **Đối soát 05/10/2026:** xem [trạng thái hiện hành](../README.md#trạng-thái-theo-tuần). Dataset project8_v0.3 đã duyệt/khóa; R1 và R2 hoàn tất tổng 5 epoch. Những mô tả trước đó trong tài liệu là lịch sử hoặc đề xuất; G1/G2/G3 chưa đủ nghiệm thu toàn bộ.

# Bắt đầu tuần 3 — CPU local, review trước mọi lần train

Cập nhật 04/10/2026. Người dùng đã chọn **CPU trên máy hiện tại** và **hoàn tất review, khóa dữ liệu trước mọi lần train**. Đã triển khai Training UI và hoàn tất lượt CPU một epoch trên toàn train; baseline evaluator chung chưa chạy. Tài liệu này chuẩn bị quy trình tối thiểu, tái dùng trainer Ultralytics và evaluator đã có; không tạo notebook, tải model hoặc nhân bản dataset.

## Trạng thái đã kiểm

- Dataset duy nhất: `data/dataset/project8_v0.3`, 5.002 ảnh/TXT, 26.566 box; train/val/test = 3.883/798/321.
- `configs/data.yaml` trỏ đúng kho hiện hành và mapping tám lớp; validator cùng checksum đạt kiểm kỹ thuật.
- Môi trường hiện tại: PyTorch 2.14.0+cpu, CUDA không khả dụng, Ultralytics 8.4.165; pretrained `weights/yolo26n.pt` có sẵn.
- 44/44 kiểm thử đạt, không skip; schema cấu hình CPU được phiên bản cài đặt chấp nhận. Đây không phải kết quả train hay baseline model.
- Người dùng đã xác nhận review toàn bộ 5.002 ảnh/hộp nhãn đủ tám lớp, nguồn/quyền và gần trùng/nhóm phiên. Metadata đã ghi approved và snapshot đã khóa tại chỗ; 4.566 cờ nguồn đều có review approved.
- Test chưa có nhãn backpack/cup. Train và val có nhãn cả tám lớp; người dùng đã xác nhận đủ các vật thuộc tám lớp trong toàn bộ ảnh.

Bằng chứng: [báo cáo readiness](../../README_DA_HOAN_THANH/reports/results/week3_readiness_20261004/REPORT.md), [JSON kỹ thuật](../../reports/results/week3_readiness_20261004/readiness.json), [profile hiện hành](PROJECT8_V0_3_DATA_PROFILE.md). Lượt đầu đã hoàn tất và kiểm checkpoint: tổng 22 phút 6 giây, final validation mAP50 7,09%, mAP50–95 4,95%. [Báo cáo lượt CPU](../../README_DA_HOAN_THANH/reports/results/project8_v03_26n_cpu_20261004_181257_1eb3de94/REPORT.md).

## 1. Review và khóa dữ liệu — đã hoàn thành

Người dùng đã xác nhận kiểm toàn bộ 5.002 ảnh cùng hộp nhãn, đủ các vật thuộc tám lớp, và kiểm/chấp nhận nguồn/quyền cùng gần trùng/nhóm phiên. Ghi nhận trực tiếp trong [human_review_confirmation.json](../../reports/results/week3_readiness_20261004/human_review_confirmation.json); reviewer dùng mã human_user vì chưa cung cấp tên thật, không phải AI reviewer.

Bốn bảng metadata hiện có đã ghi annotation/review/near-duplicate approved; annotator nguồn, media, nhãn TXT và split giữ nguyên. 4.566 cờ needs_review cũ vẫn được giữ, tất cả đã có review approved. [Danh sách tham chiếu](../../reports/results/week3_readiness_20261004/review_queue.csv) gồm 5.002 dòng hiện đã duyệt.

Kho duy nhất vẫn là project8_v0.3. [RELEASE.json](../../data/dataset/project8_v0.3/RELEASE.json) ghi release project8_v0.3-approved-20261004; checksum sau khóa và validator release trên đủ 5.002 ảnh đều đạt. Không dùng lệnh week2.py lock sao chép dataset. Mã phiên unknown/khai báo nguồn gốc được giữ trung thực; xác nhận nhóm và nguồn/quyền của người dùng được lưu riêng.

Khi kiểm kỹ thuật lại, chạy từ gốc repo, dùng output mới:

```powershell
& .\.venv\Scripts\python.exe scripts/data/week2.py check --manifest data/dataset/project8_v0.3/manifest.csv --output reports/results/week3_locked_validation_run01.json
```

Không sửa ảnh/nhãn/split trong một run hoặc thay checksum để bỏ qua thay đổi. Nếu cần chỉnh dữ liệu sau khóa, phải ghi phiên bản snapshot mới và chấm lại các kết quả liên quan; không nhân bản media trong bước chuẩn bị này.

## 2. Baseline validation trên release đã khóa

Dataset hiện đã qua validator release và checksum sau xác nhận của người dùng. Evaluator tiếp tục kiểm validate(release=True) và checksum trước/sau inference; dùng release hiện hành và output riêng cho từng model/run.

```powershell
& .\.venv\Scripts\python.exe -m src.evaluation --manifest data/dataset/project8_v0.3/manifest.csv --weights weights/yolo26n.pt --split val --device cpu --output reports/results/project8_v03_pretrained_val_run01
```

Protocol chung: `imgsz=640`, `conf=0.001`, `nms=True`, `iou=0.70`, `max_det=300`, COCO bbox IoU 0.50:0.05:0.95, maxDets `[1,10,100]`. Mapping COCO80 → project8 là `0→0, 60→1, 56→2, 63→3, 67→4, 24→5, 73→6, 41→7`; checkpoint project8 dùng identity. Không chấm pretrained 80 lớp trực tiếp trên YAML tám lớp rồi so hai bảng khác protocol.

Evaluator xuất `ground_truth.json`, `predictions.json`, `metrics.json`, `run.json`, kèm hash dataset/checkpoint, mapping và môi trường. Không ghi đè output cũ. Xem [EVALUATION_PROTOCOL.md](../EVALUATION_PROTOCOL.md).

## 3. Lượt CPU đầu — một epoch trên toàn train

Đã bổ sung [Training UI](../../TRIEN_KHAI_TUAN_3_TIEP_TUC.md). Lượt đầu được người dùng chấp thuận dùng **toàn bộ 3.883 train và 798 validation**, epochs 1, imgsz 640, batch 2, workers 0, cache False, fraction 1.0, MuSGD, lr 0.001, seed 42. Sau 44/44 kiểm thử đạt, đã bắt đầu run `project8_v03_26n_cpu_20261004_181257_1eb3de94` ngày 04/10/2026. Run đã hoàn tất đúng một epoch, best.pt/last.pt/epoch0.pt nạp được và đúng tám lớp; dataset/pretrained không đổi. Chưa chạy thêm epoch 2–3. [Kết quả](../../README_DA_HOAN_THANH/reports/results/project8_v03_26n_cpu_20261004_181257_1eb3de94/REPORT.md) có loss, metric, thời gian, RSS/peak working set và hash.

UI theo dõi tiến trình Python riêng, sống qua tải lại/đóng trình duyệt. Mỗi máy chỉ một run của app; dừng sau khi lưu epoch. Lưu best.pt, last.pt và epoch0.pt (epoch đầu tiên), log, cấu hình, loss/validation, thời gian và RAM thực tế. Có thể khởi tạo run mới từ checkpoint; UI v1 chưa resume nguyên trạng optimizer.

Verifier của snapshot tại chỗ chỉ bỏ qua ba file dẫn xuất labels/train.cache, labels/val.cache, labels/test.cache do trainer sinh. Ảnh, TXT và mọi metadata/file khác vẫn được kiểm hash/danh sách; không sửa checksum để vượt kiểm.

Lượt một epoch đo khả năng chạy và tài nguyên CPU. Validation của trainer dùng NMS-free; chưa thay bảng baseline/fine-tuned theo evaluator chung. Không dùng fraction nhỏ làm số đo của epoch đầy đủ.

## 4. Fine-tune YOLO26n và so sánh

Sau smoke đạt, dùng toàn train và một run mới, giữ cấu hình chính: imgsz 640, epochs 50, MuSGD, lr0 0.001, patience 10, seed 42. Batch CPU khởi đầu 2 là override đề xuất, chưa benchmark; chỉnh theo RAM và ghi rõ giá trị thực.

```powershell
& .\.venv\Scripts\yolo.exe detect train cfg=configs/train_n.yaml model=weights/yolo26n.pt data=configs/data.yaml device=cpu batch=2 workers=0 cache=False fraction=1.0 nms=False project=runs/train name=project8_v03_26n_cpu_supervised_run01
```

Lưu `args.yaml`, `results.csv`, loss/validation plots, `best.pt`, `last.pt`, phiên bản môi trường, manifest/checksum hash và thời gian thực. Lấy thư mục run thật vì trainer có thể thêm hậu tố. Kiểm hash dữ liệu trước/sau, không trộn nhãn sửa giữa một run.

Sau train, chấm `best.pt` bằng cùng evaluator ở bước 2, thay weights và output riêng. Validation trong trainer giúp chọn checkpoint; bảng pretrained/supervised chính dùng evaluator chung. Head NMS-free của demo/train và head NMS của protocol đánh giá phải được ghi riêng, thống nhất giữa các model trong cùng bảng.

Giữ test ngoài mọi tối ưu. Lớp thiếu ground truth phải báo chưa đánh giá/null theo protocol, không ghi AP=0 như thể đã có bộ test lớp đó. YOLO26s chỉ xét sau YOLO26n và khi đủ tài nguyên; SSL sau supervised, không thuộc bước chuẩn bị này.

## Điều kiện nghiệm thu G3

- [x] Dữ liệu được review và khóa, có xác nhận người dùng về nguồn/phiên/gần trùng, validator release và checksum.
- [ ] Baseline pretrained validation bằng evaluator chung.
- [x] Lượt thử CPU một epoch trên toàn train có kết quả và checkpoint thật.
- [ ] Fine-tune YOLO26n chính thức và đánh giá đủ; lượt thử một epoch đã có checkpoint/args/log/loss/metric nhưng chất lượng còn thấp.
- [ ] Bảng so sánh cùng dataset/protocol; báo cấu hình và thời gian CPU thực.
- [ ] YOLO26s chỉ khi có run; mọi phần chưa chạy ghi rõ chưa thực hiện.

Nguồn kỹ thuật đã đối chiếu: [Ultralytics Train](https://docs.ultralytics.com/modes/train/), [Validation](https://docs.ultralytics.com/modes/val/). Cú pháp override đã kiểm bằng schema phiên bản cài đặt; khả năng chạy và chất lượng chỉ được xác nhận sau run thật.
