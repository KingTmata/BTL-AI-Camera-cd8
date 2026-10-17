# Triển khai tuần ba tiếp tục

**Cập nhật 06/10/2026:** đã thực thi **30 epoch trong 5 run**, gồm CPU R1/R2/R3 và hai lượt Kaggle 10 + 14 epoch. [Bảng đủ 30 epoch và đối soát file/trọng số](../completed/reports/results/training_30_epochs_20261006/report.md); báo cáo Kaggle 10 epoch ở **mục 7** là lịch sử, kết quả 14 epoch mới ở **mục 8**. Tài liệu thay thế `docs/TRAINING_UI.md`.

## 1. Điểm xuất phát hiện tại

Đã chạy **năm lượt có kết quả: R1 1 epoch + R2 4 epoch + R3 1 epoch + Kaggle 05/10 10 epoch + Kaggle 06/10 14 epoch = 30 epoch đã thực thi**. R3 dừng an toàn sau epoch đầu theo yêu cầu; các lượt còn lại hoàn tất. STT 1–30 là lịch sử thực thi qua các run mới; không phải 30 epoch resume liên tục hoặc 30 lượt train độc lập:

- **R1:** một epoch từ YOLO26n pretrained của Ultralytics, sau đó tạo checkpoint tám lớp.
- **R2:** bốn epoch mới từ **R1 best**, giữ dataset đã khóa. Optimizer, warmup và scheduler khởi tạo lại khi mở R2.
- **R3:** một epoch CPU từ **R2 best**, imgsz 768/batch 4; dừng sau epoch và kiểm checkpoint đạt. R3 best là nguồn đưa lên Kaggle.
- **Kaggle 05/10 (K1):** 10/10 epoch từ **R3 best**, GPU T4 x2, imgsz thực 1088/batch tổng 4; model tốt nhất sau **epoch Kaggle 9**, model cuối sau epoch 10.
- **Kaggle 06/10 (K2):** 14/14 epoch từ **K1 best sau epoch 9**, GPU T4 x2, imgsz 1280/batch tổng 16/mosaic 0.5. Metadata best khớp epoch 14; chưa có last/epoch13 để so tensor độc lập. Epoch K1 10 có trong lịch sử thực thi nhưng không nằm trong nguồn khởi tạo K2.
- **Lượt tiếp theo:** chọn **Kaggle best mới** dưới đây. Không mặc định dùng lại pretrained, R2 hoặc R3 khi muốn tiếp tục từ kết quả mới nhất. Mỗi lượt fine-tune mới vẫn reset optimizer/scheduler/warmup.

Checkpoint nên dùng cho lượt mới:

```text
weights/BEST_Project8.pt
```

[Trọng số Kaggle best mới đã nhập local](../../../weights/BEST_Project8.pt) có SHA-256 `ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf`, 5.546.053 byte; đúng byte best.pt trong ZIP và nạp đúng tám lớp person, table, chair, laptop, cell phone, backpack, book, cup, tensor hữu hạn. train_results khớp đủ 14 dòng CSV, train_metrics/fitness khớp epoch 14; chưa so độc lập với last.pt/epoch13.pt vì chưa tải hai file này. Best đã strip optimizer, dùng cho nhận dạng hoặc **fine-tune mới với resume=False**.

So sánh chung trên 798 ảnh validation, COCO bbox ở imgsz 640/NMS: pretrained mAP50/mAP50–95 **28,176%/18,895%**, K1 best **47,779%/28,619%**, K2 best mới **53,848%/33,847%**. K2 tăng **+6,068/+5,229 điểm %** so K1. Không gộp với metric epoch cuối của trainer 1280/NMS-free **56,546%/35,617%**. Chưa dùng test hoặc SSL, chưa đo inference FPS và chưa nghiệm thu toàn bộ tuần 3.

## 2. Phân biệt toàn bộ trọng số để tránh nhầm

| Nguồn | Checkpoint | Đã học dữ liệu dự án đến đâu / dùng khi nào |
|---|---|---|
| Ultralytics gốc | [weights/yolo26n.pt](../../../weights/yolo26n.pt) | Pretrained COCO 80 lớp, chưa học dataset project8. Dùng làm điểm xuất phát/đối chứng gốc. |
| R1 | [runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt](../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt) | Tốt nhất của lượt một epoch; là nguồn khởi tạo R2. |
| R1 | [runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/last.pt](../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/last.pt) | Cuối lượt R1: sau epoch tổng 1. |
| R1 | [runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/epoch0.pt](../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/epoch0.pt) | Bản lưu riêng sau epoch R1 1 = epoch tổng 1. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/best.pt](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/best.pt) | Tốt nhất theo validation R2, đã kiểm xong; nguồn R3 và mốc lịch sử. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/last.pt](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/last.pt) | Cuối R2: sau epoch R2 4 = epoch tổng 5; không tự resume optimizer qua app. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch0.pt](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch0.pt) | Sau epoch R2 1 = epoch tổng 2. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch1.pt](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch1.pt) | Sau epoch R2 2 = epoch tổng 3. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch2.pt](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch2.pt) | Sau epoch R2 3 = epoch tổng 4. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch3.pt](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch3.pt) | Sau epoch R2 4 = epoch tổng 5. |
| R3 | [best.pt](../../../runs/train/project8_v03_26n_cpu_r2best_e5_20261005_134221_540fdcbf/weights/best.pt) | Sau một epoch R3 từ R2 best; nguồn của lượt Kaggle, không phải best mới nhất. |
| R3 | [last.pt](../../../runs/train/project8_v03_26n_cpu_r2best_e5_20261005_134221_540fdcbf/weights/last.pt) / [epoch0.pt](../../../runs/train/project8_v03_26n_cpu_r2best_e5_20261005_134221_540fdcbf/weights/epoch0.pt) | Mốc cuối/lưu riêng sau epoch R3 thứ 1; R3 dừng có kiểm chứng. |
| Kaggle K1 | [best đã nhập](../../../weights/kaggle/project8_26n_kaggle_20261005_092653_fd9cb282f05f.pt) / [best bản gốc tải về](../../../runs/kaggle/project8_26n_kaggle_20261005_092653/artifacts/weights/best.pt) | Cùng hash/tensor, tốt nhất sau epoch Kaggle 9. **Nguồn của K2, mốc lịch sử/quay lui.** |
| Kaggle | [last.pt](../../../runs/kaggle/project8_26n_kaggle_20261005_092653/artifacts/weights/last.pt) | Sau epoch Kaggle 10; validation thấp hơn best một chút. Best/last đều đã strip optimizer. |
| Kaggle | [epoch0.pt–epoch9.pt](../../../runs/kaggle/project8_26n_kaggle_20261005_092653/artifacts/weights) | Sau epoch Kaggle 1–10, còn optimizer; lưu lịch sử hoặc khôi phục khi đã kiểm lịch train. `epoch8.pt` là mốc epoch 9, `epoch9.pt` là mốc epoch 10. |
| Kaggle K2 | [best mới đã nhập](../../../weights/BEST_Project8.pt) | **Ưu tiên nhận dạng hoặc fine-tune mới.** Metadata khớp epoch 14, optimizer đã strip; result ZIP chưa có last/checkpoint từng epoch. |

**Tên file giống nhau chưa chắc cùng nguồn:** R1 epoch0 là epoch tổng 1, R2 epoch0 là epoch tổng 2. Luôn đối chiếu cả thư mục run. `best` là checkpoint được chọn theo validation; `last` là trạng thái cuối lượt, không mặc định tốt hơn best. Checkpoint riêng theo epoch giữ mốc để quay lại so sánh. File có thể khác hash/dung lượng vì trạng thái optimizer hoặc bước hoàn thiện checkpoint; không dựa riêng tên/dung lượng để suy ra chất lượng.

R2 có đủ sáu file best/last/epoch0–epoch3; tám lớp, dataset và checkpoint nguồn đã qua kiểm cuối lượt. Hash R1/pretrained gốc vẫn khớp bản ghi trước R2. [Kiểm checkpoint R2](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/checkpoint_validation.json) · [Kiểm bảo toàn nguồn](../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/tracking_verification.json).

Kaggle có đủ 12 checkpoint best/last/epoch0–epoch9, tất cả đã kiểm nạp đúng tám lớp. [Kiểm checkpoint và mốc epoch Kaggle](../../../runs/kaggle/project8_26n_kaggle_20261005_092653/checkpoint_validation.json). Không dựa vào `stored_epoch=-1` của best/last sau strip để suy ra model chưa học.

## 3. Các bước train tiếp từ Kaggle best mới

### Trên Kaggle — nền tảng ưu tiên đã chọn

1. Trong app local, chọn Training → **Kaggle Notebook**, chọn đúng file K2 best ở mục 1 rồi **Tạo gói Kaggle**. Gói mới ghi hash model mới; không thay thủ công `initial_project8.pt` trong bundle cũ vì sẽ sai checksum. K1 đã học từ R3 best, K2 đã học từ K1 best; bundle cũ không tự đổi nguồn theo model mới.
2. Tải gói mới lên Dataset **Private**, gắn vào notebook bằng **Add Input**. Nếu gắn nhiều bundle, đặt `BUNDLE_INPUT` sát dataset mới để preflight không chọn nhầm nguồn.
3. Giữ `START_TRAINING=False` khi chạy setup/preflight. Chọn accelerator T4 x2 nếu được cấp; kiểm CUDA và log device `0,1`, hash nguồn **`ad136a17611f...`**, đủ 3.883 train/798 val và tám lớp trước khi cho phép train.
4. K2 đã đo: 14 epoch, imgsz 1280, batch tổng 16, mosaic 0.5, workers 2/rank, cache False, close_mosaic=5. K1 dùng 1088/batch 4/mosaic 1.0; do đổi nhiều yếu tố không gọi đây là benchmark riêng batch. Cấu hình/hạn mức lượt tiếp theo chưa thực thi; ghi riêng và giữ protocol đánh giá chung.
5. Khi đã chọn hạn mức và nguồn đúng, mới bật `START_TRAINING=True` và chạy train một lần. Dùng thư mục run mới, lưu checkpoint từng epoch. Sau hoàn tất, chấm baseline/model theo cùng protocol, tải result ZIP/full run và kiểm checksum trước khi Stop Session.

Gói/model mới đã nằm trên máy local; chưa tạo hoặc upload bundle cho lượt kế tiếp. Xem quy trình export trong [Kaggle handoff](../guides/kaggle_training_handoff.md).

### Trên web local — nếu muốn dùng CPU

1. Mở app và chọn **Màn hình → Training**. Nếu chưa chạy server, dùng lệnh ở phần hướng dẫn local phía dưới.
2. Trong **Checkpoint khởi tạo**, chọn đúng **Kaggle best đã nhập** theo đường dẫn ở mục 1. App đọc cả các file trong `weights/kaggle/`. Không nhầm ô Checkpoint của Recognition với ô nguồn Training; Recognition chỉ chọn model để nhận dạng.
3. Ví dụ một lượt theo dõi ngắn: chọn **3 epoch mới**, batch **2**, kích thước **640**, workers **0**, cache RAM **tắt**. Đặt tên `project8_v03_26n_cpu_kagglebest_e3`. Đây là ví dụ để bạn tự quyết định số epoch, chưa phải lượt đã chạy.
4. Lượt đối chứng đầu giữ Nâng cao như mặc định: Mosaic=1.0, close_mosaic=10, MixUp=0, CutMix=0, xoay=0, lật ngang=0.5, lật dọc=0, cls_pw=0, Box=7.5, CLS=0.5, L1=1.5. Với ba epoch và close_mosaic=10, các phép ghép đóng từ đầu, cùng cách vận hành của các lượt ngắn trước.
5. Bấm **Xem trước cấu hình**, đọc cảnh báo lịch Mosaic và bảng phân phối box/weight. Xác nhận đủ 3.883 train, 798 validation, đúng checkpoint và RAM khả dụng. App dùng toàn train; không dùng test để chọn model.
6. Khi đã chọn hạn mức, bấm **Bắt đầu training**. App tạo run/thư mục mới, kiểm snapshot/checksum/nguồn/cấu hình rồi train; checkpoint khởi tạo được nạp đầu lượt, không nạp lại nguồn mỗi epoch. Giữ máy hoạt động, cắm nguồn và tránh để máy sleep trong khi train.
7. Theo dõi Train/Validation, log, loss, mAP và Recall. Muốn kết thúc sớm, dùng **Dừng sau epoch hiện tại** để chờ lưu checkpoint. Đóng trang không dừng worker.
8. Khi trạng thái completed hoặc stopped đã kiểm checkpoint, xem thư mục run mới và tải best/last/epoch. Nếu failed, đọc nguyên nhân; không coi checkpoint tồn tại là bằng chứng hoàn tất.

Đây là **run mới từ trọng số đã fine-tune**, app dùng `resume=False`. Epoch trong lượt mới đếm lại từ 1; optimizer/scheduler/warmup cũng khởi tạo lại. Ghi nguồn “K2 best run 20261006_083226, hash ad136a17...” và số epoch của lượt mới; không gọi là resume nguyên trạng. Khi lượt mới hoàn tất, chọn best của đúng lượt mới để dùng; không ghi đè trọng số của các lượt trước.

Với RAM trống khoảng 5–7 GB, cấu hình batch 2/cache tắt là điểm xuất phát đã đo. Không có bảo đảm tăng batch/cache làm nhanh hơn; benchmark từng thay đổi riêng, theo dõi RAM và thời gian thật. Trong trainer CPU hiện cài, workers được đưa về 0; tăng ô workers không bảo đảm tăng tốc CPU.

## 4. Mô hình đang học nhanh hay chậm?

**Kết quả hiện tại:** Kaggle best đạt mAP50/mAP50–95 **47,78%/28,62%** theo evaluator chung, tăng **19,60/9,72 điểm phần trăm** so pretrained. AP trong CSV Kaggle tăng đến epoch 9, giảm nhẹ ở epoch 10; chưa đủ kết luận overfitting kéo dài. Person/table/book/cell phone cải thiện rõ, nhưng laptop/backpack/cup giảm so baseline. Chi tiết đủ 10 epoch và tám lớp nằm ở mục 7.

Tốc độ Kaggle đo từ CSV: **156,520 giây/epoch (~2 phút 37 giây)** gồm train/validation; cell train tổng khoảng **27 phút 15 giây**, có thêm khởi tạo/final validation. Chưa đo FPS nhận dạng. CPU và GPU dùng cấu hình ảnh/batch/augmentation khác nhau nên không coi chênh thời gian là benchmark phần cứng cùng điều kiện.

### Lịch sử năm epoch CPU đầu (R1 + R2)

Khả năng nhận dạng cải thiện đều trong năm epoch CPU đầu. Bảng dưới giữ kết quả lịch sử, không phải chỉ số hiện tại của Kaggle; không gộp hai protocol thành một đường học liên tục.

| Epoch theo lịch sử trọng số | Precision | Recall | mAP50 | mAP50–95 | mAP50 tăng so epoch trước |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.975% | 11.303% | 7.084% | 4.946% | — |
| 2 | 37.374% | 12.620% | 11.144% | 7.370% | +4.060 điểm % |
| 3 | 37.604% | 20.116% | 15.975% | 10.299% | +4.831 điểm % |
| 4 | 57.333% | 24.081% | 21.764% | 13.804% | +5.789 điểm % |
| 5 | 50.317% | 27.786% | 25.783% | 16.246% | +4.019 điểm % |

Bảng dùng CSV cùng validation trainer; final validation của R2 best tách riêng: Precision **50,1544%**, Recall **27,7754%**, mAP50 **25,8002%**, mAP50–95 **16,2407%**. Tổng Recall/AP là trung bình theo lớp, không phải tỷ lệ đếm gộp mọi đối tượng trong tập val. Recall thấp chưa chứng minh riêng nguyên nhân imbalance. Backpack Recall=0%, cup=6,44%, cell phone=17,1% là các lớp cần kiểm nhãn/ảnh lỗi, kích thước và che khuất trước.

Về tốc độ CPU: ba epoch R2 ít có gián đoạn hơn (epoch 1, 3, 4) có thời gian train + validation suy từ chênh lệch CSV lần lượt khoảng **26 phút 11 giây, 24 phút 11 giây, 23 phút 23 giây**, trung bình khoảng **24 phút 35 giây/epoch**. Đây là số đo trên máy/lượt hiện tại, không phải cam kết thời gian lượt mới. Epoch R2 thứ hai kéo dài khoảng **3 giờ 26 phút**, nên không dùng trung bình cả lượt làm dự báo. Tổng lượt R2 khoảng **4 giờ 43 phút**, gồm cả các bước ngoài train/val. Không có dữ liệu benchmark để kết luận kiến trúc “học nhanh” hơn model khác.

## 5. Chọn thử nghiệm kế tiếp và chống nhầm kết quả

- **Theo dõi học thêm:** dùng **Kaggle best mới** và một hạn mức được chọn sau khi kiểm ảnh lỗi. Đối chứng pretrained/fine-tuned cùng [protocol](../../research/evaluation_protocol.md) đã có ở mục 7; các thử nghiệm mới giữ cùng protocol, test giữ cho đánh giá cuối.
- **Thử weighting:** A cls_pw=0, B=0.25, C=0.5 đều phải bắt đầu từ cùng **Kaggle best mới**, cùng số epoch, seed, optimizer, split và augmentation. Không dùng best của A làm nguồn B rồi coi đó là thử nghiệm đối chứng. Chỉ thay một biến; đánh giá P/R/AP từng lớp.
- **Thử Mosaic:** chọn lịch có epoch cho phép Mosaic, ví dụ 3 epoch/close_mosaic=0 hoặc close_mosaic=1. Đây là một thử nghiệm khác với đối chứng close_mosaic=10, chưa tự thực hiện. Nếu close_mosaic vẫn lớn hơn hoặc bằng số epoch, tăng Mosaic/MixUp không kiểm được tác dụng của chúng.
- **NaN/Inf, checksum sai, thiếu checkpoint hoặc process failed:** không mở thêm lượt chồng lên; kiểm log, cấu hình và dữ liệu. Không tiếp tục từ file chưa kiểm nạp được. Dừng sau epoch là dừng có kiểm soát; trong lỗi khẩn cấp cần kiểm đúng tiến trình trước khi can thiệp riêng.
- **AP tăng:** cân nhắc học thêm trong ngân sách thời gian đã chọn. **AP ngang hoặc giảm:** kiểm AP từng lớp, ảnh lỗi, nhãn và dấu hiệu overfit trước; không tăng hàng loạt augmentation/weight để phản ứng theo một epoch dao động.

[Sổ log và kế hoạch tuần 3](../guides/week3_training_log_and_plan.md) · [Báo cáo R2](../../../reports/archive/results/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/report.md).

## 6. Hướng dẫn Training local và Nâng cao đã hợp nhất

Cập nhật 05/10/2026. Màn hình Training dùng trainer Ultralytics trong tiến trình Python riêng; trình duyệt chỉ điều khiển và theo dõi. Hướng dẫn này áp dụng cho môi trường CPU đã cài của dự án.

## Mở và chạy

Từ thư mục gốc dự án:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/classroom_ai/interfaces/web/app.py --server.address 127.0.0.1 --server.port 8501
```

Mở http://127.0.0.1:8501, chọn **Màn hình → Training**. Nếu thanh bên đang đóng, bấm mũi tên ở góc trên trái.

Mặc định dùng `weights/yolo26n.pt`, CPU, một epoch, ảnh 640, batch 2, workers 0 và cache ảnh tắt. Toàn bộ 3.883 ảnh train được dùng; validation có 798 ảnh. Tập test 321 ảnh nằm ngoài huấn luyện và chọn tham số. Optimizer MuSGD, lr 0.001, patience 10 và seed 42 lấy từ cấu hình dự án.

Chỉnh form rồi bấm **Bắt đầu training**. Tên run nhận chữ/số ASCII, gạch dưới và gạch nối, tối đa 64 ký tự. Checkpoint phải có sẵn trong `weights/` hoặc `runs/train/`. Preflight kiểm snapshot đã khóa, checksum, review, YAML tám lớp, checkpoint và schema cấu hình; lỗi được ghi vào trạng thái/log và không bắt đầu trainer.

Batch là số ảnh trong một nhóm; workers 0 đọc dữ liệu trong tiến trình train và vẫn dùng nhiều luồng CPU. Cache tắt không giữ sẵn toàn bộ ảnh trong RAM, vẫn lưu checkpoint bình thường. Với RAM trống khoảng 5–7 GB, lượt đầu giữ batch 2 và cache tắt để đo trước khi điều chỉnh. RAM cache có kiểm ước lượng trước khi bật; số đo RAM thực tế vẫn cần theo dõi.

## Nâng cao và giải nghĩa (05/10/2026)

Trong form Training, mở **Nâng cao · tăng cường dữ liệu, cân bằng lớp và loss**. Bấm **Xem trước cấu hình** để cập nhật lịch augmentation và bảng class weights mà chưa train. Bấm **Bắt đầu training** mới tạo lượt mới. Các giá trị được kiểm ở cả RunManager và preflight, lưu vào request/effective_config và truyền vào trainer. Không ảnh hưởng checkpoint hoặc kết quả của lượt cũ.

| Tham số | Mặc định giữ nguyên | Ý nghĩa |
|---|---:|---|
| `mosaic` | 1.0 | Xác suất ghép bốn ảnh, trong [0,1]; `mosaic=10` không hợp lệ. Không phải cơ chế lấy mẫu cân bằng lớp. |
| `mixup` | 0.0 | Xác suất trộn hai ảnh và nhãn; 0.1 là 10% cơ hội áp dụng, không phải tỷ lệ pha ảnh. |
| `cutmix` | 0.0 | Xác suất cắt vùng ảnh ghép vào ảnh khác. |
| `degrees` | 0.0 | Xoay ngẫu nhiên trong khoảng ±giá trị độ; 0 tắt xoay. |
| `fliplr` / `flipud` | 0.5 / 0.0 | Xác suất lật ngang / dọc. Lật dọc có thể không phù hợp camera trong phòng. |
| `close_mosaic` | 10 | Số epoch cuối đóng Mosaic, MixUp, CutMix và Copy-Paste khi tác vụ hỗ trợ. 0 không đóng. Các phép lật/HSV/hình học vẫn có thể hoạt động. |
| `cls_pw` | 0.0 | Mức weighting theo tần suất box: 0 tắt, 0.25 nhẹ, 0.5 mạnh hơn, 1 inverse-frequency đầy đủ. |
| `box` | 7.5 | Trọng số loss định vị bounding box. |
| `cls` | 0.5 | Trọng số loss phân loại chung; khác `cls_pw` điều chỉnh tương đối giữa các lớp. |
| `dfl` | 1.5 | Trọng số thành phần L1 trên YOLO26; tên tham số vẫn là `dfl`. |

**Lịch Mosaic:** 100 epoch với close_mosaic=10 cho phép Mosaic trong epoch 1–90, đóng epoch 91–100. Với chỉ 1–4 epoch và close_mosaic=10, trainer 8.4.165 đóng ngay từ epoch đầu. Đây là cấu hình hợp lệ, app cảnh báo và không tự đổi thành 0. Nếu muốn thử augmentation ở lượt ngắn, cần chủ động chọn lịch có epoch cho phép ghép; chỉ tăng MixUp mà vẫn đóng toàn lượt sẽ không thử được MixUp.

**Class weights:** Ultralytics 8.4.165 hỗ trợ `cls_pw`; tính `w_i=(1/max(n_i,1))**cls_pw` từ số box `n_i` của từng lớp trong train rồi chia trung bình các weight để trung bình bằng 1. Bảng xem trước dùng counts của RELEASE đã khóa, log trainer ghi weight thực tế. Ví dụ tỷ lệ person/backpack của train hiện tại là 6.708/415 ≈ 16,16; với cls_pw=0.25, tỷ lệ weight backpack/person ≈ 2,00, không phải 16,16. Weighting không thêm ảnh và không sửa nhãn. Oversampling là lấy mẫu ảnh chứa lớp hiếm thường xuyên hơn; ảnh có nhiều lớp vẫn có thể tăng cả lớp phổ biến, cần một thử nghiệm dữ liệu riêng.

**Confidence:** slider Recognition mặc định 0.25 và chỉ lọc dự đoán khi nhận dạng. Validation trainer dùng ngưỡng đầu vào 0.001 khi conf chưa chỉ định để tính đường Precision–Recall/AP; P/R báo bởi trainer được chọn trên đường này, không tương đương Recall tại slider 0.25. Hạ slider có thể tăng số phát hiện và false positive, không làm model học lại. YOLO26 anchor-free: không có danh sách kích thước anchor boxes cần chỉnh; anchor points nội bộ khác anchor boxes định trước.

**Đọc chất lượng:** Precision là độ đúng của phát hiện, Recall là khả năng tìm được đối tượng có nhãn, AP là diện tích đường Precision–Recall từng lớp; mAP là trung bình AP các lớp. mAP50 dùng IoU=0.5; mAP50–95 trung bình nhiều ngưỡng IoU 0.5–0.95 nên khắt khe hơn về vị trí box. Không suy nguyên nhân Recall thấp chỉ từ imbalance hoặc tổng số ảnh. Kiểm số instance, nhãn thiếu/sai, kích thước, che khuất, domain và thời lượng học. Không dùng các ngưỡng Box/CLS/L1 cố định để phán tốt/xấu giữa các cấu hình loss khác nhau; loss đã đổi thang khi đổi weight.

**Đối chứng tiếp theo:** kiểm AP/Recall từng lớp và ảnh lỗi trước; nếu thử weighting, A=0, B=0.25, C=0.5 phải cùng checkpoint xuất phát, cùng số epoch, seed, optimizer, lịch augmentation và split. Không chạy B từ kết quả của A rồi gọi đó là đối chứng. Nếu đánh giá ảnh hưởng Mosaic, đảm bảo lịch close_mosaic thực sự cho phép nó hoạt động. Transfer learning đã có qua pretrained/R1; freeze backbone (khóa một phần trọng số) chỉ là lựa chọn thử riêng. Tune là tìm tham số qua nhiều lượt, tốn CPU, nên làm sau kiểm dữ liệu. Không tự chạy 100 epoch hoặc oversample từ việc bổ sung giao diện này.

Nguồn: [Train/cls_pw](https://docs.ultralytics.com/modes/train/), [Augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/), và mã nguồn trong môi trường Ultralytics 8.4.165 của dự án (`models/yolo/detect/train.py`, `data/dataset.py`, `engine/trainer.py`, `utils/loss.py`).

## Theo dõi và dừng

Mỗi máy chỉ chạy một lượt train của app. Trạng thái dựa trên PID, thời điểm tạo tiến trình và token của run; tải lại trang tìm lại tiến trình hiện hành. Đóng trình duyệt không dừng worker. Máy vẫn cần hoạt động để worker tiếp tục.

Giao diện hiển thị giai đoạn kiểm dữ liệu, chuẩn bị, train, validation, final validation và kiểm checkpoint; batch thực, epoch đã lưu, log, loss, metric, thời gian và RAM đã đo. Train batch 2 có thể dùng validation batch 4 theo trainer; hai số được hiển thị riêng. Chưa có CSV/metric thì giao diện chờ số đo, không tự điền kết quả. File JSON/CSV đang ghi được đọc an toàn.

**Dừng sau epoch hiện tại** gửi yêu cầu chờ validation và checkpoint của epoch hoàn tất, sau đó trainer làm final validation và kiểm artefact. Nút không ngắt cưỡng bức giữa batch. Nếu worker chết, run được đánh dấu lỗi; việc có file `.pt` không đủ để báo hoàn thành.

Webcam của phiên hiện tại được giải phóng khi chuyển sang Training. App chặn suy luận mới trong lúc train; Recognition cho biết đang tạm dừng và tự mở lại sau khi worker kết thúc.

## Checkpoint và kết quả

Mỗi run có tên chứa thời điểm và mã riêng dưới `runs/train/`; không ghi đè run cũ hoặc checkpoint khởi tạo. Lưu `request.json`, `effective_config.yaml`, `environment.json`, `console.log`, `state.json`, `args.yaml`, `results.csv`, hình loss/validation, `timings.json`, `checkpoint_validation.json` và weights.

`weights/last.pt` là epoch cuối; `best.pt` được chọn theo validation; `epoch0.pt` là **epoch 1**, `epoch1.pt` là epoch 2. App dùng `save=True, save_period=1`. Checkpoint epoch có thể tải sau khi lưu; best/last chỉ tải khi worker kết thúc vì final validation còn ghi lại hai file này. Chọn run trong lịch sử để xem và tải.

Có thể chọn checkpoint đã lưu để khởi tạo **run mới**. Phiên bản này không resume nguyên trạng optimizer. Dữ liệu và checkpoint khởi tạo được kiểm hash lại sau train; chỉ ba cache nhãn dẫn xuất được verifier bỏ qua, không bỏ qua ảnh/TXT hay metadata.

Thời gian từng giai đoạn đo theo callback; tổng gồm cả kiểm dữ liệu, chuẩn bị, train, validation, final validation và kiểm checkpoint. Đỉnh RSS của cây tiến trình được lấy mẫu; trên Windows còn ghi peak working set của worker và RAM khả dụng thấp nhất. Đây là số đo của máy/lượt chạy cụ thể.

Validation trong trainer dùng NMS-free. Bảng so sánh pretrained/fine-tuned chính thức cần evaluator chung theo [EVALUATION_PROTOCOL.md](../../research/evaluation_protocol.md). Một epoch đầu kiểm khả năng chạy và tài nguyên; chưa đủ kết luận model hoàn thiện hoặc nghiệm thu toàn bộ tuần 3. Test hiện thiếu ground truth backpack/cup.

## Kiểm thử

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

44/44 kiểm thử đạt ngày 04/10/2026 trước lượt train thật, không skip: cấu hình/form, điều phối tiến trình, tải lại, chạy đồng thời, tiến trình lỗi, checksum sai, dừng sau checkpoint, bảo toàn run/weights, file đang ghi và hồi quy webcam. Camera mô phỏng trong kiểm thử không thay nghiệm thu thiết bị thật trên từng máy thành viên.

05/10/2026: **48/48 đạt, không skip** sau khi bổ sung Nâng cao. Kiểm thêm mặc định/override, xác suất sai (`mosaic=10`), NaN/Inf, request không hợp lệ trước khi nạp model, preview không train và truyền form vào RunManager. Tham số Nâng cao cũng được kiểm lưu bền trong request và đọc lại qua RunManager mới bằng tiến trình fixture, giữ chống chạy trùng và bảo toàn run cũ.

## 7. Báo cáo tuần 3 — Kaggle 10 epoch đã hoàn tất (05/10/2026)

Báo cáo được cập nhật trực tiếp vào file hướng dẫn tuần 3 này theo yêu cầu. Run `project8_26n_kaggle_20261005_092653` đã hoàn thành 10/10 epoch, evaluator baseline/best chạy xong và artefact tải về máy đã kiểm hợp lệ. Các mục lịch sử R1/R2 phía trên vẫn giữ kết quả cũ.

**Đánh giá:** chất lượng tổng cải thiện rõ so pretrained ở cùng protocol (mAP50 +19,60 điểm %, mAP50–95 +9,72 điểm %). Model best là mốc epoch 9, chưa hội tụ được chứng minh; laptop/backpack/cup còn thấp hơn baseline, nên việc cần làm tiếp là kiểm AP và ảnh lỗi từng lớp, rồi mới quyết định ngân sách train mới. Không gọi model hoàn thiện chỉ vì đã đủ epoch; không dùng test để tối ưu.

### Điều kiện thực tế

- Nguồn R3 best: `4034b0ec824b2e9a5e5b945f4626dbbdf2bf506ac26e5575c73e240f4a52f5d3`.
- Dataset khóa `project8_v0.3-approved-20261004`: 3.883 ảnh train, 798 ảnh validation; không dùng 321 ảnh test để chọn model/tham số. Manifest nguồn vẫn khớp SHA-256 đã duyệt.
- YOLO26n, GPU Tesla T4 x2, `device=0,1`, DDP hai tiến trình; Ultralytics 8.4.165, Python 3.13.15, Torch 2.11.0+cu128.
- 10 epoch; imgsz yêu cầu 1080, trainer tự làm tròn 1088; batch tổng 4 (2/GPU); workers 2/rank, cache False, MuSGD, lr0=.001, seed 42, fraction=1, close_mosaic=5, save=True/save_period=1.
- Fine-tune mới từ R3, `resume=False`: optimizer/scheduler reset. Đây là 10 epoch của một lượt, không phải 10 lượt độc lập.
- Snapshot tài nguyên trong hai phút đầu: CPU 303%, RAM 8,3 GiB/30 GiB; GPU0 40%/1,4 GiB, GPU1 37%/1,3 GiB. Đây là số đo tại một thời điểm, không phải peak hay trung bình toàn lượt.

### So sánh chung với pretrained

Cả hai model được đánh giá trên cùng 798 ảnh validation bằng evaluator COCO bbox: imgsz=640, conf=.001, IoU=.70, NMS=True, max_det=300, augment=False. `actual_end2end=False`. Các số này tách khỏi validation NMS-free của trainer ở imgsz 1088.

| Model | mAP50 | mAP50–95 |
|---|---:|---:|
| Pretrained COCO80 | 28.18% | 18.89% |
| Kaggle best | 47.78% | 28.62% |
| Tăng tuyệt đối | +19.60 điểm % | +9.72 điểm % |

| Lớp | Box GT val | AP50 pretrained | AP50 best | Chênh điểm % | AP50–95 best |
|---|---:|---:|---:|---:|---:|
| person | 1242 | 35.98% | 79.51% | +43.53 | 48.92% |
| table | 222 | 11.92% | 61.32% | +49.40 | 35.44% |
| chair | 290 | 35.59% | 48.01% | +12.42 | 33.76% |
| laptop | 279 | 79.84% | 78.03% | -1.80 | 52.29% |
| cell phone | 579 | 2.92% | 27.11% | +24.19 | 11.56% |
| backpack | 110 | 21.39% | 14.51% | -6.88 | 5.63% |
| book | 2458 | 5.71% | 47.31% | +41.60 | 24.23% |
| cup | 202 | 32.07% | 26.45% | -5.63 | 17.12% |

Model cải thiện rõ ở person, table, book và cell phone. Laptop, backpack và cup có AP thấp hơn baseline ở protocol chung; không thể kết luận tất cả lớp đều tốt hơn. Backpack (AP50 14,51%) vẫn yếu nhất; cell phone 27,11% và cup 26,45% cần kiểm ảnh false positive/false negative, nhãn, kích thước và che khuất. Chưa có bằng chứng quy nguyên nhân riêng cho mất cân bằng.

### Lịch sử đủ 10 epoch

Bảng này lấy từ results.csv của trainer ở imgsz 1088, NMS-free. Loss và P/R là chỉ số thực tại mỗi epoch. Giây/epoch là chênh cumulative time trong CSV, gồm train/validation và overhead ghi nhận.

| Epoch | Giây | Train box | Train cls | Train L1 | Val box | Val cls | Val L1 | P | R | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 186.208 | 1.56015 | 2.97867 | 0.01703 | 1.46948 | 2.91512 | 0.02134 | 48.92% | 32.81% | 29.02% | 16.75% |
| 2 | 155.252 | 1.49784 | 2.80401 | 0.01578 | 1.42842 | 2.76931 | 0.02004 | 44.23% | 33.92% | 31.32% | 18.73% |
| 3 | 153.600 | 1.47801 | 2.62683 | 0.01563 | 1.45908 | 2.59928 | 0.0206 | 45.78% | 34.16% | 33.63% | 20.05% |
| 4 | 154.412 | 1.47083 | 2.51529 | 0.01447 | 1.43416 | 2.39934 | 0.0207 | 47.98% | 37.76% | 37.29% | 22.30% |
| 5 | 155.801 | 1.44882 | 2.37717 | 0.01507 | 1.42743 | 2.38982 | 0.02017 | 40.91% | 42.45% | 38.50% | 23.11% |
| 6 | 153.213 | 1.45636 | 2.39769 | 0.02049 | 1.4281 | 2.41261 | 0.01983 | 54.16% | 38.82% | 41.37% | 24.56% |
| 7 | 152.324 | 1.4191 | 2.16809 | 0.01953 | 1.4245 | 2.32007 | 0.01953 | 50.88% | 40.98% | 42.14% | 25.08% |
| 8 | 151.370 | 1.41495 | 2.09283 | 0.01911 | 1.40967 | 2.19138 | 0.01947 | 50.11% | 43.31% | 42.77% | 25.68% |
| 9 | 151.610 | 1.39937 | 2.04166 | 0.01937 | 1.39857 | 2.18563 | 0.01915 | 51.52% | 44.47% | 43.83% | 26.45% |
| 10 | 151.410 | 1.36952 | 1.99965 | 0.0186 | 1.38743 | 2.13498 | 0.01887 | 50.46% | 42.57% | 42.82% | 25.84% |

- CSV tổng 1.565,200 giây; trung bình 156,520 giây/epoch (2 phút 36,52 giây). Epoch đầu 186,208 giây; epoch 2–10 khoảng 151–156 giây.
- Log trainer báo 10 epoch trong 0,435 giờ (số làm tròn). Cell train thực thi 1.634,937 giây (~27 phút 15 giây), gồm khởi tạo và final validation; đây không phải tổng thời gian toàn Run All/setup/evaluator/đóng gói.
- Epoch 9 có mAP50–95 cao nhất trong CSV (26,451%); epoch 10 còn 25,836%. Đây là giảm ở một epoch, chưa đủ kết luận overfitting kéo dài.
- Final validation của best trong log trainer: P≈51,2%, R≈44,6%, mAP50≈43,8%, mAP50–95≈26,5%. Không gộp chúng với evaluator 640/NMS ở bảng baseline.
- Mosaic đóng ở epoch 6 vì close_mosaic=5. Train L1 tăng tại chuyển giai đoạn, rồi giảm dần; không dùng một ngưỡng loss tuyệt đối để kết luận lỗi khi augmentation thay đổi.

### File đã tải và xác minh

- Full ZIP: `runs/kaggle/project8_26n_kaggle_20261005_092653/project8_26n_kaggle_20261005_092653_full_run.zip`, 166.119.984 byte; SHA-256 `242a05978400103a7c47c5dab3f394b6c37eb5b1b50fb2ff0ce4be7722a7b43c`.
- Result ZIP nhỏ: `runs/kaggle/project8_26n_kaggle_20261005_092653/project8_26n_kaggle_20261005_092653_result.zip`, 4.858.177 byte; SHA-256 `3907c6386f6f60ee23583b42092e3fa86ee3e7797c16fdc9973b3f05729e5795`.
- Đã kiểm 44 file payload theo DOWNLOAD_MANIFEST, CRC archive và SHA-256 từng file; bản giải nén: `runs/kaggle/project8_26n_kaggle_20261005_092653/artifacts/`.
- Cả 12 checkpoint nạp được, đúng task detect và thứ tự person/table/chair/laptop/cell phone/backpack/book/cup. Xem `checkpoint_validation.json`.
- Model best đã nhập và đăng ký local: `weights/kaggle/project8_26n_kaggle_20261005_092653_fd9cb282f05f.pt`.
- CSV, cấu hình, baseline/fine-tuned metrics, đường cong và confusion matrix được chép vào thư mục báo cáo này. Confusion matrix/đường cong của trainer thuộc protocol trainer 1088/NMS-free.
- Log/code notebook local: `runs/colab_handoff_audit_20261005/`.

| Trọng số | Sau epoch | Optimizer | Dùng tiếp |
|---|---:|---|---|
| best.pt / bản weights/kaggle | 9 | Đã strip | Nhận dạng hoặc khởi tạo fine-tune mới |
| last.pt | 10 | Đã strip | Trọng số cuối; CSV hơi thấp hơn best |
| epoch0.pt–epoch9.pt | 1–10 | Còn | Lưu lịch sử/khôi phục; chỉ resume sau khi xác định đúng epoch và lịch train |

`best.pt` khớp toàn bộ tensor EMA trong epoch8.pt; `last.pt` khớp EMA trong epoch9.pt. Đã kiểm trực tiếp tensor local. best/last có stored_epoch=-1 do strip optimizer, không đọc -1 như số epoch đã học.

### Sự cố lưu/tải và giới hạn

Kaggle báo Failed to save draft / ConcurrencyViolation (draft sequence lệch), khiến Download qua sidebar không hoàn tất. Đã lấy các archive từ dữ liệu file xuất thành liên kết trong notebook, ghép local rồi xác minh đúng hash cloud. Chưa xác nhận sửa xong lưu draft hoặc có Kaggle version lưu bền; bản sao local đã đầy đủ. Không chạy Save & Run All để tải vì nó tạo lượt train mới.

Cảnh báo Ray Tune get_trial_id không ngắt train. Không thấy Traceback/CUDA OOM/nonfinite trong output training đã quan sát; CSV đủ 10 dòng hữu hạn. Không đo inference FPS, không gọi model hoàn thiện và không suy ra tốc độ tính toán thuần từ mọi elapsed time.

### Hướng đi tiếp theo — chưa thực thi

1. Dùng best đã nhập để thử nhận dạng ảnh thực tế; kiểm lỗi theo lớp, đặc biệt backpack/cell phone/cup và các lớp giảm so baseline.
2. Giữ snapshot đã khóa; nếu sửa nhãn/tập dữ liệu, tạo release mới có review/checksum. Không dùng test để chọn model hay tham số.
3. Nếu cần train tiếp, chọn best của run này làm nguồn fine-tune mới. Không gọi resume nguyên trạng từ best/last vì optimizer đã strip.
4. Đo tốc độ trước: giữ imgsz 1088, thử batch tổng 16 trên hai T4, workers 2/rank, cache False; đổi từng yếu tố, so giây/epoch và RAM/VRAM. Batch lớn có thể thay đổi quá trình tối ưu, phải ghi cấu hình riêng.
5. Sau kiểm lỗi và benchmark mới quyết định lượt thêm có hạn mức. Không bật cùng lúc class weighting, oversampling, MixUp và các thay đổi khác; giữ protocol đánh giá chung.
6. Khi đã kiểm/tải đủ file, người dùng có thể Stop Session để tiết kiệm quota GPU. Không bấm Run All chỉ để tải file; notebook backup đang START_TRAINING=True.

### Trạng thái đầu việc tuần 3 sau cập nhật

- [x] Dataset đã duyệt/khóa, train/val/test 3.883/798/321 giữ nguyên.
- [x] CPU R1/R2 hoàn thành 5 epoch; R3 dừng an toàn sau 1 epoch, có checkpoint và kiểm cuối.
- [x] Kaggle hoàn thành 10 epoch trên T4 x2; có cấu hình, CSV, loss/validation và checkpoint thật.
- [x] Pretrained và Kaggle best đã chấm cùng protocol COCO bbox 640/NMS trên validation; có AP từng lớp.
- [x] Đã tải/xác minh 44 file và 12 checkpoint, nhập model mới vào weights/kaggle; có bản sao local đầy đủ.
- [ ] Phân tích ảnh lỗi có review, đặc biệt backpack/cell phone/cup và các lớp giảm so baseline.
- [ ] Benchmark nhận dạng thực tế: FPS, p95 latency và RAM trên máy/app; thời gian train không thay được số đo inference.
- [ ] Nghiệm thu các phần recognition/SSL/YOLO26s theo phạm vi kế hoạch; chưa chạy SSL hoặc YOLO26s trong các lượt nêu trên.
- [ ] Khắc phục lưu draft Kaggle và xác nhận version cloud persist nếu cần giữ output trên Kaggle; bản sao local hiện đã kiểm đủ.

Chưa tự thực thi lượt tiếp theo. Tài liệu vẫn ở thư mục gốc vì còn hướng dẫn và đầu việc đang tiếp tục; không chuyển vào README_DA_HOAN_THANH chỉ vì lượt train này đã xong.

## 8. Báo cáo tuần 3 — Kaggle 14 epoch và tổng 30 epoch (06/10/2026)

Run `project8_26n_kaggle_20261006_083226` hoàn thành 14/14 epoch từ K1 best (hash nguồn `fd9cb282f05f...`), resume=False. Cấu hình thực: YOLO26n, T4 x2/device=0,1, imgsz 1280, batch tổng 16, MuSGD/lr0=.001, workers=2/rank, cache=False, seed=42, mosaic=.5, close_mosaic=5, cls_pw=0. CSV đủ 14 dòng hữu hạn, time cuối 1.573,48 giây; không dùng số này làm tổng thời gian notebook hoặc FPS inference.

| Model — evaluator chung 640/NMS, 798 ảnh val | mAP50 | mAP50–95 |
|---|---:|---:|
| Pretrained COCO80 | 28,176% | 18,895% |
| K1 best (05/10) | 47,779% | 28,619% |
| K2 best (06/10) | 53,848% | 33,847% |
| K2 − K1 | +6,068 điểm % | +5,229 điểm % |

K2 tăng AP50/AP50–95 ở cả 8 lớp so K1; backpack/cup và AP50–95 laptop vẫn thấp hơn pretrained. Protocol chung dùng conf=.001, IoU=.70, max_det=300, agnostic_nms=False, augment=False. Metric trainer cuối ở 1280/NMS-free là P=58,718%, R=54,028%, mAP50=56,546%, mAP50–95=35,617%; tách khỏi bảng đối chứng trên.

- [Bảng đầy đủ 30 epoch, loss/P/R/AP/thời gian và lineage](../completed/reports/results/training_30_epochs_20261006/report.md).
- [Báo cáo riêng K2](../completed/reports/results/kaggle/project8_26n_kaggle_20261006_083226/report.md).
- [CSV đủ 30 epoch, gồm learning rate](../../../reports/archive/results/training_30_epochs_20261006/epochs_30.csv) và [audit hash/checkpoint/dataset](../../../reports/archive/results/training_30_epochs_20261006/audit.json).
- [ZIP K2](../../../runs/kaggle/project8_26n_kaggle_20261006_083226/project8_26n_kaggle_20261006_083226_result.zip): 4.877.504 byte, SHA-256 `1bd9b9ffa191f68dcf70c7ee6391b5685549962149ef49dbab0fdc2cee9e375d`; 6 payload khớp hash/byte và CRC đạt.
- [Best K2](../../../weights/BEST_Project8.pt): 5.546.053 byte, SHA-256 `ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf`; đúng detect/8 lớp, stride 8/16/32, tensor hữu hạn. train_results/train_metrics khớp CSV tới epoch 14; chưa có last.pt/epoch13.pt để xác nhận tensor best=last độc lập. Optimizer đã strip, chỉ inference hoặc fine-tune mới resume=False.

Tổng 30 epoch là lịch sử thực thi qua 5 run. K2 lấy K1 best ở epoch 9, vì thế epoch 10 của K1 không được dùng làm nguồn K2. Không gọi 30 là lịch resume liên tục. ZIP K2 chưa chứa last/checkpoint từng epoch/full output; chỉ tải result ZIP không đủ bằng chứng đã giữ toàn bộ checkpoint. Dataset khóa/checksum giữ nguyên, test chưa dùng, SSL/FPS/ảnh FP-FN và nghiệm thu vẫn còn việc. Chưa tự chạy lượt tiếp theo.
