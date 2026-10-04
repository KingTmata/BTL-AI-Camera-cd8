# Triển khai tuần ba tiếp tục

Tài liệu thống nhất hướng dẫn Training local và phần Nâng cao trước đây, bổ sung cách học tiếp từ trọng số sau tổng năm epoch. Tài liệu này thay thế `docs/TRAINING_UI.md`; nhật ký kết quả vẫn lưu riêng.

## 1. Điểm xuất phát hiện tại

Đã hoàn tất **hai lượt train, tổng 1 + 4 = 5 epoch học**, không phải năm lượt đánh giá trên test:

- **R1:** một epoch từ YOLO26n pretrained của Ultralytics, sau đó tạo checkpoint tám lớp.
- **R2:** bốn epoch mới từ **R1 best**, giữ dataset đã khóa. Optimizer, warmup và scheduler khởi tạo lại khi mở R2.
- **Lượt tiếp theo:** chọn **R2 best** dưới đây để kế thừa kết quả đã học. Form vẫn mặc định chọn pretrained gốc; cần chủ động đổi đúng ô **Checkpoint khởi tạo** trong Training.

Checkpoint nên dùng cho lượt mới:

```text
runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/best.pt
```

[Trọng số R2 best](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/best.pt) đã được kiểm nạp đúng tám lớp: person, table, chair, laptop, cell phone, backpack, book, cup. Trong giao diện Windows, đường dẫn có thể hiện dấu gạch chéo ngược; tên thư mục run và file phải khớp như trên.

## 2. Phân biệt toàn bộ trọng số để tránh nhầm

| Nguồn | Checkpoint | Đã học dữ liệu dự án đến đâu / dùng khi nào |
|---|---|---|
| Ultralytics gốc | [weights/yolo26n.pt](weights/yolo26n.pt) | Pretrained COCO 80 lớp, chưa học dataset project8. Dùng làm điểm xuất phát/đối chứng gốc. |
| R1 | [runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt](runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt) | Tốt nhất của lượt một epoch; là nguồn khởi tạo R2. |
| R1 | [runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/last.pt](runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/last.pt) | Cuối lượt R1: sau epoch tổng 1. |
| R1 | [runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/epoch0.pt](runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/epoch0.pt) | Bản lưu riêng sau epoch R1 1 = epoch tổng 1. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/best.pt](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/best.pt) | Tốt nhất theo validation R2, đã kiểm xong. Ưu tiên cho lượt học tiếp/nhận dạng. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/last.pt](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/last.pt) | Cuối R2: sau epoch R2 4 = epoch tổng 5; không tự resume optimizer qua app. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch0.pt](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch0.pt) | Sau epoch R2 1 = epoch tổng 2. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch1.pt](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch1.pt) | Sau epoch R2 2 = epoch tổng 3. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch2.pt](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch2.pt) | Sau epoch R2 3 = epoch tổng 4. |
| R2 | [runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch3.pt](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch3.pt) | Sau epoch R2 4 = epoch tổng 5. |

**Tên file giống nhau chưa chắc cùng nguồn:** R1 epoch0 là epoch tổng 1, R2 epoch0 là epoch tổng 2. Luôn đối chiếu cả thư mục run. `best` là checkpoint được chọn theo validation; `last` là trạng thái cuối lượt, không mặc định tốt hơn best. Checkpoint riêng theo epoch giữ mốc để quay lại so sánh. File có thể khác hash/dung lượng vì trạng thái optimizer hoặc bước hoàn thiện checkpoint; không dựa riêng tên/dung lượng để suy ra chất lượng.

R2 có đủ sáu file best/last/epoch0–epoch3; tám lớp, dataset và checkpoint nguồn đã qua kiểm cuối lượt. Hash R1/pretrained gốc vẫn khớp bản ghi trước R2. [Kiểm checkpoint R2](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/checkpoint_validation.json) · [Kiểm bảo toàn nguồn](runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/tracking_verification.json).

## 3. Các bước train tiếp từ R2 best trên web local

1. Mở app và chọn **Màn hình → Training**. Nếu chưa chạy server, dùng lệnh ở phần hướng dẫn local phía dưới.
2. Trong **Checkpoint khởi tạo**, chọn đúng **R2 best** theo đường dẫn ở mục 1. Không nhầm ô Checkpoint của Recognition với ô nguồn Training; Recognition chỉ chọn model để nhận dạng.
3. Ví dụ một lượt theo dõi ngắn: chọn **3 epoch mới**, batch **2**, kích thước **640**, workers **0**, cache RAM **tắt**. Đặt tên `project8_v03_26n_cpu_r2best_e3`. Đây là ví dụ để bạn tự quyết định số epoch, chưa phải lượt đã chạy.
4. Lượt đối chứng đầu giữ Nâng cao như mặc định: Mosaic=1.0, close_mosaic=10, MixUp=0, CutMix=0, xoay=0, lật ngang=0.5, lật dọc=0, cls_pw=0, Box=7.5, CLS=0.5, L1=1.5. Với ba epoch và close_mosaic=10, các phép ghép đóng từ đầu, cùng cách vận hành của các lượt ngắn trước.
5. Bấm **Xem trước cấu hình**, đọc cảnh báo lịch Mosaic và bảng phân phối box/weight. Xác nhận đủ 3.883 train, 798 validation, đúng checkpoint và RAM khả dụng. App dùng toàn train; không dùng test để chọn model.
6. Khi đã chọn hạn mức, bấm **Bắt đầu training**. App tạo run/thư mục mới, kiểm snapshot/checksum/nguồn/cấu hình rồi train; checkpoint khởi tạo được nạp đầu lượt, không nạp lại nguồn mỗi epoch. Giữ máy hoạt động, cắm nguồn và tránh để máy sleep trong khi train.
7. Theo dõi Train/Validation, log, loss, mAP và Recall. Muốn kết thúc sớm, dùng **Dừng sau epoch hiện tại** để chờ lưu checkpoint. Đóng trang không dừng worker.
8. Khi trạng thái completed hoặc stopped đã kiểm checkpoint, xem thư mục run mới và tải best/last/epoch. Nếu failed, đọc nguyên nhân; không coi checkpoint tồn tại là bằng chứng hoàn tất.

Đây là **run mới từ trọng số đã fine-tune**, app dùng `resume=False`. Epoch trong lượt mới đếm lại từ 1; optimizer/scheduler/warmup cũng khởi tạo lại. Có thể ghi “R3 từ R2 best, thêm ba epoch, lịch sử trọng số 5 + 3” để theo dõi, nhưng không gọi là resume liên tục epoch 6–8. Khi lượt mới hoàn tất, chọn best của đúng lượt mới để dùng; không tự thay checkpoint của form hoặc ghi đè R2.

Với RAM trống khoảng 5–7 GB, cấu hình batch 2/cache tắt là điểm xuất phát đã đo. Không có bảo đảm tăng batch/cache làm nhanh hơn; benchmark từng thay đổi riêng, theo dõi RAM và thời gian thật. Trong trainer CPU hiện cài, workers được đưa về 0; tăng ô workers không bảo đảm tăng tốc CPU.

## 4. Mô hình đang học nhanh hay chậm?

Khả năng nhận dạng **cải thiện đều trong năm epoch đầu**, chưa thấy mAP đứng im trong chuỗi này. Đây là tốc độ cải thiện chất lượng, khác tốc độ tính toán của máy. Không thể gọi model đã tốt hoặc hội tụ chỉ từ năm epoch.

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

- **Theo dõi học thêm:** có thể dùng R2 best và một hạn mức ngắn đã chọn để xem AP còn tăng hay không. Kiểm dữ liệu và baseline pretrained/fine-tuned theo cùng [protocol](EVALUATION_PROTOCOL.md); test giữ cho đánh giá cuối.
- **Thử weighting:** A cls_pw=0, B=0.25, C=0.5 đều phải bắt đầu từ cùng **R2 best**, cùng số epoch, seed, optimizer, split và augmentation. Không dùng best của A làm nguồn B rồi coi đó là thử nghiệm đối chứng. Chỉ thay một biến; đánh giá P/R/AP từng lớp.
- **Thử Mosaic:** chọn lịch có epoch cho phép Mosaic, ví dụ 3 epoch/close_mosaic=0 hoặc close_mosaic=1. Đây là một thử nghiệm khác với đối chứng close_mosaic=10, chưa tự thực hiện. Nếu close_mosaic vẫn lớn hơn hoặc bằng số epoch, tăng Mosaic/MixUp không kiểm được tác dụng của chúng.
- **NaN/Inf, checksum sai, thiếu checkpoint hoặc process failed:** không mở thêm lượt chồng lên; kiểm log, cấu hình và dữ liệu. Không tiếp tục từ file chưa kiểm nạp được. Dừng sau epoch là dừng có kiểm soát; trong lỗi khẩn cấp cần kiểm đúng tiến trình trước khi can thiệp riêng.
- **AP tăng:** cân nhắc học thêm trong ngân sách thời gian đã chọn. **AP ngang hoặc giảm:** kiểm AP từng lớp, ảnh lỗi, nhãn và dấu hiệu overfit trước; không tăng hàng loạt augmentation/weight để phản ứng theo một epoch dao động.

[Sổ log và kế hoạch tuần 3](docs/WEEK3_TRAINING_LOG_AND_PLAN.md) · [Báo cáo R2](reports/results/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/REPORT.md).

## 6. Hướng dẫn Training local và Nâng cao đã hợp nhất

Cập nhật 05/10/2026. Màn hình Training dùng trainer Ultralytics trong tiến trình Python riêng; trình duyệt chỉ điều khiển và theo dõi. Hướng dẫn này áp dụng cho môi trường CPU đã cài của dự án.

## Mở và chạy

Từ thư mục gốc dự án:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501
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

Validation trong trainer dùng NMS-free. Bảng so sánh pretrained/fine-tuned chính thức cần evaluator chung theo [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). Một epoch đầu kiểm khả năng chạy và tài nguyên; chưa đủ kết luận model hoàn thiện hoặc nghiệm thu toàn bộ tuần 3. Test hiện thiếu ground truth backpack/cup.

## Kiểm thử

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

44/44 kiểm thử đạt ngày 04/10/2026 trước lượt train thật, không skip: cấu hình/form, điều phối tiến trình, tải lại, chạy đồng thời, tiến trình lỗi, checksum sai, dừng sau checkpoint, bảo toàn run/weights, file đang ghi và hồi quy webcam. Camera mô phỏng trong kiểm thử không thay nghiệm thu thiết bị thật trên từng máy thành viên.

05/10/2026: **48/48 đạt, không skip** sau khi bổ sung Nâng cao. Kiểm thêm mặc định/override, xác suất sai (`mosaic=10`), NaN/Inf, request không hợp lệ trước khi nạp model, preview không train và truyền form vào RunManager. Tham số Nâng cao cũng được kiểm lưu bền trong request và đọc lại qua RunManager mới bằng tiến trình fixture, giữ chống chạy trùng và bảo toàn run cũ.
