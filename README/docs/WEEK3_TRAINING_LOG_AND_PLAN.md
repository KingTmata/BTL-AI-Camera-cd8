> **Đối soát 05/10/2026:** xem [trạng thái hiện hành](../README.md#trạng-thái-theo-tuần). Dataset project8_v0.3 đã duyệt/khóa; R1 và R2 hoàn tất tổng 5 epoch. Những mô tả trước đó trong tài liệu là lịch sử hoặc đề xuất; G1/G2/G3 chưa đủ nghiệm thu toàn bộ.

# Tuần 3 — Sổ theo dõi huấn luyện và kế hoạch quyết định

Cập nhật **04/10/2026**, múi giờ Asia/Saigon. Tài liệu ghi số đo thật của lần 1 và đề xuất các bước tiếp theo. **Baseline chung chưa chạy. Lượt R2 đã được khởi động theo yêu cầu: học thêm 4 epoch từ best.pt của R1; xem mục 10.**

## Mục lục

- [1. Trạng thái hiện tại và đánh giá lần 1](#1-trạng-thái-hiện-tại-và-đánh-giá-lần-1)
- [2. Dữ liệu, cấu hình và kết quả đã đo](#2-dữ-liệu-cấu-hình-và-kết-quả-đã-đo)
- [3. Sổ lịch sử các lượt](#3-sổ-lịch-sử-các-lượt)
- [4. Baseline, chọn checkpoint và kiểm ảnh](#4-baseline-chọn-checkpoint-và-kiểm-ảnh)
- [5. Kế hoạch các mốc 3, 5, 10 epoch](#5-kế-hoạch-các-mốc-3-5-10-epoch)
- [6. Các tình huống và hành động](#6-các-tình-huống-và-hành-động)
- [7. Chống lỗi trước, trong và sau run](#7-chống-lỗi-trước-trong-và-sau-run)
- [8. Mẫu ghi thêm kết quả](#8-mẫu-ghi-thêm-kết-quả)
- [9. Nguồn và giới hạn nghiệm thu](#9-nguồn-và-giới-hạn-nghiệm-thu)

## 1. Trạng thái hiện tại và đánh giá lần 1

**mAP50 = 7,0884% và mAP50–95 = 4,9520% là thấp để dùng thực tế.** Recall tổng khoảng 11,35%; một số lớp có AP bằng 0. Tuy nhiên, chỉ một epoch chưa đủ xác định quá trình fine-tune thất bại, hội tụ chậm hay có lỗi dữ liệu/cấu hình. Hiện chưa có baseline pretrained trên cùng validation bằng evaluator chung, nên **chưa kết luận fine-tuned hơn hoặc kém pretrained**. Chưa có bằng chứng để quy nguyên nhân cho một yếu tố cụ thể.

Lần 1 đã kiểm được khả năng chạy: đi hết train, loss ghi trong CSV hữu hạn, validation và final validation hoàn tất, ba checkpoint nạp được với đúng tám lớp, dataset và checkpoint khởi tạo không đổi. Không xảy ra OOM hoặc worker crash trong lượt này. Kiểm kỹ thuật đạt không đồng nghĩa model đã đạt chất lượng sử dụng.

**Backend đã tắt theo yêu cầu:** worker lần 1 đã kết thúc trước khi tắt web; không còn run train active. Đã đối chiếu command line, thư mục làm việc và thời điểm tạo tiến trình rồi dừng đúng hai tiến trình Streamlit của dự án, PID 17856 và wrapper PID 28688. Kiểm sau dừng: không còn backend của dự án, kết nối TCP đến `127.0.0.1:8501` thất bại vì server đã đóng; hash cả ba checkpoint vẫn khớp. Không dừng hàng loạt Python, không archive chat. Các PID này là lịch sử, không dùng chúng để dừng một lần chạy khác.

Bằng chứng lần 1: [báo cáo](../../README_DA_HOAN_THANH/reports/results/project8_v03_26n_cpu_20261004_181257_1eb3de94/REPORT.md), [state](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/state.json), [CSV](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/results.csv), [timings](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/timings.json), [config thực](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/args.yaml), [kiểm checkpoint](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/checkpoint_validation.json).

## 2. Dữ liệu, cấu hình và kết quả đã đo

### Dữ liệu và hash cố định

Dataset duy nhất `data/dataset/project8_v0.3`, release `project8_v0.3-approved-20261004`: **5.002 ảnh, 26.566 hộp**, train/val/test = **3.883/798/321**, phát triển = 4.681 ảnh. Người dùng đã xác nhận review toàn bộ ảnh/hộp đủ tám lớp, nguồn/quyền và gần trùng/nhóm phiên; không hỏi lại các xác nhận này. Một sai nhãn mới nếu phát hiện phải có ảnh/bằng chứng cụ thể, không tự coi toàn bộ review cũ là chưa làm.

| ID | Lớp | Hộp train | Hộp val | Hộp test |
|---:|---|---:|---:|---:|
| 0 | person | 6.708 | 1.242 | 219 |
| 1 | table | 2.162 | 222 | 95 |
| 2 | chair | 1.841 | 290 | 42 |
| 3 | laptop | 1.104 | 279 | 142 |
| 4 | cell phone | 1.191 | 579 | 242 |
| 5 | backpack | 415 | 110 | 0 |
| 6 | book | 5.467 | 2.458 | 726 |
| 7 | cup | 830 | 202 | 0 |

Nguồn: [environment.json](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/environment.json). Có chênh lệch số hộp giữa các lớp; đây là số liệu để kiểm giả thuyết mất cân bằng, chưa chứng minh nguyên nhân AP thấp. Dataset hiện không có ảnh âm tính theo manifest. Test thiếu GT backpack/cup, AP hai lớp này trên test phải là **N/A/null**, không gán 0 như lớp đã được đánh giá. Val có GT cả tám lớp.

- SHA-256 release/checksums: `e7a891899268d614fbb58eafb093fe6b37a0ec9367c78baf60a392cd31dc2b0a`.
- SHA-256 manifest: `bdcc6d2f5cee1ce7e973a453cc0f53e533829a94fbfdfd18a618077c8bd8fc64`.
- SHA-256 pretrained `weights/yolo26n.pt`: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

### Cấu hình lần 1

YOLO26n pretrained COCO80 → project8; CPU; `epochs=1`, `fraction=1.0`, `imgsz=640`, `batch=2`, `workers=0`, `cache=False`, `optimizer=MuSGD`, `lr0=0.001`, `lrf=0.01`, `patience=10`, `seed=42`, `deterministic=True`, `cls_remap=True`, `nbs=64`, `save=True`, `save_period=1`, `resume=False`, `nms=False`. Torch dùng 8 luồng CPU. Train có 1.942 batch; val/final val mỗi lần có 200 batch, batch thực 4 theo trainer.

Các tham số kế thừa cần lưu khi so sánh: `warmup_epochs=3.0`, `warmup_bias_lr=0.1`, `warmup_momentum=0.8`, `close_mosaic=10`, `mosaic=1.0`, trọng số loss `box=7.5`, `cls=0.5`, `dfl=1.5`. `amp=True` trong args không chứng minh CPU chạy mixed precision; trainer kiểm khả năng AMP theo device. Không tự thay optimizer/LR hoặc tăng batch từ các số loss này.

**Tên loss thật là `box_loss`, `cls_loss`, `l1_loss`.** CSV lần này ghi L1; không đổi tên cột `dfl_loss` của model/phiên bản khác thành L1 để ghép bảng. YOLO26 DFL-free dùng L1 khoảng cách hộp khi `reg_max=1`; key cấu hình `dfl` vẫn là trọng số của thành phần này. Loss đã chịu trọng số/chuẩn hóa; L1 = 0,01751 không có nghĩa sai số tọa độ bằng 1,751% ảnh. Đã đối chiếu `BboxLoss.forward` trong phiên bản cài 8.4.165 và [tham số loss chính thức](https://docs.ultralytics.com/modes/train/).

### Loss và metric tổng

| Nguồn số đo | Box | CLS | L1 | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train CSV epoch 1 | 1,57004 | 5,70516 | 0,01751 | — | — | — | — |
| Val CSV epoch 1 | 1,45456 | 5,08331 | 0,01417 | 19,975% | 11,303% | 7,084% | 4,946% |
| Final validation trong state | — | — | — | 19,9791% | 11,3472% | 7,0884% | 4,9520% |

Giữ riêng CSV của epoch và final validation, không chép số cuối đè lên lịch sử epoch. Hai lần đo ở đây đều từ trainer **NMS-free**, chưa phải kết quả evaluator chung của dự án. P/R là số tổng hợp của validator, không phải “accuracy” hay tỷ lệ học sinh được đếm đúng. Đọc AP cùng P/R và ảnh lỗi; [định nghĩa metric](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

### Metric từng lớp — final validation

Số dưới được in trong [console.log](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/console.log), đã làm tròn. “Ảnh chứa lớp” không cộng lại thành 798 vì một ảnh có nhiều lớp.

| Lớp | Ảnh chứa lớp | GT hộp | P (%) | R (%) | AP50 (%) | AP50–95 (%) |
|---|---:|---:|---:|---:|---:|---:|
| person | 229 | 1.242 | 26,5 | 11,5 | 17,9 | 10,6 |
| table | 109 | 222 | 2,6 | 2,25 | 0,879 | 0,420 |
| chair | 151 | 290 | 5,24 | 29,0 | 9,44 | 8,14 |
| laptop | 179 | 279 | 18,2 | 48,0 | 28,4 | 20,4 |
| cell phone | 164 | 579 | 0 | 0 | 0 | 0 |
| backpack | 74 | 110 | 100 | 0 | 0 | 0 |
| book | 268 | 2.458 | 7,21 | 0,0407 | 0,120 | 0,0864 |
| cup | 80 | 202 | 0 | 0 | 0 | 0 |

P = 100% ở dòng backpack đi cùng R/AP = 0 không chứng minh lớp này tốt; cần xem dự đoán và TP/FP/FN. AP thấp/0 ở một epoch là tín hiệu cần theo dõi, chưa tự kết luận nhãn sai hoặc model không thể học lớp đó.

### Thời gian, bộ nhớ và weights

Lần 1 bắt đầu **18:12:58**, kết thúc **18:35:04 ngày 04/10/2026**, Asia/Saigon.

| Giai đoạn | Giây thực đo |
|---|---:|
| Preflight | 80,578 |
| Setup | 18,578 |
| Train | 1.088,297 — khoảng 18 phút 8 giây |
| Validation | 65,125 |
| Lưu checkpoint | 1,250 |
| Final validation | 39,547 |
| Hoàn thiện trainer | 1,843 |
| Kiểm checkpoint/hash | 30,719 |
| Tổng worker | **1.325,937 — khoảng 22 phút 6 giây** |

CSV `time=1153.5` giây là thời gian tích lũy train + val, không gồm toàn bộ preflight/final val/kiểm artefact. Riêng train đạt khoảng 3,57 ảnh/giây. RAM máy 13,79 GiB; khả dụng trước train 6,86 GiB; thấp nhất lấy mẫu 6,145 GiB. Đỉnh RSS cây tiến trình lấy mẫu **0,961 GiB**; peak working set worker do Windows ghi **1,098 GiB**. RSS lấy mẫu có thể bỏ lỡ đỉnh ngắn, không phải toàn bộ RAM hệ thống. Số đo này hỗ trợ giữ batch 2/cache tắt khi máy còn khoảng 5–7 GB RAM trống; vẫn kiểm RAM thực trước run, chưa coi đây là bảo đảm cho cả khoảng RAM hoặc mọi cấu hình. Chưa suy ra batch 4 nhanh hơn.

| File | Vai trò | Byte | SHA-256 |
|---|---|---:|---|
| [best.pt](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt) | Tốt nhất theo validation trainer; ứng viên đánh giá | 5.362.949 | `c36a5ebe74f48bb116415168c42e0c52de3c8f813702eba030e36fbdfa87c53b` |
| [last.pt](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/last.pt) | Cuối run | 5.362.949 | `1b80d8e02380215271a2f3606f66c6b9840f37da5bdedb474cf3e36f42eba495` |
| [epoch0.pt](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/epoch0.pt) | Sau epoch 1, giữ riêng | 15.524.899 | `616999481a1922fe9a64811dd1272ea7519e3eb35bd38e6f438fd505f440fcbf` |

Cả ba nạp được và có đúng `person, table, chair, laptop, cell phone, backpack, book, cup`. Lần 1 đã qua 44/44 kiểm thử trước khi train; [bằng chứng UI/test](../../runs/week3_ui/verification.json) không thay kiểm webcam thật trên từng máy.

## 3. Sổ lịch sử các lượt

**Một run/lượt huấn luyện có thể chứa nhiều epoch.** Đánh giá baseline không cập nhật weights và không phải một lượt train. Mặc định đề xuất: hai lượt đánh giá B0/B1 để tạo đối chứng, một run chính R2, sau đó **0–2 run kiểm giả thuyết** nếu có bằng chứng; không chạy lặp nhiều run một epoch chỉ để có thêm dòng bảng.

| Mã | Trạng thái | Model/nguồn checkpoint | Run ID hoặc output | Epoch trong run; quan hệ | Kết quả/quyết định |
|---|---|---|---|---|---|
| R1 | **Đã chạy, hoàn tất** | YOLO26n từ pretrained gốc | `project8_v03_26n_cpu_20261004_181257_1eb3de94` | 1/1; run độc lập | Final NMS-free 7,0884%/4,9520%; giữ làm kiểm kỹ thuật, chưa kết luận cải thiện |
| B0 | Đề xuất, chưa chạy | Pretrained COCO80 gốc | Dự kiến `reports/results/week3_pretrained_val_nms_run01` | 0 epoch; chỉ đánh giá 798 val | Tạo baseline bằng evaluator chung |
| B1 | Đề xuất, chưa chạy | R1 `best.pt`, project8 | Dự kiến `reports/results/week3_r1_best_val_nms_run01` | 0 epoch; chỉ đánh giá cùng val | So sánh R1 với B0 đúng head/protocol |
| R2 | Đề xuất, chưa chạy | **Pretrained gốc**, không lấy R1 làm mặc định | Slug dự kiến `project8_v03_26n_cpu_main_e10`; ID thật do app cấp | 10 epoch mới, xem ở 3/5/10; lịch liên tục trong run | Dừng nếu có bằng chứng lỗi; quyết định bằng đường cong và từng lớp |
| R3–R4 | Chỉ xét khi cần, chưa chạy | Pretrained gốc để kiểm đối chứng; nếu dùng checkpoint khác phải ghi parent/hash | Chưa cấp ID | Tối đa 2 run, mỗi run không quá 10 epoch trong vòng thăm dò | Một giả thuyết, một thay đổi cấu hình hoặc một phiên bản dữ liệu mỗi run |
| Giai đoạn dài hơn | Chưa quyết định | Checkpoint được chọn sau phân tích | Chưa cấp ID | Chốt ngân sách/lịch riêng sau mốc 10 | Không tự tăng epoch hoặc chạy 26s/SSL |

ID/output dự kiến không phải artefact đã tồn tại. Mọi run cùng phiên bản dữ liệu phải ghi lại hai hash release/manifest ở mục 2; nếu đổi dữ liệu, so sánh đó thuộc một phiên bản khác và cần chấm lại các đối chứng liên quan. Không ghi đè R1 hoặc output đánh giá cũ.

## 4. Baseline, chọn checkpoint và kiểm ảnh

### Đánh giá trước khi so sánh

Thực hiện B0 rồi B1 bằng evaluator hiện có, trên **cùng 798 ảnh val** theo [EVALUATION_PROTOCOL.md](../EVALUATION_PROTOCOL.md): `imgsz=640`, `conf=0.001`, `nms=True`, `iou=0.70`, `max_det=300`, `agnostic_nms=False`, `augment=False`; COCO bbox IoU 0.50:0.05:0.95, maxDets `[1,10,100]`. Mỗi output dùng thư mục mới. Lệnh sau là kế hoạch, **chưa thực thi**:

```powershell
& .\.venv\Scripts\python.exe -m src.evaluation --manifest data/dataset/project8_v0.3/manifest.csv --weights weights/yolo26n.pt --split val --device cpu --output reports/results/week3_pretrained_val_nms_run01
& .\.venv\Scripts\python.exe -m src.evaluation --manifest data/dataset/project8_v0.3/manifest.csv --weights runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt --split val --device cpu --output reports/results/week3_r1_best_val_nms_run01
```

Pretrained COCO80 phải map `0→0, 60→1, 56→2, 63→3, 67→4, 24→5, 73→6, 41→7`; project8 dùng identity. Không gọi trực tiếp pretrained 80 lớp trên YAML tám lớp rồi ghép hai bảng không cùng mapping. COCO `dining table` là đối chứng gần đúng của `table`; ghi khác biệt phạm vi và khả năng overlap ảnh nguồn pretrained khi diễn giải.

**Tách hai bảng:** bảng theo dõi epoch của trainer dùng `nms=False`; bảng so sánh chính dùng evaluator chung `nms=True`. Không lấy mAP NMS-free của R1 so với baseline NMS. YOLO26 có hai head; lựa chọn head ảnh hưởng prediction và validation. Nếu cần so demo NMS-free, chấm thêm cả pretrained và ứng viên bằng cùng `--no-nms` trong một bảng phụ, chỉ khi có nhu cầu và ngân sách. [Tài liệu chọn head](https://docs.ultralytics.com/guides/end2end-detection/).

Evaluator chung hiện xuất AP tổng/từng lớp, số GT, ảnh âm tính và số dự đoán; **chưa xuất một cặp P/R tại ngưỡng triển khai**. P/R từ trainer cần ghi rõ nguồn và head. Nếu chấm P/R tại confidence cố định, phải ghi quy tắc ghép hộp/IoU/ngưỡng và đo cho cả hai model; chưa coi công cụ đó đã triển khai. Confidence 0,25 của demo không thay confidence 0,001 dùng tính AP.

### Chọn weights và kiểm visual trên val

1. Xem `results.csv`, [results.png](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/results.png), [confusion matrix](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/confusion_matrix.png), PR/F1 curves. Mỗi lớp ghi GT, AP50/AP50–95, P/R nếu có cùng quy tắc; ưu tiên lớp AP 0/thấp và lỗi person.
2. Kiểm ảnh train có nhãn sau transform, ví dụ [train_batch0.jpg](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/train_batch0.jpg). Kiểm val [labels](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/val_batch0_labels.jpg) với [prediction](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/val_batch0_pred.jpg); không coi ba ảnh batch là review đủ validation.
3. Lập danh sách cố định tối đa 40 ảnh val để kiểm lặp: 3 ảnh chứa mỗi lớp (ưu tiên cell phone/backpack/book/cup/table), bổ sung ca vật nhỏ, che khuất, nhiều người, bàn/ghế sát nhau và cảnh khác nguồn; bỏ ảnh trùng trong danh sách. Chọn ảnh FN/FP/sai lớp khi đã có kết quả, không chỉ ảnh đẹp. Ghi image_id, GT, checkpoint/hash, confidence và lỗi quan sát được.
4. Nếu có lỗi nhãn cụ thể, ghi issue và review ảnh đó; muốn sửa nhãn phải tạo quyết định phiên bản release mới, không sửa TXT/checksum giữa run. Không tự thêm giả nhãn hoặc chỉ gán lớp của người phụ trách mà bỏ các vật thuộc lớp khác.
5. `best.pt` là ứng viên mặc định do trainer chọn theo validation của head đã cấu hình. Khi run nhiều epoch, cân nhắc thêm `epoch2.pt` (epoch 3), `epoch4.pt` (epoch 5), `epoch9.pt` (epoch 10) trên evaluator chung; checkpoint tốt nhất ở head NMS-free chưa chắc tốt nhất ở NMS. Chỉ đánh giá nặng khi train đã dừng/kết thúc để tránh tranh CPU/RAM.

Giữ test ngoài mọi tuning/chọn model/threshold. Chỉ đánh giá test sau khi khóa lựa chọn cuối; lớp không có GT trả N/A/null. Mở lại test nhiều lần để chọn tham số sẽ làm mất vai trò kiểm độc lập.

## 5. Kế hoạch các mốc 3, 5, 10 epoch

### Mặc định thực tế trên CPU

R1 là lượt kiểm kỹ thuật. Sau B0/B1 và kiểm ảnh, đề xuất **một run R2 với tổng 10 epoch từ pretrained gốc**, giữ cấu hình CPU batch 2, workers 0, cache False, ảnh 640 và optimizer/seed hiện có. Đổi `epochs=10` trong UI; chưa thực hiện thao tác này. Theo dõi CSV/checkpoint ở mốc 3, 5, 10, không dừng và khởi tạo lại chỉ để chụp mỗi mốc.

Dự trù: 10 × (1.088,297 + 65,125) giây ≈ **3 giờ 12 phút** cho train/val nếu tốc độ như R1; cộng setup/final val/kiểm artefact và biến động máy, dự kiến **3–4 giờ**. Đây là ước lượng từ **một epoch đã đo**, không cam kết. Lấy tốc độ thực của những epoch đầu R2 để cập nhật; xem RAM, nhiệt/giảm xung, tác vụ khác và thời gian validation. Thêm RAM không bảo đảm CPU train nhanh hơn.

Lịch có ảnh hưởng thực: trong source 8.4.165, warmup thực = `min(warmup_epochs, max(total_epochs - 1, 0))`. R1 một epoch có warmup thực 0; R2 mười epoch có warmup thực 3. Scheduler phụ thuộc tổng epoch. `close_mosaic=10` khiến cả run 1 và run 10 tắt mosaic từ đầu, dù args còn `mosaic=1.0`. Ghi điều này vào hồ sơ; không gọi run 10 là cùng điều kiện động học với mười run một epoch. Giữ nguyên mặc định này ở R2 để hạn chế thay đổi; nếu kiểm mosaic/optimizer sau đó, dùng thí nghiệm riêng có một thay đổi được ghi nhận. Không quy các cơ chế này thành nguyên nhân AP thấp khi chưa có đối chứng.

Mốc 2–3 chủ yếu phát hiện lỗi và xu hướng sớm; mốc 5–10 cung cấp thêm điểm để xem đường cong, **không phải số epoch bảo đảm hội tụ hoặc đủ nghiệm thu**. Tiêu chí dưới là heuristic thăm dò ban đầu của dự án, cần hiệu chỉnh theo biến động thực và số GT từng lớp.

| Mốc | Việc xem | GO — tiếp tục trong giới hạn | HOLD — dừng sau epoch để kiểm | STOP — không tiếp tục lượt lỗi |
|---|---|---|---|---|
| Trước R2 | B0/B1, mapping, prediction, checksum, tài nguyên | Đối chứng đo đúng; artefact/hộp/lớp hợp lệ; biết điểm yếu cần theo dõi | Chưa có đối chứng hoặc nghi nhãn/khác protocol | Hash/mapping hỏng, không nạp weights, không đủ tài nguyên |
| Epoch 2–3 | Loss từng epoch/batch, LR/warmup, val AP/P/R từng lớp | Hữu hạn, không lỗi hệ thống, prediction hợp lệ; có dấu hiệu học hoặc chưa đủ điểm sau warmup | Mất lớp, hộp lệch hệ thống, khác class ID, log có bất thường | NaN/Inf, diverge rõ, OOM hoặc crash chưa xử lý |
| Epoch 5 | Cửa sổ khoảng 3 epoch, train+val, lớp AP 0, ảnh lỗi | Có tiến bộ; hoặc ngang nhưng kiểm ảnh/config sạch, cho thêm 2–3 epoch trong R2 rồi xem lại | AP thấp/ngang kèm lỗi dữ liệu/config, hoặc val xấu đi khi train tốt lên | Lỗi kỹ thuật tái diễn hoặc mất tính toàn vẹn |
| Epoch 10 | Đường cong toàn run; chấm ứng viên cùng protocol, visual/per-class | Dừng theo hạn mức 10; nếu còn tăng rõ, đề xuất ngân sách giai đoạn dài hơn | Nếu ngang/thấp: ưu tiên phân tích lỗi, không tự cộng epoch | Diverge/lỗi chưa sửa: đóng thí nghiệm, giữ bằng chứng và checkpoint tốt |

“GO tại 10” là đồng ý chuẩn bị đề xuất bước sau, không tự khởi chạy thêm. `patience=10` không bảo đảm sớm dừng ở mốc 3/5 và không thay quyết định của người kiểm. Nếu chọn HOLD và bấm dừng, run kết thúc; UI v1 không tự giữ lịch optimizer để tiếp tục run đó.

### Bao nhiêu lượt là đủ để quyết định?

Trong vòng đầu: **B0 + B1 (hai lần đánh giá), R2 (một run train 10 epoch)** là mức đề xuất để có đối chứng và đường cong; R1 đã tồn tại. Chỉ thêm **0–2 run R3/R4, tối đa 10 epoch mỗi run**, khi có giả thuyết rõ. Không chạy search hàng chục cấu hình, không chạy đồng thời, không tự chuyển sang 26s/SSL. Sau hạn mức này, tổng hợp bằng chứng và quyết định có cần dữ liệu/tài nguyên hoặc giai đoạn dài hơn; số lượt không bảo đảm model đạt.

Để kiểm một yếu tố, giữ checkpoint khởi tạo gốc, seed, dữ liệu/hash, protocol và tổng lịch giống R2, thay đúng yếu tố đang thử. Nếu dùng R2 best để học thêm, đó là thí nghiệm khởi tạo khác; phải ghi parent/hash và reset optimizer/scheduler. Không so trực tiếp loss giữa hai cấu hình khác trọng số loss, head/model, dữ liệu, kích thước ảnh, batch/chuẩn hóa hoặc augmentation như cùng một thước đo.

## 6. Các tình huống và hành động

### Cách dùng bảng ngưỡng người dùng đề xuất

Bảng Box/CLS/L1 do người dùng đưa là **gợi ý quan sát sau 2–3 epoch**, không phải chuẩn chính thức hay tiêu chuẩn vạn năng. Loss tuyệt đối phụ thuộc model, cách tính/weighting, mật độ hộp, dữ liệu và lịch train. “Giảm dưới 5%” cũng không đủ kết luận học rất chậm nếu AP/per-class còn cải thiện. Cần đọc **cả train lẫn val**, prediction và số GT.

| Nhãn gợi ý của người dùng | Khoảng loss được đề xuất | Cách sử dụng trong hồ sơ này |
|---|---|---|
| Rất tốt | Box ≤1,3; CLS ≤4; L1 ≤0,017; mAP tăng rõ | Có thể ghi “loss giảm mạnh”, chỉ GO khi val/AP và ảnh cũng tốt lên |
| Tốt | Box 1,3–1,5; CLS 4–5; L1 0,017–0,020; mAP tăng | Tham khảo trong cùng cấu hình; không thay đánh giá từng lớp |
| Đứng im | Box 1,5–1,7; CLS 5,3–6,1; L1 khoảng 0,020; mAP ngang | Cần chuỗi epoch để xác nhận ngang; không gán R1 vào nhóm này từ một điểm |
| Học rất chậm | Loss giảm <5%, mAP tăng ít | Kiểm cửa sổ/trend, warmup, GT/lớp và lỗi visual trước khi đổi tham số |
| Xấu | Box >1,8; CLS >6,5; L1 >0,025; mAP giảm | Cảnh báo để kiểm log/ảnh/config; chưa chứng minh lỗi chỉ từ ba số |
| Rất xấu | Loss tăng liên tục, mAP giảm mạnh | HOLD/STOP nếu xu hướng xác nhận qua nhiều điểm và không do khác protocol |
| Lỗi | NaN/Inf | STOP và debug; không dùng checkpoint chỉ vì nạp được |

Lần 1 Box 1,57004; CLS 5,70516; L1 0,01751 chưa đồng thời khớp một nhãn trong bảng và **không có điểm trước để đo đứng im**. Không gọi “rất tốt/tốt/xấu” chỉ bằng giao các khoảng này.

### Heuristic xu hướng và đơn vị

Ưu tiên mAP50–95, kèm mAP50 và AP/recall từng lớp. Trong **cùng chuỗi run, data/head/protocol**, có thể dùng cửa sổ 3 epoch: tăng ≥1 **điểm phần trăm** mAP50–95 là tín hiệu để xem xét tiến bộ; dao động <1 điểm là vùng chưa rõ; giảm ≥2 điểm qua nhiều điểm cùng train loss giảm là cảnh báo cần kiểm overfit. Các mức 1/2 điểm là heuristic ban đầu, **không phải khoảng tin cậy hay ngưỡng thống kê đã đo**; lớp ít GT có thể dao động lớn. Không tự dừng chỉ vì một epoch xuống.

Ví dụ mAP từ 7,09% lên 9,09% là **+2 điểm phần trăm**, khoảng **+28,2% tương đối**. Hai cách báo này khác nhau. Loss giảm tương đối = `(loss_trước - loss_sau) / loss_trước × 100%`, chỉ tính khi mẫu trước dương, hữu hạn và cùng cấu hình. Không so baseline của head NMS với trend NMS-free bằng các ngưỡng này.

| Tình huống | Bằng chứng cần xem | Hành động cụ thể | Khi nào cho thêm 2–3 epoch? |
|---|---|---|---|
| Tiến bộ rõ | Val AP tăng qua nhiều điểm, nhiều lớp cải thiện, train/val loss hữu hạn; FN/FP giảm trên danh sách cố định | Giữ cấu hình, lưu checkpoint, theo dõi đến mốc tiếp theo | Trong hạn mức R2; sau 10 chỉ đề xuất giai đoạn mới |
| Học chậm | Loss giảm ít nhưng AP/recall vài lớp tăng; không có lỗi ảnh/hộp/class ID | Kiểm warmup/LR và số GT; không tăng LR/model ngay | Có dấu hiệu học, không overfit/lỗi, còn hạn mức; xem lại sau 2–3 |
| Ngang | AP gần ngang khoảng 3 epoch; train/val ổn định, prediction vẫn yếu | Kiểm FN/FP/từng lớp, nguồn/kích thước vật, bbox, nhãn đầy đủ, config thực | Nếu chưa đủ sau warmup và kiểm sạch, thử thêm một cửa sổ trong R2; ngang tiếp thì HOLD, không cộng vô hạn |
| Nghi overfit | Train loss tiếp tục giảm, val loss tăng hoặc AP giảm qua nhiều điểm | Dừng sau epoch, giữ best trước suy giảm; kiểm nguồn/split/ảnh lỗi rồi mới đề xuất một thay đổi dữ liệu/augmentation | Không dùng “thêm epoch” làm phản ứng mặc định |
| Diverge | Train và val loss tăng kéo dài, AP/P/R sụt; có thể kèm gradient/LR bất thường | Dừng, lưu args/log/LR và checkpoint tốt gần nhất; kiểm dữ liệu trước, chỉ thử giảm LR hoặc đổi optimizer ở run riêng có lý do | Sau khi sửa và kiểm lỗi; không thêm vào run chưa rõ nguyên nhân |
| NaN/Inf | Raw console/CSV/tensor loss hoặc metric không hữu hạn; UI thiếu số không đủ chứng minh hữu hạn | Dừng worker ngay có xác minh danh tính; giữ log/batch lỗi/checkpoint trước lỗi; kiểm bbox/dữ liệu/LR/precision | Không; chỉ chạy thí nghiệm mới sau debug |
| OOM/RAM hoặc đĩa đầy | MemoryError/OS kill/lỗi ghi file; RAM/disk/swap bất thường | Dừng đúng tiến trình, giữ artefact đã ghi; đóng tác vụ không cần; giữ cache tắt, thử batch 1 trước. Nếu cần giảm imgsz, đó là thay đổi tiếp theo riêng | Chỉ sau kiểm tài nguyên và preflight; checkpoint dở không dùng |
| Worker crash/treo/mất điện | PID hết sống, log dừng, state không terminal; hoặc không có batch mới kéo dài | Đối chiếu PID+ctime+cmdline/cwd, CPU/I/O, log; không lấy file `.pt` làm bằng chứng run hoàn tất. Kiểm checkpoint/hash rồi quyết định phục hồi | Không tiếp tục mù; xác định cách resume/new run và phần epoch mất |
| Sai mapping/nhãn/vật thiếu | Hộp vẽ lệch, lớp đảo, ảnh có vật nhưng thiếu GT, split/hash đổi | HOLD, ghi lỗi cụ thể, review và lập release mới nếu phải sửa; chấm lại đối chứng | Không trước khi sửa/xác nhận ca lỗi |

Sau khi tìm thấy vấn đề, mỗi thí nghiệm chỉ đổi **một yếu tố chính**: LR hoặc optimizer hoặc batch hoặc imgsz hoặc chính sách dữ liệu. Với đổi dataset, ghi release/hash mới; không tự xóa ảnh khó hay lớp yếu để làm metric đẹp.

## 7. Chống lỗi trước, trong và sau run

### Trước khi chạy

- [ ] Chốt run ID mới/nguồn weights, tổng epoch và điểm GO/HOLD/STOP; đủ khoảng thời gian theo dõi, không tự khởi chạy từ tài liệu này.
- [ ] Verify snapshot/checksum, manifest, YAML cùng tám ID/tên; chỉ bỏ qua ba cache nhãn dẫn xuất được verifier cho phép. Không sửa checksum để vượt kiểm.
- [ ] Kiểm kỹ thuật ảnh/TXT: đọc được, mọi giá trị hữu hạn, class 0–7, tọa độ YOLO hợp lệ, box có diện tích, không label mồ côi. TXT rỗng chỉ là âm tính khi ảnh được xác nhận không có tám lớp; không tự biến missing label thành negative.
- [ ] Review người dùng đã được ghi nhận; kiểm lại train_batch/ca lỗi để tìm bằng chứng mới. Kiểm nhóm nguồn/phiên và split khi đổi release, không tách crop/frame cùng phiên sang split khác.
- [ ] Checkpoint tồn tại/hash khớp/task detect; pretrained phải đúng profile COCO80 và mapping; checkpoint fine-tuned phải đúng project8. Không ép pretrained có tám tên trước khi train hoặc dùng nhầm checkpoint pose/classify.
- [ ] Ghi phiên bản môi trường và args thực; CPU batch 2, workers 0, cache False, full train; theo dõi RAM khả dụng thực thay vì suy từ tổng RAM.
- [ ] Kiểm dung lượng còn trống. R1 có epoch file khoảng 14,8 MiB; 10 epoch file tương tự khoảng 148 MiB, chưa gồm best/last/ảnh/log. Đề xuất để dư ít nhất 1 GiB cho vòng này và ghi số thực; đây là dự phòng vận hành, không yêu cầu cố định của YOLO.
- [ ] Không có worker/app run khác; dừng webcam/inference; không train và chạy evaluator nặng đồng thời. Máy cắm nguồn, không sleep/tắt máy giữa run; việc thay cài đặt hệ thống nếu cần là thao tác riêng.

### Trong khi chạy

- [ ] Theo dõi phase thực, batch/epoch, raw console, CSV, train+val loss và AP; không chỉ xem thời gian ước tính hoặc trường metrics đã lọc.
- [ ] Mỗi epoch hoàn thành giữ `epochN.pt`, best/last, CSV; `epoch0.pt` là epoch 1. Dừng chủ động dùng nút **Dừng sau epoch hiện tại** để chờ validation/checkpoint; có thể phải chờ hết phần còn lại của epoch.
- [ ] Kiểm mốc 3/5 theo bảng, ghi quyết định và lý do; giữ log cả khi trainer có retry/recovery. Chưa có số đo thì ghi “chưa đo”, không điền 0.
- [ ] Nếu train không tiến bộ batch trong thời gian dài hơn nhiều so với batch bình thường, xem CPU/I/O và log trước khi kết luận treo; preflight/validation/ghi checkpoint có khoảng chờ riêng. Chưa triển khai watchdog tự dừng.
- [ ] **NaN/Inf cần dừng ngay:** UI v1 chỉ hỗ trợ dừng sau epoch, chưa có fail-fast tại từng batch. `numbers()` của monitor bỏ giá trị không hữu hạn; số không xuất hiện trên UI không phải bằng chứng run sạch. Trainer 8.4.165 có recovery NaN/Inf, có thể retry hoặc tiếp tục ở epoch đầu, và có xử lý EMA khi lưu. Không coi đây là bảo đảm khỏi lỗi. Cần giám sát raw log/CSV; khi gặp NaN/Inf, dừng đúng worker đã xác minh bằng công cụ hệ thống, không kill tất cả Python. Cơ chế fail-fast tự động là hạng mục cần yêu cầu triển khai riêng trước vận hành dài không người theo dõi.

### Sau run và khi bị ngắt

- [ ] PID đã kết thúc, state completed/stopped/failed được đối chiếu với log; có đủ epoch CSV/checkpoint, tất cả số cần dùng hữu hạn. Checkpoint nạp được hoặc state completed riêng lẻ chưa đủ chứng minh chất lượng và không có NaN.
- [ ] Nạp checkpoint, xác nhận tám lớp/task detect; kiểm tensor trọng số hữu hạn nếu có dấu hiệu NaN/Inf. Kiểm SHA-256 dataset/weights khởi tạo trước/sau; file `.pt` đang ghi dở không dùng.
- [ ] Ghi timing train/val/final val/tổng và peak RSS/working set/RAM khả dụng; chấm ứng viên cùng protocol, lưu P/R/per-class/ảnh lỗi và quyết định.
- [ ] Khi mất điện/crash giữa epoch, chỉ giữ kết quả epoch đã lưu hợp lệ; phần batch đang làm không phải một epoch hoàn tất. Sao lưu log/args/state và kiểm checkpoint trước phục hồi; không ghi đè run cũ.

**Resume thật và run mới khác nhau.** Đã đọc checkpoint R1: best/last có `epoch=-1`, `optimizer=None` sau `final_eval/strip_optimizer`; epoch0 có `epoch=0`, optimizer còn và `train_args.epochs=1`. Source `resume_training` yêu cầu `0 < start_epoch < total_epochs`; với epoch0 của run gốc 1 epoch, `start_epoch=1` không nhỏ hơn 1. Vì vậy, việc epoch0 còn optimizer **không bảo đảm resume được run đã hoàn thành**. [Quy trình resume chính thức](https://docs.ultralytics.com/modes/train/).

Nếu R2 bị ngắt trước tổng 10 và có checkpoint epoch hoàn thành còn optimizer, cần kiểm epoch/config gốc/trạng thái/tính hữu hạn để lập bước resume riêng; không chỉnh `epoch` hay `epochs` trong file checkpoint để vượt kiểm. CLI có đường resume, nhưng **UI v1 hiện luôn `resume=False`, chỉ khởi tạo run mới**. Dừng an toàn qua UI cũng gọi final validation và strip best/last; checkpoint epoch giữ riêng cần được xét theo tổng epoch gốc. Không tự chạy CLI resume song song khi app chưa quản lý quyền sở hữu run đó.

Sau run 10 đã hoàn tất, nếu muốn học thêm 2–3 hoặc nhiều hơn từ best, UI tạo run mới: epoch lại đếm từ 1, optimizer/warmup/scheduler khởi tạo lại. Ghi rõ ví dụ `R3 ← R2 best, 3 epoch mới`; có thể ghi số epoch weights đã trải qua để tham khảo, nhưng không gọi đây là “epoch 11–13 liên tục của R2”. Muốn resume nguyên trạng cần hỗ trợ/quy trình riêng, chưa có trong yêu cầu tài liệu này. [Hướng dẫn UI hiện có](../../TRIEN_KHAI_TUAN_3_TIEP_TUC.md).

## 8. Mẫu ghi thêm kết quả

Chỉ bổ sung kết quả khi có artefact thật. Không đánh dấu xong từ một kế hoạch hoặc tiến trình đã được dispatch.

### Mẫu hồ sơ một lượt

| Trường | Nội dung cần ghi |
|---|---|
| Mã, ngày, trạng thái | R2/B0/...; ngày giờ Asia/Saigon; proposed/running/completed/stopped/failed |
| Model/run/output | Tên model/task/profile; run ID và đường dẫn thật, không dùng slug dự kiến như ID đã cấp |
| Checkpoint nguồn | Path + SHA-256; pretrained/parent run + epoch của parent; new run hay resume thật |
| Dữ liệu | Release ID, manifest/hash release/hash, train/val/test, số GT từng lớp, ảnh âm tính |
| Epoch và quan hệ runs | Epoch trong run, tổng định trước, epoch được chọn best; lịch reset nếu tạo run mới |
| Hyperparameters | args đầy đủ; batch/imgsz/workers/cache/fraction/device, optimizer/LR/lrf/nbs, seed, warmup thực, close_mosaic/augmentation, loss weights, head/NMS; yếu tố thay đổi và giả thuyết |
| Train/val loss | Tên cột đúng model/phiên bản, từng epoch; không ghép DFL với L1; finite/missing và nguồn CSV/log |
| Metric | mAP50/mAP50–95 tổng, AP từng lớp + GT, P/R tổng/từng lớp nếu đã đo; nguồn, head, confidence/IoU/quy tắc; lớp thiếu GT = N/A |
| Timing/RAM/disk | Train/val/final val/tổng, epoch times, peak RSS lấy mẫu/peak working set, RAM khả dụng thấp nhất, dung lượng artefact |
| Weights/artefact | best/last/epochN path/hash; config/environment/state/log/CSV/plots, checkpoint load và dataset trước/sau |
| Visual và quyết định | Danh sách image_id lỗi; GO/HOLD/STOP, bằng chứng, người kiểm, ngày; một thay đổi dự kiến và giới hạn lượt sau |

### Mẫu dòng theo từng epoch trong run chính

| Run | Epoch trong run | Train Box/CLS/L1 | Val Box/CLS/L1 | P/R | mAP50 / mAP50–95 | AP từng lớp/GT | Giây epoch; RAM | Checkpoint | Quyết định/lý do |
|---|---:|---|---|---|---|---|---|---|---|
| R2 — chưa chạy | 1 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa có | Chưa quyết định |
| R2 — mốc đề xuất | 3 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Dự kiến epoch2.pt | Kiểm lỗi/warmup trước quyết định |
| R2 — mốc đề xuất | 5 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Dự kiến epoch4.pt | Kiểm trend và ảnh lỗi |
| R2 — mốc đề xuất | 10 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Dự kiến epoch9.pt/best.pt | Dừng theo hạn mức, lập quyết định sau đo |

Lưu mọi epoch từ CSV, không chỉ ba mốc trong bảng. Metric chung bổ sung ở bảng đánh giá riêng sau run, không thay số trainer. Nếu chưa đo P/R theo protocol cần dùng thì ghi N/A/chưa đo và lý do.

## 9. Nguồn và giới hạn nghiệm thu

Số đo R1 và hash lấy trực tiếp từ artefact liên kết trong mục 1–2. Đã kiểm source **Ultralytics 8.4.165 đang cài** tại `.venv/Lib/site-packages/ultralytics`: `engine/trainer.py` (`_get_warmup_iterations`, `_setup_scheduler`, `_do_train`, `save_model`, `final_eval`, `_handle_nan_recovery`, `resume_training`) và `utils/loss.py` (`BboxLoss.forward`). Khả năng optimizer trong ba file R1 được đọc trực tiếp, không suy từ tên file. Khi đổi phiên bản, kiểm lại source/schema và ghi version mới.

Nguồn chính thức tham khảo: [Train/settings/checkpoint](https://docs.ultralytics.com/modes/train/), [Validation](https://docs.ultralytics.com/modes/val/), [YOLO26](https://docs.ultralytics.com/models/yolo26/), [metric và visual](https://docs.ultralytics.com/guides/yolo-performance-metrics/), [chọn head/NMS](https://docs.ultralytics.com/guides/end2end-detection/). Heuristic mốc 3/5/10, cửa sổ AP và hạn mức thí nghiệm trong file là **đề xuất của dự án**, không gán cho Ultralytics.

G3 chưa nghiệm thu đủ: còn baseline chung, đường cong fine-tune dài hơn, so sánh cùng protocol và phân tích từng lớp/visual. Sổ này giúp chọn bước có bằng chứng; chưa đặt một mục tiêu mAP sử dụng thực tế khi nhóm chưa chốt yêu cầu và độ bao phủ validation. Không tự huấn luyện thêm, thay split, sửa algorithm/UI hoặc áp dụng kế hoạch dài chỉ vì đã có tài liệu.


## 10. R2 — học thêm 4 epoch từ trọng số R1

**Yêu cầu đã chốt ngày 04/10/2026:** chạy trong chat hiện tại, dùng trọng số đã train lần 1 để học thêm 4 epoch và tiếp tục ghi log vào sổ này. Lượt 4 epoch này thay bước R2 10 epoch dự kiến ở trên; các lượt dài hơn vẫn chỉ là đề xuất, chưa được phép tự chạy.

- Run ID: `project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41`.
- Khởi động: **19:27:09 Asia/Saigon**, 04/10/2026.
- Nguồn: [R1 best.pt](../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt); SHA-256 `c36a5ebe74f48bb116415168c42e0c52de3c8f813702eba030e36fbdfa87c53b`.
- Đây là **run mới từ checkpoint đã fine-tune**, `resume=False`: optimizer/scheduler/warmup khởi tạo lại. Trọng số cuối có lịch sử 1 + 4 epoch học nếu hoàn thành, nhưng không phải resume liên tục epoch 2–5 của R1. Không khởi tạo lại từ COCO pretrained gốc.
- CPU, epochs=4, batch=2, imgsz=640, workers=0, cache=False, fraction=1.0; optimizer MuSGD, lr0=0.001, seed=42 theo cấu hình hiện có. Không SSL, không đổi dataset/split, không dùng test để chọn model.
- Dữ liệu: snapshot đã khóa project8_v0.3; toàn bộ **3.883 train / 798 validation**, tám lớp như mục 2. Preflight tự kiểm checksum/mapping/checkpoint trước train.
- RAM khả dụng lúc tạo run: **5,69 GiB**; đĩa trống **134,96 GiB**. Đây là số đo lúc bắt đầu, chưa phải peak của lượt mới.
- Backend Streamlit tiếp tục tắt; chỉ worker training riêng của lượt này chạy. Không ghi đè R1 hoặc pretrained gốc.
- Artefact: [state](../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/state.json), [log thô](../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/console.log), [request/config](../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/request.json). CSV, timing và checkpoint được cập nhật khi worker ghi xong.

### Tiến độ R2 (số đo thực)

<!-- R2_LIVE_START -->
Trạng thái cuối: completed, đủ 4/4 epoch R2. Xem đánh giá cuối sau tổng 5 epoch ở cuối tài liệu.
<!-- R2_LIVE_END -->

### Nhật ký mốc R2

| Giờ Asia/Saigon | Mốc | Bằng chứng / xử lý |
|---|---|---|
| 19:27:09 | Tạo run 4 epoch từ R1 best | Run mới độc lập, giữ R1; worker bắt đầu preflight. |
| 19:53:47 | val: epoch 1/4; saved 0/4 | Status running; batch 40/200; elapsed 1597.0s. |
| 19:55:17 | train: epoch 2/4; saved 1/4 | Status running; batch 44/1942; elapsed 1687.0s. |
| 23:19:19 | val: epoch 2/4; saved 1/4 | Status running; batch 32/200; elapsed 13928.7s. |
| 23:21:34 | train: epoch 3/4; saved 2/4 | Status running; batch 52/1942; elapsed 14064.9s. |

Sau từng epoch lưu `epoch0.pt` đến `epoch3.pt` tương ứng epoch 1–4 của R2, cùng last/best. Mốc 3 dùng kiểm xu hướng; mốc 4 dừng theo hạn mức đã yêu cầu, kiểm đủ checkpoint và báo kết quả trước đề xuất lượt tiếp theo. Không tự mở rộng epoch khi metric yếu.


### Báo cáo tiến độ và kiểm giữa run

Đã tạo [báo cáo riêng R2](../../README_DA_HOAN_THANH/reports/results/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/REPORT.md), tự cập nhật từ state/CSV đến khi worker kết thúc. Đây là bản tiến độ khi run còn chạy, không phải báo cáo nghiệm thu đủ bốn epoch.

Kiểm lúc **04/10/2026 23:04:37**: epoch R2 1 đã lưu (cách đếm người dùng: epoch 2); epoch R2 2 đang train (epoch 3), epoch 4–5 chưa có kết quả. Backend Streamlit không listen port 8501; worker train và bộ ghi log/báo cáo còn sống.

Đã nạp riêng best.pt, last.pt, epoch0.pt của R2: **detect, đúng tám lớp, toàn bộ tensor trọng số hữu hạn, file ổn định lúc kiểm**. Cả ba hiện có SHA-256 `3d164920da6e43f31eb43b8d06195d7f407a843a71fd54fc75a947fab79f2933`; không phát hiện NaN/Inf/Traceback/OOM trong log thô tới thời điểm kiểm. [Bằng chứng kiểm giữa run](../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/checkpoint_interim_validation.json). best/last sẽ thay đổi khi epoch sau lưu; kiểm này không thay thế kiểm cuối run.

Ghi nhận elapsed của epoch R2 2 đã kéo dài hơn ba giờ nhưng batch vẫn tăng và CPU worker khoảng 793% tại thời điểm kiểm (khoảng tám luồng). Chưa xác định nguyên nhân khoảng thời gian không tiến triển trước đó; có thể gồm khoảng máy/tác vụ bị gián đoạn, không quy kết là treo hay coi toàn bộ elapsed là thời gian CPU tính toán.


### Mở lại giao diện theo dõi

04/10/2026 23:07:08 — Theo yêu cầu người dùng, đã bật lại Streamlit tại http://127.0.0.1:8501 để xem Training. Health HTTP 200/ok; worker R2 tiếp tục chạy, không khởi động lại training và không tạo run mới. Các ghi nhận backend tắt phía trên là trạng thái trước khi mở lại. Chọn Màn hình → Training để theo dõi run r1best_e4; đóng trang không dừng worker.


## So sánh từ epoch đầu — 04/10/2026 23:25

Theo cách đếm của người dùng: epoch 1 là R1; epoch 2–5 là epoch 1–4 của R2. R2 khởi tạo từ R1 best nhưng reset optimizer/scheduler/warmup; đây không phải resume liên tục. Cùng dataset và validation NMS-free; dùng CSV từng epoch, không trộn với final validation.

| Epoch theo cách đếm | Train Box / CLS / L1 | Val Box / CLS / L1 | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---|---:|---:|---:|---:|
| 1 | 1.57004 / 5.70516 / 0.01751 | 1.45456 / 5.08331 / 0.01417 | 19.975% | 11.303% | 7.084% | 4.946% |
| 2 | 1.49333 / 4.26692 / 0.01645 | 1.44019 / 3.95075 / 0.01371 | 37.374% | 12.620% | 11.144% | 7.370% |
| 3 | 1.51250 / 3.65964 / 0.01713 | 1.47401 / 3.68009 / 0.01419 | 37.604% | 20.116% | 15.975% | 10.299% |
| 4 — đang chạy, chưa có validation | Chưa chốt | Chưa đo | — | — | — | — |
| 5 — chưa hoàn thành | Chưa chốt | Chưa đo | — | — | — | — |

So epoch hoàn thành gần nhất với epoch 1:
- metrics/precision(B): +17.629 điểm phần trăm; tương đối +88.26%.
- metrics/recall(B): +8.813 điểm phần trăm; tương đối +77.97%.
- metrics/mAP50(B): +8.891 điểm phần trăm; tương đối +125.51%.
- metrics/mAP50-95(B): +5.353 điểm phần trăm; tương đối +108.23%.
- train/box_loss: giảm 3.66% (giá trị âm nghĩa là tăng).
- train/cls_loss: giảm 35.85% (giá trị âm nghĩa là tăng).
- train/l1_loss: giảm 2.17% (giá trị âm nghĩa là tăng).
- val/box_loss: giảm -1.34% (giá trị âm nghĩa là tăng).
- val/cls_loss: giảm 27.60% (giá trị âm nghĩa là tăng).
- val/l1_loss: giảm -0.14% (giá trị âm nghĩa là tăng).

Nhận xét: AP và recall cải thiện qua các epoch đã đo; CLS giảm rõ. Box/L1 không giảm đều, val Box/L1 tăng nhẹ ở điểm mới nhất. Chưa đủ điểm để kết luận hội tụ/overfit; chất lượng vẫn thấp. Tiếp tục đúng lượt 4 epoch hiện tại, chưa tự đổi batch/cache/workers hoặc mở thêm run.
Người dùng yêu cầu để việc sửa code giao diện sang sau; tạm dừng công việc giao diện.


## Đánh giá cuối sau tổng 5 epoch — 05/10/2026

R2 completed, exit_code=0, đã lưu đủ 4/4 epoch; cộng R1 là 5 epoch học. RunManager không còn lượt train hoạt động. Đủ best.pt, last.pt và epoch0.pt–epoch3.pt; verifier xác nhận nạp đúng tám lớp, dataset/nguồn không đổi. Đã đối chiếu SHA-256 cả sáu checkpoint với checkpoint_validation.json và bốn file R1/pretrained với hash ghi trước R2; đều khớp. Không tìm thấy NaN/Inf/Traceback/out of memory trong log thô. Bằng chứng bảo toàn: runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/tracking_verification.json.

| Epoch tổng | Train Box / CLS / L1 | Val Box / CLS / L1 | Precision | Recall | mAP50 | mAP50–95 |
|---:|---|---|---:|---:|---:|---:|
| 1 | 1.57004 / 5.70516 / 0.01751 | 1.45456 / 5.08331 / 0.01417 | 19.975% | 11.303% | 7.084% | 4.946% |
| 2 | 1.49333 / 4.26692 / 0.01645 | 1.44019 / 3.95075 / 0.01371 | 37.374% | 12.620% | 11.144% | 7.370% |
| 3 | 1.51250 / 3.65964 / 0.01713 | 1.47401 / 3.68009 / 0.01419 | 37.604% | 20.116% | 15.975% | 10.299% |
| 4 | 1.51328 / 3.14622 / 0.01719 | 1.44702 / 3.48400 / 0.01370 | 57.333% | 24.081% | 21.764% | 13.804% |
| 5 | 1.49993 / 2.86969 / 0.01694 | 1.45340 / 2.81175 / 0.01404 | 50.317% | 27.786% | 25.783% | 16.246% |

Sau năm epoch học, mAP50 theo CSV tăng từ 7,084% lên 25,783% (+18,699 điểm phần trăm), mAP50–95 từ 4,946% lên 16,246% (+11,300 điểm). Train CLS giảm khoảng 49,7%, val CLS giảm 44,7%; Box/L1 cải thiện ít và dao động. AP tăng ở mọi epoch đã đo, chưa có dấu hiệu đứng im trong chuỗi này, nhưng chất lượng còn thấp và chưa đủ chứng minh hội tụ hay khả năng dùng thực tế. Final validation riêng của best: Precision 50,1544%, Recall 27,7754%, mAP50 25,8002%, mAP50–95 16,2407%. Recall 24,081% là CSV epoch tổng 4, không phải cuối lượt. R2 reset optimizer/scheduler/warmup khi nạp R1 best nên chuỗi là hai lượt fine-tune, không phải năm epoch resume liên tục; đánh giá trainer NMS-free chưa thay thế đối chứng pretrained cùng EVALUATION_PROTOCOL.

### Từng lớp ở final validation best

Số metric dưới đây đọc từ log cuối đã làm tròn của Ultralytics, không phải chạy lại evaluation. Box train là số instance của snapshot; Box val là ground truth trong log.

| Lớp | Box train | Box val | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| person | 6708 | 1242 | 49.30% | 51.30% | 47.30% | 30.50% |
| table | 2162 | 222 | 18.40% | 30.20% | 15.80% | 8.20% |
| chair | 1841 | 290 | 63.00% | 21.70% | 29.50% | 22.70% |
| laptop | 1104 | 279 | 35.50% | 73.10% | 61.00% | 42.60% |
| cell phone | 1191 | 579 | 59.10% | 17.10% | 17.60% | 7.43% |
| backpack | 415 | 110 | 100.00% | 0.00% | 0.10% | 0.03% |
| book | 5467 | 2458 | 43.70% | 22.40% | 26.10% | 12.50% |
| cup | 830 | 202 | 32.10% | 6.44% | 9.00% | 5.92% |

Backpack chỉ có 415 box train và Recall=0%; Precision=100% tại điểm đo này không có nghĩa là nhận dạng backpack tốt. Cup có 830 box, Recall=6,44%; cell phone có 1.191 box, Recall=17,1%. Đây là các lớp cần kiểm trước; chưa đủ bằng chứng quy lỗi riêng cho imbalance. Person/backpack lệch khoảng 16,16 lần, nhưng laptop ít box hơn cell phone vẫn có Recall cao hơn nhiều. Recall tổng là trung bình theo lớp của trainer, không phải phép đếm gộp mọi ground truth bị bỏ sót.

### Hướng đi tiếp theo (chưa thực thi)

1. Xem ảnh false negative/false positive và confusion matrix của backpack, cup, cell phone; kiểm thiếu/sai nhãn, kích thước nhỏ, che khuất, phân bố cảnh. Không đổi snapshot đã khóa trực tiếp; nếu sửa dữ liệu thì tạo phiên bản mới có review/checksum.
2. Chạy đối chứng pretrained và fine-tuned trên cùng validation theo EVALUATION_PROTOCOL trước khi báo cải thiện chính thức; giữ test ngoài chọn tham số.
3. Vì AP vẫn tăng, cân nhắc một lượt thêm có hạn mức sau khi kiểm dữ liệu. Không mặc định nhảy lên 100 epoch trên CPU.
4. Nếu thử cls_pw: A=0, B=0.25, C=0.5 phải cùng checkpoint nguồn, epoch, seed, optimizer và lịch augmentation. Giữ MixUp=0 ở đối chứng đầu, chỉ đổi một biến; đọc P/R/AP từng lớp. Weighting và oversampling là hai cơ chế khác nhau, không bảo đảm cải thiện.
5. Mosaic=1.0/close_mosaic=10 là hợp lệ nhưng cả R1 (1 epoch) và R2 (4 epoch) đóng Mosaic từ đầu; không gọi đây là thử nghiệm Mosaic đang hoạt động. Thử augmentation riêng phải chọn lịch có epoch cho phép ghép. Giữ mặc định này trong UI theo lựa chọn người dùng.

R2 elapsed 16.982,656 giây (4 giờ 43 phút 02,7 giây); train phase 16.433,547 giây, validation theo epoch 351,625 giây, final validation 66,390 giây. Epoch R2 thứ hai có khoảng kéo dài/gián đoạn lớn, không coi elapsed là CPU tính toán thuần hoặc lấy trung bình này làm dự báo tốc độ. Peak RSS lấy mẫu 0,980 GiB; peak working set 1,186 GiB; RAM khả dụng thấp nhất 2,878 GiB. Backend Streamlit đã mở lại và kiểm health HTTP 200; các ghi nhận backend tắt phía trên thuộc thời điểm trước.

Mục Nâng cao đã bổ sung vào Training, có giải nghĩa và xem trước cấu hình; mặc định không đổi. Xem TRIEN_KHAI_TUAN_3_TIEP_TUC.md. 48/48 kiểm thử đạt 05/10/2026. Không tạo thêm run trong lần cập nhật này.
