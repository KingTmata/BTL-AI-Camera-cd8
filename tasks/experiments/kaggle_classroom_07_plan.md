# Kế hoạch Kaggle: 6 cấu hình fine-tune từ checkpoint 07 — YOLO26n Project8 v0.4

Ngày 12/10/2026, múi giờ Asia/Saigon. Trạng thái: **người dùng đã yêu cầu thực hiện và theo dõi**. Tasklist riêng: [kaggle_classroom_07_todo.md](kaggle_classroom_07_todo.md). Không ghi đè plan/todo đang có công việc review dataset v05 chưa hoàn tất.

## Quyết định thực hiện ngày 12/10/2026

Người dùng chọn rõ: “Chỉ thử phương pháp trên dữ liệu hiện có, chấp nhận nhãn chưa review; không kết luận cải thiện lớp học”. Vì vậy nhánh đang thực hiện dùng nguyên dataset Project8 v0.5 12.000 train/2.500 val (500 test không stage), full_visual_review_complete=False được giữ đúng trong metadata; không nâng training_ready của dataset lên True. Cổng semantic review ở nhánh classroom chính vẫn chưa đạt và vẫn cần khi kết luận lớp học.

Người dùng yêu cầu rõ dùng **cả hai T4**: runner CLI/DDP `device=0,1`, batch tổng 32 cho tất cả configs. Sáu run vẫn tuần tự; theo dõi CSV/log từ process cha, không giả định callback notebook chạy được trong DDP child. Quota UI đã xác nhận 29h16m còn lại. Ngân sách 6 giờ bên dưới là đề xuất ban đầu; khi thực hiện dùng trần 10 giờ wall-clock (9h30 để chạy, 30 phút archive), tối đa 110 phút/job, vẫn không vượt 10 epoch/config. Smoke đo thời gian trước E1; nếu ước lượng trần vượt 9h30 thì dừng để điều chỉnh, không tự tăng epochs hoặc giảm dữ liệu. Native AP1280 dùng theo dõi trước, COCO640/application + YOLO26n gốc là bước đánh giá riêng sau training, không trộn số.

Notebook: https://www.kaggle.com/code/awm3636/cp07-yolo26n-project8-v0-4-best19-e1-e6-e10-p3 . Theo dõi 30 phút/lần trong chat, thông báo khi có mốc/lỗi cần xử lý. Trạng thái launch phải căn cứ UI/log, không coi click Save là đã train. Trên Kaggle đã gặp ConcurrencyViolation khi commit do draft sequence lệch; đang đồng bộ editor, chưa ghi epoch nào nếu chưa có log thật.

## 1. Mục tiêu và phạm vi đã chốt

Tên đầy đủ: **checkpoint 07 — YOLO26n Project8 v0.4, phiên Kaggle ngày 08/10/2026, best epoch 19**. 07 là thứ tự dropdown Streamlit, **không phải run R07 của chiến dịch Project8 v0.5**. Ký hiệu checkpoint07/raw07/07 bên dưới đều chỉ đúng checkpoint có đường dẫn và hash tại mục 2. Báo cáo và notebook sau này luôn phải kèm tên đầy đủ, đường dẫn và checksum.

Người dùng thấy checkpoint này nhận diện lớp học tốt hơn một số bản fine-tune, nhưng vẫn bỏ sót person, laptop, backpack, table. Mục tiêu là đo và giảm lỗi này bằng dữ liệu có nhãn và thử nghiệm có đối chứng, bảo vệ các lớp còn lại.

- **6 cấu hình train; tối đa 10 epoch/cấu hình; patience=3.** Không phải 10 runs hoặc 10 seeds. Một seed chung 42 để sàng lọc; chưa có mean/std nhiều seed.
- Hai model đối chứng chỉ đánh giá: checkpoint07 nguyên trạng và YOLO26n pretrained gốc. Không tính hai đối chứng vào 6 cấu hình train.
- Mọi cấu hình bắt đầu từ cùng byte checkpoint07, tạo model/optimizer/scheduler mới, `resume=False`; không lấy E1 làm trọng số đầu E2–E6.
- Chạy tuần tự trên Kaggle; không tự thay checkpoint ứng dụng, không push/merge, không publish hoặc chia sẻ ảnh lớp học.
- 10 epoch là ngân sách sàng lọc, không cam kết hội tụ hay chắc chắn tốt hơn 07. Nếu không cấu hình nào tốt hơn, giữ 07 và báo nguyên nhân có bằng chứng.

## 2. Bằng chứng đã kiểm và những điều chưa biết

| Nội dung | Bằng chứng / trạng thái |
|---|---|
| Checkpoint07 | `runs/kaggle/project8_v04_12k_from_best50_20261008_151936/weights/best.pt` |
| SHA256 kiểm lại local | `d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb` |
| Lịch sử | YOLO26n, 8 lớp, dataset Project8 v0.4 12.000 ảnh; fine-tune mới từ best K3 Project8 v0.3, reset optimizer/scheduler (`resume=False`), dừng thủ công 20/50 epoch; best epoch 19 |
| Config lịch sử | MuSGD, lr0=0.001, imgsz=1280, batch tổng 32, device=0,1, seed=42, AMP=True; không mặc định GPU hiện tại giống lịch sử |
| Số đo lưu trong report | v04 val COCO 640/NMS=True: mAP50=70.706%, mAP50–95=50.694%; trainer 1280/NMS-free: 75.356%/53.488%; đây là hai protocol khác nhau, không phải đo lại hôm nay |
| Dữ liệu v04 | Report ghi 12.000 ảnh, split 9.540/1.754/706; 7.081 ảnh có cờ cần review đủ lớp, scene/near-duplicate audit chưa đầy đủ |
| Dữ liệu v05 | Plan hiện có ghi 15.000 ảnh, split 12.000/2.500/500, full_review_pending, training_ready=False; chưa audit lại toàn bộ trong phiên này |
| Ảnh lớp học người dùng | Hai screenshots là mẫu chẩn đoán, chưa thành train/val/test có ground truth; không đủ để kết luận định lượng |
| Kaggle | Đã mở `https://www.kaggle.com/code`, tài khoản đăng nhập A W M / `awm3636`, thấy các notebook Project8 lịch sử |
| GPU/quota Kaggle | Chưa xác nhận GPU thực được cấp, VRAM, thời gian session, quota còn lại, phiên bản runtime; không kích hoạt GPU để lập plan |
| Ultralytics local | Đọc source 8.4.165: detection fitness=mAP50–95; EarlyStopping strict `>` với ngoại lệ best_fitness=0. Phải kiểm lại source runtime Kaggle |
| Source phục vụ chuẩn bị sau này | Worktree `C:/Users/aawm8/.codex/worktrees/20a1/BTL-AI-Camera-cd8`; tài liệu này ở checkout hiện tại. Không sửa source của chat CPU/GPU đang làm song song |

Nguồn local: [report checkpoint07](../../docs/archive/completed/reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/report.md), [args lịch sử](../../reports/archive/results/kaggle/project8_v04_12k_from_best50_20261008_151936/args.yaml), [class map v04](../../configs/data_v04.yaml), [plan chất lượng v05](../plan.md).

## 3. Cổng dữ liệu trước khi train

### 3.1 Hợp đồng 8 lớp không được đổi

| ID | Tên | Lưu ý |
|---|---|---|
| 0 | person | Mọi người đủ điều kiện trong ảnh, kể cả xa/che khuất theo rubric |
| 1 | table | Bàn lớp học, thống nhất box với các nguồn cũ |
| 2 | chair | Không nhầm bàn/ghế hoặc background |
| 3 | laptop | Cả màn hình mở/đóng theo rubric thống nhất |
| 4 | cell phone | Chú ý vật nhỏ và che khuất |
| 5 | backpack | Chú ý treo sau ghế, dưới bàn |
| 6 | book | Không nhập nhầm tập giấy theo tiêu chí chưa chốt |
| 7 | bottle | Lớp gộp **cup+bottle**, không báo AP lớp này như AP chai riêng |

Rà toàn bộ vật thuộc 8 lớp trong mỗi ảnh được đưa vào thí nghiệm. Nhãn một lớp nhưng bỏ sót các lớp khác có thể dạy model coi vật thật là background. Viết rubric box, vật bị che, vật mơ hồ, negative thật; lưu người review, lý do sửa, nguồn, version. Auto-label chỉ là đề xuất sau này, mọi nhãn phải được review trước khi dùng.

### 3.2 Dữ liệu lớp học đại diện

Thu ảnh gốc từ nhiều buổi/cảnh/góc máy, ánh sáng, khoảng cách; có cảnh đông, laptop/ba lô dưới bàn, bàn che khuất, vật nhỏ và background thật. Hai screenshots đang có có thể đặt ở tập chẩn đoán riêng; không đưa cùng ảnh vào train và dùng lại để chứng minh cải thiện.

Đề xuất split theo **buổi/cảnh/clip**, khoảng 70/15/15 theo nhóm, không chia ngẫu nhiên từng frame. Số ảnh/nhóm cần được thống kê trước khi khóa; không bịa số tối thiểu bảo đảm mAP. Mỗi lớp cần đủ ảnh và nhóm ở validation/test để metric có ý nghĩa; thiếu thì báo thiếu và bổ sung trước nghiệm thu. Test giữ kín, không dùng chọn config hoặc sửa dữ liệu theo lỗi model.

Kiểm byte/pixel duplicates và near-duplicates giữa các split; tên file khác không chứng minh độc lập. Giữ provenance v03/v04/v05, không chuyển holdout lịch sử vào train. Dùng train cũ đã review + classroom train; khóa cùng một manifest/sampling cho cả 6 cấu hình. Báo riêng classroom val và legacy val để thấy domain shift/forgetting.

### 3.3 Chọn nhánh thực hiện

**Nhánh chính:** dữ liệu lớp học có nhãn, class map/format/semantic review và split audit đạt → chạy đủ ma trận để đo cải thiện miền lớp học.

**Nhánh phương pháp:** nếu chưa có classroom labels, có thể chuẩn bị notebook hoặc so optimizer trên một phiên bản dữ liệu hiện có đã vượt cổng chất lượng. Không tự dùng v05 full_review_pending để train, không gọi kết quả này là khắc phục lỗi lớp học. Hiện chưa có bằng chứng cổng dữ liệu này đạt.

## 4. Ma trận 6 cấu hình

| ID | Optimizer | lr0 | Augmentation / freeze | So sánh có kiểm soát |
|---|---|---|---|---|
| E1 | MuSGD | 0.0001 | Aug chuẩn chung; không freeze backbone | Mốc fine-tune nhẹ |
| E2 | MuSGD | 0.0003 | Như E1 | E2/E1 chỉ đổi LR |
| E3 | AdamW | 0.0001 | Như E1 | E3/E1 chỉ đổi optimizer |
| E4 | AdamW | 0.0003 | Như E1 | E4/E3 chỉ đổi LR; E4/E2 chỉ đổi optimizer |
| E5 | MuSGD | 0.0001 | Chỉ giảm `scale` từ 0.5 xuống 0.2; như E1 ở mọi tham số khác | Thử scale jitter nhẹ cho vật nhỏ; chưa chứng minh tốt hơn |
| E6 | MuSGD | 0.0001 | Aug như E1; freeze toàn backbone đã xác minh layer list | E6/E1 chỉ đổi khả năng cập nhật backbone |

E1–E4 là factorial 2 optimizer × 2 LR; không kết luận optimizer độc lập LR chỉ bằng một cặp. **Augmentation** là biến đổi ảnh lúc train (màu, vị trí, tỷ lệ, ghép mosaic) và biến đổi box tương ứng; không biến đổi ngẫu nhiên validation/test. E5 chỉ giảm biên độ phóng/thu `scale` từ 0.5 xuống 0.2 để thử giữ vật nhỏ ít bị thay đổi tỷ lệ mạnh; đây là giả thuyết, chưa chứng minh tốt hơn.

**Freeze backbone** là giữ cố định trọng số phần trích xuất đặc trưng, chỉ cập nhật phần còn được phép học; giả thuyết là hạn chế quên kiến thức cũ. E6 giữ backbone cố định cả run tối đa 10 epoch; không thêm pha unfreeze vì làm khác ngân sách/schedule. Phải đọc architecture checkpoint và log tên/lượng tham số frozen; không mặc định `freeze=10` chắc chắn là toàn backbone. Dùng list chỉ số được xác minh; failure gate nếu freeze head ngoài ý định. Mọi E1–E6 bắt đầu từ checkpoint 07 — YOLO26n Project8 v0.4, phiên Kaggle ngày 08/10/2026, best epoch 19; không nối chuỗi trọng số giữa các run.

### 4.1 Cấu hình chung đề xuất

- `epochs=10`, `patience=3`, `val=True`, `resume=False`, `seed=42`, `deterministic=True`, `pretrained=True`, `cls_remap=False` vì ID 8 lớp giữ nguyên.
- `imgsz=1280`; batch khởi điểm đề xuất 16 cho một GPU, workers=2, cache=False, AMP=True sau kiểm tương thích. Khóa batch/GPU sau preflight cho toàn matrix. Nếu dùng hai GPU phải kiểm DDP bằng script và tổng batch; không mặc định dùng device=0,1 theo run cũ.
- Scheduler cấu hình cho 10 epoch ngay từ đầu: `cos_lr=False`, `lrf=0.01`, `warmup_epochs=1.0`, `warmup_bias_lr=0.0001`, `warmup_momentum=0.8`, `momentum=0.937`, `weight_decay=0.0005`, `nbs=64`. Bias warmup thấp tránh dùng 0.1 kế thừa cho fine-tune LR thấp. Đây là thiết kế ngắn hạn, không tái hiện lịch sử 50 epoch.
- Aug chung: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0, translate=0.1, scale=0.5 (E5=0.2), shear=0, perspective=0, flipud=0, fliplr=0.5, mosaic=0.5, mixup=0, cutmix=0, copy_paste=0; `close_mosaic=2` thay vì copy 10 vào run 10 epoch. Log epoch thực đóng mosaic; early stop trước pha này là khả năng có thật.
- Giữ box=7.5, cls=0.5, dfl=1.5, cls_pw=0, multi_scale=0, rect=False, fraction=1.0, compile=False; lưu toàn bộ args hiệu lực, cả defaults không ghi ở đây.
- Không dùng optimizer=auto vì có thể tự chọn optimizer/LR làm hỏng đối chứng. Khóa Ultralytics/Torch/CUDA và evaluator sau kiểm tương thích, ưu tiên đối chiếu 8.4.165; không tự nâng package mỗi run.

## 5. Dừng sớm: định nghĩa chính xác

Fitness validation detection local 8.4.165 là mAP50–95, không phải train loss. Nếu best epoch=4 và epoch 5,6,7 không tốt hơn thì kết thúc sau epoch 7; mỗi cấu hình vẫn có trần cứng epoch 10. Điểm tăng rất nhỏ vẫn là cải thiện vì built-in không có min_delta. Không hứa “không thay đổi” theo điểm làm tròn ở bảng log.

Source local: `fitness > best_fitness or best_fitness == 0`; ties sau best dương không reset, nhưng khi best vẫn 0, chuỗi 0 reset best_epoch nên patience có thể không dừng. Nếu runtime khác dùng `>=`, ties sẽ reset. Trước train phải lưu source/version và kiểm các chuỗi tăng, đứng yên, giảm, toàn 0. Nếu người dùng yêu cầu đứng yên kể cả toàn 0 luôn dừng sau 3 epoch, cần stopper strict được kiểm riêng trước launch; không tự patch thư viện trong bước plan. `epochs=10` vẫn chặn trường hợp ngoại lệ. EarlyStopping theo best trong run mới; không dùng fitness của raw07 làm mốc stopper. Dùng raw07 làm đối chứng khi chọn model cuối.

NaN/Inf, dataset/hash/class map lệch, nhãn bất hợp lệ hoặc checkpoint lệch hash → dừng run và không tiếp tục ma trận cho đến khi xác định lỗi.

## 6. Notebook và vận hành Kaggle dự kiến

Notebook mới riêng: `notebooks/kaggle/classroom07_matrix_e10_p3.ipynb` (chưa tạo/chạy trong lượt plan). Cell được tách thành: cấu hình/manifest → môi trường/hash/gates → baseline eval → smoke riêng → runner tuần tự → val comparison → archive. Không auto-run khi mở notebook; flag launch mặc định False và dataset_ready gate phải là kết quả audit, không nút bỏ qua.

1. Read-only kiểm tài khoản, GPU được phép chọn, quota/session limit và storage; chỉ bật accelerator khi bước launch được chốt.
2. Stage checkpoint đúng hash và dataset version đã review vào input riêng tư; verify ZIP/path traversal/CRC, image decode, label format, số lớp, bounds, hash và split manifest. Không upload ảnh lớp học khi chưa có dữ liệu hợp lệ/phạm vi upload cụ thể.
3. Khi được chuyển sang thực hiện: smoke 1 epoch riêng, không tính là E1, kiểm train_batch/val labels, thời gian epoch, VRAM, AMP và checkpoint/log. Smoke xong thì bỏ model smoke, E1 vẫn bắt đầu raw07.
4. Baselines và E1–E6 dùng cùng dữ liệu/protocol. Mỗi job output riêng, `exist_ok=False`. Sau mỗi run lưu kết quả và archive trước job tiếp theo; không chạy 6 notebook đồng thời.
5. Trong lúc train theo dõi loss hữu hạn, validation, GPU/VRAM, disk, epoch/stop reason; chỉ báo thay đổi có ý nghĩa hoặc lỗi/cần người dùng xử lý.

### Ngân sách và khôi phục

Trần **60 training epoch chính**, cộng 1 smoke và chi phí baseline/val/test/archive. Chưa có ước lượng runtime mới. Lịch sử 20 epoch ~79.47 phút chỉ là tham khảo, không áp thành ETA do khác data/GPU/optimizer/batch. Sau smoke: ước tính `61 × median epoch time + eval + stage/archive`, báo range và overhead; theo dõi GPU-hours riêng theo số GPU.

Đề xuất tổng wall-clock budget 6 giờ, trần mỗi job 60 phút, dự phòng tối thiểu 30 phút session cho archive. Chốt lại theo quota/session thực và số đo smoke trước launch. Không dùng `time=` thiếu kiểm vì Ultralytics có thể điều chỉnh horizon epochs; supervisor bên ngoài phải giữ trần 10, dừng ở ranh giới epoch khi còn thời gian và đánh dấu budget_stop, không early_stop. Không gọi run chưa đủ ngân sách là completed.

OOM → lưu failure, giảm batch cho **toàn ma trận**, smoke lại và chạy lại các config bị ảnh hưởng từ raw07 ở version experiment mới; không đổi riêng E4 rồi so như cùng protocol. Chỉ hạ imgsz thành matrix version khác khi budget không cho phép 1280. Quota/session sắp hết → ngừng launch job mới, archive job đã xong. Run gián đoạn dùng last.pt của **chính run đó**, `resume=True`, giữ tổng tối đa 10 epoch và kiểm optimizer/scheduler/stopper state được phục hồi; nếu stopper state không phục hồi, ghi history và khôi phục/kiểm riêng hoặc chạy lại, không reset ngầm patience. Không resume lịch sử07.

Artifact từng run: best.pt, last.pt, args.yaml, results.csv, log, plots, failure/stop reason, initial/final SHA256, seed, library/CUDA/GPU versions, dataset/split hashes, timestamps, peak VRAM, timing, source stopper, metadata protocol. Download/archive có checksum và kiểm đọc trước khi hết session.

## 7. Đánh giá và chọn model

### Hai phép đo riêng

**AP:** cùng imgsz=640, conf=0.001, iou=0.7, NMS=True, max_det=300, augment=False, COCO maxDets=[1,10,100] để nối report lịch sử; dùng evaluator có ground truth/predictions lưu lại. Trainer AP 1280/NMS-free chỉ để theo dõi/early stop, không trộn với COCO AP 640/NMS=True.

**Ứng dụng lớp học:** chốt imgsz/mode triển khai từ source runtime trước đo; conf=0.25, NMS iou=0.6 cùng max_det=300 và augment=False. Nếu pipeline YOLO26 đang end-to-end và iou không có tác dụng, ghi rõ; dùng cùng NMS flag/head cho mọi model, không giả định đặt iou là đã bật NMS. Báo riêng mode ứng dụng và mode AP. Giữ cùng ảnh/thứ tự/conf/preprocessing, không tune threshold mỗi model.

Ghép dự đoán-GT cùng lớp one-to-one tại IoU≥0.5 để tính TP/FP/FN, precision/recall từng lớp; unmatched prediction trùng vật thật ghi duplicate riêng (vẫn là FP), không đếm nhiều TP cho một GT. Lưu gallery lỗi trên validation và báo theo kích thước/che khuất/nguồn. Screenshot chẩn đoán không thay ground truth.

YOLO26n gốc 80 lớp cần mapping rõ COCO→8ID: person, dining table→table, chair, laptop, cell phone, backpack, book, cup+bottle→ID7; gộp class ở evaluator và xử lý duplicate theo cùng luật. Đây là đối chứng transfer, tên dining table có khác ngữ nghĩa bàn lớp học, phải ghi giới hạn.

### Quy tắc chọn bằng validation, chốt trước khi chạy

Đề xuất candidate đạt khi: classroom mAP50–95 tăng ≥1 điểm phần trăm so raw07; macro recall của person/table/laptop/backpack tăng ≥2 điểm tại conf=0.25; FP/ảnh tăng không quá 10% (nếu baseline=0 dùng trần tuyệt đối 0.05 FP/ảnh); legacy mAP50–95 và AP50 từng lớp không giảm quá 2 điểm. Nếu dataset nhỏ, nêu khoảng bất định theo bootstrap nhóm cảnh và tránh nói chênh lệch nhỏ là chắc chắn. Các ngưỡng này là **đề xuất review**, không mục tiêu đã được người dùng chốt.

Trong các candidate đạt, xếp classroom mAP50–95 giảm dần, sau đó macro recall 4 lớp, rồi FP/ảnh thấp hơn. Không chọn bằng train loss, không chọn bằng test, không chọn chỉ bằng ảnh minh họa. Khi không có candidate đạt hoặc chưa đủ dữ liệu, giữ07 và báo no_promote. Sau khi khóa config/model trên val, đánh giá test một lần với raw07 và model được chọn; không đổi config theo test. Báo đủ8 lớp và coverage, không chỉ4 lớp ưu tiên. Nếu test thất bại, chưa promote; vòng nghiên cứu tiếp theo cần holdout mới.

## 8. Thứ tự công việc và cổng hoàn thành

T0 đối chiếu/plan (đã làm) → T1 chốt metric/budget/stop semantics + T2 audit/khóa dữ liệu → cổng A → T3 notebook draft không launch → T4 preflight/runtime/stopper → cổng B (plan được chốt, data đạt, quota đủ) → T5 smoke → T6 baseline và E1–E6 → T7 chọn bằng val → cổng C → T8 test/archive/report. Từng run T6 là đơn vị riêng để có thể ngừng trước job kế tiếp. Không có subagent hoặc job train song song trong kế hoạch này.

Hoàn tất khi có bảng baseline+6configs (hoặc run thất bại/budget-stop được giải thích), per-class metrics, hashes/protocol/timing, model được chọn bằng val và test độc lập, artifact khôi phục được. Tài liệu kết quả phải phân biệt completed/early_stop/budget_stop/failed và không đổi model deployment tự động.

## 9. Điểm đang chặn launch

1. Chưa có classroom train/val/test đã review và khóa theo nhóm cảnh; dữ liệu hiện có chưa được xác nhận training_ready. Đây là thiếu bằng chứng dữ liệu, không phải thiếu quyền chạy train chung.
2. GPU, quota/session, runtime source early-stop/optimizer và VRAM chưa kiểm trên Kaggle; chưa có smoke timing.
3. Plan mới nhất đang ở bước review: ngưỡng chọn model, wall-clock budget và ngoại lệ zero-fitness cần chốt khi chuyển sang thực hiện. Ý nghĩa10lần đã giải quyết là tối đa10epochs/config, không hỏi lại.

Tài liệu chính thức đã đối chiếu ngày12/10/2026: [Train settings và resume](https://docs.ultralytics.com/modes/train/), [Training best practices](https://docs.ultralytics.com/guides/model-training-tips/). Docs mô tả resume phục hồi optimizer/scheduler/epoch; source runtime là căn cứ cuối cho stopper, freeze và config hiệu lực.
