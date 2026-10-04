> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

# R2 — báo cáo học thêm 4 epoch từ R1 best

Cập nhật 05/10/2026 00:10:27, Asia/Saigon. **Trạng thái: completed / finished**.

Bản tiến độ tự cập nhật; chỉ xác nhận kết thúc sau trạng thái terminal và kiểm checkpoint. Không coi tồn tại best.pt/last.pt là run đã xong.

Run ID: `project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41`. Bắt đầu 04/10/2026 19:27:09; kết thúc 05/10/2026 00:10:12.
Đã lưu **4/4 epoch**. Epoch hiện tại 4/4; batch 200/200 trong phase finished.

## Nguồn và cấu hình

Nguồn: R1 `project8_v03_26n_cpu_20261004_181257_1eb3de94/weights/best.pt`. Lượt mới resume=False, optimizer/warmup/scheduler reset; trọng số kế thừa pretrained qua R1, không nạp lại COCO gốc.
SHA-256 nguồn theo preflight: `c36a5ebe74f48bb116415168c42e0c52de3c8f813702eba030e36fbdfa87c53b`.
YOLO26n detect, CPU, batch=2, imgsz=640, workers=0, cache=False, epochs=4, fraction=1.0, MuSGD, lr0=0.001, seed=42; save=True, save_period=1. Không SSL.
Full train 3.883 ảnh, validation 798 ảnh, test 321 ảnh giữ riêng; snapshot project8_v0.3 đã duyệt/khóa, tám lớp: person, table, chair, laptop, cell phone, backpack, book, cup.

## Kết quả từng epoch

Cột “epoch theo cách đếm 2–5” là 1 epoch của R1 cộng các epoch R2; không phải scheduler resume liên tục.

| Epoch theo cách đếm 2–5 | Epoch R2 | Trạng thái | Box / CLS / L1 train | Box / CLS / L1 val | P / R | mAP50 / mAP50–95 |
|---:|---:|---|---|---|---|---|
| 2 | 1 | Đã lưu | 1.49333 / 4.26692 / 0.01645 | 1.44019 / 3.95075 / 0.01371 | 37.3740% / 12.6200% | 11.1440% / 7.3700% |
| 3 | 2 | Đã lưu | 1.51250 / 3.65964 / 0.01713 | 1.47401 / 3.68009 / 0.01419 | 37.6040% / 20.1160% | 15.9750% / 10.2990% |
| 4 | 3 | Đã lưu | 1.51328 / 3.14622 / 0.01719 | 1.44702 / 3.48400 / 0.01370 | 57.3330% / 24.0810% | 21.7640% / 13.8040% |
| 5 | 4 | Đã lưu | 1.49993 / 2.86969 / 0.01694 | 1.45340 / 2.81175 / 0.01404 | 50.3170% / 27.7860% | 25.7830% / 16.2460% |

So với CSV R1 (mAP50 7,0840%, mAP50–95 4,9460%), epoch đã đo gần nhất thay đổi **+18.6990 / +11.3000 điểm phần trăm**. Cùng validation/head NMS-free của trainer; chưa thay thế baseline chung NMS. Không kết luận model đủ tốt từ vài epoch.

Final validation của best (số riêng, không ghi đè CSV):
- metrics/precision(B): 50.1544%
- metrics/recall(B): 27.7754%
- metrics/mAP50(B): 25.8002%
- metrics/mAP50-95(B): 16.2407%

## Log, checkpoint và kiểm tra

[Log thô](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/console.log), [CSV](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/results.csv), [state](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/state.json), [args](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/args.yaml), [sổ tuần 3](../../../../README/docs/WEEK3_TRAINING_LOG_AND_PLAN.md).

| Trọng số hiện có | Bytes |
|---|---:|
| [best.pt](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/best.pt) | 5363653 |
| [epoch0.pt](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch0.pt) | 15525027 |
| [epoch1.pt](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch1.pt) | 15525219 |
| [epoch2.pt](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch2.pt) | 15525411 |
| [epoch3.pt](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/epoch3.pt) | 15525539 |
| [last.pt](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/weights/last.pt) | 5363653 |

`epoch0.pt`–`epoch3.pt` tương ứng epoch R2 1–4 (cách đếm người dùng 2–5). best/last đã hoàn tất final validation; R1 và pretrained gốc giữ riêng.
Kiểm giữa run lúc 04/10/2026 23:02:05: best.pt: detect, 8 lớp, tensor hữu hạn=True, file ổn định khi kiểm=True; epoch0.pt: detect, 8 lớp, tensor hữu hạn=True, file ổn định khi kiểm=True; last.pt: detect, 8 lớp, tensor hữu hạn=True, file ổn định khi kiểm=True. Chỉ áp dụng đúng các file/thời điểm có SHA trong [kiểm giữa run](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/checkpoint_interim_validation.json).
Kiểm cuối run: valid=True, dataset không đổi=True, nguồn không đổi=True. Bảo toàn tất cả weights R1/pretrained: True (đối chiếu SHA-256 đã ghi trước R2).

## Thời gian và RAM

Elapsed theo worker: 16982.7 giây; đây là thời gian trôi qua, có thể gồm khoảng máy ngủ/tác vụ bị gián đoạn. Chưa xác định nguyên nhân khoảng thời gian kéo dài; không coi elapsed là CPU tính toán thuần.
Peak RSS lấy mẫu 0.980 GiB; peak working set 1.186 GiB; RAM khả dụng thấp nhất 2.878 GiB.

| Phase | Giây đã đo |
|---|---:|
| preflight | 78.296 |
| setup | 10.329 |
| train | 16433.547 |
| val | 351.625 |
| checkpoint | 2.906 |
| final_val | 66.390 |
| finalizing | 1.750 |
| verification | 37.813 |
| finished | 0.000 |
Tổng theo timings.json: 16982.656 giây.

## Quyết định

Giữ hạn mức 4 epoch đã yêu cầu, không tự chạy thêm. Backend Streamlit đã kiểm tắt (port 8501 không listen lúc kiểm); worker train và bộ ghi log/báo cáo được phép tiếp tục. Sau khi đủ epoch và kiểm cuối run mới quyết định lượt tiếp theo.


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
