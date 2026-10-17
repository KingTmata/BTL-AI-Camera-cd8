# Kaggle fine-tune Project8 — 14 epoch

> **Cập nhật vị trí 08/10/2026:** checkpoint K2 trong hồ sơ này được giữ tại `weights/kaggle/BEST_Project8_20261006.pt`. Alias `weights/BEST_Project8.pt` hiện chứa K3 best mới; các JSON audit cũ giữ đường dẫn tại thời điểm 06/10, cần đối chiếu theo hash.

Ngày ghi nhận: 06/10/2026. Run Kaggle đã hoàn tất 14/14 epoch; result ZIP đã tải về, kiểm tra hash và nhập checkpoint local.

**Đối soát lại 06/10/2026:** số liệu/hash trong báo cáo khớp ZIP gốc. Đã nạp model local và kiểm tensor hữu hạn; nội dung checkpoint train_results khớp đủ 14 dòng CSV. Xem [bảng đủ 30 epoch trong 5 run](../../training_30_epochs_20261006/report.md) và [audit máy đọc được](../../../../../../../reports/archive/results/training_30_epochs_20261006/audit.json). 30 là tổng epoch thực thi (1+4+1+10+14), không phải 30 run hoặc 30 epoch resume liên tục.

## Kết luận ngắn

- Checkpoint mới dùng cho inference: weights/kaggle/BEST_Project8_20261006.pt, SHA-256 ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf (5.546.053 byte).
- Mô hình được fine-tune từ best của run Kaggle 05/10, SHA nguồn fd9cb282f05f0053166174e6c8d13221f7bfff54e8f06673b21bf8134951a3a8; resume=false, tức lượt fine-tune mới chứ không tiếp tục optimizer giữa run.
- Trên cùng evaluator validation, mAP50 tăng từ 28,18% lên 53,85% (+25,67 điểm phần trăm); mAP50–95 tăng từ 18,89% lên 33,85% (+14,95 điểm).
- Backpack giảm 1,81 điểm AP50 và 2,60 điểm AP50–95; cup giảm nhẹ 0,88/0,44 điểm; laptop tăng AP50 nhưng giảm AP50–95. Cần xem lỗi theo ảnh/lớp trước khi kết luận dùng thực tế.
- Đánh giá trên 798 ảnh validation, không phải test độc lập hay nghiệm thu sản phẩm. Dataset có 3.883 ảnh train và 798 ảnh val; không dùng 321 ảnh test.

## Đối soát artifact

| Artifact | Vị trí / giá trị |
|---|---|
| Result ZIP đã tải | project8_26n_kaggle_20261006_083226_result.zip trong Downloads |
| Bản lưu cùng dự án | runs/kaggle/project8_26n_kaggle_20261006_083226/project8_26n_kaggle_20261006_083226_result.zip |
| SHA-256 ZIP | 1bd9b9ffa191f68dcf70c7ee6391b5685549962149ef49dbab0fdc2cee9e375d |
| Kích thước ZIP | 4.877.504 byte |
| Checkpoint nhập local | weights/kaggle/BEST_Project8_20261006.pt |
| SHA-256 checkpoint | ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf |
| Checkpoint nguồn | weights/kaggle/project8_26n_kaggle_20261005_092653_fd9cb282f05f.pt |
| SHA-256 nguồn | fd9cb282f05f0053166174e6c8d13221f7bfff54e8f06673b21bf8134951a3a8 |
| Release / SHA-256 source manifest | project8_v0.3-approved-20261004 / bdcc6d2f5cee1ce7e973a453cc0f53e533829a94fbfdfd18a618077c8bd8fc64 |

Đã so SHA-256 và byte của cả 6 payload trong FILES.json; tất cả đều khớp. SHA-256 result ZIP khớp giá trị ghi từ Kaggle. Đã nạp checkpoint mới bằng Ultralytics local: task detect, đúng 8 lớp person, table, chair, laptop, cell phone, backpack, book, cup; stride 8/16/32; nạp thành công.

ZIP có 7 file: best.pt, result.json, FILES.json, args.yaml, results.csv, baseline_val_metrics.json, finetuned_val_metrics.json. ZIP không có ảnh/nhãn dataset, last.pt hay optimizer state.

Metadata/config/CSV được chép từ ZIP bằng đúng byte gốc vào `runs/kaggle/project8_26n_kaggle_20261006_083226/` và `reports/results/kaggle/project8_26n_kaggle_20261006_083226/` để truy cập trực tiếp; model giữ tại weights/kaggle. ZIP gốc được giữ nguyên. Chưa tải full ZIP, ảnh/log train hoặc checkpoint từng epoch của K2; không hiểu save_period=1 là bằng chứng các file đó đã tải về máy.

## Cấu hình train ghi trong args.yaml

| Tham số | Giá trị |
|---|---|
| Run / nguồn dữ liệu | project8_26n_kaggle_20261006_083226 / project8_v0.3-approved-20261004 |
| GPU | 2× T4, device=0,1; DDP |
| Python / Torch / CUDA / Ultralytics | 3.13.15 / 2.11.0+cu128 / 12.8 / 8.4.165 |
| Epoch / imgsz / batch tổng | 14 / 1280 / 16 |
| Thời gian cuối trong results.csv | 1.573,48 giây, khoảng 26 phút 13 giây |
| Optimizer / lr0 / patience | MuSGD / 0,001 / 10 |
| Workers / cache / AMP / seed | 2 mỗi rank / False / True / 42 |
| close_mosaic / resume | 5 / False |
| Mosaic / fliplr / flipud | 0,5 / 0,5 / 0 |
| degrees / translate / scale | 0 / 0,1 / 0,5 |
| mixup / cutmix | 0 / 0 |
| Dataset | 3.883 train / 798 val; không dùng test |

args.yaml ghi nms=false cho validation trong vòng train. Vì thế metric từng epoch bên dưới là NMS-free ở imgsz 1280 và không so trực tiếp với evaluator chung NMS ở imgsz 640.

## Đối chứng trên cùng validation protocol

Protocol cho pretrained baseline và fine-tuned best: bbox; split val; 798 ảnh; imgsz 640; conf 0,001; IoU 0,70; NMS bật; max_det 300; agnostic NMS tắt; augment tắt.

| Model | mAP50 | mAP50–95 |
|---|---:|---:|
| Pretrained baseline | 28,18% | 18,89% |
| Fine-tuned best | 53,85% | 33,85% |
| Chênh lệch | **+25,67 điểm %** | **+14,95 điểm %** |

### AP theo lớp — cùng protocol

Các số là AP (%); chênh lệch tính fine-tuned trừ baseline.

| Lớp | GT val | AP50 baseline → tuned (Δ) | AP50–95 baseline → tuned (Δ) |
|---|---:|---:|---:|
| person | 1.242 | 35,98 → 82,94 (+46,96) | 20,80 → 53,44 (+32,63) |
| table | 222 | 11,92 → 73,61 (+61,70) | 8,08 → 49,19 (+41,11) |
| chair | 290 | 35,59 → 52,44 (+16,86) | 27,12 → 37,13 (+10,01) |
| laptop | 279 | 79,84 → 83,47 (+3,63) | 58,32 → 56,60 (−1,72) |
| cell phone | 579 | 2,92 → 32,18 (+29,26) | 1,17 → 14,82 (+13,65) |
| backpack | 110 | 21,39 → 19,58 (−1,81) | 11,25 → 8,65 (−2,60) |
| book | 2.458 | 5,71 → 55,36 (+49,65) | 3,14 → 30,11 (+26,98) |
| cup | 202 | 32,07 → 31,20 (−0,88) | 21,28 → 20,84 (−0,44) |

AP cải thiện rõ ở person, table, cell phone và book; backpack và cup giảm nhẹ. Laptop có AP50 cao hơn nhưng AP50–95 thấp hơn baseline, nên độ chính xác định vị ở IoU nghiêm ngặt hơn chưa tăng.

## Tiến triển 14 epoch trong results.csv

Đây là metric của trainer (imgsz 1280, NMS-free), không phải bảng đối chứng phía trên.

| Epoch | mAP50 | mAP50–95 |
|---:|---:|---:|
| 1 | 44,738% | 27,096% |
| 2 | 46,397% | 28,385% |
| 3 | 46,735% | 28,831% |
| 4 | 47,471% | 28,748% |
| 5 | 50,448% | 30,980% |
| 6 | 51,591% | 31,450% |
| 7 | 52,904% | 32,962% |
| 8 | 53,864% | 33,544% |
| 9 | 54,804% | 34,144% |
| 10 | 55,548% | 34,348% |
| 11 | 55,291% | 34,306% |
| 12 | 56,348% | 35,116% |
| 13 | 56,604% | 35,347% |
| 14 | 56,546% | 35,617% |

Epoch cuối trong CSV có P=58,718%, R=54,028%, mAP50=56,546%, mAP50–95=35,617%. mAP50 cao nhất theo CSV là epoch 13; mAP50–95 cao nhất là epoch 14. Không suy ra checkpoint được chọn theo riêng một cột này; dùng best.pt của Kaggle và metric protocol đã ghi trong JSON.

Kiểm nội dung best local: stored_epoch=-1, optimizer=None do strip; train_metrics/fitness khớp epoch 14 và train_results khớp tất cả cột CSV. Đây là bằng chứng metadata của checkpoint; chưa có last.pt/epoch13.pt để so tensor độc lập. Nếu last và best cùng tensor, chúng cho cùng kết quả nhận dạng; hash file có thể khác do metadata, và resume còn cần trạng thái optimizer/scheduler/epoch.

So **K1 best ngày 05/10**, K2 best tăng mAP50 **+6,068 điểm %** và mAP50–95 **+5,229 điểm %**; cả 8 lớp tăng AP50 và AP50–95. Nhận xét backpack/cup/laptop giảm phía trên là **so pretrained baseline**, không phải giảm so K1. K2 thay đổi đồng thời imgsz 1088→1280, batch 4→16 và mosaic 1.0→0.5, nên chưa quy cải thiện riêng cho yếu tố nào.

## Những file nào cần giữ?

- Chỉ chạy inference: giữ best.pt mới. Nó đã được sao vào weights/kaggle/.
- Dùng best để mở lượt fine-tune mới: cũng cần best.pt, cùng dataset/config code tương ứng. Run này resume=false.
- Tái lập báo cáo/kiểm toán: giữ toàn bộ result ZIP. result.json ghi provenance và số đo; FILES.json kiểm hash; args.yaml ghi cấu hình; results.csv ghi từng epoch; hai JSON metrics ghi baseline và fine-tuned cùng protocol.
- Các file metrics/config rời không cần giữ riêng nếu result ZIP nguyên vẹn.
- Checkpoint cũ fd9cb... không cần để dùng model mới. Có thể giữ làm bản quay lui/đối chiếu nguồn; hash nguồn đã nằm trong result record. ZIP mới không có last.pt nên không thể khôi phục optimizer để resume chính xác giữa lượt đã hoàn thành.
- ZIP không chứa dataset. Muốn train với dataset thì cần có ảnh/nhãn tương ứng; result ZIP một mình không đủ dữ liệu để huấn luyện lại.

## Lưu ý

Run hoàn tất và metric hợp lệ trên validation; chưa xem ảnh false positive/false negative, chưa đo tốc độ, chưa chạy test độc lập và chưa xác nhận triển khai. Log có cảnh báo Ray get_trial_id được gọi ngoài Tuner/Trainer và trả None; cảnh báo không làm run dừng. Không xem kết quả này là nghiệm thu model.
