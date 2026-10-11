# Báo cáo đối soát 30 epoch huấn luyện Project8

> **Cập nhật vị trí 08/10/2026:** checkpoint K2 trong hồ sơ này được giữ tại `weights/kaggle/BEST_Project8_20261006.pt`. Alias `weights/BEST_Project8.pt` hiện chứa K3 best mới; các JSON audit cũ giữ đường dẫn tại thời điểm 06/10, cần đối chiếu theo hash.

Ngày đối soát: **06/10/2026**, Asia/Saigon. Nguồn là CSV/JSON/checkpoint thật trên máy; không chạy train hoặc chấm validation lại.

**Đủ 30 epoch đã thực thi, thuộc 5 run: R1 1 + R2 4 + R3 1 + K1 10 + K2 14.** R3 dừng an toàn sau 1/5 epoch yêu cầu. STT dưới đây là thứ tự thực thi, không phải số epoch resume liên tục hay số lượt train độc lập.

## Vị trí và tính toàn vẹn

- ZIP gốc ở `runs/kaggle/<run_id>/`; model sử dụng ở `weights/kaggle/`; số liệu/kiểm toán ở `reports/results/`; Markdown đã hoàn thành ở `README_DA_HOAN_THANH/reports/results/`. Hướng dẫn còn dùng tiếp ở `README/` và file tuần 3 tại gốc.
- Hai result ZIP đạt CRC và SHA-256 toàn bộ **12 payload**; full ZIP K1 đạt hash/CRC, 44 artifact giải nén khớp DOWNLOAD_MANIFEST. Model nhập local bằng đúng byte `best.pt` trong ZIP. **26 file checkpoint Project8** được kiểm hash, nạp detection đúng 8 lớp và tensor hữu hạn.
- Dataset khóa `project8_v0.3-approved-20261004`: xác minh **10017 file** trong checksums.json, manifest khớp `bdcc6d2f5cee1ce7e973a453cc0f53e533829a94fbfdfd18a618077c8bd8fc64`; train/val/test **3.883/798/321** giữ nguyên.
- K2 chỉ tải result ZIP 7 file, có best.pt; thiếu last.pt, checkpoint từng epoch, full ZIP và ảnh/log train. Cấu hình save_period=1 không chứng minh các file đó đã tải về. Đã chép 6 file metadata từ ZIP vào thư mục run/báo cáo bằng byte gốc để dễ đối chiếu; ZIP vẫn được giữ nguyên.
- Audit chi tiết: [audit.json](../../../../reports/results/training_30_epochs_20261006/audit.json); số liệu đầy đủ cả learning rate: [epochs_30.csv](../../../../reports/results/training_30_epochs_20261006/epochs_30.csv).

## Năm run và nguồn trọng số

| Run | Run ID / CSV | Epoch thực/yêu cầu | Thiết bị | imgsz thực | Batch | Nguồn |
|---|---|---:|---|---:|---:|---|
| R1 | [project8_v03_26n_cpu_20261004_181257_1eb3de94](../../../../runs/train/project8_v03_26n_cpu_20261004_181257_1eb3de94/results.csv) | 1/1 | CPU | 640 | 2 | pretrained COCO80 |
| R2 | [project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41](../../../../runs/train/project8_v03_26n_cpu_r1best_e4_20261004_192709_35d01d41/results.csv) | 4/4 | CPU | 640 | 2 | R1 best (epoch 1) |
| R3 | [project8_v03_26n_cpu_r2best_e5_20261005_134221_540fdcbf](../../../../runs/train/project8_v03_26n_cpu_r2best_e5_20261005_134221_540fdcbf/results.csv) | 1/5 | CPU | 768 | 4 | R2 best (epoch 4) |
| K1 | [project8_26n_kaggle_20261005_092653](../../../../runs/kaggle/project8_26n_kaggle_20261005_092653/artifacts/results.csv) | 10/10 | T4 ×2 (device=0,1) | 1088 | 4 | R3 best (epoch 1) |
| K2 | [project8_26n_kaggle_20261006_083226](../../../../runs/kaggle/project8_26n_kaggle_20261006_083226/results.csv) | 14/14 | T4 ×2 (device=0,1) | 1280 | 16 | K1 best (epoch 9) |

**Mọi run dùng resume=False:** optimizer/scheduler/warmup khởi tạo lại. K1 thực thi 10 epoch nhưng chọn best ở epoch 9; K2 học từ best đó. Vì thế không gọi model hiện tại là “resume đủ 30 epoch”. Chuỗi checkpoint được chọn bỏ epoch 10 của K1; STT 16 vẫn có mặt trong lịch sử thực thi.

K1 yêu cầu imgsz=1080 trong args.yaml, trainer dùng 1088. K2 đổi đồng thời imgsz 1280, batch 16 và mosaic 0.5; không quy cải thiện/tốc độ riêng cho một thay đổi.

## Bảng đủ 30 epoch

P/R/mAP lấy đúng dòng CSV của trainer, NMS-free; R1/R2 ở 640, R3 ở 768, K1 ở 1088, K2 ở 1280. Không dùng bảng này làm đối chứng chất lượng trực tiếp giữa các run. Giây là chênh time tích lũy trong từng CSV, gồm train/val/overhead được ghi nhận, không phải tổng worker hay tốc độ tính toán thuần. L1 giữ đúng tên cột gốc.

| STT | Run | Epoch | Giây | Train box | Train cls | Train L1 | Val box | Val cls | Val L1 | P (%) | R (%) | mAP50 (%) | mAP50–95 (%) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | R1 | 1 | 1153.500 | 1.57004 | 5.70516 | 0.01751 | 1.45456 | 5.08331 | 0.01417 | 19.975 | 11.303 | 7.084 | 4.946 |
| 2 | R2 | 1 | 1571.100 | 1.49333 | 4.26692 | 0.01645 | 1.44019 | 3.95075 | 0.01371 | 37.374 | 12.620 | 11.144 | 7.370 |
| 3 | R2 | 2 | 12362.500 | 1.5125 | 3.65964 | 0.01713 | 1.47401 | 3.68009 | 0.01419 | 37.604 | 20.116 | 15.975 | 10.299 |
| 4 | R2 | 3 | 1450.500 | 1.51328 | 3.14622 | 0.01719 | 1.44702 | 3.484 | 0.0137 | 57.333 | 24.081 | 21.764 | 13.804 |
| 5 | R2 | 4 | 1402.500 | 1.49993 | 2.86969 | 0.01694 | 1.4534 | 2.81175 | 0.01404 | 50.317 | 27.786 | 25.783 | 16.246 |
| 6 | R3 | 1 | 1894.360 | 1.46987 | 2.65664 | 0.01384 | 1.43679 | 2.51084 | 0.01352 | 50.370 | 33.414 | 29.535 | 18.348 |
| 7 | K1 | 1 | 186.208 | 1.56015 | 2.97867 | 0.01703 | 1.46948 | 2.91512 | 0.02134 | 48.922 | 32.811 | 29.015 | 16.746 |
| 8 | K1 | 2 | 155.252 | 1.49784 | 2.80401 | 0.01578 | 1.42842 | 2.76931 | 0.02004 | 44.235 | 33.925 | 31.321 | 18.730 |
| 9 | K1 | 3 | 153.600 | 1.47801 | 2.62683 | 0.01563 | 1.45908 | 2.59928 | 0.0206 | 45.777 | 34.162 | 33.635 | 20.049 |
| 10 | K1 | 4 | 154.412 | 1.47083 | 2.51529 | 0.01447 | 1.43416 | 2.39934 | 0.0207 | 47.975 | 37.758 | 37.289 | 22.296 |
| 11 | K1 | 5 | 155.801 | 1.44882 | 2.37717 | 0.01507 | 1.42743 | 2.38982 | 0.02017 | 40.911 | 42.453 | 38.502 | 23.115 |
| 12 | K1 | 6 | 153.213 | 1.45636 | 2.39769 | 0.02049 | 1.4281 | 2.41261 | 0.01983 | 54.165 | 38.823 | 41.366 | 24.560 |
| 13 | K1 | 7 | 152.324 | 1.4191 | 2.16809 | 0.01953 | 1.4245 | 2.32007 | 0.01953 | 50.875 | 40.981 | 42.136 | 25.077 |
| 14 | K1 | 8 | 151.370 | 1.41495 | 2.09283 | 0.01911 | 1.40967 | 2.19138 | 0.01947 | 50.114 | 43.309 | 42.768 | 25.679 |
| 15 | K1 | 9 | 151.610 | 1.39937 | 2.04166 | 0.01937 | 1.39857 | 2.18563 | 0.01915 | 51.522 | 44.469 | 43.830 | 26.451 |
| 16 | K1 | 10 | 151.410 | 1.36952 | 1.99965 | 0.0186 | 1.38743 | 2.13498 | 0.01887 | 50.463 | 42.573 | 42.821 | 25.836 |
| 17 | K2 | 1 | 165.245 | 1.41347 | 1.9082 | 0.01238 | 1.42232 | 2.0271 | 0.01753 | 50.569 | 44.881 | 44.738 | 27.096 |
| 18 | K2 | 2 | 112.850 | 1.38422 | 1.81521 | 0.01236 | 1.41101 | 1.96283 | 0.01784 | 49.554 | 46.819 | 46.397 | 28.385 |
| 19 | K2 | 3 | 110.893 | 1.37266 | 1.78649 | 0.01238 | 1.40216 | 1.93809 | 0.01714 | 48.818 | 47.052 | 46.735 | 28.831 |
| 20 | K2 | 4 | 109.782 | 1.36466 | 1.7145 | 0.0111 | 1.40273 | 1.92629 | 0.01724 | 50.566 | 46.287 | 47.471 | 28.748 |
| 21 | K2 | 5 | 107.956 | 1.33772 | 1.6618 | 0.01169 | 1.40212 | 1.83682 | 0.01722 | 53.970 | 48.569 | 50.448 | 30.980 |
| 22 | K2 | 6 | 104.363 | 1.32893 | 1.6106 | 0.01116 | 1.38068 | 1.77276 | 0.01699 | 53.694 | 50.293 | 51.591 | 31.450 |
| 23 | K2 | 7 | 108.324 | 1.31433 | 1.52479 | 0.01094 | 1.36782 | 1.75955 | 0.01654 | 56.290 | 52.327 | 52.904 | 32.962 |
| 24 | K2 | 8 | 109.791 | 1.31841 | 1.5087 | 0.01066 | 1.35851 | 1.71773 | 0.01617 | 57.284 | 52.727 | 53.864 | 33.544 |
| 25 | K2 | 9 | 108.766 | 1.29441 | 1.46539 | 0.01049 | 1.36454 | 1.70045 | 0.01639 | 58.868 | 51.592 | 54.804 | 34.144 |
| 26 | K2 | 10 | 110.390 | 1.29487 | 1.40876 | 0.01277 | 1.35745 | 1.69721 | 0.01634 | 57.377 | 54.909 | 55.548 | 34.348 |
| 27 | K2 | 11 | 106.670 | 1.29304 | 1.34276 | 0.01259 | 1.36311 | 1.69226 | 0.01637 | 57.743 | 53.783 | 55.291 | 34.306 |
| 28 | K2 | 12 | 105.950 | 1.24825 | 1.29257 | 0.0117 | 1.3676 | 1.65042 | 0.01658 | 60.503 | 52.908 | 56.348 | 35.116 |
| 29 | K2 | 13 | 106.860 | 1.25085 | 1.29387 | 0.0115 | 1.35142 | 1.63041 | 0.01624 | 60.183 | 54.002 | 56.604 | 35.347 |
| 30 | K2 | 14 | 105.640 | 1.24704 | 1.27833 | 0.01171 | 1.35204 | 1.60643 | 0.01614 | 58.718 | 54.028 | 56.546 | 35.617 |

## Đối chứng best trên cùng protocol

COCO bbox, 798 ảnh val, imgsz=640, conf=.001, IoU=.70, NMS=True, max_det=300, agnostic_nms=False, augment=False, actual_end2end=False. Đây là số đo đã lưu trong JSON, chưa chạy đánh giá lại tại lần audit này.

| Model | mAP50 | mAP50–95 |
|---|---:|---:|
| Pretrained COCO80 | 28.176% | 18.895% |
| K1 best (05/10) | 47.779% | 28.619% |
| K2 best (06/10) | 53.848% | 33.847% |
| K2 − K1 (điểm %) | +6.068 | +5.229 |
| K2 − pretrained (điểm %) | +25.671 | +14.952 |

| Lớp | GT val | AP50 baseline | AP50 K1 | AP50 K2 | Δ K2−K1 | AP50–95 K2 |
|---|---:|---:|---:|---:|---:|---:|
| person | 1242 | 35.979% | 79.506% | 82.944% | +3.438 | 53.435% |
| table | 222 | 11.916% | 61.315% | 73.613% | +12.297 | 49.187% |
| chair | 290 | 35.587% | 48.009% | 52.443% | +4.434 | 37.130% |
| laptop | 279 | 79.837% | 78.034% | 83.469% | +5.435 | 56.597% |
| cell phone | 579 | 2.917% | 27.107% | 32.179% | +5.072 | 14.823% |
| backpack | 110 | 21.388% | 14.506% | 19.577% | +5.071 | 8.653% |
| book | 2458 | 5.713% | 47.311% | 55.362% | +8.052 | 30.114% |
| cup | 202 | 32.073% | 26.446% | 31.196% | +4.750 | 20.840% |

K2 tăng AP50 và AP50–95 ở cả 8 lớp so K1. So pretrained, backpack/cup vẫn thấp hơn ở cả hai chỉ số; laptop AP50–95 còn thấp hơn. Không gọi tất cả lớp đều tốt hơn baseline.

## Trọng số mới nên dùng

[K2 best đã nhập](../../../../weights/kaggle/BEST_Project8_20261006.pt): **5.546.053 byte**, SHA-256 `ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf`. Đúng detect, 8 lớp theo thứ tự `person/table/chair/laptop/cell phone/backpack/book/cup`, stride 8/16/32, 2.506.920 tham số; toàn bộ tensor hữu hạn. Recognition và Training liệt kê đệ quy weights/, nên file nằm trong nguồn lựa chọn của app; việc này không đổi model mặc định trong configs/app.yaml.

Checkpoint có stored_epoch=-1 và optimizer=None do đã strip. train_results bên trong khớp đủ 14 dòng CSV, train_metrics/fitness khớp epoch 14. Đây là bằng chứng metadata của best mới; chưa thể đối chiếu tensor với epoch13.pt/last.pt vì hai file đó chưa tải. Best cũ K1 được đối chiếu tensor EMA độc lập đúng epoch 9. Dùng model mới cho inference hoặc fine-tune mới resume=False; không resume optimizer nguyên trạng.

## Markdown được sửa và giới hạn

- Sửa tổng hiện hành 5/16 thành 30 epoch, trỏ nguồn mới ad136a17 thay fd9cb cũ trong hướng dẫn tiếp tục; giữ báo cáo 10 epoch ở phần lịch sử.
- Sửa danh mục tài liệu/ngày đối soát và mục sổ log đang dùng; bổ sung liên kết đến bảng 30 epoch. Báo cáo K2 gốc có số metric/hash khớp ZIP, bổ sung trạng thái thiếu full checkpoint và bằng chứng kiểm model.
- Chưa phân tích FP/FN theo ảnh, đo FPS/p95/RAM, chạy test độc lập hoặc nghiệm thu. Test hiện thiếu GT backpack/cup. Không lấy thời gian train để kết luận tốc độ inference.

Tham chiếu cách dùng resume: [Ultralytics Train](https://docs.ultralytics.com/modes/train/); hành vi checkpoint trong audit được kiểm theo Ultralytics **8.4.165** cài tại dự án.
