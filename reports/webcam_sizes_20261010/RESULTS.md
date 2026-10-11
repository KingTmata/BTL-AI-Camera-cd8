# Webcam: cân bằng tốc độ và độ chính xác

Cập nhật11/10/2026. **Ứng viên hiện tại: R04 ở640, một forward, NMS theo lớp IoU0,6, confidence0,25.** Độ trễ ưu tiên số1; mAP50/mAP50–95 và F1 cao trong các cấu hình đáp ứng tốc độ. Không chọn512 chỉ để tiết kiệm vài ms khi F1 giảm nhiều. Đây là lựa chọn trong các cấu hình đã đo, chưa phải checkpoint thắng toàn bộ R01–R11.

Chưa train, chưa đổi cấu hình ứng dụng hoặc alias. Checkpoint: `weights/campaign_20261010/BEST_Project8_v05.pt`, SHA256 `087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758`.

## So sánh đủ2500 ảnh validation trên CPU

8.359 GT; FP32,4 threads. mAP từ conf0,001; P/R/F1micro cùng conf0,25, match IoU0,5. Mọi cấu hình chỉ một forward và NMS theo lớp0,6. P/R này khác P/R tối ưu do evaluator báo.

| Size | mAP50% | mAP50–95% | Precision% | Recall% | F1% | p50 ms | p95 ms | >100ms /120 | FPS replay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 65.28 | 44.34 | 73.84 | 49.65 | 59.37 | 47.04 | 75.79 | 2 | 19.94 |
| 640 | 71.58 | 49.62 | 76.55 | 58.73 | 66.46 | 59.30 | 82.48 | 2 | 16.43 |
| 768 | 75.26 | 52.67 | 77.45 | 64.59 | 70.44 | 75.99 | 110.65 | 13 | 12.95 |
| 960 | 77.80 | 54.58 | 78.72 | 68.25 | 73.11 | 107.08 | 147.17 | 77 | 9.30 |

Runtime: cùng40 ảnh validation đã decode,5 warmup/size,3 lần lặp, thứ tự size/ảnh xen kẽ ngẫu nhiên. Tính predict+NMS+vẽ khung; chưa có capture/tracker/UI. FPS là throughput replay tuần tự, không phải FPS webcam. **640 tăng7,09điểm F1 so với512, p95 tăng6,69ms.**768/960 vượt100ms ởp95.512/640 vẫn có2/120frame vượt100ms; chưa bảo đảm mọi frame dưới100ms.

## Mười mục theo dõi

| Mục | Bằng chứng |
|---|---|
| 1. Latency | p50/p95 và số frame vượt100ms trong bảng; ưu tiên điều kiện tốc độ. |
| 2. mAP50 | Đủ2500val, tất cả8 lớp; bảng trên. |
| 3. mAP50–95 | Đủ2500val; kiểm chất lượng định vị box. |
| 4. F1 | Micro tại cùng ngưỡng0,25 và IoU0,5; báo ngưỡng nếu tối ưu riêng. |
| 5. Precision | Cùng ngưỡng triển khai; đánh đổi false positives. |
| 6. Recall | Cùng ngưỡng triển khai; đánh đổi bỏ sót. |
| 7. FPS | Có FPS replay; webcam thực chưa đo được vì camera0 không mở được. |
| 8. Tài nguyên | Đã đo RSS toàn tiến trình và CPU time; không phải bộ nhớ model riêng biệt. |
| 9. Cân bằng lớp | AP theo lớp có trong comparison.json; phone/book yếu hơn ở size nhỏ. |
| 10. Robustness | 299val theo nguồn,5 chế độclean/dark/blur/JPEG/noise; chưa chứng minh cảnh mới hoặc che khuất. |

## Tài nguyên và robustness

| Size | RSS đỉnh MiB | CPU ms/frame median (tổng các core) | F1 clean subset% | F1 corruption thấp nhất% | Giảm điểm F1 | AP50 phone% | AP50 book% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 709.20 | 375.00 | 56.33 | 48.51 | 7.82 | 44.83 | 46.10 |
| 640 | 680.71 | 500.00 | 63.52 | 54.03 | 9.49 | 53.10 | 62.47 |
| 768 | 689.06 | 625.00 | 69.84 | 60.88 | 8.96 | 61.68 | 70.38 |
| 960 | 701.52 | 875.00 | 73.99 | 63.07 | 10.92 | 70.02 | 73.78 |

RSS lấy trong cùng tiến trình dùng các size xen kẽ, không dùng chênh lệch nhỏ này để xếp hạng memory. CPU ms là tổng thời gian trên nhiều core, không phải latency wall-clock. Robustness: seed20261010, tối đa20 ảnh/source,299ảnh; mẫu theo nguồn không giữ tỷ lệ toànval. Photometric corruption không sửa hình học/GT. Xem `extended_metrics.json` để đọc từng chế độ.

## Kaggle và checkpoint còn thiếu

| Model / size | mAP50% | mAP50–95% | F1 conf0,25% |
|---|---:|---:|---:|
| R04_512 | 65.26 | 44.33 | 59.44 |
| R04_640 | 71.60 | 49.60 | 66.46 |
| R04_768 | 75.24 | 52.65 | 70.40 |
| R04_960 | 77.83 | 54.59 | 73.23 |
| BASELINE_v05_512 | 66.13 | 46.17 | 59.68 |
| BASELINE_v05_640 | 71.18 | 50.38 | 66.14 |
| BASELINE_v05_768 | 74.61 | 53.17 | 69.82 |
| BASELINE_v05_960 | 75.84 | 54.06 | 71.95 |

Kaggle console được lưu với trạng thái `complete`. GPU T4 FP16/batch8; không lấy tốc độ validator làm latency webcamCPU. Upload là bản R04 bỏ optimizer, đã đối chiếu từng tensor model khớp raw checkpoint; SHA bản inference `2fedf0ef4cab22911ea9786b86cbb0ec37a0802012a8cb0982ec76d3c73c2105`.

R01 epoch0 đã khôi phục từ version8 và kiểm SHA256 `3323fb2f72a91e2b827d4d7dc696fa66e57ed168d6e138e9e8af9a7c41ee5ece`; chưa có kết quả bốn size mới cho R01. R02/R07 và các checkpoint campaign khác chưa có artifact xác nhận để chấm lại. Không train nhằm giả khôi phục. **Chưa chạy hếtR01–R11 ở bốn size và chưa chọn được model thắng toàn bộ.**

## Giới hạn nghiệm thu

Webcam probe: `unavailable`; không đọc được camera0 qua các backend đã thử. Không lưu ảnh camera. Chưa đo được capture/tracker/UI/frame age hoặc jitter theo thời gian trên webcam.

Full visual review chưa hoàn tất; split còn nhóm cảnh dùng chung train/val. Không xóa ảnh khó, không đổi nhãn/split, không chấm500test trong phép so sánh này. Các số mAP/F1 trên là validation, không phải chất lượng đã xác nhận trên webcam có nhãn độc lập.

## Ngưỡng confidence và F1

Replay cache, không thêm forward. Chọn ngưỡng theo validation có thể tăng recall/F1 nhưng tăng false positives; chưa đo lại runtime tại ngưỡng mới nên cấu hình đề xuất vẫnconf0,25. Không thay mAP bằng F1 hoặc lấy F1 tính từ P/R macro của evaluator.

| Size | Ngưỡng tốt nhất trong grid | Precision% | Recall% | F1% |
|---|---:|---:|---:|---:|
| 512 | 0.15 | 66.05 | 55.17 | 60.12 |
| 640 | 0.20 | 73.31 | 61.55 | 66.92 |
| 768 | 0.25 | 77.45 | 64.59 | 70.44 |
| 960 | 0.20 | 75.75 | 70.68 | 73.13 |

Trạng thái sweep: `complete`. Grid:0,05/0,10/0,15/0,20/0,25/0,30/0,40/0,50.
