# Tiến độ và metric đã xác nhận

Kiểm tra 16:41 ngày 12/10/2026 (Asia/Saigon). Checkpoint 07 — YOLO26n Project8 v0.4, phiên Kaggle ngày 08/10/2026, best epoch 19. SHA256 d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb.

Version 2 vẫn Running, GPU T4 x2, device=0,1, batch tổng 32. E1 hoàn tất 10/10 epoch, best CSV epoch8; E2 đã hoàn tất epoch1/10 (session3988,1s), E3–E6 đang chờ. Không thấy lỗi mới. Không khởi chạy job khác.

| Model / giai đoạn | mAP50 | mAP50–95 | Precision | Recall | F1 xấp xỉ từ P/R tổng hợp |
|---|---:|---:|---:|---:|---:|
| Checkpoint 07 nguyên trạng | 71,8% | 50,8% | 73,5% | 66,3% | 69,71% |
| Smoke riêng 1 epoch | 76,8% | 54,1% | 80,7% | 70,2% | 75,08% |
| E1 best, trainer final (3566,3s) | 78,5% | 56,2% | 82,0% | 71,1% | 76,16% |
| E1 best, native batch8 (3627,1s) | 76,1% | 54,6% | 80,4% | 68,1% | 73,74% |
| E2 epoch7 (đang chạy) | Chưa có | 56,400% | Chưa có | Chưa có | Chưa có |

Baseline/smoke: số làm tròn từ final validation log trên 2.500 ảnh, 8.359 boxes, native1280. F1 ở bảng là 2PR/(P+R) từ P/R tổng hợp đã làm tròn, không phải macro-F1 trung bình từng lớp; F1 chính thức và ngưỡng confidence cần lấy từ evaluator/artifacts. Smoke tách riêng, không dùng khởi tạo E1.

| E1 epoch | mAP50–95 | Thời điểm trong session (giây) |
|---|---:|---:|
| 1 | 53,987% | 925,9 |
| 2 | 55,123% | 1226,0 |
| 3 | 55,395% | 1516,0 |
| 4 | 55,527% | 1796,0 |
| 5 | 55,478% | 2086,1 |
| 6 | 55,884% | 2376,1 |
| 7 | 56,078% | 2656,1 |
| 8 | 56,154% | 2946,3 |
| 9 | 55,823% | 3226,3 |
| 10 | 55,911% | 3506,3 |

Hai lượt validation của E1 có số khác nhau; giữ nguyên cả hai, chưa xác định nguyên nhân. Native riêng dùng batch8, conf=.001, iou=.7, nms=False, max_det300. Cần đối chiếu args/checkpoint và evaluator sau khi tải artifacts; không gộp thành một metric. Live epoch feed E2 chỉ xuất mAP50–95.

Archive E1 được kiểm trên Kaggle tại3575,1s: SHA256 `d9837461ee3001108c5f3ae6a6ac624e7a9488f22eb501b14357fea31cc15181`. Chưa tải về/kiểm checksum local vì toàn phiên vẫn chạy, Output0B. COCO640/application và đối chứng YOLO26n gốc vẫn còn thiếu.

E1 best native riêng: số theo lớp làm tròn từ log. F1 xấp xỉ từng lớp sẽ tính từ P/R nếu cần; F1 chính thức cần evaluator.

| Lớp | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| person | 81,4% | 74,4% | 83,9% | 57,2% |
| table | 80,2% | 67,8% | 76,6% | 55,7% |
| chair | 75,4% | 52,5% | 63,4% | 42,5% |
| laptop | 75,0% | 74,4% | 75,0% | 49,6% |
| cell phone | 79,7% | 59,7% | 69,6% | 42,1% |
| backpack | 86,3% | 76,3% | 82,5% | 66,6% |
| book | 76,9% | 62,6% | 72,7% | 46,9% |
| bottle (cup+bottle) | 88,2% | 77,3% | 85,6% | 76,5% |

Dữ liệu v0.5 chưa review được dùng theo lựa chọn của người dùng để thử phương pháp. Không kết luận cải thiện lớp học, không chọn model bằng test, không thay model ứng dụng.

Nguồn: [log Kaggle Version 2](https://www.kaggle.com/code/awm3636/cp07-yolo26n-project8-v0-4-best19-e1-e6-e10-p3?scriptVersionId=357596703).

## Cập nhật 17:11 ngày 12/10/2026

E2 đã xác nhận 7/10 epoch, phiên vẫn Running trên T4x2, không thấy lỗi mới. E3–E6 chưa bắt đầu. E2 best tạm epoch6 đạt56,644%, cao hơn best CSV E1 (56,154%) 0,490 điểm phần trăm; đây là so sánh trainer tạm thời, chưa phải đánh giá chung best.pt hay kết luận cải thiện lớp học.

| E2 epoch | mAP50–95 | Session giây |
|---|---:|---:|
| 1 | 54,309% | 3988,1 |
| 2 | 55,011% | 4277,7 |
| 3 | 55,384% | 4567,7 |
| 4 | 55,650% | 4847,8 |
| 5 | 55,906% | 5137,7 |
| 6 | 56,644% | 5427,8 |
| 7 | 56,400% | 5717,9 |

mAP50, precision, recall và F1 của E2 chưa xuất trong live feed; giữ thiếu cho tới final validation/artifacts.

## Cập nhật 17:29 ngày 12/10/2026

E2 hoàn tất10/10, best CSV epoch10 đạt56,787% (epoch8:56,782%, epoch9:56,411%). E3 đang chạy, epoch1 đạt53,850% tại7029,1s. Phiên Running T4x2; không thấy lỗi mới. Không khởi chạy job mới.

| E2 best validation | mAP50 | mAP50–95 | Precision | Recall | 2PR/(P+R) xấp xỉ |
|---|---:|---:|---:|---:|---:|
| Trainer final,6627,9s | 79,1% | 56,8% | 81,9% | 71,6% | 76,40% |
| Native riêng batch8,6688,5s | 76,6% | 55,2% | 82,0% | 67,6% | 74,11% |

Native riêng E2 so với E1: AP50 tăng0,5 điểm phần trăm; AP50–95 tăng0,6; precision tăng1,6; recall giảm0,5. Số làm tròn từ log, cùng runner native riêng; chưa phải COCO640/application hoặc kết luận lớp học. F1 xấp xỉ từ P/R tổng hợp, không phải macro-F1 chính thức.

Archive E2 đã kiểm trên Kaggle tại6636,6s, SHA256 `853093fa749a1c4dc600c4703929542b2ba2c127c7b3072b16df4896af2e9854`; chưa tải/kiểm local.

| E2 native riêng, lớp | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| person | 83,2% | 74,7% | 84,4% | 57,7% |
| table | 81,6% | 69,3% | 77,8% | 56,9% |
| chair | 79,8% | 51,8% | 65,3% | 43,9% |
| laptop | 74,6% | 72,9% | 74,4% | 49,1% |
| cell phone | 83,7% | 57,9% | 70,1% | 42,9% |
| backpack | 86,7% | 74,8% | 82,3% | 67,9% |
| book | 77,1% | 63,0% | 73,2% | 47,1% |
| bottle (cup+bottle) | 89,5% | 76,6% | 85,6% | 76,4% |

## Cập nhật 17:41 ngày 12/10/2026

E3 đang chạy3/10 epoch: epoch2 mAP50–95=54,677% tại7309,1s; epoch3=55,249% tại7599,1s. Phiên Running7814,3s trên T4x2. Không thấy lỗi/OOM/mốc hoàn tất mới. mAP50/P/R/F1 của E3 chưa có trong live feed. E1/E2 hoàn tất, E4–E6 đang chờ.

## Cập nhật 18:11 ngày 12/10/2026

E3 đạt10/10 epoch, best CSV epoch10 mAP50–95=57,204% tại9549,4s. Chưa xác nhận final validation/archive, không đánh dấu run hoàn tất. Phiên Running9617,9s trên T4x2, không lỗi mới. E4–E6 chưa có epoch.

| E3 epoch | mAP50–95 | Session giây |
|---|---:|---:|
| 4 | 55,695% | 7879,2 |
| 5 | 55,835% | 8159,2 |
| 6 | 56,536% | 8439,2 |
| 7 | 56,445% | 8719,3 |
| 8 | 56,948% | 8989,3 |
| 9 | 56,891% | 9269,4 |
| 10 | 57,204% | 9549,4 |

E3 trainer best cao hơn E2 0,417 điểm phần trăm; chỉ là so sánh trainer. mAP50/precision/recall/F1 E3 vẫn chưa có bằng chứng; chờ lượt đánh giá riêng để so công bằng.
