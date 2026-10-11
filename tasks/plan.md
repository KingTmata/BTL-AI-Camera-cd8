# Mục tiêu và kế hoạch phát triển BTL-AI-Camera-cd8

**Ngày đối chiếu: 08/10/2026 — Asia/Saigon. Trạng thái: kế hoạch đã cập nhật sau khi ghép v0.5 được người dùng ủy quyền; full review, training và triển khai checkpoint tiếp theo chưa hoàn tất.**

## 1. Kết luận dành cho người dùng

Ưu tiên tiếp theo là **làm rõ chất lượng nhãn và bộ đánh giá camera, rồi mới tiêu GPU**. YOLO26n mới tốt hơn rõ rệt trên validation v0.4: mAP50 **70,706%**, mAP50–95 **50,694%**, theo protocol đối chứng CPU 640/NMS=True. Kết quả trainer 1280/NMS-free là **75,356%/53,488%**, thuộc phép đo khác. Chưa có kết quả test độc lập hay FPS camera đủ để nghiệm thu.

Mục tiêu ban đầu đề xuất vẫn là **mAP50 ≥85%, mAP50–95 ≥60%, AP50 mỗi lớp ≥75%** trên bộ test đại diện đã khóa. Đây là mục tiêu, không phải điểm đã đạt hay cam kết chắc chắn. Mốc 90%/70% chỉ xét sau; 95%/80% là hướng nghiên cứu. Không tự nâng mục tiêu thành 95%.

Ba ưu tiên chất lượng là **cell phone → chair → book**, đồng thời bảo vệ backpack trên miền cũ. V0.5 hiện có 15.000 ảnh, tất cả mang review_stage=full_review_pending. 7.081 ảnh có cờ review ở nguồn v0.4 là một nhánh ưu tiên lịch sử; không thể kết luận các ảnh đó đều sai. Lô merged_3k ban đầu thêm2.969 unique; sau loại3 pizza còn2.966; đãthêm34phoneunique vàđủ15.000.

**Mục tiêu số lượng đã đạt:**15.000 ảnh duy nhất,12.000 train /2.500 validation /500 test. Đã bổ sung34 ảnhphoneunique sau khi loại3 pizza sai nhãn; còn thiếu0 ảnh. Tổng3.000 ảnh mới đượcgiữ so v04=2.966 merged_3k +34 phone_topup.

**Cổng chất lượng chưa đạt:**toàn bộ15.000 ảnh full_review_pending; cầntruynguồn mapping vàreviewđủ8lớp, kiểmscene/near-duplicate. Chưa train hoặc cómetricv05; training_ready=False. Đủsốkhông bảođảm đạtmetric.

Đường đi đề xuất: khóa tiêu chí và provenance → pilot review → rà nhãn và xác minh nhóm cảnh → khóa phiên bản dữ liệu/test → YOLO26n với dữ liệu đã sửa → cân nhắc YOLO26s pretrained nếu nano còn thiếu năng lực → kiểm CPU 640/NMS-free → nghiệm thu → promote có rollback. Danh sách công việc nằm trong [todo.md](todo.md).

## 2. Cách đọc bằng chứng và phạm vi đã làm

- **[X] Xác minh trực tiếp trong phiên này:** đọc manifest UTF-8 BOM, đếm các file nhãn, đối chiếu số box, hash ba checkpoint và manifest, kiểm CRC ZIP, đọc cấu hình/mã nguồn và artifact đánh giá. Không chạy lại suy luận hoặc tính lại COCO AP từ predictions.
- **[L] Kết quả đã lưu:** audit, run metadata, CSV và comparison hiện có; xác nhận các tệp nhất quán, không biến thành phép đo mới của phiên này.
- **[G] Giả thuyết:** lời giải thích cần kiểm bằng mẫu ảnh hoặc thí nghiệm có đối chứng.
- **[Đ] Đề xuất:** tiêu chí, ngân sách và lịch trình chưa được nghiệm thu/chốt, trừ các ràng buộc người dùng đã quyết định.

Không phát hiện AGENTS.md trong phạm vi kiểm tra repo và các thư mục cha đã kiểm. Không có graphify-out/graph.json; đối chiếu trực tiếp các tệp liên quan, không dựng graph mới. Áp dụng quy trình lập kế hoạch, ghi quyết định và hướng dẫn YOLO cho dữ liệu/training. Lượt lập kế hoạch ban đầu chỉ tạo tài liệu. Sau đó chat chính đã được người dùng ủy quyền ghép v0.5, chuyển 206 test cũ sang val và cập nhật pointer dataset; tài liệu này ghi nhận các thao tác đã thực hiện đó. Lượt cập nhật tài liệu này chỉ đọc đối chiếu và sửa tài liệu, không thực hiện thêm merge, sửa nhãn, training hoặc thay checkpoint.

Backend đã được bàn giao là chạy tại `http://127.0.0.1:8501/`, health từng trả `ok`. Lần thử GET health trong phiên tài liệu bị môi trường chặn socket; **không suy ra backend đã dừng**, cũng không xác nhận lại trạng thái sống ở thời điểm viết. Không khởi động lại/dừng backend để giải quyết việc này.

## 3. Hiện trạng đã đối chiếu

### 3.0 Dataset chính v0.5 hiện hành

[X] Đọc trực tiếp manifest v0.5: **15.000 dòng**, 15.000 giá trị image_sha256 khác nhau, 15.000 pixel_sha256 khác nhau; đếm file nhãn khớp số box từng dòng. Tổng **43.276 box**, chia33.272 train /8.359 val /1.645 test. Phiên này không hash lại toàn bộ ảnh/pixel; báo cáo ghép đã lưu kết quả kiểm byte/pixel. Không đồng nhất unique exact với độc lập cảnh.

| Split v0.5 | Hiện tại | Mục tiêu | Còn thiếu |
|---|---:|---:|---:|
| Train | 12.000 | 12.000 | 0 |
| Validation | 2.500 | 2.500 | 0 |
| Test | 500 | 500 | 0 |
| Tổng | **15.000** | **15.000** | **0** |

[L] Đã giữ 500 test cũ, gồm tất cả 11 ảnh bottle; chuyển 206 test cũ sang val. Không chuyển val/test v03/v04 vào train. Mapping nguồn mới:0 book→6,1 chair→2,2 mobile phone→4; ID7 bottle gộp giữ nguyên. Nhóm filename/clip mới được giữ cùngsplit theo heuristic; full scene/near-duplicate audit chưa hoàn tất. Tất cả 15.000 ảnh có review_stage=full_review_pending; annotation_review lịch sử được giữ đểprovenance, không chứng minh đã full review.

Pointer hiện hành là `configs/dataset_current.json`, trỏ `data/dataset/project8_v0.5_merged_20261008`; cấu hình train dữ liệu là `configs/data_v05.yaml`. V04 và weights được giữ. SHA256 manifest v0.5: `22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd`.

ZIP Mobile Phone Dataset.v2i.yolo26 (1).zip có100/100 ảnh trùng pixel với merged_3k, đóng góp0 unique và đã được xóa theo yêu cầu người dùng. Bằng chứng overlap được giữ; không tính100 ảnh đó vào34 ảnh còn thiếu.

### 3.0.1 Lỗi ngữ nghĩa đã xác nhận — cổng training đang không đạt

[X] Đã xem `label_mapping_samples.jpg`: mẫu pizza/collage có box **chair ID 2**, không phù hợp nội dung nhìn thấy. image_id `merge3k__dc02793e38668b6af38f`; nguồn `cellphone_train_3class_filtered__000000000397_jpg`; nhóm cùng nguồn có3 ảnh. Issue và peerIDs được lưu trong `visual_label_issues.json`. Sau đó người dùng đã yêu cầu xóa; **cả3 ảnh và3 nhãn trong nhóm đã bị loại khỏi active v05**, intake gốc giữ nguyên. `removed_pizza_images.json` lưu exclusion ledger; khi reimport phải áp dụng ledger để không nhập lại nhóm này. Loại3 mẫu không chứng minh phần còn lại đã sạch nhãn.

[L] Bản verification trước loại3 pizza ghi **valid=False**, format_and_hash_valid=True, semantic_labels_verified=False, training_ready=False; full visual review=False vàscene audit=False. Đã kiểm29.938 payload ảnh/nhãn v05 và24.000 payload v04 cũ theo báo cáo. Đạt hash/YAML/reader không thay được chất lượng nhãn.

**Ưu tiên trước training:** truy nguồn YAML/classorder và labels trước/sau remap của nhóm `cellphone_train_3class_filtered`, rà 3 ảnh cùng nhóm rồi nguồn này vàtoàn bộ merge3k. Phân biệt nhãn nguồn đã sai, mapping sai, hay augmentation/conversion làm lệch ảnh/box; chưa đủ evidence để đảo ID hàng loạt. Sửa chỉ theo rubric vàversioning saukhi xác minh, không đổi thứ tự8ID hoặc xóaảnh tùyý. Một mẫu sai xácnhận cólỗi, không ước lượng được tỷ lệ sai toàn nguồn. Fullreview15k vẫn làgate.

### 3.1 Dataset v0.4 lịch sử và hợp đồng lớp

Dataset: `data/dataset/project8_v0.4_merged_20261008/`. [X] Manifest có 12.000 dòng, 12.000 giá trị image_sha256 khác nhau và 12.000 pixel_sha256 khác nhau. Đây là kiểm trên trường manifest, **chưa hash lại byte/pixel toàn bộ ảnh** và chưa loại trừ near-duplicate.

| Split v0.4 lịch sử — không phải split hiện hành | Ảnh | Box | Tỷ lệ ảnh |
|---|---:|---:|---:|
| Train | 9.540 | 28.443 | 79,500% |
| Validation | 1.754 | 6.861 | 14,617% |
| Test | 706 | 2.149 | 5,883% |
| Tổng | 12.000 | 37.453 | 100% |

[X] Đếm trực tiếp file nhãn khớp manifest; không phát hiện sai số lượng box, cấu trúc 5 trường, ID ngoài 0–7 hoặc giá trị tọa độ chuẩn hóa ngoài [0,1]/kích thước không dương trong phép kiểm này. **Không đồng nghĩa box sát vật thể, lớp đúng, hoặc nhãn đầy đủ.** Cả ba split đều không có file nhãn rỗng; chưa có bộ background thật đã xác minh.

| ID cố định | Tên hiện hành | Box train | Ảnh val chứa lớp | Box val | Ảnh test chứa lớp | Box test |
|---|---|---:|---:|---:|---:|---:|
| 0 | person | 7.449 | 278 | 1.443 | 44 | 335 |
| 1 | table | 2.352 | 123 | 241 | 35 | 100 |
| 2 | chair | 2.178 | 171 | 339 | 46 | 54 |
| 3 | laptop | 1.266 | 175 | 274 | 88 | 144 |
| 4 | cell phone | 4.861 | 522 | 1.139 | 305 | 569 |
| 5 | backpack | 1.645 | 346 | 408 | 169 | 189 |
| 6 | book | 5.427 | 262 | 2.439 | 142 | 720 |
| 7 | bottle = cup + bottle | 3.265 | 402 | 578 | 11 | 38 |

Ảnh có thể chứa nhiều lớp, không cộng cột ảnh theo lớp thành tổng dataset. AP của ID 7 là **AP lớp gộp cup+bottle**, không phải AP chai riêng. Giữ thuộc tính phụ `subtype=cup/bottle/uncertain` trong hồ sơ review để phân tích, không thêm ID detection thứ chín. v0.3 giữ nguyên ID 7 = cup cho đối chứng lịch sử. Phạm vi table và book/sách-vở cần theo quy tắc dự án, không mặc định đồng nhất mọi nguồn với COCO dining table/book.

| Nhóm nguồn trong manifest | Train | Val | Test | Tổng cần rà đủ 8 lớp |
|---|---:|---:|---:|---:|
| addition1 — new_zip_8classes_20261008 | 2.428 | 424 | 247 | 3.099 |
| addition2 — new_zip_bottle_merged_20261008 | 3.285 | 551 | 146 | 3.982 |
| Tổng requires_full_class_review | 5.713 | 975 | 393 | 7.081 |
| base — trạng thái approved | 3.827 | 779 | 313 | 4.919 |

[X] 7.081/12.000 = **59,008%** mang cờ review. Có **487/4.919** bản ghi approved mà JSON provenance vẫn chứa ghi chú `Human review pending`. Đây là mâu thuẫn trạng thái cần làm rõ, chưa đủ để phủ định hoặc xác nhận nhãn của các ảnh đó. Pilot phải có một nhánh kiểm lại base, không chỉ các nguồn mới.

[L] Audit hash v03→v04 không thấy byte ảnh cũ chuyển val/test→train hoặc train→val/test; đã ghi 3.827 ảnh train, 779 val, 313 test được giữ đúng split. 56 train, 19 val, 8 test cũ không còn trong v04 vẫn thuộc lịch sử sử dụng. Không nhập lại chúng như dữ liệu mới “chưa từng thấy”. Chưa có chứng cứ đầy đủ về near-duplicate, cảnh và buổi quay độc lập.

### 3.2 Training và checkpoint

[L] 50 epoch chính lịch sử + 20 epoch lượt mới = 70 epoch chính, smoke tách riêng. Lượt mới fine-tune từ best v03, `resume=False`, reset optimizer/scheduler; không phải resume liên tục 70 epoch. `epochs=50` nhưng dừng thủ công tại 20. Best là **epoch 19**, tương ứng file `epoch18.pt` đánh số từ 0.

| Điểm trainer 1280/NMS-free | P (%) | R (%) | mAP50 (%) | mAP50–95 (%) |
|---|---:|---:|---:|---:|
| Epoch 18 — mAP50 cao nhất | 78,080 | 67,979 | 75,415 | 52,792 |
| Epoch 19 — best theo fitness | 77,882 | 67,352 | 75,356 | 53,488 |
| Epoch 20 — last | 76,166 | 68,194 | 75,024 | 53,004 |

`imgsz=1280, batch=32, device=0,1, workers=2, cache=False, optimizer=MuSGD, lr0=0.001, patience=5, mosaic=0.5, close_mosaic=10, cls_remap=False, seed=42`. Device yêu cầu không chứng minh hai GPU đã được sử dụng hiệu quả. CSV ghi 4.768,48 giây (~79,47 phút); chưa gồm stage dữ liệu, tải artifact, smoke hoặc chấm CPU. Không dùng thời gian này làm cam kết cho lượt khác/hardware khác.

Lịch 50 epoch đóng mosaic ở phần cuối, chưa tới khi dừng tại 20. Một epoch giảm chưa đủ kết luận overfit. [X] ZIP có 29 entry, kiểm CRC không phát hiện entry hỏng.

| Tệp | Dung lượng byte [X] | SHA256 [X] |
|---|---:|---|
| weights/BEST_Project8_v04_20261008.pt | 15.725.347 | d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb |
| weights/BEST_Project8.pt — alias mặc định | 15.725.347 | d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb |
| weights/BEST_Project8_20261008.pt — v03 | 5.547.077 | d4244d26e69d78a066944b22ba64f80ea68602f2f994d6cf6bbae7970ff123f9 |

[L] Best mới còn optimizer/state do dừng giữa lịch. Chênh lệch dung lượng tệp không chứng minh kiến trúc nano đã lớn lên. Giữ bản gốc; nếu tối ưu dung lượng, tạo **bản triển khai riêng** và đo parity trước khi dùng. Manifest SHA256: `03a5e12eecb751f69ff81990201e6f4fc026b9a1ef036fbe96f10264608a0955`.

### 3.3 Hai miền validation, cùng protocol đối chứng

[L] CPU, imgsz 640, conf 0.001, iou0.7, NMS=True, max_det 300, augment=False, agnostic_nms=False; COCO IoU 0.50:0.05:0.95, maxDets[1,10,100], thứ tự ảnh cố định. Metadata: Ultralytics8.4.165, Torch2.14.0+cpu, NumPy2.4.6, pycocotools2.0.11, actual_end2end=False.

| Miền | Ảnh/GT | Best đầu mAP50 / mAP50–95 (%) | Best mới (%) | Thay đổi điểm phần trăm |
|---|---|---|---|---|
| v03 val, ID7=cup | 798 / 5.382 | 56,727 / 35,896 | 55,279 / 35,049 | −1,448 / −0,847 |
| v04 val, ID7=cup+bottle | 1.754 / 6.861 | 47,593 / 31,047 | 70,706 / 50,694 | +23,113 / +19,647 |

Không so trực tiếp điểm v03 với v04 để quy toàn bộ khác biệt cho model: dữ liệu và phạm vi ID7 khác nhau. So trước/sau **trong cùng hàng** mới là đối chứng đúng. Old-domain backpack AP50 giảm từ 25,568% xuống 17,529% (−8,039 điểm); phone 35,547→35,927, chair 54,979→54,216, laptop84,314→81,611, book 61,463→64,841.

| ID/lớp trên v04 val | AP50 đầu (%) | AP50 mới (%) | AP50–95 mới (%) | Khoảng thiếu tới sàn AP50=75 (điểm) |
|---|---:|---:|---:|---:|
| 0 person | 62,518 | 77,521 | 49,714 | 0 |
| 1 table | 57,778 | 73,962 | 50,356 | 1,038 |
| 2 chair | 43,993 | 57,806 | 41,726 | 17,194 |
| 3 laptop | 80,204 | 80,204 | 54,747 | 0 |
| 4 cell phone | 22,768 | 49,881 | 30,206 | 25,119 |
| 5 backpack | 22,631 | 79,916 | 69,153 | 0 |
| 6 book | 59,293 | 64,738 | 36,401 | 10,262 |
| 7 bottle gộp | 31,559 | 81,620 | 73,252 | 0 |

mAP là trung bình AP tám lớp, không phải tỷ lệ dự đoán đúng trên toàn bộ box. Hai số laptop AP50 chỉ bằng nhau sau làm tròn. Khoảng tới mục tiêu tổng 85/60 hiện là **14,294/9,306 điểm** trên v04 val hiện hành; chưa phải khoảng thiếu trên test mới.

### 3.4 Chẩn đoán lỗi, không phải AP hoặc kết quả backend

[L] Saved predictions640/NMS=True, conf≥0.25, ghép greedy một-một cùng lớp IoU≥0.5:

| Lớp | TP | FN | FP chưa ghép | FN/GT |
|---|---:|---:|---:|---:|
| person | 1.067 | 376 | 503 | 26,06% |
| table | 179 | 62 | 95 | 25,73% |
| chair | 180 | 159 | 127 | 46,90% |
| laptop | 212 | 62 | 82 | 22,63% |
| cell phone | 500 | 639 | 149 | 56,10% |
| backpack | 301 | 107 | 74 | 26,23% |
| book | 1.533 | 906 | 857 | 37,15% |
| bottle | 448 | 130 | 150 | 22,49% |

FP chưa ghép có thể chứa vật thật chưa có GT. Không tự sửa/xóa nhãn dựa vào dự đoán. Summary gắn 496/639 FN phone là `no_matching_detection`; 413/906 FN book dưới confidence0.25 và 117 là ứng viên lệch box; chair có 81 không ghép, 60 dưới confidence. Đây là **nhóm chẩn đoán tự động**, không phải nguyên nhân đã chứng minh.

Báo cáo ảnh đã lưu nêu: lớp học phone14/16 FN; nội thất chair 8/11; kệ sách book 38/44; nhà ga backpack5/5. Chúng được chọn vì FN cao, không đại diện ngẫu nhiên toàn dataset. Cảnh nhà ga cần phân biệt backpack với suitcase/bag; phiên này không tự kết luận nhãn sai.

## 4. Bottleneck và kiểm chứng cần làm

| Ưu tiên | Bằng chứng | Giả thuyết cần kiểm | Phép kiểm/đầu ra quyết định |
|---|---|---|---|
| P0 — nhãn đủ lớp | 15.000 full reviewpending; lỗi pizza→chair đã xác nhận; lịch sử7.081 flags/487 pendingnote; FP chưa ghép nhiều | Nguồn đơn lớp bỏ sót vật thuộc lớp khác, làm sai cả học và đánh giá | Pilot phân tầng + review cả 8 lớp; tỷ lệ ảnh thiếu/sai lớp/sai box theo nguồn; không quy lỗi theo số FP đơn thuần |
| P0 — holdout | Test bottle 11 ảnh/38 box; không có negative; nhóm cảnh chưa xác minh; testv05 hiện hành500 | Điểm cao trên một số nguồn chưa đại diện camera | Data card test 500, kiểm độc lập, xác minh coverage test 500 hiện hành, báo giới hạn và trình release tương lai nếu cần; không xem prediction test để chọn ảnh train |
| P1 — phone/chair/book nhỏ và che khuất | AP thấp, FN cao; ví dụ ảnh đã lưu | Vật nhỏ sau resize640, che khuất, mật độ đông hoặc blur làm mất đặc trưng | Chia slice theo kích thước sau resize, occlusion, blur, nguồn; xem ngẫu nhiên cùng ảnh khó; chỉ thử imgsz khác khi evidence ủng hộ |
| P1 — localization | mAP50–95 thấp hơn mAP50; ứng viên lệch box trong audit | Box không nhất quán hoặc dự đoán thiếu chính xác | Double-review box, IoU disagreement; AP75/AP90 phụ, recall theo size; không suy nguyên nhân từ khoảng AP đơn lẻ |
| P1 — domain shift | v04 tăng nhưng v03 giảm; backpack−8,039 điểm | Mất năng lực miền cũ, nguồn mới khác kiểu túi/cảnh, hoặc nhãn không tương thích | Giữ bảng old-domain; phân tích nguồn/cảnh và ranh giới lớp, bổ sung train cùng kiểu miền từ cảnh độc lập |
| P2 — capacity nano | Chưa có cặp n/s kiểm soát | Nano có thể thiếu capacity sau khi dữ liệu đã sạch | So n-pretrained với s-pretrained cùng split/recipe/seed và đo CPU; chưa gọi nano là nút thắt đã xác minh |
| P2 — lịch tối ưu | Dừng20/50, chưa đóng mosaic | Lịch phù hợp budget có thể cải thiện | Lượt20 epoch khai báo20 từ đầu, đóng mosaic 5 epoch cuối; không hứa gain và không gọi overfit từ epoch 20 |

### 4.1 Nhận biết overfitting, underfitting và các lỗi trên dự án này

Các triệu chứng dưới đây là tín hiệu chẩn đoán, không là quy tắc cho độ chắc chắn100%. Mọi phân tích chọn model/threshold dùng **train và validation**, tuyệt đối không dùng lỗi test để thiết kế training. Trước tiên loại sai protocol, dữ liệu và mapping; sau đó mới quy cho capacity hay tối ưu.

| Vấn đề/triệu chứng | Dữ liệu cần đo | Cách xác minh trong dự án | Hướng xử lý theo evidence |
|---|---|---|---|
| **Overfit:** train loss tiếp tục giảm nhưng val metric giảm kéo dài, val loss xấu dần | results.csv nhiều epoch, train/val loss, AP 8 lớp, theo nguồn/size, seeds | Cố định validation sạch/hash/GT/protocol; xem chuỗi 5–10 epoch và đường gốc bên cạnh smoothing 3–5 epoch. Smoothing chỉ giúp nhìn xu hướng. Kiểm drift nguồn/augmentation và seed variance. Best 19 rồi giảm ở20 hiện **chưa đủ bằng chứng**; plateau riêng lẻ cũng không chứng minh overfit | Giữ best, xem early stopping; giảm độ phức tạp hoặc tăng đa dạng/regularization phù hợp khi đã kiểm nhãn. Mỗi thay đổi là một lượt có đối chứng, không mặc định tăng augmentation sẽ giải quyết |
| **Underfit:** chất lượng train và val đều thấp/plateau | Inference trên một train-subset cố định đủ nguồn/lớp, **không augmentation**, cùng640/head/conf/evaluator với val; loss/LR/gradient logs nếu có | Không dùng loss của batch có mosaic làm mAP train. So AP train-subset và val, kiểm nhãn/khả năng học các ví dụ rõ; loại lỗi pipeline trước. Cả hai thấp có thể do dữ liệu khó, nhãn thiếu hoặc lịch tối ưu, chưa chắc nano thiếu capacity | Sửa nhãn trước; kiểm LR/warmup/lịch đã đủ chưa; sau đó thử ngân sách/độ phân giải hoặc cặp n/s có kiểm soát |
| **Domain shift hoặc suy giảm miền cũ:** v04 tăng, v03 giảm | AP/P/R theo nguồn, cảnh camera, size/occlusion; mô tả nhãn từng nguồn | Backpack mới 79,916% nhưng old17,529%; so before/after trong từng miền và kiểm old-domain giảm25,568→17,529. Không so hai AP khác bộ để khẳng định nguyên nhân forgetting hoặc overfit | Rà semantic backpack/bag/suitcase và phân bố nguồn; giữ mẫu train miền cũ hợp lệ, bổ sung cảnh tương ứng độc lập; old-domain guardrail |
| **Label noise/thiếu nhãn:** nhiều unmatched FP, confusion lệch lớp | 7.081 flags,487 approved có pending note; GT/pred overlays train/val, review ledger, confusion matrix | Full review 15.000 ảnh; rà cả 8 lớp, kiểm systematic missing labels và consistency box. Off-diagonal có thể do model hoặc GT. Unmatched FP không mặc định là false positive thật | Sửa theo rubric độc lập, hai người hòa giải; tạo GT version mới, chấm lại baseline và candidate cùng GT; không tự xóa dự đoán/GT khó |
| **Localization:** AP50 cao hơn AP50–95 nhiều | AP75/AP90 phụ; phân bố IoU của cặp cùng lớp, FN riêng; overlay box theo size; inter-reviewer agreement | Kiểm đồng thời chất lượng GT và predicted box. IoU chỉ trên matched boxes có selection bias, nên báo cả unmatched FN. Khoảng AP tự nó không chứng minh box sai; vật nhỏ nhạy với vài pixel | Chuẩn hóa box, thử kích thước ảnh/lịch loss khi có evidence; đo lại cùngprotocol, không bẻ boxGT cho khớp prediction |
| **Low confidence / missed detection / wrong class** | PR curves, recall/P và FP theo threshold trên val; tìm detection cùng lớp/IoU, ứng viên khác lớp hoặc lệch box | Audit saved640/NMS=True: phoneFN639/1.139;117 FN có detection cùng lớpIoU≥0,5 với score 0,05–<0,25 theo tiêu chí audit đã bàn giao. 496 là no_matching_detection trong quy tắc tìm candidate, **không đồng nghĩa không có output ở mọi score/head**. Dùng CSV/JSON và ảnh để phân biệt | Hạ threshold chỉ thử trên val, báo recall gain cùng FP/precision tradeoff. Không đảm bảo tăng mAP hoặc precision. Nhầm lớp cần kiểm GT; không có candidate cần xem size/blur/che khuất và dữ liệu |
| **Leakage:** val cao nhưng camera kém, nhóm lặp qua split | SHA/pixel hash, pHash/embedding candidates, source/session/video grouping, exposure của từng checkpoint | Val cao/camera kém chỉ là dấu hiệu, cũng có thể domain shift. Exact hash không phát hiện chuyển split chưa chứng minh độc lập cảnh. Kiểm toàn nhóm và lịch sử các ảnh đã bỏ khỏi release | Chặn nghiệm thu nếu holdout contaminated; version hóa migration, giữ lịch sử; không lặng lẽ chọn ảnh dễ khác |
| **Lỗi protocol/config:** điểm giữa báo cáo lệch mạnh | Release/hash/GT, class mapping cup/bottle,640/1280,NMS/head,conf/max_det/maxDets, phiên bản và actual_end2end | Kiểm8 ID có đúng thứ tự; cùng checkpoint/GT/order dưới cùngA hoặcB; AP50–95 là trung bìnhIoU, không phảiAP95; score confidence không phải accuracy | Sửa phép đo trước khi đụng model; vô hiệu hóa kết luận từ bảng không tương thích; ghi lại run metadata và baseline đúng |
| **Training instability:** NaN/Inf, loss spikes, dao động bất thường | Batch ảnh/nhãn, LR/optimizer/AMP logs, VRAM/OOM, gradient finite nếu instrument, curves gốc | Kiểm batch gây lỗi/box invalid, lịch warmup/LR, numerical precision; tái hiện smoke nhỏ khi được giao. Hiện chưa có bằng chứng run này bất ổn numerical | Dừng run không hợp lệ; sửa data trước; thử từng yếu tốLR/AMP/batch phù hợp. Không tự thay optimizer hoặc tắt AMP hàng loạt trong phiên tài liệu |

Train-subset chẩn đoán phải giữ đúng roletrain, chọn trước theo nguồn/size với seed cố định, không augmentation lúc chấm; ghi subset hash và tránh chỉ lấy ví dụ dễ. Mẫu train không dùng để nghiệm thu tổng quát. Có thể đo ở đầu/giữa/cuối để tiết kiệm thay vì inference toàn train mỗi epoch. Với underfit/overfit phải xét cả đường cong, chất lượng nhãn, miền và độ dao động seed; không dùng một heuristic thay kết luận có kiểm chứng.

**Artifact cần lưu cho mỗi run hợp lệ:** results.csv và curves gốc/smoothed có nhãn; train/val losses; AP50/AP50–95/P/R đủ 8 lớp, confusion matrix vàPR curves; FN/FP gallery trêntrain/val; GT/predictions/run/metrics; config đầy đủ và hash checkpoint/evaluator/release/manifest; seeds, thiết bị/thời gian/VRAM; breakdown source/scene/size/occlusion; latency/RAM khi đo vận hành. Slices size nên ghi cả COCO small/medium/large theo diện tích gốc và kích thước sau resize640 (ví dụ cạnh ngắn<16,16–32,>32 pixel), không trộn hai định nghĩa. Mỗi slice ghi mẫu số, nhóm và CI, tránh kết luận từ vài ảnh.

## 5. Tiêu chí nghiệm thu và phép đo

### 5.1 Ràng buộc đã chốt

1. Tổng đúng 15.000 ảnh duy nhất:12.000 train/2.500 val/500 test, bổ sung đúng 3.000 so với12.000 ảnh v0.4 lịch sử. Augmentation/crop sinh từ ảnh gốc không được tính là mẫu độc lập mới. Ghi cả số file và số ảnh gốc/nhóm để tránh phồng count.
2. Đúng tám ID cố định, cup+bottle→7; v03 giữ hợp đồng lịch sử cup. Hash và provenance theo phiên bản, không ghi đè release cũ.
3. Giữ release/split/báo cáo lịch sử bất biến; tạo release mới cho split mục tiêu. Không đưa test cũ vào train hoặc val cũ đã tuning vào test độc lập. Test cuối không train, không chọn threshold/hyperparameter, không khai thác lỗi để thu train tương ứng.
4. Không giảm độ khó holdout để đạt điểm. Mọi đổi annotation/split về sau cần phiên bản mới và lý do độc lập với việc tăng metric.
5. Full review 15.000 ảnh là điều kiện trước training: kiểm tự động đường dẫn/hash/format/ID/box100%, rồi review trực quan từng ảnh đủ 8 lớp, provenance và bất đồng được giải quyết. Bắt đầu với 15.000 ảnh v0.5 hiện có, review34topup đãthêm; chưa coi thao tác này đã hoàn tất.

### 5.2 Ba protocol tách biệt

| Mã đề xuất | Mục đích | Thiết lập và vai trò |
|---|---|---|
| A — đối chứng chất lượng | Mục tiêu 85/60 và sàn 75 | CPU 640, conf 0.001, NMS=True, iou0.70, max_det 300, augment=False, agnostic=False; pycocotools2.0.11, IoU 0.50:0.05:0.95, maxDets[1,10,100], all-area, thứ tự image_id cố định; ghi mapping category JSON và actual_end2end=False |
| B — vận hành | Chất lượng camera và latency | CPU 640, nms=False, xác nhận actual_end2end=True, batch 1; cùng tiền xử lý backend; conf vận hành khóa trên validation; thêm lượt conf 0.001 để báo AP head vận hành riêng; ghi rõ max_det/effective defaults và tracking bật/tắt |
| C — trainer | Chọn best mỗi lượt | imgsz 1280/NMS-free theo recipe; lưu đầy đủ CSV; không trộn điểm với A/B |

A giữ chính sách hộp dự đoán zero-area sau clip: bỏ và đếm; NaN/Inf gây lỗi. Metadata bốn lượt cũ ghi số bỏ33/25/94/71; lượt best mới trên v04 là 71. Không được bỏ GT khó/nhỏ vì model không tìm thấy. Evaluator cũ `kaggle_tools/evaluate.py` vẫn hard-code cup và có hành vi bỏ nonfinite; khi chuẩn hóa evaluator v04 phải bám protocol đã ghi ở run.json, không gọi thẳng luồng v03 rồi đổi tên kết quả. Lưu hash evaluator để tái lập.

AP dùng danh sách confidence thấp cố định để tích phân đường precision–recall; **confidence0.25 trên giao diện không phải độ chính xác model**. Tăng ngưỡng có thể đổi precision/recall vận hành nhưng không phải cách tạo ra mAP85%. Không có hệ số cố định đổi mAP50 sang mAP50–95; ví dụ51/71×95≈68 chỉ là phép minh họa, không là dự báo hay trần.

### 5.3 Ngưỡng đề xuất cần người dùng chốt

| Hạng mục [Đ] | Đề xuất ban đầu | Quy tắc báo cáo |
|---|---|---|
| Chất lượng A trên test đại diện đã khóa | mAP50≥85%; mAP50–95≥60%; AP50 cả 8 lớp≥75% | Báo đủ 8 lớp, n ảnh/n box/n nhóm; không thay test bằng val để tuyên bố đạt |
| Camera B, IoU≥0.5 | Micro precision≥90%, recall≥80%; mỗi lớp precision≥85%, recall≥75% | Cùng threshold đã chọn trên val; báo macro và micro, FP/FN theo cảnh; coi nhãn thiếu là vấn đề dữ liệu cần giải quyết trước |
| Background thật B | ≤0,10 FP/frame và ≤5% frame có ≥1 FP | Frame không có bất kỳ lớp nào trong 8 lớp; báo cả số nhóm cảnh; vật ngoài8 lớp vẫn là negative đối với hợp đồng này |
| Độ chắc chắn | CI95% bằng bootstrap theo nhóm cảnh, ≥1.000 lần resample | Không resample box như mẫu độc lập; thiếu nhóm hoặc CI rộng thì ghi chưa đủ bằng chứng, không làm tròn để đạt |
| Old-domain regression | So baseline v04 hiện tại: giảm không quá1 điểm mAP50–95 và không quá2 điểm AP50 của bất kỳ lớp 0–6 | Tiêu chí đề xuất; kiểm cả CI/seed. Backpack phải không tiếp tục giảm; vẫn nêu khoảng cách tới best v03. ID7 lịch sử báo riêng vì khác phạm vi |
| FPS/latency | Cân nhắc mức 5 FPS, p95 end-to-end≤300 ms hoặc mức 10 FPS, p95≤200 ms | Hai lựa chọn chưa chốt, chưa đo đạt; không coi preview_fps10 là inference10 FPS |
| Dung lượng | Ghi bytes checkpoint gốc, artifact deploy, RAM peak, dataset trên đĩa | Chưa có trần MB/RAM người dùng chốt; không tối ưu size bằng việc xóa bản gốc |

Threshold đề xuất thử trên validation camera:0.10/0.15/0.20/0.25/0.30/0.40/0.50; chọn theo ràng buộc P/R và FP, không chọn bằng test. Khởi đầu dùng một threshold chung; threshold theo lớp chỉ xét sau nếu có đủ val từng lớp và được version hóa. Báo reliability theo dải confidence nếu đủ mẫu, không mặc định score 0.9 là xác suất đúng 90%.

Mốc90/70 chỉ sau khi mốc85/60, sàn lớp và vận hành đạt. Mốc95/80 không có lịch cam kết. Nếu CI cắt qua ngưỡng, kết luận “điểm ước lượng đạt, độ chắc chắn chưa đủ”, không tự nghiệm thu.

## 6. Kế hoạch full review 15.000 ảnh và chất lượng nhãn

### 6.1 Pilot trước khi sửa hàng loạt

[Đ] Tạo **review queue chỉ đọc** từ manifest + error CSV/JSON. Dùng seed 42, lưu danh sách ID và phương pháp chọn. Phân tầng theo nguồn gốc (`source_id`, archive/version trong source_metadata_json), split, nhóm cảnh, kích thước box, mật độ, lớp và trạng thái. Hai input_group ở trên là tầng lớn, chưa thay cho từng nguồn thực. Rà15.000 ảnh hiện có ngay khi bước review được giao, đồng thời review34topup đãthêm; full review chưa được thực hiện trong lượt cập nhật tài liệu này.

- Pilot **400 ảnh ngẫu nhiên phân tầng** trong 7.081: bảo đảm mỗi nguồn có ít nhất 20 nếu số lượng cho phép; phân bổ phần còn lại theo quy mô/rủi ro, ghi xác suất chọn. Chọn theo nhóm để không lấy nhiều frame gần giống.
- Thêm **100 ảnh có chủ đích** ở train/val theo lỗi phone/chair/book và backpack miền cũ, không trùng 400 ảnh. Không dùng phần có chủ đích để ước lượng tỷ lệ lỗi toàn bộ nguồn.
- Nhánh **100 ảnh base approved** riêng: ưu tiên kiểm tính mâu thuẫn487 ghi chú pending, kèm mẫu approved không mâu thuẫn. Không tính100 ảnh này vào tiến độ7.081.
- Người rà test chỉ xem ảnh/GT/provenance theo rubric cố định, **không xem dự đoán test**. Tổng hợp cho nhóm model chỉ tình trạng đủ lớp/nguồn và QC; không chuyển ví dụ lỗi test thành yêu cầu thu train.
- Hai người rà độc lập ít nhất 20% pilot; hòa giải mọi bất đồng semantic, bỏ sót và ranh giới box. Chưa có người thứ hai thì ghi “single-review”, chưa qua cổng QA.

Không công bố sampling này đã được thực hiện: phiên này mới lập kế hoạch và đếm file nhãn, chưa rà trực quan500/600 ảnh.

### 6.2 Rubric áp dụng cho mọi ảnh

Quét đủ 8 lớp theo thứ tự cố định, cả vùng rìa/che khuất/vật nhỏ; không chỉ rà lớp nguồn gán. Ghi `missing_object`, `wrong_class`, `loose/tight_box`, `duplicate_box`, `uncertain_identity`, `ambiguous_visibility`, `true_negative`, `source_scope_mismatch`. Kiểm person/table/chair/laptop/phone/backpack/book/bottle trong cùng ảnh.

Chốt quy tắc box visible hay extent trước pilot; đề xuất box bao phần vật nhìn thấy, nhất quán với hướng dẫn và tập chuẩn đã hòa giải, không tự đoán phần vật bị che. Ghi truncated/occlusion và subtype ID7 trong sidecar. Với ảnh mờ quá mức không phân loại chắc, chuyển adjudication; không ép nhãn theo model, không tự xóa vật khó. Nếu rubric mới khác chuẩn nguồn, tạo migration note/version thay vì sửa riêng một split.

Riêng backpack: phân biệt balô với vali/túi xách; book: một cuốn hay chồng/kệ, gáy sách có tách được; phone: điện thoại với remote/calculator; chair: ghế so với sofa; bottle: không đổi lại cup ID7. Không coi ảnh là negative chỉ vì file nhãn trống: người rà phải xác nhận không có bất kỳ đối tượng hợp lệ thuộc8 lớp.

Review ledger ghi image_id, image/label SHA256 trước/sau, nguồn/archive và quyền sử dụng, scene/session/group, split cũ và split kế hoạch, người rà/thời gian/rubric version, lỗi, quyết định và lý do. Trạng thái: pending → reviewed → adjudicated → approved. Nghi ngờ chưa giải quyết không được đánh approved để đủ tiến độ.

### 6.3 Mở rộng và nghiệm thu review

Sau pilot, rà **toàn bộ 12.000 ảnh hiện có**, gồm 7.081 flagged và 4.919 approved; rà toàn bộ 2.966 ảnh mới còn giữ và34topup đãthêm. Mỗi ảnh trong 15.000 phải có lượt trực quan đủ 8 lớp; không dùng pilot hoặc trạng thái approved cũ để miễn rà. Chia lô 200–400 ảnh; ưu tiên nguồn đơn lớp và nhóm rủi ro phone/chair/book, sau đó backpack miền cũ và các lớp còn lại. QC độc lập10% mỗi lô +100% ảnh tranh chấp. [Đ] Nếu mẫu QC có >2% ảnh thiếu vật đúng hợp đồng, >1% sai lớp, hoặc bất kỳ lỗi hệ thống mapping: mở lại cả lô/nguồn liên quan; chưa kết luận các ảnh chưa xem đều lỗi. Các ngưỡng là đề xuất và cần hiệu chỉnh bằng pilot.

Definition of done: **15.000/15.000 ảnh** có kết quả kiểm tự động và review trực quan/provenance; mọi tranh chấp được quyết định; QC đạt rubric, trạng thái base pending được làm rõ. Kiểm tự động gồm ảnh/nhãn tồn tại, đọc được, kích thước, hash ảnh/nhãn thực tế, ID, tọa độ hữu hạn/box không suy biến và chính sách biên, trùng exact/pixel/near-duplicate và nhóm chia split. Đếm file hoặc format hợp lệ không đủ gọi full review. Nếu ảnh hỏng/không hợp lệ cần thay, phải ghi version và bù ảnh hợp lệ để vẫn đúng 15.000, không xóa âm thầm. Nhãn val/test được sửa theo cùng chuẩn, không theo điểm model; giữ bản lịch sử, chấm lại tất cả checkpoint so sánh trên **cùng bản nhãn mới**. Điểm trước/sau sửa GT phải tách bảng.

### 6.4 Công sức có giả định

| Việc | Giả định để lập ngân sách | Công sức đề xuất |
|---|---|---:|
| Rubric, queue và pilot | ~600 ảnh, ảnh khó nhiều hơn bình thường | 20–45 giờ-người tham khảo; phần review ảnh pilot đã tính trong tổng bên dưới, không cộng lặp |
| Nhóm7.081 cần review, nằm trong 12.000 ảnh | 1,5–4 phút/ảnh | 177–472 giờ-người; dùng để ưu tiên, không cộng thêm vào tổng 15.000 ảnh |
| Lượt review/sửa 12.000 hiện có | 1,5–4 phút/ảnh gồm quét đủ lớp và sửa cơ bản | 300–800 giờ-người |
| Review2.966merged_3k +34topup | Cùng giả định; chưa tính thu/chụp ảnh | 75–200 giờ-người |
| QC 10% =1.500 ảnh | 2–4 phút/ảnh | 50–100 giờ-người |
| Hòa giải/phân nhóm/provenance | Dự phòng15–25% thời gian review chính15.000 ảnh | 56–250 giờ-người |
| Tổng15.000 ảnh, tránh cộng pilot hai lần | Review + QC + hòa giải | ~481–1.350 giờ-người; cộng công thiết kế rubric/thu ảnh nếu chưa nằm trong khoản thực đo |

Đây là estimate, không phải thời gian đã tiêu. Với hai người, mỗi người4 giờ tập trung/ngày: khoảng 61–169 ngày làm việc cho481–1.350 giờ, chưa kể thu camera. Không hứa full review 15.000 ảnh trong vài ngày. Pilot sẽ đo median/p90 phút mỗi nguồn và cập nhật lại; cảnh kệ sách đông có thể vượt4 phút. Có thể dùng model gợi ý để tiết kiệm thao tác nhưng vẫn rà độc lập toàn ảnh, không tự nhận nhãn máy là đúng; với test không dùng prediction của model đang chọn để định hướng sửa GT.

## 7. Topup đã hoàn tất và kế hoạch kiểm chất lượng

[L/X] Đãghép2.966 ảnhmerged_3k còn giữ và34 ảnhphone_topup unique vào12.000 v04. Hiện **15.000 ảnh:12.000/2.500/500**, thiếu0. Topup34ảnh/53phonebox đượcghi ởselection.json. Đãkiểm30.000 hashpayload vàformat nhãn; chưa reviewtoàn bộ.

| Thành phần | Train | Val | Test | Tổng |
|---|---:|---:|---:|---:|
| V04 trướcmerge | 9.540 | 1.754 | 706 | 12.000 |
| Oldtest→val | 0 | +206 | −206 | 0 |
| Merged_3k giữ sau loại3pizza | 2.426 | 540 | 0 | 2.966 |
| Phone_topup | 34 | 0 | 0 | 34 |
| V05 hiện hành | **12.000** | **2.500** | **500** | **15.000** |

Khôngcòn nhiệm vụthu thêm34ảnh; chuyểnưu tiên sangfullreview15k. Mappingnguồn0book→6,1chair→2,2mobilephone→4 làhợpđồngremap đãghi, chưa chứngminh semanticlabel đúng. Truyvết nguồn bịpháthiệnpizza→chair; áp dụngexclusionledger đểkhôngnhập lại3ảnh đãloại. Bổsungthêm dữliệu trongtươnglai phảiđược thiết kếversion/budget riêng, không tựvượt15k hoặc bỏảnhkhó theo điểmmodel.

Source splitmoves629mục làlịch sử thờiđiểmmerge; khi đốichiếureleasehiệnhành phảiápdụng removal vàtopup.500test đềuoldtest,11bottleđược giữ; không đưaoldholdoutv03/v04 vào train. Nhómfilename/clip chỉđược kiểmheuristic, fullscene/near-duplicate chưahoànthành.

## 8. Holdout500 hiện hành, provenance và chống leakage

[X/L] Testv05 có500 ảnh,1.645 GT box, đều từtest cũ; bottle 11 ảnh/38 box được giữ nguyên. Coverage hiện tại:

| Lớp | Ảnh test chứa lớp | GT box |
|---|---:|---:|
| person | 40 | 287 |
| table | 25 | 66 |
| chair | 31 | 38 |
| laptop | 58 | 101 |
| cell phone | 223 | 454 |
| backpack | 116 | 133 |
| book | 106 | 528 |
| bottle gộp | 11 | 38 |

So với mốc đề xuất≥50 ảnh/lớp và≥100 box, person/table/chair/bottle còn thiếu ít nhất một điều kiện; chưa có negative thật đã xác minh. Test 500 đúng số lượng nhưng chưa đủ chứng cứ nghiệm thu mọi lớp/cảnh.Topup34 ảnh đã vào train, không thay đổi coverage test. Nếu sau full review vẫn thiếu, báo giới hạn và trình kế hoạch holdout đại diện ở release tương lai với migration/count rõ; không tự thay test hiện tại hoặc loạiảnh khó theo metric.

1. Lập exposure history cho mọi checkpoint: ever_train, ever_val_tuned, test_reserved, pretraining_overlap vàscene/session_group, kể cảảnh đãloại khỏirelease.
2. Kiểm lại500 test vànhóm liên quan: không oldtrain/oldval đã tuning→test; không test cũ→train.206 test cũ chuyểnval đã làdữliệu phát triển cho các lượt sau, không đưa ngược lại test rồi gọi chưa tuning.
3. Kiểm exact byte/pixel, perceptualhash/embedding vàscene/session; filename/clip grouping làheuristic, **full scene/near-duplicate independence chưa xác minh**. Hiệu chỉnh similarity trên cặp mẫu, rà candidate vàconnectedcomponent; không mặc định mọi cặp gầnnhau làduplicate.
4. Nếu phát hiệnleakage: giữrelease/báocáo lịch sử, ghi conflict vàchặn tuyênbố independent; trìnhmigration cólýdo dữliệu, không dùng điểmmodel đểchọn test đẹp. Nhóm nguồn mới có split gốc test/valid cần ghi provenance vàkiểm rằng chưa bị dùng làmholdout tuning của dựán.
5. Ghi overlapCOCOpretrained vàcoveragecamera tựthu; chưa fine-tune trên test không đồngnghĩa pretrained chưa từng thấyảnh. Train lại từpretrained không tự chữa overlap này.

Testreview chỉ sửaGT theo rubric độclập, không dùng predictionmodel đang phát triển đểđịnhhướng sửa/chọntrain/tuning. Mọi chỉnhGT cóversioning vàhash; chỉkhóa test sauQA/scene audit trước tối ưu. Nhóm training nhậncoverage/QC, không nhận danh sáchlỗi modeltest đểthutraining.

Baseline70,706%/50,694% vẫn làv04val1.754/GTcũ, **không làmetricv05**. Sauclean vàkhóarelease, tạo baseline mới trên val 2.500/GTmới bằngprotocolA, chấm checkpoint hiện tại vàcandidate cùngGT/hash. mAP50–95 làtrungbìnhIoU0.50…0.95, không phảiAP95 riêng. Chỉ mở test 500 đểnghiệm thu saukhóamodel/threshold; test thấtbại thìbáoevidence vàphạmvi vòng sau, không khai thác lỗi đểtrain rồi coi test chưa thấy.

## 9. Thí nghiệm tuần tự tiết kiệm GPU

### 9.1 Điều kiện vào

Chỉ thực hiện sau khi được giao triển khai: **full review 15.000 ảnh đã qua QC**, split mới 12.000/2.500/500 và additions khóa, release/split lịch sử được bảo tồn, không có leakage unresolved trong tập dùng nghiệm thu, rubric 8 lớp thống nhất, storage/GPU budget và tiêu chí vận hành được chốt. Không chạy local Training/export hiện tại để giả định đã train v04: các luồng đó còn v03 cup.

| Bước | Thay đổi/chủ đích | Đối chứng và cổng đi tiếp |
|---|---|---|
| E0 — baseline | Chấm checkpoint hiện tại trên val bản nhãn/release mới, protocol A và B; giữ bảng v03/v04 lịch sử | Không train; dùng đúng GT mới cho mọi ứng viên. Chưa mở test. Ghi AP/P/R/nguồn/size/occlusion, CPU latency |
| E1 — corrected-data YOLO26n | Fine-tune từ best v04 trên train đã sửa, recipe20 epoch cố định | So E0 để biết hiệu quả gói cập nhật; không quy toàn bộ gain cho sửa nhãn vì cũng thêm tối ưu và có thể thêm ảnh |
| E1a — chỉ khi cần chứng minh tác động dữ liệu | Một cặp train cùng init/seed/recipe/split, chỉ thay tập nhãn/ảnh train được định nghĩa trước | Chi phí tùy chọn; không cần train lại trên nhãn đã biết sai chỉ để có biểu đồ. Nếu bỏ control, ghi rõ giới hạn nhân quả |
| E2 — một ablation nano có evidence | Chỉ chọn một yếu tố: augmentation hoặc imgsz hoặc LR/lịch; giữ yếu tố khác | Tối đa1 lượt sàng lọc ban đầu; chọn theo lỗi val, không grid search tất cả. Resize inference khác 640 là protocol phụ, không ghi vào cột640 |
| E3 — n/s pretrained nếu còn nút thắt | So YOLO26n pretrained và YOLO26s pretrained cùng train/val, recipe, seed, budget | Nếu chưa có n khởi đầu pretrained tương ứng, cần run n-fresh trước s để không nhầm lợi ích architecture với init. Warm-start v04 n so s pretrained chỉ là so ứng viên thực dụng |
| E4 — kiểm độ lặp | Lặp ứng viên và đối chứng trực tiếp bằng seed 43,44 ngoài seed 42 | Báo trung bình/độ lệch chuẩn và từng seed; không chọn seed thắng để phóng đại. Nếu budget chỉ1 seed, kết luận sơ bộ |
| E5 — khóa và nghiệm thu | Chọn checkpoint/threshold trên val theo quy tắc trước; đo CPU; mở test cuối | Đạt tất cả cổng mới promote; không lấy best test để chọn model |

Nano `.pt` không tự chuyển thành kiến trúc small khi đổi tên file. Small phải khởi tạo từ **`yolo26s.pt` pretrained**, xác minh kiến trúc/class head và nguồn/hash. [Tài liệu YOLO26 chính thức](https://docs.ultralytics.com/models/yolo26/) liệt kê riêng n/s và head NMS-free; benchmark công bố không phải tốc độ ứng dụng này.

### 9.2 Recipe khởi điểm — giả thuyết, không phải lệnh đã chạy

E1 đề xuất: `epochs=20, resume=False, imgsz=1280, batch=32` nếu cùng phần cứng đủ VRAM, `workers=2, cache=False, optimizer=MuSGD, lr0=0.001, lrf=0.01, mosaic=0.5, close_mosaic=5, seed=42, deterministic=True, cls_remap=False`. Giữ các tham số còn lại và phiên bản bằng cấu hình lưu đầy đủ, không dựa vào default đang đổi. `patience=20` cho pilot20 epoch để thường có đủ 5 epoch cuối đóng mosaic; dừng ngay nếu lỗi dữ liệu/NaN/OOM không xử lý được. Đây là thay đổi lịch có chủ đích so với run cũ, phải được ghi chứ không gọi resume.

Một smoke1 epoch chỉ để kiểm pipeline/mapping/ảnh augmented sau này, không tính vào run chính; khởi tạo lại đúng checkpoint ban đầu cho run20. Smoke đo VRAM, sec/epoch và xác nhận chế độ GPU thực tế. Nếu batch 32 không phù hợp, chọn batch an toàn và khóa trước cặp thí nghiệm; thay batch giữa n/s phải ghi hạn chế so sánh và effective batch/accumulation. Không mặc định hai GPU nhanh gấp đôi.

MuSGD/LR trên là điểm xuất phát theo lịch sử, chưa chứng minh tối ưu cho s pretrained. Khi muốn thử optimizer/LR khác, đó là một ablation riêng, không đồng thời đổi kiến trúc, data và augmentation rồi quy gain cho một yếu tố. Nếu đường cong còn tăng và ngân sách cho phép, đề xuất lượt mới 30 epoch, close_mosaic 5, patience 10; không tự kéo dài vô hạn hoặc sửa lịch giữa run để che lần dừng sớm.

`close_mosaic` là số epoch cuối tắt mosaic; `resume=True` phục hồi trạng thái lượt trước, còn `resume=False` bắt đầu lượt mới. Cần đối chiếu version cài thực tế trước khi chạy. [Ultralytics Train](https://docs.ultralytics.com/modes/train/).

### 9.3 Ngân sách, dừng và quay lui

[Đ] Gói sàng lọc ban đầu: tối đa60 epoch chính (E1=20, một E2=20 nếu có lý do, một lượt tiếp theo=20); nếu cần cặp n-fresh/s-fresh và seeds sẽ lập ngân sách bổ sung rõ ràng, không giả vờ mọi bước vừa trong 60. Khóa đồng thời trần wall-clock/GPU-hours sau smoke; không suy GPU-hours từ `device=0,1`. 79,47 phút/20 epoch lịch sử chỉ là tham khảo; đo lại trước dự toán tiền/thời gian.

- Chọn best theo mAP50–95 validation trainer trong từng run; chọn giữa ứng viên bằng A/B và các guardrail đã định, không bằng test. Nếu đổi checkpoint-selection sang protocolA cần khóa quy tắc và ngân sách chấm trước.
- [Đ] Chỉ coi là tiến bộ đủ để đầu tư tiếp nếu mAP50–95 A tăng ≥1 điểm hoặc AP50 lớp ưu tiên tăng ≥3 điểm, không vi phạm old-domain/camera guardrail; chênh nhỏ hơn cần CI/seeds, không gọi chiến thắng chắc chắn.
- Dừng mở rộng khi hai lượt có giả thuyết rõ không cải thiện qua nhiễu đo, vi phạm budget hoặc dữ liệu còn lỗi hệ thống. Không kết luận overfit trừ khi nhiều epoch/seeds cho thấy train tốt dần nhưng val xấu bền vững, đã loại yếu tố dữ liệu/protocol.
- Khi lỗi mapping, hash, split hoặc nonfinite xuất hiện: run không hợp lệ, không promote. Giữ artifact và lý do; rollback ứng dụng về alias/checkpoint được xác minh trước nếu một lần deploy tương lai gặp lỗi.
- Nếu85/60 không đạt dưới budget: công bố bảng thất bại và khoảng cách từng lớp, đưa lựa chọn giảm phạm vi cảnh được hỗ trợ, tăng model/compute chấp nhận chậm, ưu tiên thêm review trong giới hạn ảnh, hoặc tăng budget lượt sau. Thu hẹp phạm vi sản phẩm phải công khai và có bộ đánh giá phiên bản riêng; không bỏ ảnh khó khỏi holdout hiện tại để nâng điểm.

## 10. Kiểm CPU backend và triển khai có rollback

[X] `configs/app.yaml` chọn aliasbest, CPU 640/nmsFalse, display_conf0.25; `src/inference/detector.py` gọi predict CPU 640/nmsFalse và chấp nhận hai ordered8profiles. `src/ui/app.py` đọc pointer `configs/dataset_current.json` để lấy manifest v0.5 (fallback v0.4 nếu chưa có pointer), bằngutf-8-sig, lấy50 ảnh mẫu train, đường dẫn ảnh/thumbnail theo dataset root; đây là tiện ích demo, không là bộ test. `src/app.py` là entrypoint Streamlit.

[X] `src/dataset.py`, `src/training/worker.py`, `src/training/kaggle_export.py` vẫn giữ legacy cup. Trang training có cảnh báo v03; notebook v04 tách riêng. `COCO_TO_PROJECT` hiện vẫn mapping cup41→7 lịch sử, không tự khẳng định baseline COCO80 đã gộp bottle 39 vào lớp 7. Khi làm baseline merged pretrained phải định nghĩa mapping đúng và version hóa riêng. Không tuyên bố toàn bộ ứng dụng đã chuyển train v04.

[Đ] Đo trong phiên riêng được giao, không ảnh hưởng backend đang dùng: cùng CPU/threads/OS/package/power mode, cùng input640 và video/camera, warmup 30 frame, ít nhất 300 frame mỗi kịch bản hoặc 60 giây nếu dài hơn, lặp3 lần. Tách detection-only và pipeline thật có tracking/rendering. Dùng đồng hồ monotonic, ghi capture timestamp, enqueue/dequeue và display time.

Ghi p50/p95 latency model, decode/preprocess/postprocess, end-to-end; processed FPS, displayed FPS, dropped/skipped frames, frame age và RAM peak. Throughput=n frame hoàn thành/thời gian đo, không lấy1/latency model làm FPS toàn ứng dụng. Kiểm live_latest/queue để hệ thống không đạt FPS bằng cách hiển thị ảnh cũ. Preview10 FPS trong config chỉ là cấu hình hiển thị.

Chấm cả baseline và candidate đúng **head 640/NMS-free**; không chuyển điểm COCO640/NMS=True thành điểm vận hành. Ngưỡng tracker_input_conf0.10 khác display_conf0.25: báo rõ detection được tính ở ngưỡng nào. Tracking/đếm người cần GT và tiêu chí sự kiện riêng nếu mở rộng scope, chưa nằm trong nghiệm thu AP detection.

Trình tự promote tương lai: kiểm ordered8mapping/subtype contract → hash weights/manifest/config → smoke ảnh/video/camera đủ 8 lớp → parity bản deploy → benchmarkCPU và bộ test đã khóa → lưu model card → sao lưu alias/config hiện hành → chuyển alias có kiểm checksum → health/smoke sau chuyển → rollback nếu regression. Bản rollback ngay trước candidate là v04 hiện tại; v03 giữ thêm để đối chứng, tránh vô tình đổi phạm vi ID7 khi rollback. Không strip/checkpoint overwrite bản gốc. Export ONNX/OpenVINO hoặc bản bỏ optimizer là nhánh tối ưu riêng sau khi xác minh parity A/B, không tự triển khai trong phiên này.

## 11. Milestone, deliverable và Definition of Done

Lịch theo **cổng hoàn thành**, không hứa xong review hàng nghìn ảnh trong một tuần. Ngày bắt đầu triển khai chưa chốt.

| Mốc | Deliverable | Phụ thuộc | Điều kiện ra cổng | Người đảm nhiệm đề xuất |
|---|---|---|---|---|
| M0 — hồ sơ hiện trạng | Tài liệu này, checklist và biên bản kiểm chứng | Không | Số liệu có nguồn; quyết định split mới nhất đúng; không đổi dataset/runtime | Người lập kế hoạch + người dùng review |
| M1 — pilot/rubric | Queue, rubric 8 lớp, ledger pilot, estimate cập nhật | M0 và giao triển khai review | 400 random+100 targeted, nhánh100 base; sai khác reviewer hòa giải; chưa dùng test predictions | Người phụ trách dữ liệu + reviewer |
| M2 — dữ liệu đạt chuẩn | Full review 12.000 cũ +3.000 mới, QC, grouping/exposure report | M1 | 15.000/15.000 đã kiểm tự động và trực quan đủ 8 lớp; không còn lỗi hệ thống/tranh chấp; provenance đầy đủ | Nhóm dữ liệu |
| M3 — release và holdout khóa | Manifest/hash/label version/data card, migration đã thực hiện và coverage val/test | M2 và bổ sung34 train độclập | Đúng 12.000/2.500/500; release lịch sử bất biến; test cũ không sang train, val cũ không sang test; nhóm độc lập và hash đã khóa | Dữ liệu + người đánh giá |
| M4 — baseline và nano | E0/E1, bảng A/B/C, lỗi nguồn/size/occlusion | M3, budget/metric chốt | Run hợp lệ, giả thuyết và quyết định đi tiếp minh bạch | Người training/evaluation |
| M5 — lựa chọn model | Ablation/small nếu cần, seeds, CPU benchmark | M4 | Lợi ích vượt nhiễu hoặc nêu chưa đủ; không vi phạm guardrail | Người training + ứng dụng |
| M6 — nghiệm thu | Test report, model/data card, quyết định đạt/chưa đạt | M5, model/threshold khóa | Đủ8 lớp và camera/background, P/R/latency/size; kết quả không bị tuning bằng test | Người đánh giá + người dùng |
| M7 — phát hành | Bản deploy có hash, alias/config và rollback | M6 đạt, được giao deploy | Smoke/health/parity đạt; rollback kiểm được | Người ứng dụng |

**DoD cho nhiệm vụ hiện tại:** tài liệu tiếng Việt đầy đủ, đối chiếu report/code, tách số đo/giả thuyết/mục tiêu, bảng8 lớp, kế hoạch review/budget/split/test/training/deploy và link review; chưa thực thi các bước sau M0.

**DoD cho sản phẩm:** vượt các ràng buộc đã chốt và ngưỡng nghiệm thu cuối cùng, test sạch đủ chứng cứ, repeatability/old-domain/CPU vận hành có số đo, provenance và rollback đầy đủ. Nếu còn thiếu coverage/CI hoặc ngưỡng FPS chưa chốt thì ghi trạng thái đó, không đánh dấu hoàn thành chỉ vì mAP tổng cao.

## 12. Rủi ro và các quyết định cần chốt

| Rủi ro/tradeoff | Hệ quả | Cách xử lý đề xuất |
|---|---|---|
| Review dài hơn training nhiều lần | Chậm tiến độ | Pilot đo tốc độ, ưu tiên nguồn/lớp bằng evidence; không bỏ QA để chạy nhanh |
| Phát hiện leakage cũ trong kế hoạch test 500 | Không đủ cơ sở test độc lập | Giữ lịch sử và chặn tuyên bố nghiệm thu; chọn migration theo nhóm có lý do, không theo metric |
| Test chỉ500 ảnh, bottle cũ ít | CI rộng, thiếu cảnh/subtype | Báo coverage test500 hiện hành; nếu cần thay đổi holdout trong release tương lai phải có thiết kế camera/nhóm độc lập và migration rõ, không tự đổi theo metric |
| Nguồn pretrained trùng nguồn test | Điểm lạc quan | Data card overlap, camera tự thu độc lập, báo theo nguồn |
| Higher resolution/small model | Có thể tăng chất lượng nhưng tăng RAM/latency | Mỗi thay đổi là thí nghiệm riêng; so đúng CPU 640 vận hành |
| Bản nhãn mới đổi metric | Dễ nhầm model tiến bộ | Re-evaluate mọi checkpoint trên cùng GT mới; tách bảng lịch sử |
| Cup+bottle rộng | AP gộp che một subtype yếu | Báo coverage và lỗi subtype trong sidecar; không đổi8 ID |
| Confidence cao nhưng bỏ sót nhiều | Demo trông sạch nhưng thiếu vật | PR curve và FN/FP theo camera; threshold khóa trên val |
| Trần ảnh không phải trần dung lượng | File lớn/cache/checkpoints vẫn nặng | Đo bytes thực, giữ manifest/nhãn và phiên bản; không tự xóa dữ liệu gốc |

Người dùng cần chốt khi chuyển sang triển khai: (1) phạm vi camera/cảnh/khoảng cách cần hỗ trợ và cách xử lý coverage test còn thiếu sau full review; (2) chấp nhận ngưỡngA85/60 và sàn 75, các ngưỡngP/R/background/old-domain ở mục 5 hay điều chỉnh; (3) chọn mức tốc độ5 FPS/10 FPS hoặc mục tiêu khác và máy đo; (4) GPU-hours, số lượt/seed, thời gian review và người hòa giải; (5) trần MB artifact/RAM nếu có. **Không hỏi lại target 15.000 và split 12.000/2.500/500 hoặc full review**, vì đã chốt. Các câu còn lại không chặn việc hoàn thành tài liệu hiện tại.

**Bước tiếp theo có thể làm mà không train hoặc đổi dataset:** lập review queue chỉ đọc và mẫu rubric/ledger, kèm bảng provenance/nhóm cần kiểm từ manifest và saved error audit. Không tự đánh dấu ảnh reviewed, không tự sửa nhãn. Sau đó thực hiện pilot/full review trên 15.000 ảnh hiện có và cập nhật công sức bằng số đo thật; topup đã hoàn tất, không còn cần thu thêm34 ảnh.

## 13. Nguồn đối chiếu và hồ sơ tái lập

- [Báo cáo merge v05](../reports/results/dataset_merge_v05_20261008/REPORT.md), [summaryv05](../reports/results/dataset_merge_v05_20261008/summary.json), [629 splitmoves](../reports/results/dataset_merge_v05_20261008/split_moves.json), [overlap ZIP phone](../reports/results/merge_3k_intake_20261008/replacement_zip_overlap.json), [verification v05](../reports/results/dataset_merge_v05_20261008/verification.json), [lỗi nhãn trực quan](../reports/results/dataset_merge_v05_20261008/visual_label_issues.json), [ảnh mapping mẫu](../reports/results/dataset_merge_v05_20261008/label_mapping_samples.jpg), [ledger loại3 pizza](../reports/results/dataset_merge_v05_20261008/removed_pizza_images.json).
- [Manifestv05](../data/dataset/project8_v0.5_merged_20261008/manifest.csv), [pointerdataset](../configs/dataset_current.json), [data_v05](../configs/data_v05.yaml), [đối chiếu cập nhậtv05](verification_v05_20261008.json).

Các đường dẫn bên dưới tương đối với thư mục tasks để tài liệu đi cùng workspace:

- [Báo cáo chính lượt 20 epoch](../README_DA_HOAN_THANH/reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/REPORT.md).
- [Audit](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/audit.json), [comparison bốn phép chấm](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/comparison.json), [results.csv](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/results.csv), [args.yaml](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/args.yaml).
- [Run metadata v04 best](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/after_on_v04_val/run.json); cùng thư mục có GT/predictions/metrics, ba phép chấm còn lại nằm cạnh.
- [Error report](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/error_audit_conf025/REPORT.md), [summary](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/error_audit_conf025/summary.json), [CSV ảnh/lớp](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/error_audit_conf025/image_class_errors.csv), [vị trí box](../reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/error_audit_conf025/images.json).
- [Manifest v04](../data/dataset/project8_v0.4_merged_20261008/manifest.csv), [data config](../configs/data_v04.yaml), [app config](../configs/app.yaml), [hợp đồng lớp](../src/classes.py), [detector](../src/inference/detector.py), [UI](../src/ui/app.py), [entrypoint](../src/app.py).
- [Dataset legacy](../src/dataset.py), [worker legacy](../src/training/worker.py), [exporter legacy](../src/training/kaggle_export.py), [training UI](../src/ui/training.py), [evaluator Kaggle](../kaggle_tools/evaluate.py), [protocol lịch sử](../README/EVALUATION_PROTOCOL.md).
- [Checklist công việc](todo.md), [biên bản kiểm chứng phiên lập kế hoạch](verification_20261008.json).

Đối chiếu hướng dẫn chính thức ngày 08/10/2026: [YOLO26](https://docs.ultralytics.com/models/yolo26/) về n/s và head; [Train](https://docs.ultralytics.com/modes/train/) về resume/close_mosaic. Tài liệu online có thể đổi default; run metadata/version cài và code mới là chuẩn tái lập số liệu dự án. Mọi quota, ngưỡngQA, thời gian review và recipe tương lai ở đây là đề xuất của kế hoạch, không phải bảo đảm từ nhà cung cấp.


### Cập nhật Kaggle theo ủy quyền mới

08/10/2026: người dùng cho phép chạy thử trước khi full review hoàn tất. Chuẩn bị notebook v0.5 và gói train/val + best v0.4; chỉ smoke 1 epoch với fraction=0.05, không gọi full run 20 epoch khi gate review chưa đạt. Không đổi nhãn, split, checkpoint chính hoặc ghi dữ liệu đã reviewed.
