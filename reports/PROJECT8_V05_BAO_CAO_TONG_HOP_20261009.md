# Báo cáo tổng hợp Project8: kết quả huấn luyện và hướng phát triển

**Ngày:** 09/10/2026  
**Phạm vi:** Tổng hợp những nội dung đã trao đổi về YOLO26n, kết quả v0.4/v0.5, tải trọng số, learning rate, augmentation, fine-tune hai giai đoạn, vật thể nhỏ, SAHI, ensemble, dữ liệu khó, YOLO26s, distillation và RF-DETR.

> **Cập nhật mới nhất — phần 25:** mục tiêu trước mắt theo yêu cầu mới là **80% mAP50**. Hai lượt chạy bù đã xong và được chấm chung: baseline 74,710%; giảm LR 74,829%; augmentation nhẹ 74,318%. Đã thử inference 1536, lọc box trùng và audit ảnh val/train. L1A1/L2A0 đã hoàn tất, đạt 75,484%/75,423%. BEST gốc + lọc box trùng đạt 76,715% trên đủ2.500 val bằng metric Ultralytics local; trọng số mới + NMS chưa được chấm. Trạng thái chi tiết ở phần25. Các phần cũ là lịch sử/kế hoạch tại thời điểm viết. Phần 14–16 chưa cân nhắc; phần 17 để đọc sau.

> Các bảng kết quả dưới đây là số liệu đã đọc từ log hoặc tệp kết quả. Các cấu hình thử nghiệm và ngưỡng lựa chọn là đề xuất, chưa phải kết quả đã chạy. Đã tải/kiểm tra trọng số, thống kê dữ liệu, chạy inference local và chỉnh notebook Kaggle. Lượt đối chứng GPU đã hoàn thành 2 epoch; hai nhánh còn thiếu được chuẩn bị chạy lại sau lỗi kết nối phiên; không dùng tập test để tuning.

**Cách đọc:** phần 1–6 ghi kết quả đã có; phần 7–13 là các hướng bạn đang cân nhắc; phần 14–17 để sau; phần 18–21 chốt lịch thử, bảng ghi kết quả và quyết định.

## 1. Mục tiêu và điều kiện hiện tại

Mục tiêu ban đầu bạn nêu là **85% mAP50**. Yêu cầu mới đặt **80% mAP50 làm mục tiêu trước mắt**, ưu tiên những việc làm được nhanh và giữ mô hình đủ nhẹ. Đây là mAP trung bình của tám lớp, không phải yêu cầu từng lớp đạt 80%, cũng không phải mục tiêu 80% mAP50–95. Những phép tính tới 85% dưới đây giữ lại để ghi lịch sử thảo luận; quyết định mới xem phần 25.

Theo AP50 lịch sử ở phần 5, khoảng cách từ 74,8% tới 85% là khoảng **10,2 điểm phần trăm**. Nếu giả định chair và cell phone đạt 100% còn sáu lớp kia giữ nguyên, mAP50 chỉ khoảng **83,6%**. Đây là phép tính minh họa, không phải dự đoán đạt 100%; cần cải thiện thêm các lớp khác ngoài hai lớp yếu.

Các điều kiện cần giữ trong kế hoạch:

- Hiện chưa thể thu thập dữ liệu để train từ camera thực tế.
- Bạn đã thử làm ảnh nét hơn và chưa thấy thay đổi đáng kể.
- Bạn nhận thấy điểm số đang đứng yên, muốn ưu tiên điều chỉnh nhỏ trước khi chuyển sang mô hình lớn.
- Bạn cho rằng cách chia train/validation hiện tại khá hợp lý. Có thể giữ cách chia để so sánh nhất quán, đồng thời kiểm tra tính độc lập giữa các cảnh và chất lượng nhãn.
- Bạn đã yêu cầu “tải trọng số về thôi, cái khác từ từ”. Sau đó bạn đã yêu cầu chạy các thử nghiệm đơn giản; trạng thái và kết quả thực tế cập nhật ở phần 22–24.

**Đánh giá hiện tại:** v0.5 best đạt khoảng **74,8% mAP50**. Khoảng cách đến mục tiêu là **10,2 điểm phần trăm**. Việc điểm ít tăng trong một lượt ngắn chưa chứng minh YOLO26n đã đạt giới hạn kiến trúc; cũng chưa có bằng chứng rằng chỉ giảm learning rate hoặc augmentation sẽ đủ đạt 85%.

## 2. Công việc đã hoàn thành

### 2.1. Tải và kiểm tra trọng số v0.5

| Tệp trong dự án | Kích thước | Trạng thái |
|---|---:|---|
| `weights/kaggle/BEST_Project8_v05_20261009.pt` | 5.544.517 byte | Đã tải, kiểm tra SHA256 và nạp được bằng Ultralytics |
| `weights/kaggle/LAST_Project8_v05_20261009.pt` | 5.544.517 byte | Đã tải, kiểm tra SHA256 và nạp được bằng Ultralytics |

SHA256:

```text
BEST v0.5
92adfeb2516861a6958e8adc4a15cf764faa3956078da1ebc81db1fbb7014e31

LAST v0.5
328d5f7e80f3cf0b32f02c61fb37821345d60f4b2182d75902b784621a4db25f
```

Cả hai checkpoint được nhận diện là mô hình detection, với đúng tám lớp theo thứ tự:

```text
0 person
1 table
2 chair
3 laptop
4 cell phone
5 backpack
6 book
7 bottle
```

Các trọng số mặc định của ứng dụng chưa được thay thế. Việc nạp được checkpoint xác nhận tệp sử dụng được ở mức cấu trúc; chưa xác nhận độ chính xác trong ứng dụng hoặc tốc độ chạy camera.

### 2. 2. Những việc chưa hoàn thành

- Đã thay nhánh smoke gây lỗi trong notebook Kaggle bằng pipeline ba pilot; chưa ghi nhận pipeline mới hoàn tất.
- Ba pilot GPU đã bắt đầu; chưa có kết quả cuối tại thời điểm ghi trạng thái. Kết quả thật sẽ bổ sung ở phần 23.
- SAHI đã chạy pilot inference 300 ảnh (phần 22). Ensemble, YOLO26s, distillation và RF-DETR chưa chạy.
- Chưa kiểm tra trực quan các mẫu FP/FN của hai lớp yếu.
- Chưa có so sánh v0.4 và v0.5 trên cùng một tập đánh giá, cùng một quy trình, trong phiên này.

Đã kiểm tra hash của 500 ảnh test và 500 tệp nhãn theo manifest. Một lần chuẩn bị đánh giá test bị dừng ở bước tạo cấu hình tạm, trước khi model chạy inference. Vì vậy **chưa có mAP test mới** và không được diễn giải việc kiểm tra hash thành kết quả đánh giá mô hình.

## 3. Dữ liệu đang sử dụng

Dataset hiện tại là `project8_v0.5_merged_20261008`:

| Split | Số ảnh | Số bounding box |
|---|---:|---:|
| Train | 12.000 | 33.272 |
| Validation | 2.500 | 8.359 |
| Test | 500 | 1.645 |
| **Tổng** | **15.000** | **43.276** |

Manifest SHA256:

```text
22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd
```

Kiểm tra hash và cú pháp nhãn của 14.500 cặp ảnh/nhãn train + validation đã qua preflight trên Kaggle. Tuy nhiên, trạng thái dự án vẫn ghi **chưa hoàn thành kiểm tra trực quan toàn bộ** và **training_ready=false**. Hai mức kiểm tra này khác nhau: cú pháp hợp lệ không đảm bảo nhãn đúng vật thể hoặc đầy đủ.

Các điểm cần lưu ý khi đọc kết quả:

- Lớp ID 7 mang tên `bottle` đang gộp dữ liệu cup + bottle; AP của lớp này không phải AP riêng cho chai.
- Test chỉ có 11 ảnh và 38 box thuộc lớp này, nên độ đại diện cho lớp còn hạn chế.
- Ảnh khác nhau ở mức pixel chưa chắc độc lập về cảnh: nhiều khung hình cùng cảnh có thể rất giống nhau.
- Train scan ghi nhận 0 ảnh background. Đây là dấu hiệu nên kiểm tra khả năng báo nhầm trên nền, chưa phải bằng chứng dataset sai.
- Trước đây đã phát hiện một lỗi ánh xạ ngữ nghĩa pizza/collage thành chair và loại ba ảnh theo phạm vi đã được cho phép. Việc sửa ba ảnh không chứng minh mọi nhãn còn lại đã sạch.
- 500 ảnh test được giữ lại từ tập test cũ; 206 ảnh test cũ đã chuyển sang validation trong lần tổ chức dữ liệu trước. Cần ghi nhận lịch sử này khi mô tả nguồn gốc split.

Trong các vòng phát triển tiếp theo, dùng validation để lựa chọn cấu hình. Giữ test cho đánh giá cuối, tránh sử dụng test liên tục để chỉnh tham số.

## 4. Kết quả các epoch và ý nghĩa của “best”

### 4.1. v0.4: epoch tốt nhất tùy theo chỉ số

Run: `project8_v04_12k_from_best50_20261008_151936`. Run được lên kế hoạch 50 epoch nhưng dừng theo yêu cầu người dùng sau 20 epoch; không phải một lượt đã train đủ 50 epoch.

Số liệu trong CSV của trainer ở imgsz 1280:

| Epoch | Precision | Recall | mAP50 | mAP50–95 | Ý nghĩa |
|---|---:|---:|---:|---:|---|
| 18 | 78,080% | 67,979% | **75,415%** | 52,792% | mAP50 cao nhất trong run |
| 19 | 77,882% | 67,352% | 75,356% | **53,488%** | Checkpoint được trainer chọn làm best |
| 20 | 76,166% | 68,194% | 75,024% | 53,004% | Epoch cuối đã chạy |

Trong phiên bản Ultralytics đã kiểm tra, fitness detection chọn theo mAP50–95. Vì vậy **best.pt không nhất thiết là checkpoint có mAP50 cao nhất**, dù mục tiêu của bạn là mAP50. Tên checkpoint theo chỉ số bắt đầu từ 0 cũng có thể gây nhầm: `epoch18.pt` tương ứng epoch 19 hiển thị trong bảng.

Trọng số v0.4 dùng làm khởi tạo cho v0.5 có SHA256:

```text
d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb
```

### 4. 2. v0.5: best ở epoch 2, dừng sau epoch 5

Run: `project8_v05_15k_from_best_v04_20261009_012407`, Kaggle Version 3.

| Tham số | Giá trị thực tế trong run |
|---|---|
| Mô hình | YOLO26n, khởi tạo từ best v0.4 |
| imgsz | 1280 |
| batch | 32 tổng trên hai GPU |
| GPU | T4 × 2 |
| Epoch kế hoạch / thực tế | 25 / 5 |
| patience | 3 |
| optimizer | MuSGD |
| lr0 / lrf | 0,001 / 0,01 |
| cos_lr | False |
| warmup_epochs | 3 |
| mosaic / close_mosaic | 0,5 / 5 |
| scale / translate | 0,5 / 0,1 |
| fliplr / flipud / degrees | 0,5 / 0 / 0 |
| mixup / cutmix | 0 / 0 |
| workers / cache | 2 / False |
| seed / deterministic | 42 / True |
| resume | False |

Log xác nhận best ở epoch 2 và early stopping sau epoch 5. Chưa có bảng CSV đầy đủ từng epoch của v0.5 được tải trong phiên này, nên báo cáo không dựng thêm số liệu epoch 1–4.

| Checkpoint/kết quả | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Best, validation cuối run | 80,2% | 66,6% | **74,8%** | **53,3%** |
| Epoch 5, cuối quá trình train | 78,9% | 65,4% | 74,5% | 52,3% |

Train + validation cho năm epoch hoàn thành trong khoảng 0,430 giờ, tương đương 25,8 phút. Toàn notebook chạy khoảng 1.757,5 giây, tương đương 29,3 phút. Log epoch 5 ghi GPU memory khoảng 10,4 GB trên mỗi rank; đây không phải tổng VRAM của hai GPU.

Best và last đã bị loại trạng thái optimizer ở cuối run. Có thể dùng để khởi tạo lượt fine-tune mới, nhưng không nên coi chúng là checkpoint có đầy đủ trạng thái để tiếp tục optimizer/scheduler chính xác như một lần resume hoàn chỉnh.

### 4.3. AP từng lớp của v0.5 best

| Lớp | Ảnh có lớp | GT box | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| person | 282 | 1.491 | 84,3% | 70,8% | 82,3% | 55,6% |
| table | 133 | 275 | 79,8% | 67,5% | 75,3% | 54,5% |
| **chair** | 503 | 989 | 74,3% | **52,7%** | **61,3%** | **40,3%** |
| laptop | 205 | 317 | 71,7% | 71,9% | 71,5% | 46,6% |
| **cell phone** | 765 | 1.443 | 80,9% | **56,0%** | **68,1%** | **41,5%** |
| backpack | 399 | 464 | 84,3% | 75,9% | 82,3% | 65,2% |
| book | 388 | 2.802 | 75,3% | 62,9% | 72,5% | 45,7% |
| bottle, gồm cup + bottle | 402 | 578 | 90,6% | 75,1% | 84,9% | 76,5% |
| **Toàn bộ** | ** 2.500** | ** 8.359** | **80,2%** | **66,6%** | **74,8%** | **53,3%** |

Hai lớp yếu nhất theo AP50 là chair và cell phone. Recall thấp gợi ý nên kiểm tra các mẫu bị bỏ sót, nhưng chưa thể kết luận nguyên nhân là vật thể nhỏ, thiếu dữ liệu hay kiến trúc. Precision/recall do validator báo không đồng nghĩa với kết quả ứng dụng tại một confidence cố định như 0,25.

mAP50 lấy trung bình đều tám lớp. Nếu chỉ hai lớp yếu tăng 10 điểm AP50 mỗi lớp và sáu lớp còn lại giữ nguyên, mAP50 toàn bộ tăng khoảng ** 2,5 điểm**. Nếu mỗi lớp yếu tăng 20 điểm thì tăng khoảng **5 điểm**. Vì vậy đạt 85% có thể cần cải thiện thêm các lớp khác, dù không đặt yêu cầu quá cao riêng cho hai lớp yếu.

### 4.4. Không so trực tiếp hai mAP khác validation

v0.4 dùng validation 1.754 ảnh; v0.5 dùng 2.500 ảnh. Do đó so 75,356% của v0.4 với 74,8% của v0.5 chưa đủ kết luận mô hình tốt lên hay kém đi.

Một đánh giá trước đây của v0.4 trên cùng quy trình COCO, CPU, imgsz 640 cho thấy:

| Tập đánh giá | Trọng số trước v0.4: AP50 / AP50–95 | Best v0.4: AP50 / AP50–95 |
|---|---:|---:|
| Validation v0.4, 1.754 ảnh | 47,593% / 31,047% | 70,706% / 50,694% |
| Validation v0.3, 798 ảnh | 56,727% / 35,896% | 55,279% / 35,049% |

Đây là dấu hiệu mô hình có thể cải thiện trên phân bố mới nhưng giảm trên phân bố cũ. Nó hỗ trợ việc theo dõi AP theo nguồn/cảnh, không chỉ một con số trung bình. Các số COCO 640 này cũng không nên trộn với số trainer 1280.

## 5. Hai lượt smoke cho biết điều gì?

Hai lượt dùng cùng trọng số cha v0.4, 600 ảnh train và 2.500 ảnh validation. Sau đó cùng được đánh giá ở imgsz 1536, batch 8, GPU 0, conf 0,001, IoU 0,7, NMS bật, max_det 300, augment=False, half=True.

| Cấu hình train | Thời gian một epoch gồm train + val | mAP50 khi đánh giá lại | mAP50–95 |
|---|---:|---:|---:|
| imgsz 1536, batch 16 | 117,777 giây | 71,963% | 47,580% |
| imgsz 1280, batch 32 | 102,916 giây | 72,754% | 48,191% |

Trong hai lượt này, cấu hình 1280/batch32 có kết quả cao hơn và thời gian ngắn hơn. Tuy nhiên, **đã thay đồng thời imgsz và batch**, chỉ train một epoch trên 5% train và dữ liệu chưa được duyệt trực quan đầy đủ. Không thể quy toàn bộ khác biệt cho một tham số hoặc kết luận đó là cấu hình tối ưu.

Thông số tốc độ inference xuất hiện trong các lần validator chạy nối tiếp chưa đủ kiểm soát warmup/cache để coi là benchmark tốc độ triển khai. Quy trình đánh giá smoke cũng khác full run ở resolution và chế độ NMS.

## 6. Vì sao notebook báo lỗi dù trọng số vẫn có?

Log Kaggle xác nhận train đã kết thúc và có checkpoint. Lỗi xuất hiện ở cell so sánh smoke chạy sau full run:

```python
assert not TRAIN_COMPLETED, 'Cell này chỉ so sánh smoke; không dùng kết quả full run'
```

Đây là lỗi điều khiển luồng notebook: cell chỉ dành cho smoke được thực thi khi full run đã hoàn thành. Nó không chứng minh huấn luyện bị hỏng hoặc checkpoint không dùng được. Nguồn: [Kaggle Version 3 — log](https://www.kaggle.com/code/awm3636/project8-v05-fast-archive-from-best-v04/log?scriptVersionId=356500799).

Hướng sửa khi tiếp tục làm notebook:

1. Thêm cờ rõ ràng `RUN_SMOKE_COMPARISON=False` mặc định.
2. Bỏ qua toàn bộ cell so sánh khi full run hoàn thành, chỉ chuẩn bị dữ liệu hoặc chưa có smoke.
3. Giữ bước đánh giá và đóng gói checkpoint chạy độc lập với việc train lại.
4. Đặt `START_TRAINING=False` mặc định cho luồng cứu kết quả.
5. Thống nhất và ghi rõ resolution, head/NMS và evaluator trong so sánh.

Không nên chỉ xóa assertion rồi chạy tiếp: cell còn dựa vào biến smoke và điều kiện cố định imgsz1536/batch16. Notebook local hiện cũng chứa cấu hình smoke cũ và cell so sánh full run ở 640/NMS bật; không được coi cấu hình local đó là cấu hình thực tế của Version 3.

## 7. Thử learning rate: chạy bao nhiêu epoch và so sánh thế nào?

### 7.1. Trả lời trực tiếp

**Vòng đầu: chạy 2 epoch cho mỗi cấu hình**, xem kết quả sau cả epoch 1 và epoch 2. Không cần chạy một lượt 1 epoch rồi chạy lại từ đầu một lượt 2 epoch, vì một lượt 2 epoch đã có hai mốc quan sát.

Sau đó:

- Nhánh có dấu hiệu tốt: xác nhận bằng ** 8–10 epoch**, ưu tiên 10 epoch khi đủ ngân sách.
- Nhánh giảm rõ hoặc không ổn định: kiểm tra loss/LR và mẫu lỗi trước khi tiếp tục.
- Nhánh chỉ tăng nhẹ hoặc giảm nhẹ: chưa loại ngay; 2 epoch chưa đủ phân biệt hội tụ chậm với cấu hình không tốt.
- Nếu đã tăng và muốn học thêm: từ checkpoint được chọn, chạy thêm **5 epoch**, đánh giá rồi mới quyết định thêm. Không tự động nối nhiều lượt chỉ vì loss train giảm.

Mỗi vòng xác nhận phải có đối chứng với **cùng số epoch và lịch LR**. Không so ứng viên học thêm 10 epoch với đối chứng chỉ học thêm 2 epoch rồi quy toàn bộ chênh lệch cho tham số.

### 7. 2. Ba mức learning rate

| Mã | lr0 | Thay đổi so với 0,001 | Câu hỏi |
|---|---:|---:|---|
| L0 | 0,001 | Giữ nguyên | Học thêm với LR hiện tại có tăng không? |
| L1 | 0,0005 | Giảm 50% | Bước nhỏ hơn có giúp tinh chỉnh ổn định không? |
| L2 | 0,0001 | Giảm 90% | Giảm mạnh có bảo toàn kiến thức hay làm học quá chậm? |

Đây là lựa chọn thí nghiệm, chưa biết L1/L2 sẽ tốt hơn. LR thấp nhất có thể ít thay đổi sau 2 epoch; không đồng nghĩa chắc chắn kém sau 10 epoch.

Tất cả nhánh độc lập bắt đầu từ:

    weights/kaggle/BEST_Project8_v05_20261009.pt
    SHA256: 92adfeb2516861a6958e8adc4a15cf764faa3956078da1ebc81db1fbb7014e31

Không để L1 bắt đầu từ kết quả L0 hoặc L2 bắt đầu từ kết quả L1. Khi đó lịch sử học và số lần thấy dữ liệu sẽ khác nhau, không còn là phép so độc lập.

### 7.3. Lịch LR cho vòng thử ngắn

Đề xuất chung cho lượt 2 epoch:

    epochs: 2
    patience: 0
    optimizer: MuSGD
    lrf: 1.0
    cos_lr: false
    warmup_epochs: 0.25
    warmup_bias_lr: 0.0
    resume: false

- lrf=1,0 giữ hệ số scheduler không giảm trong vòng sàng lọc; chỉ có warmup đầu lượt. Đây là phép thử ba mức LR, không tái hiện nguyên xi lịch 25 epoch cũ.
- warmup_epochs=0,25 cho một đoạn khởi động ngắn, giống nhau ở cả ba mức.
- warmup_bias_lr=0 tránh dùng bias LR khởi động 0,1 quá lớn so với LR fine-tune đang so.
- patience=0 vô hiệu early stopping trong ngân sách 2 epoch; không có nghĩa dừng ngay.
- Giữ MuSGD tường minh; optimizer=auto có thể bỏ qua LR thủ công.

**Đính chính chi tiết warmup:** mã Ultralytics 8.4.165 đang cài giới hạn warmup theo min(warmup_epochs, epochs − 1). Với lượt 2 epoch, giá trị 3 sẽ bị giới hạn khoảng 1 epoch; với lượt 1 epoch thì warmup về 0. Không hiểu rằng lượt 2 epoch thực sự có đủ 3 epoch warmup. Dùng warmup mặc định vẫn có thể làm phần lớn epoch đầu chưa phản ánh LR cần kiểm tra.

LR của các nhóm tham số MuSGD có thể khác lr0 danh nghĩa. Phải lưu cột LR thực tế trong results.csv, không chỉ ghi lr0.

Nguồn tham số: [Ultralytics training](https://docs.ultralytics.com/modes/train). Chi tiết warmup được kiểm tra trực tiếp trong ultralytics/engine/trainer.py của bản 8.4.165 cài tại dự án.

### 7.4. Đối chứng gồm hai dạng

1. **B0: không học thêm** — đánh giá best v0.5 hiện tại, 0 epoch.
2. **L0A0: học thêm nhưng giữ LR/augmentation** — train 2 epoch với lịch sàng lọc chung.

Chúng trả lời hai câu khác nhau: học thêm có lợi không, và đổi LR/augmentation có lợi thêm so với đơn thuần học thêm không?

Do lịch sàng lọc khác lịch run cũ, L0A0 là đối chứng của thí nghiệm mới, không phải resume y nguyên run 25 epoch.

## 8. Giảm augmentation: giữ nguyên, giảm nhẹ và giảm mạnh

### 8.1. Thay đổi đồng bộ cả bộ augmentation

Theo yêu cầu của bạn, vòng đầu so ba **bộ cấu hình đầy đủ**, không chỉ thay mosaic rồi giữ toàn bộ thành phần khác.

| Tham số | A0: giữ nguyên | A1: giảm nhẹ nhiều thành phần | A2: giảm mạnh, tắt augmentation |
|---|---:|---:|---:|
| mosaic | 0,5 | 0,2 | 0 |
| scale | 0,5 | 0,25 | 0 |
| translate | 0,1 | 0,05 | 0 |
| fliplr | 0,5 | 0,25 | 0 |
| hsv_h | 0,015 | 0,0075 | 0 |
| hsv_s | 0,7 | 0,35 | 0 |
| hsv_v | 0,4 | 0,2 | 0 |
| degrees | 0 | 0 | 0 |
| flipud | 0 | 0 | 0 |
| shear/perspective | 0/0 | 0/0 | 0/0 |
| mixup/cutmix | 0/0 | 0/0 | 0/0 |
| bgr | 0 | 0 | 0 |
| close_mosaic | 0 cho vòng thử | 0 | 0 |

**Các giá trị A0 trong bảng đã được đọc từ train_args của best v0.5**, gồm HSV, shear, perspective và bgr. A1/A2 là đề xuất. Trước khi chạy, vẫn đối chiếu args hiệu lực để tránh kế thừa tham số khác ngoài ý muốn.

close_mosaic=0 trong vòng 2 epoch để A0/A1 không tự tắt mosaic giữa lượt, làm mất khác biệt cần so. Resize/letterbox đưa ảnh vào model vẫn thực hiện ở A2; tắt augmentation không có nghĩa bỏ tiền xử lý.

A2 kiểm tra giới hạn của giảm augmentation, không phải cấu hình chắc chắn nên dùng lâu dài. Với detect, không lấy tham số chỉ dành cho classification/segmentation để gọi là thay đổi augmentation của detector.

### 8. 2. Giả thuyết cần kiểm tra

| Bộ | Khả năng có lợi | Khả năng bất lợi |
|---|---|---|
| A0 | Giữ đa dạng, hạn chế học thuộc | Biến đổi có thể quá mạnh khi tinh chỉnh |
| A1 | Ảnh gần tự nhiên hơn, vẫn còn đa dạng | Có thể giảm khả năng tổng quát |
| A2 | Tinh chỉnh theo ảnh tự nhiên, giảm nhiễu biến đổi | Có thể học thuộc, kém khi ánh sáng/góc nhìn đổi |

Loss A2 có thể thấp hơn vì bài toán train dễ hơn. Không dùng riêng điều đó để kết luận model tốt hơn.

So A1/A2 với A0 cho biết hiệu quả **cả bộ thay đổi**. Nếu bộ thắng và cần tìm thành phần đóng góp, mới làm lượt sau chỉ đổi mosaic hoặc scale. Không cần chạy mọi ablation nhỏ ngay từ đầu.

### 8.3. Ma trận LR × augmentation

| Run | LR | Augmentation | Epoch | So sánh chính |
|---|---:|---|---:|---|
| L0A0 | 0,001 | Giữ nguyên | 2 | Đối chứng học thêm |
| L1A0 | 0,0005 | Giữ nguyên | 2 | Giảm LR nhẹ |
| L2A0 | 0,0001 | Giữ nguyên | 2 | Giảm LR mạnh |
| L0A1 | 0,001 | Giảm nhẹ | 2 | Chỉ thay bộ augmentation |
| L0A2 | 0,001 | Tắt augmentation | 2 | Chỉ giảm mạnh augmentation |
| L1A1 | 0,0005 | Giảm nhẹ | 2 | Cùng giảm nhẹ LR và augmentation |
| L1A2 | 0,0005 | Tắt augmentation | 2 | LR giảm nhẹ, augmentation giảm mạnh |
| L2A1 | 0,0001 | Giảm nhẹ | 2 | LR giảm mạnh, augmentation giảm nhẹ |
| L2A2 | 0,0001 | Tắt augmentation | 2 | Cùng giảm mạnh |

**Ngân sách: 9 lượt × 2 epoch = 18 epoch train**, không phải một lượt nối tiếp 18 epoch. Mỗi lượt khởi tạo lại từ B0.

Có thể chạy trước L0A0, L1A0, L2A0, L0A1, L0A2 rồi bốn nhánh kết hợp. Không đổi tiêu chí đánh giá sau khi nhìn kết quả.

Nguồn ý nghĩa các phép biến đổi: [Ultralytics augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation). Các mức A1/A2 là thiết kế cho dự án, chưa được chạy.

### 8.4. Cách đọc sự tương tác

Ví dụ minh họa, **không phải số đo**: giảm LR riêng tăng 0,5 điểm, giảm augmentation riêng tăng 0,4 điểm, kết hợp chỉ tăng 0,2 điểm. Không cộng hai mức tăng rồi kỳ vọng chắc chắn được 0,9 điểm.

Kết hợp có thể hỗ trợ nhau hoặc làm cập nhật quá yếu. So từng nhánh với B0 và L0A0, cùng lúc xem AP chair/cell phone và các lớp đang mạnh.

## 9. Thử hai kiểu fine-tune hai giai đoạn

Có thể sàng lọc bằng **1 epoch giai đoạn 1 + 1 epoch giai đoạn 2**. Chuỗi này chỉ cho dấu hiệu ban đầu; cần lượt dài hơn để xác nhận.

### 9.1. Kiểu F: freeze backbone rồi unfreeze

| Nhánh | Giai đoạn 1: 1 epoch | Giai đoạn 2: 1 epoch | Câu hỏi |
|---|---|---|---|
| F0: đối chứng | Không freeze, LR 0,001, A0 | Không freeze, LR 0,0005, A0 | Hạ LR theo hai giai đoạn tác động thế nào? |
| F1: thử freeze | Freeze backbone, LR 0,001, A0 | Unfreeze, LR 0,0005, A0 | Freeze đầu lượt có lợi thêm không? |

Giữ cùng lịch và thao tác reset optimizer giữa hai nhánh. Nếu F1 reset optimizer ở stage 2 còn F0 chạy liên tục không reset, phép so bị lẫn tác động optimizer.

Theo cấu trúc YOLO26 đã kiểm tra, backbone gồm layer 0–10: freeze=11 đóng toàn backbone; freeze=10 vẫn để layer 10 học. Trước khi chạy cần xác nhận cấu trúc checkpoint, log lớp frozen và số tham số trainable. Giai đoạn 2 dùng freeze=0 hoặc không freeze, trừ thành phần thư viện vốn luôn đóng băng.

Giai đoạn 2 khởi tạo từ **checkpoint cuối giai đoạn 1**, không chọn best khác epoch tùy nhánh. Với stage 1 chỉ có một epoch, best/last thường cùng mốc nhưng vẫn ghi lineage.

B0 đã fine-tune trên dữ liệu hiện tại, nên freeze không mặc định cần thiết. Không tăng thì bỏ hướng này trước khi làm phức tạp hơn.

### 9. 2. Kiểu R: thích nghi rồi tinh chỉnh nhẹ/mạnh

Ba nhánh cùng dùng kết quả stage 1 không freeze, LR 0,001, A0:

| Nhánh | Giai đoạn 2: 1 epoch | Mục đích |
|---|---|---|
| R0 | LR 0,001, A0 | Giữ cấu hình |
| R1 | LR 0,0005, A1 | Tinh chỉnh nhẹ |
| R2 | LR 0,0001, A2 | Tinh chỉnh mạnh |

Không đổi seed hoặc mẫu train giữa các nhánh stage 2. R1/R2 thay đổi kết hợp; ma trận phần 8 giúp giải thích từng thành phần.

### 9.3. Tránh warmup che mất phép thử 1 epoch

Mỗi stage ngắn dùng:

    epochs: 1
    lrf: 1.0
    warmup_epochs: 0
    warmup_bias_lr: 0
    patience: 0
    optimizer: MuSGD
    resume: false

Đây là sàng lọc từ checkpoint đã học, không phải khuyến nghị bỏ warmup cho mọi lượt train. Có thể chuyển stage trong một trainer để giữ optimizer khi xác nhận dài, nhưng phải ghi rõ vì quy trình khác hai lượt độc lập.

Nếu tái dùng stage 1 chung, hai kiểu F/R cần tổng **7 epoch train mới**:

- Stage1 không freeze chung: 1.
- Stage1 freeze: 1.
- Stage2 F0/F1: 2.
- Stage2 R0/R1/R2: 3.

### 9.4. Xác nhận ứng viên

Nếu F có triển vọng: thử 2 epoch freeze + 8 epoch unfreeze, so đối chứng 2+8 không freeze, cùng LR từng stage.

Nếu R có triển vọng: thử 5 epoch thích nghi + 5 epoch tinh chỉnh, so đối chứng cùng điểm chuyển/ngân sách. LR/augmentation lấy từ ma trận thắng, không mặc định L2A2 tốt nhất.

Không kết luận hai giai đoạn thắng chỉ vì train nhiều hơn đối chứng. Nguồn: [Fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide).

## 10. Vật thể nhỏ: resolution và SAHI

### 10.1. Tăng nét: tạm dừng ưu tiên

Bạn đã thử tăng nét, hiệu quả thấp. Không tiếp tục đây là nhánh chính. Ảnh trông nét hơn không bảo đảm bổ sung thông tin thật cho vật nhỏ.

### 10. 2. Số liệu mới từ dataset hiện tại

Thống kê dưới đây từ toàn bộ manifest/nhãn local. Kết quả dự đoán trên mẫu 300 ảnh được tách riêng ở phần 22.

“Nhỏ” nghĩa là diện tích box trên **ảnh gốc** < 32² pixel. Đây là mô tả dữ liệu, không phải AP_small của model hoặc kích thước sau letterbox.

| Lớp | Box nhỏ train / toàn bộ | Tỷ lệ train | Box nhỏ val / toàn bộ | Tỷ lệ val |
|---|---:|---:|---:|---:|
| person | 502 / 7.449 | 6,7% | 143 / 1.491 | 9,6% |
| table | 28 / 2.352 | 1,2% | 3 / 275 | 1,1% |
| chair | 167 / 5.023 | 3,3% | 35 / 989 | 3,5% |
| laptop | 163 / 1.266 | 12,9% | 20 / 317 | 6,3% |
| cell phone | 2.444 / 5.856 | 41,7% | 907 / 1.443 | **62,9%** |
| backpack | 97 / 1.645 | 5,9% | 34 / 464 | 7,3% |
| book | 3.581 / 6.416 | 55,8% | 1.977 / 2.802 | **70,6%** |
| bottle, gồm cup + bottle | 432 / 3.265 | 13,2% | 110 / 578 | 19,0% |

Cell cell phone/book nên theo dõi khi slicing. Chair yếu nhưng chỉ 3,5% box val là nhỏ theo ngưỡng này, nên tìm thêm nguyên nhân; chưa có cơ sở coi SAHI là giải pháp chính cho chair.

Validation có 538 ảnh cạnh dài > 640 và 510 ảnh cạnh dài > 1024. **1.962/2.500 ảnh có cạnh dài ≤ 640.** Tăng đầu vào 1280→1536 sẽ phóng to nhiều ảnh ít pixel, không tạo chi tiết đã mất.

### 10.3. Thử resolution trước ở inference

Dùng cùng B0, ảnh và evaluator:

| Mã | imgsz inference | Epoch thêm | Cần đo |
|---|---:|---:|---|
| I0 | 1280 | 0 | Baseline AP/latency |
| I1 | 1536 | 0 | AP có tăng đủ so chi phí không? |
| I2 | 960 | 0 | Có nhanh hơn mà giữ AP không? |

Đây là resolution lúc dự đoán, không chứng minh train 1536 tốt hơn. Nếu I1 có lợi và muốn thử train: chạy **1280/batch16 và 1536/batch16**, mỗi nhánh 2 epoch, cùng checkpoint/LR/aug. Không so 1280/batch32 với1536/batch16 rồi gọi là tác động riêng của resolution.

Hai lượt train tùy chọn thêm 4 epoch. Kiểm tra VRAM/batch theo số GPU; trước đây1536/batch32 gặp OOM.

### 10.4. SAHI khác tăng nét thế nào?

SAHI cắt ảnh thành các ô chồng lấn, chạy detector trên từng ô, chuyển tọa độ về ảnh gốc rồi ghép box. Vật nhỏ chiếm phần lớn hơn trong đầu vào mỗi ô. Có thể dùng checkpoint hiện tại, **không cần train thêm epoch** cho phép thử inference.

Có thể bắt được thêm vật, nhưng cũng tăng FP, box trùng và thời gian. Vật lớn bị cắt hoặc cần ngữ cảnh có thể bị ảnh hưởng.

### 10.5. Ba cấu hình SAHI

Do nhiều ảnh chỉ 640/512 pixel, không mặc định cắt 1024: phần lớn ảnh đó sẽ không được chia.

| Mã | Ô cắt trên ảnh gốc | Chồng lấn | Lượt toàn cảnh | imgsz model | Mức |
|---|---|---:|---|---:|---|
| S0 | Không cắt | — | Có | 1280 | Đối chứng |
| S1 | 512×512 | 20% | Có | 1280 | Slicing nhẹ |
| S2 | 320×320 | 25% | Có | 1280 | Slicing mạnh |

Giữ checkpoint, confidence, head và evaluator. Quy tắc NMS/NMM ghép ô ghi riêng, không lẫn với NMS của detector.

Một ảnh 640×640 có thể cần khoảng 4 ô ở S1 hoặc9 ô ở S2, cộng một lượt toàn cảnh. Đây là số gọi minh họa; tốc độ không nhất thiết tăng đúng5/10 lần vì còn batch/preprocessing/ghép box. Đo từ nhận ảnh đến kết quả cuối.

### 10.6. Sàng lọc SAHI

1. Chọn danh sách cố định khoảng300 ảnh **validation**, đủ 8 lớp, nhiều nguồn, có vật nhỏ/lớn. Dùng cùng danh sách cho S0/S1/S2 và I0/I1/I2.
2. Warmup rồi đo AP theo cùng GT ảnh gốc.
3. Ghi AP50, AP chair/cell cell phone/book, FP/FN, AP/recall theo kích thước, latency median/p95.
4. Tập ưu tiên vật nhỏ có thể khác phân bố toàn val:300 ảnh chỉ sàng lọc, không thay74,8% bằng mAP từ tập khác.
5. Xác nhận cấu hình triển vọng trên đủ 2.500 ảnh.

Với AP, dùng score threshold thấp, giữ score từng prediction cho đường PR. Với audit vận hành, dùng0,1/0,25/0,4. Không so AP của SAHI cắt prediction ở 0,25 với baseline giữ ở 0,001.

Trước chạy cài/ghi phiên bản SAHI, kiểm tra adapter với Ultralytics 8.4.165 và tọa độ sau ghép. Nếu phải đổi head để adapter hoạt động, thêm đối chứng không slicing với head tương ứng.

Nguồn: [Ultralytics SAHI](https://docs.ultralytics.com/guides/sahi-tiled-inference), [SAHI](https://github.com/obss/sahi).

## 11. Cân bằng độ khó với dữ liệu hiện có

### 11.1. Giữ split, lấy mẫu có kiểm soát

Giữ 12.000 train,2.500 val,500 test. Dùng lại ảnh train qua lịch sampling có kiểm soát. Không chuyển val sang train hoặc gọi ảnh lặp là ảnh độc lập mới.

Dataset có 15.000 ảnh nhưng độ đa dạng/độ khó từng lớp không đều. Một ảnh chứa nhiều lớp, nên không cộng số ảnh theo lớp để suy ra tổng dataset.

### 11. 2. Số lượng từng lớp train

| Lớp | Ảnh train có lớp | Box train | Điều cần xem |
|---|---:|---:|---|
| person | 1.264 | 7.449 | Nhiều box/ảnh, chưa chắc nhiều cảnh |
| table | 1.044 | 2.352 | Vật nhỏ ít, xem ngữ cảnh/nhầm lớp |
| chair | 2.652 | 5.023 | Số lượng không thấp nhưng AP yếu |
| laptop | 788 | 1.266 | Ít ảnh/box hơn nhiều lớp |
| cell phone | 3.996 | 5.856 | Nhiều ảnh nhưng vật nhỏ nhiều |
| backpack | 1.417 | 1.645 | AP khá, tránh giảm khi ưu tiên lớp khác |
| book | 1.102 | 6.416 | Nhiều box nhỏ trên một ảnh |
| bottle, gồm cup + bottle | 1.891 | 3.265 | Giữ định nghĩa gộp nhất quán |

Không nói chair/cell phone yếu chỉ vì thiếu số lượng: chúng đã có nhiều ảnh. Xem chất lượng, cảnh, che khuất, kích thước và nhãn.

### 11.3. Tạo nhóm độ khó có lý do

Mỗi ảnh train cần chỉ mục:

    image_id
    source_id / source_group_key
    các lớp hiện diện
    số box, kích thước tương đối
    mức che khuất/cắt mép nếu đã kiểm tra
    lỗi baseline nếu đã dự đoán
    nhãn: đã kiểm tra / nghi vấn / chưa kiểm tra
    difficulty: dễ / vừa / khó
    lý do xếp nhóm

Ảnh nhỏ không mặc định khó; loss cao không mặc định nhãn đúng. Ưu tiên ảnh khó **đã kiểm tra nhãn**: phone nhỏ còn thấy được, chair che khuất nhưng box đúng, cặp dễ nhầm đủ nhãn.

Dùng lỗi val chọn **kiểu vấn đề**, sau đó tìm ảnh train tương ứng. Không dùng trực tiếp FP/FN val làm danh sách train.

### 11.4. Ba mức ưu tiên mẫu khó

| Nhánh | Mỗi epoch | Ưu tiên khó | Epoch thử |
|---|---|---|---:|
| H0 | 12.000 lượt, mỗi ảnh train một lần | Không | Tái dùng L0A0 nếu mọi cấu hình giống nhau |
| H1 | 9.600 lượt train ngẫu nhiên +2.400 lượt ưu tiên khó | Nhẹ, khoảng 20% lượt | 2 |
| H2 | 7.200 lượt ngẫu nhiên +4.800 lượt ưu tiên khó | Mạnh, khoảng 40% lượt | 2 |

H0/H1/H2 cùng LR 0,001, A0 và lịch2 epoch; chỉ đổi sampling.

Quy tắc:

- Phần ngẫu nhiên lấy theo seed, không lấy tỷ lệ ảnh đầu manifest.
- Mỗi ảnh tối đa ** 2 lần/epoch**, tính cả phần ngẫu nhiên/khó.
- Pool khó đã kiểm tra không đủ thì lấp bằng ảnh chưa chọn hoặc giảm tỷ lệ thực tế, không lặp vô hạn.
- Giữ hiện diện cả 8 lớp và nguồn; ảnh nhiều lớp tính vào mọi lớp.
- Lưu danh sách lấy mẫu, số ảnh unique/lặp, tỷ lệ lớp/nguồn mỗi epoch.
- Nếu tăng chair làm laptop/backpack ít xuất hiện, điều chỉnh hạn mức trước chạy.
- Tập train vật lý không đổi, phân bố model thấy đã đổi; có thể thấy ít hơn12.000 ảnh unique trong một epoch.

Đây là thiết kế sampling. **YOLO mặc định không tự thực hiện tỷ lệ 20%/40% bằng một tham số thông thường.** Cần sampler/loader phù hợp. Với DDP phải chia lượt giữa rank đúng, không nhân đôi ngân sách. Chưa triển khai cơ chế này.

### 11.5. Những điều chưa nên làm

- Không coi pseudo-label mới là GT đã kiểm tra.
- Không tạo nhiều bản augmentation offline của vài ảnh rồi gọi là nhiều cảnh mới.
- Không đổi class loss weights cùng sampling trong phép so H0/H1/H2.
- Không bỏ mẫu dễ; model vẫn cần ví dụ rõ và background.
- Không ưu tiên ảnh khó chưa kiểm tra vì có thể khuếch đại nhãn sai.

Chưa đủ pool nhãn tốt thì audit trước. source_id/source_group_key có trong manifest, nhưng chưa đảm bảo biết đủ scene/session.

## 12. Audit FP/FN là gì? Làm thế nào?

### 12.1. Giải thích bằng ví dụ

- **TP — True Positive:** đúng lớp, box khớp vật thể có nhãn.
- **FP — False Positive:** báo không có GT tương ứng, sai lớp, box không khớp hoặc box trùng không ghép với GT còn lại.
- **FN — False Negative:** GT không được phát hiện theo tiêu chí đang xét.

Ảnh có10 điện thoại thật, model báo8 box, trong đó6 box đúng:

    TP = 6
    FP = 8 - 6 = 2
    FN = 10 - 6 = 4
    Precision = 6 / 8 = 75%
    Recall = 6 / 10 = 60%

Phone dự đoán thành book thường là FP book và FN phone. Box trùng có thể tạo FP. GT thiếu điện thoại thật có thể khiến prediction đúng bị tính thành FP: phải xem ảnh để phát hiện.

### 12. 2. Audit là tìm nguyên nhân lỗi

Không chỉ đếm FP/FN:

- Bỏ sót do nhỏ/mờ/che khuất hay confidence thấp?
- Nhầm lớp do hình giống nhau hay nhãn không nhất quán?
- Box sai do model hay annotation?
- Nguồn ảnh chỉ gán một số lớp, bỏ sót lớp khác?
- Lỗi tập trung một nguồn/cảnh hay xuất hiện rộng?

### 12.3. Quy trình ban đầu

1. Dự đoán B0 trên val với score threshold thấp, lưu score/box.
2. Tính AP theo evaluator thống nhất.
3. Audit tại confidence 0,25, IoU matching0,5; prediction/GT ghép một lần theo lớp, không nhiều TP cho một GT.
4. Lặp ở confidence 0,1/0,4 để kiểm tra tác động threshold.
5. **Mỗi lớp chair/cell phone:**20 mẫu FN,20 FP,20 ngẫu nhiên.
6. Loại trùng ảnh.120 lượt chọn chưa chắc120 ảnh unique.
7. Xem ảnh và nhãn của cả 8 lớp.
8. Ghi loại lỗi, mức chắc chắn, hành động dự kiến.

| image_id | Lớp | TP/FP/FN | Confidence | IoU nếu có | Kích thước | Nguyên nhân | Nhãn cần sửa? | Hướng xử lý |
|---|---|---|---|---|---|---|---|---|
| Chờ chạy | — | — | — | — | — | — | — | — |

### 12.4. Từ lỗi đến hướng xử lý

| Lỗi chiếm nhiều | Ưu tiên |
|---|---|
| Nhãn thiếu/sai | Sửa theo version mới, xác lập lại baseline |
| Phone/book nhỏ, nhãn đúng | S1/S2, resolution, sampling khó |
| Chair che khuất/nhầm table | Mẫu phân biệt/che khuất train, kiểm tra box |
| FP background | Ảnh nền kiểm tra đủ 8 lớp, sampling, conf vận hành |
| Đúng lớp nhưng box lệch | Annotation, định vị, AP50–95 |
| Train loss giảm, val giảm | Xem overfitting; ưu tiên A0/A1 thay vì tắt hết |
| Lỗi còn rộng sau tinh chỉnh | Cân nhắc năng lực model, phần 13 |

Sửa nhãn val làm thay GT: đánh giá lại B0 và ứng viên trên version mới. Không gán mức tăng do sửa GT thành mức tăng do train.

## 13. YOLO26s: phương án cuối đang cân nhắc

### 13.1. Điều kiện chuyển

Bạn đã nói YOLO26n→YOLO26s; báo cáo dùng tên này để thống nhất với “YOLO 2s” trong lời nói mới.

Cân nhắc S khi:

- Đã có baseline chung/audit.
- Đã thử LR/augmentation/hai giai đoạn/sampling phù hợp.
- SAHI/resolution không đủ hữu ích hoặc quá chậm.
- Nhánh có lợi đã xác nhận và học thêm có kiểm soát.
- Vẫn chưa đạt mục tiêu hoặc lỗi gợi ý thiếu năng lực.

Nếu nhánh đang tăng rõ, có thể dùng tiếp trước; không bắt buộc chạy xong mọi lựa chọn mới học thêm. Sau hai vòng xác nhận không tăng đáng kể, tránh đổi tham số vô hạn.

### 13. 2. S có ảnh hưởng tốc độ phản hồi nhiều không?

Có thể chậm hơn, phụ thuộc CPU/GPU/runtime/imgsz/precision/pipeline. S khoảng 9,5 triệu tham số, N 2,4 triệu, gần4 lần; **không có nghĩa latency chắc chắn gấp4**.

Benchmark chính thức ở640:

| Môi trường tham khảo | N | S | S/N |
|---|---:|---:|---:|
| CPU ONNX | 38,9 ms | 87,2 ms | Khoảng2,24 lần |
| T4 TensorRT | 1,7 ms | 2,5 ms | Khoảng1,47 lần |

Không suy ra webcam1280 từ bảng này. Đo latency toàn pipeline batch1, median/p95, FPS và RAM/VRAM trên thiết bị đích.

Ví dụ **minh họa**: inference N 40 ms→S 80 ms nghĩa là khoảng 25→12,5 lượt/giây, chỉ tính inference. Nếu cần cập nhật 10 lần/giây có thể chấp nhận; nếu cần 25–30 lần/giây có thể ảnh hưởng lớn. Camera/preprocessing/vẽ kết quả làm FPS thực thấp hơn. Chưa chốt FPS mục tiêu nên chưa kết luận S phù hợp.

Nguồn benchmark: [YOLO26](https://docs.ultralytics.com/models/yolo26).

### 13.3. Cách thử cơ bản

Dùng yolo26s.pt pretrained với cùng 8 lớp/split. Không đổi tên best N thành S. 2 epoch kiểm tra pipeline, VRAM và xu hướng, **không loại S chỉ vì chưa vượt N sau 2 epoch**: N đã fine-tune nhiều lượt, S mới thích nghi.

Chọn model thực tế cần lượt đủ dài, cùng val/runtime/thiết bị. Chỉ ghi phương án cơ bản ở đây, chưa mở thêm kiến trúc.

## 14. Distillation — CHƯA CÂN NHẮC

Chưa đưa vào lịch chạy/ngân sách hiện tại theo yêu cầu của bạn.

Ghi chú đọc sau: teacher mạnh hỗ trợ train student nhỏ, triển khai chỉ student. Cần teacher tốt hơn trên dữ liệu dự án, thêm chi phí train. Chưa chọn teacher hoặc cấu hình.

Tham khảo khi cần: [Knowledge distillation](https://docs.ultralytics.com/guides/knowledge-distillation).

## 15. Ensemble — CHƯA CÂN NHẮC

Chưa đưa vào lịch hiện tại.

Chạy nhiều model rồi gộp dự đoán có thể tăng AP nếu lỗi bổ sung nhau, nhưng tăng bộ nhớ/latency. Chưa chọn số model/cách ghép. Không dùng ensemble để thay mục tiêu giữ model gọn khi chưa có lý do.

## 16. RF-DETR — CHƯA CÂN NHẮC

Chưa đưa vào lịch hiện tại.

Detector khác cần pipeline fine-tune/đánh giá riêng trên cùng dữ liệu. Chưa chuyển nhãn, tải pretrained hoặc so benchmark.

Tham khảo khi cần: [RF-DETR](https://github.com/roboflow/rf-detr).

## 17. Các hướng bổ sung — ĐỂ ĐỌC SAU

Bạn chưa cân nhắc nhóm này. Không đưa vào ma trận hiện tại:

- Đổi optimizer sang AdamW.
- Tối ưu loss/class weighting riêng.
- TTA hoặc nghiên cứu runtime/export/lượng tử hóa.
- Mở rộng kiến trúc ngoài YOLO26s.

Audit nhãn, FP/FN và đo latency vẫn cần cho thí nghiệm hiện tại; không trì hoãn chúng chỉ vì hướng bổ sung chưa chọn.

## 18. Quy trình chung, số lượt và ngân sách

### 18.1. Cấu hình chung cho lượt2 epoch ở 1280

    data: configs/data_v05.yaml
    model: weights/kaggle/BEST_Project8_v05_20261009.pt
    epochs: 2
    imgsz: 1280
    batch: 32
    device: [0, 1]  # Kaggle T4 x2; local hiện không có CUDA
    optimizer: MuSGD
    lrf: 1.0
    cos_lr: false
    warmup_epochs: 0.25
    warmup_bias_lr: 0.0
    patience: 0
    close_mosaic: 0
    seed: 42
    deterministic: true
    workers: 2
    cache: false
    fraction: 1.0
    resume: false
    save: true
    save_period: 1

Các giá trị chung khác đã xác minh trong checkpoint: weight_decay=0,0005; momentum=0,937; box=7,5; cls=0,5; dfl=1,5; cls_remap=False; nms=False. Khi đưa vào mã Python/YAML, dùng dấu chấm thập phân, ví dụ lr0=0.0005; dấu phẩy trong các bảng là cách trình bày số tiếng Việt.

- LR/augmentation lấy theo mã L/A.
- save_period=1 ở vòng 2 epoch giúp kiểm tra checkpoint từng epoch; xác nhận dài có thể không cần lưu tất cả.
- Kaggle cần đường dẫn data/model đúng input/working. Đây là cấu hình kế hoạch, chưa submit job.
- Pin Ultralytics, ghi torch/CUDA/GPU. Không nâng thư viện giữa các nhánh.
- Giữ weight_decay/momentum/loss weights và tham số khác theo một bộ chung, lưu args hiệu lực.
- fraction1.0 để 2 epoch là hai lượt train đầy đủ, trừ H1/H2 thay sampling. Tập con kiểm tra pipeline phải ghi smoke riêng, không trộn bảng hiệu quả.

### 18. 2. Thứ tự khi bắt đầu chạy

1. Sửa cell smoke chạy nhầm sau full run, tách đánh giá khỏi train lại.
2. Kiểm tra class mapping/manifest, xem batch có nhãn.
3. Đánh giá B0 chung; so v0.4 nếu cần.
4. Audit, tạo danh sách val pilot và pool train khó.
5. Chạy ma trận LR/augmentation.
6. Chạy hai kiểu F/R.
7. Chạy H1/H2 khi pool khó đủ chất lượng.
8. So I0/I1/I2 và S0/S1/S2; inference có thể chạy sớm sau baseline/audit.
9. Chọn tối đa1–2 ứng viên train và một ứng viên inference để xác nhận.
10. Chưa đạt sau xác nhận/học tiếp thì cân nhắc S.
11. Giữ phần 14–17 ngoài lịch.

### 18.3. Ngân sách sàng lọc đầy đủ

| Nhóm | Epoch train mới | Ghi chú |
|---|---:|---|
| B0/audit | 0 | Có chi phí inference |
| LR × augmentation | 18 | 9 lượt độc lập |
| Hai kiểu F/R | 7 | Stage1 chung |
| H1/H2 | 4 | H0 tái dùng L0A0 khi cấu hình giống nhau |
| Resolution inference | 0 | I0/I1/I2 |
| SAHI | 0 | S0/S1/S2; S0/I0 có thể tái dùng |
| **Tổng chính** | **29** | Chưa tính xác nhận |
| Resolution train tùy chọn | +4 | 1280/1536 cùng batch16 |

Không chạy mọi nhóm ngay một phiên; có thể dừng để xử lý nhãn/lỗi.

Run cũ mất25,8 phút/5 epoch gồm train/val; phép nhân thô29 epoch khoảng 150 phút. Chỉ tham khảo quy mô, **không phải ETA**: nhiều lần khởi tạo, I/O, freeze, lưu checkpoint, SAHI và kiểm tra có chi phí riêng. Không suy tổng thời gian từ smoke 600 ảnh.

### 18.4. Cố định evaluator

- Val 2.500 ảnh là chuẩn xác nhận.
- imgsz1280 khi so train; resolution cố ý đổi phải ghi.
- Head/end-to-end/NMS cố định.
- Conf AP thấp, IoU hậu xử lý, max_det/precision cố định.
- SAHI ghép về ảnh gốc rồi đánh giá, áp cùng cap tổng detection.
- Ghi evaluator; AP Ultralytics và COCO với giới hạn detection khác không mặc định bằng nhau.
- 74,8% là số log lịch sử. Đo lại B0 theo quy trình mới trước khi tính chênh lệch.

### 18.5. Xác nhận LR tốt trong bao nhiêu epoch?

**Chọn10 epoch** cho đối chứng/ứng viên, từ cùng B0, lịch chung:

    epochs: 10
    warmup_epochs: 1.0
    warmup_bias_lr: 0.0
    lrf: 0.1
    cos_lr: false
    patience: 0

Đây là lịch xác nhận khác vòng 2 epoch lrf1.0. Dùng cùng lịch ở hai nhánh. So augmentation giữ close_mosaic0; chủ ý đóng mosaic/hai giai đoạn phải có lịch riêng.

Hạn chế ngân sách thì8 epoch cho cả hai. Đủ10 epoch AP giảm thì giữ checkpoint tốt, không mặc định last. AP còn tăng thì có thể thêm 5 epoch từ checkpoint chọn, có đối chứng phù hợp nếu còn cần kết luận nguyên nhân.

## 19. Bảng kết quả và quyết định

### 19.1. Tách hai loại điểm

- **Học trên train:** box/cls/dfl loss, LR, độ ổn định, số ảnh thấy.
- **Tổng quát:** mAP50/mAP50–95 val, AP từng lớp, FP/FN.

Loss A0/A2 có độ khó dữ liệu khác. Loss thấp hơn không tự chứng minh model tốt hơn. Mục tiêu mAP50, nhưng các chỉ số khác phát hiện tác dụng phụ.

### 19. 2. Bảng điền sau chạy

| Run | Epoch | mAP50 | Δ so B0 | Δ so đối chứng cùng mốc | mAP50–95 | Chair AP50 | Phone AP50 | Book AP50 | Train loss | Thời gian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0 | 0 | Chờ đo lại | — | — | Chờ đo | Chờ đo | Chờ đo | Chờ đo | — | Chờ đo |
| L0A0 | 1/2 | Train đủ 2 epoch; còn log, checkpoint chưa khôi phục | — | — | — | — | — | — | — | — |
| L1A0 | 1/2 | Đã xong chạy bù; CSV AP50 73,771% / 74,850%; chờ chấm chung | — | — | — | — | — | — | — | — |
| L2A0 | 1/2 | Chưa chạy | — | — | — | — | — | — | — | — |
| L0A1 | 1/2 | Epoch 1 AP50 ≈72,9%; epoch 2 đang chạy | — | — | — | — | — | — | — | — |
| L0A2 | 1/2 | Chưa chạy | — | — | — | — | — | — | — | — |
| L1A1/L1A2/L2A1/L2A2 | 1/2 mỗi run | Chưa chạy | — | — | — | — | — | — | — | — |
| F0/F1 | Mỗi stage | Chưa chạy | — | So F0 | — | — | — | — | — | — |
| R0/R1/R2 | Mỗi stage | Chưa chạy | — | So R0 | — | — | — | — | — | — |
| H1/H2 | 1/2 | Chưa chạy | — | So H0 | — | — | — | — | — | — |

Sau chạy tách các dòng gộp thành dòng từng run/epoch. Không điền số giả hoặc kỳ vọng thành kết quả.

| Inference | mAP50 | AP vật nhỏ | Phone AP50 | FP/FN conf 0,25 | Latency median | Latency p95 | Peak memory | Nhận xét |
|---|---|---|---|---|---|---|---|---|
| I0/S0 | Có pilot 300 ảnh, phần 22; full val chưa chạy | — | — | — | — | — | — | — |
| I1 | Chưa chạy | — | — | — | — | — | — | — |
| I2 | Chưa chạy | — | — | — | — | — | — | — |
| S1 | Có pilot tile512, phần 22; full val chưa chạy | — | — | — | — | — | — | — |
| S2 | Chưa chạy | — | — | — | — | — | — | — |

### 19.3. Quy tắc sau 2 epoch

Ưu tiên nhánh mAP50 tăng so B0/đối chứng cùng mốc, không giảm mạnh lớp tốt, không có loss/LR bất thường và xu hướng còn ở epoch2.

Tăng **≥0,5 điểm phần trăm** là tín hiệu ưu tiên xác nhận, chưa phải chứng cứ chắc chắn. LR thấp giảm nhẹ0,1–0,3 điểm có thể cần lượt dài nếu loss ổn/cập nhật chậm. Không khẳng định thắng từ nhiễu vài phần nghìn.

74,8%→75,8% là tăng 1 điểm phần trăm; không phải tăng 0,01 điểm.

### 19.4. Quy tắc sau 8–10 epoch

Ngưỡng tham khảo:

- mAP50 tăng khoảng ≥ 1 điểm so đối chứng cùng ngân sách.
- mAP50–95 không giảm quá khoảng 0,5 điểm.
- Lớp quan trọng giảm > 2 điểm AP50: xem lỗi trước chọn.
- Xác nhận seed khác nếu chênh lệch nhỏ/cần chắc chắn.
- Hai vòng gần như không tăng: audit lại trước đổi LR ngẫu nhiên.

Đây là tiêu chí đề xuất, chưa bảo đảm thống kê. Không chọn cấu hình vì đã tốn nhiều GPU hơn.

### 19.5. Tăng thì dùng tiếp thế nào?

1. Chọn checkpoint phù hợp mAP50, không tác dụng phụ lớn.
2. Giữ B0/đối chứng để quay lại.
3. Thêm5 epoch cấu hình có lợi, đánh giá mỗi epoch.
4. Plateau/giảm thì giữ checkpoint tốt; last không mặc định tốt nhất.
5. Đổi thêm LR/aug thì đặt mã thử mới, ghi lineage.
6. Đạt mục tiêu val thì chốt, đánh giá test 500 ảnh cuối vòng phát triển.
7. Đo tốc độ triển khai trước thay trọng số mặc định.

Ultralytics chọn best.pt theo mAP50–95 ở bản đã kiểm tra. Mục tiêu mAP50 cần đối chiếu CSV/checkpoint epoch tương ứng, không coi tên best tự động là best mAP50.

### 19.6. SAHI tăng AP có nghĩa model học thêm không?

**Không.** SAHI/resolution inference đổi cách model nhìn ảnh, không cập nhật trọng số. Nếu S1/S2 thắng, có thể dùng làm cấu hình dự đoán khi latency chấp nhận.

Muốn lợi ích nằm trong model cần train riêng, ví dụ crop với nhãn chuyển đúng, giữ mẫu toàn ảnh/ngữ cảnh. Chưa đưa hướng train crop vào vòng hiện tại; không coi SAHI tạo ra checkpoint mới.

## 20. Trạng thái hiện tại

### Đã hoàn thành

- Tải/kiểm tra best/last v0.5 và xác minh kết quả lịch sử.
- Thống kê ảnh/box/kích thước; hoàn thiện kế hoạch thử nghiệm.
- Đánh giá 960, 1280 và SAHI trên cùng 300 ảnh validation; thống kê FP/FN bằng đối sánh nhãn (phần 22).
- Bật phiên T4 ×2 và sửa notebook cho ba pilot × 2 epoch (phần 23).

### Chưa hoàn thành

- Toàn bộ ma trận 9 nhánh, F/R/H; xác nhận 8–10 epoch và học tiếp 5 epoch.
- SAHI tile 320 và các cấu hình slice/merge khác; inference 1536.
- Audit FP/FN bằng mắt và full visual review của dataset.
- Kết quả ba pilot GPU sẽ ghi khi chạy xong, không điền số dự đoán.

### Môi trường

Local: PyTorch 2.14.0+cpu, Ultralytics 8.4.165, CPU 4 threads khi benchmark. SAHI 0.12.8 được cài riêng trong `.tools/sahi_runtime`; không thay Torch/Ultralytics của venv. Không dùng 500 test để chọn cấu hình.

Dataset vẫn chưa hoàn thành full visual review; pilot được người dùng yêu cầu, trạng thái review và training_ready không đổi thành True.

## 21. Tóm tắt quyết định

1. **LR:** 2 epoch/mức 0,001 / 0,0005 / 0,0001; tốt thì xác nhận 8–10 epoch.
2. **Augmentation:** giữ nguyên / giảm nhẹ nhiều thành phần / tắt mạnh.
3. **Tương tác:** 3 mức LR × 3 bộ augmentation, tổng 18 epoch sàng lọc.
4. **Hai giai đoạn:** freeze→unfreeze và thích nghi→tinh chỉnh, mỗi kiểu có đối chứng; 1+1 epoch ban đầu.
5. **Vật nhỏ:** tạm dừng tăng nét, thử resolution inference và SAHI 512/320, đo AP/latency.
6. **Cân bằng khó:** train hiện có, ưu tiên 0% / 20% / 40%, giới hạn lặp, kiểm tra nhãn.
7. **FP/FN:** tìm báo nhầm/bỏ sót và nguyên nhân trước thêm GPU.
8. **Tăng:** xác nhận, học tiếp từng đợt 5 epoch, giữ checkpoint tốt.
9. **Không tăng sau các hướng đang cân nhắc:** xem YOLO26s, đo AP/tốc độ cùng điều kiện.
10. **Phần 14–16 chưa cân nhắc; phần 17 đọc sau.**
11. **Ưu tiên thử đơn giản trước:** inference đã có kết quả; ba pilot GPU được cấu hình trước ma trận đầy đủ. Mục tiêu vẫn là 85% mAP50 và tốc độ phản hồi phù hợp ứng dụng.
## 22. Kết quả chạy thật: resolution và SAHI trên local

### 22.1 Protocol

- Cùng BEST v0.5, SHA256 `92adfeb2516861a6958e8adc4a15cf764faa3956078da1ebc81db1fbb7014e31`.
- Chọn ngẫu nhiên không hoàn lại 300/2.500 ảnh validation bằng seed 42, từ manifest đã sắp thứ tự; 890 box, trong đó 298 box nhỏ theo diện tích ảnh gốc <32² pixel. Đã kiểm hash ảnh và nhãn.
- YOLO26 NMS-free được đặt rõ `nms=False` và kiểm head end-to-end. Confidence 0,001, max_det 300. AP dùng pycocotools COCOeval, maxDets [1,10,100], macro tám lớp.
- SAHI: tile 512×512, overlap 20%, model imgsz 1280, có full-image pass, gộp NMS từng lớp IoU 0,5, top 300 box. Adapter đã được so với dự đoán YOLO trực tiếp trước chạy.
- Thời gian đo trên cùng 30 ảnh đầu của mẫu, thứ tự ba nhánh xoay vòng, đã warm up, CPU 4 threads/batch 1. Bao gồm preprocessing, model, postprocessing và slice; không gồm đọc ảnh, camera/render. Đây chưa phải độ trễ triển khai GPU.
- Hai nhánh baseline đã xử lý đủ 300 ảnh ở lần đầu. SAHI được chạy lại sau lỗi serialization NumPy; baseline giữ nguyên. Cả ba nhánh đo thời gian lại trên cùng 30 ảnh.

**Không so trực tiếp các số dưới đây với 74,8% lịch sử:** khác số ảnh và cách tính AP. Chênh lệch rất nhỏ giữa 960/1280 chưa chứng minh nhánh nào tốt hơn trên toàn bộ validation.

### 22.2 Bảng tổng hợp

| Nhánh | mAP50 (%) | mAP50–95 (%) | AP small 50–95 (%) | AR small 50–95 (%) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| imgsz960 | 72.36 | 53.49 | 19.05 | 36.35 | 204.80 | 265.70 |
| imgsz1280 | 72.18 | 51.94 | 18.65 | 40.07 | 167.28 | 298.84 |
| sahi512_imgsz1280 | 69.14 | 46.41 | 18.09 | 40.62 | 662.87 | 980.85 |

### 22.3 AP50 từng lớp

| Lớp | GT box | 960 (%) | 1280 (%) | SAHI (%) |
|---|---:|---:|---:|---:|
| person | 160 | 82.26 | 82.39 | 83.58 |
| table | 38 | 70.91 | 62.26 | 57.92 |
| chair | 115 | 56.72 | 55.73 | 49.36 |
| laptop | 31 | 71.12 | 69.96 | 62.49 |
| cell phone | 131 | 74.00 | 79.98 | 74.18 |
| backpack | 69 | 87.03 | 87.01 | 85.70 |
| book | 258 | 69.96 | 72.69 | 73.33 |
| bottle | 88 | 66.87 | 67.43 | 66.58 |

### 22.4 FP/FN tự động tại conf 0,25, IoU 0,5

Các box được đối sánh theo lớp với GT; FP bao gồm dự đoán không ghép được, FN là GT không được phát hiện. Thống kê phụ thuộc ngưỡng; đây chưa phải kết luận nhãn nào sai. Cần mở ảnh để phân biệt nhãn thiếu, box lệch, nhầm lớp và dự đoán dư.

| Nhánh | TP | FP | FN |
|---|---:|---:|---:|
| imgsz960 | 590 | 197 | 300 |
| imgsz1280 | 605 | 195 | 285 |
| sahi512_imgsz1280 | 645 | 402 | 245 |

SAHI giảm FN của chair từ 56 xuống 49, nhưng FP tăng từ 32 lên 109 so với 1280. Cell phone giảm FN từ 34 xuống 26, FP tăng từ 19 lên 59; book giảm FN từ 92 xuống 78, FP tăng từ 58 lên 86. Vì vậy recall có thể tăng mà AP tổng giảm.

### 22.5 Quyết định từ pilot

- SAHI cấu hình này giảm mAP50 khoảng **3,04 điểm phần trăm** so với 1280, median chậm khoảng **3,96 lần**. Chưa nên dùng cấu hình này làm mặc định hoặc gọi là cải thiện.
- 1280 giúp AP50 cell phone tăng khoảng 5,98 điểm so với 960, nhưng một số lớp giảm; mAP50 tổng gần tương đương. 960 có mAP50–95 nhỉnh hơn trên mẫu này.
- Median 1280 thấp hơn 960 trong phép đo CPU này là kết quả quan sát, không suy ra tăng resolution luôn nhanh hơn. Hai nhánh có p95 khác nhau, padding/chạy CPU khác nhau; cần benchmark trên máy triển khai để chọn tốc độ.
- Chưa train lại, chưa thử tile 320, chưa thử ensemble hay đổi model từ kết quả inference này.

### 22.6 Tệp bằng chứng

- `reports/simple_experiments_20261009/results.json`: protocol, metric, AP và FP/FN từng lớp, top ảnh FP/FN.
- `sample_ground_truth.json`: mẫu ảnh và GT; `recovered_predictions.json.gz`: dự đoán đủ ba nhánh.
- `.temp/simple_experiments_20261009.py` và `.temp/resume_simple_experiments_20261009.py`: script chạy/khôi phục.

## 23. Notebook GPU: ba pilot đơn giản đã cấu hình

| Lượt | Epoch | LR0 | Augmentation |
|---|---:|---:|---|
| L0A0_control | 2 | 0,001 | mosaic 0,5; scale 0,5; translate 0,1; flip 0,5; HSV 0,015/0,7/0,4 |
| L1A0_half_lr | 2 | 0,0005 | cùng L0A0 |
| L0A1_light_aug | 2 | 0,001 | mosaic 0,2; scale 0,25; translate 0,05; flip 0,25; HSV 0,0075/0,35/0,2 |

Cùng BEST v0.5 và toàn bộ 12.000 train/2.500 val; imgsz 1280, batch tổng 32, T4 ×2, seed 42, MuSGD, resume False, cls_remap False, nms False. Không thay ảnh/nhãn/split.

Schedule ngắn chung: lrf=1, cos_lr=False, warmup_epochs=0,25, warmup_bias_lr=0, patience=0, close_mosaic=0. Ultralytics có sàn 100 batch warmup khi bật warmup; với 375 batch/epoch, phần này xấp xỉ 0,267 epoch. **L0A0 giữ LR/augmentation nhưng không lặp toàn bộ schedule cũ** lrf=0,01, warmup=3, close_mosaic=5. Đây là đối chứng mới giữa ba pilot.

Lưu checkpoint từng epoch. Chọn epoch theo `metrics/mAP50(B)` trong CSV; `best.pt` mặc định chọn theo mAP50–95 có thể khác. Baseline và checkpoint chọn sẽ được chấm lại cùng protocol imgsz 1280, NMS-free, FP16, batch 8 trên GPU0. Không tính độ trễ camera từ timing validator.

Output riêng: `/kaggle/working/BTL-AI-Camera-v05-pilots`. Nhánh smoke cũ được thay; pipeline sẽ đóng ZIP ba lượt và kết quả so sánh. Phiên GPU đã bật, cấu hình đã lưu; **lượt đối chứng đã xong; hai nhánh còn thiếu được chạy lại theo phần 23.1, chưa có kết quả cuối**. Run timestamp: `20261009_162131`. Đã khởi động lại phiên sau khi mount Input bị kẹt. Preflight mới đã kiểm đúng BEST v0.5 và đủ 14.500 cặp train/val, 41.631 box; preview tám lớp đã được xem. L0A0 bắt đầu với MuSGD LR0=0,001, imgsz 1280 và VRAM khoảng 10,9 GB/GPU. Output notebook tự gắn trên Kaggle trỏ phiên bản cũ. Đã tạo Input riêng tư `awm3636/project8-v05-best-checkpoint-pilots-20261009` chứa BEST v0.5; preflight bắt buộc xác minh SHA256 trước train. Cả 9 cell đã compile và cả ba cấu hình đã qua get_cfg của Ultralytics 8.4.165. Bản local: `notebooks/kaggle/project8_v05_simple_pilots_2epochs.ipynb`. Runtime Kaggle đã xác nhận Torch 2.11.0+cu128/CUDA12.8 và hai Tesla T4.


### 23.1 Phục hồi hai nhánh còn thiếu

Ngày 09/10/2026, phiên tương tác bị kẹt kết nối: log LR một nửa dừng ở epoch 2, batch 134/375. Active Events vẫn báo Running; chưa có bằng chứng OOM/CUDA. Đã Quick Save Version 4 (`356717825`) với tùy chọn lưu output; Kaggle báo Successful. Tuy nhiên trang Output chưa hiển thị tệp và kiểm tra kernel mới trả `RECOVERY_OUTPUT_EXISTS False`: trọng số đối chứng chưa khôi phục được. Giữ log local, không coi snapshot thành công là bằng chứng trọng số đã được bảo toàn.

- Đối chứng L0A0 đã hoàn thành 2 epoch trong log: mAP50 theo train validation xấp xỉ 72,8% → 74,2%; lần chấm `best.pt` cuối khoảng 74,3%. Đây là số làm tròn trong log, chưa phải phép chấm chung batch 8.
- LR một nửa lượt cũ đã xong epoch 1 khoảng mAP50 73,8%; epoch 2 chưa có bằng chứng hoàn tất.
- Augmentation nhẹ chưa có bằng chứng bắt đầu ở phiên cũ.
- Chạy lại **L1A0_half_lr và L0A1_light_aug, mỗi lượt 2 epoch đầy đủ**, từ cùng BEST v0.5. Giữ nguyên các tham số so sánh; không tiếp tục từ checkpoint dở dang.
- Code kiểm tra metadata/CSV/hash để dùng lại lượt hoàn tất nếu tìm được. Nếu checkpoint đối chứng mất, không tự chạy lại đối chứng; ghi rõ `unavailable_runs` trong kết quả đánh giá.
- Đánh giá chung BEST v0.5 ban đầu và hai checkpoint mới; số đối chứng cũ nằm riêng dưới dạng bằng chứng lịch sử. Không kết luận chênh lệch chính xác giữa đối chứng và nhánh mới từ hai protocol khác nhau.

Bản notebook local đã cập nhật quy trình phục hồi; tất cả 9 cell Python compile thành công. File xuất bỏ trọng số YOLO26n 80 lớp tải cho AMP check, chỉ đưa checkpoint project trong `kaggle_runs` vào gói kết quả.


**Trạng thái chạy bù đã xác nhận:** phiên T4 ×2 mới hoạt động; preflight lại đủ 14.500 cặp/41.631 box và đúng SHA256 BEST v0.5. `L1A0_half_lr_e2_20261009_170119` đã chạy epoch 1, log tiến từ batch 34 lên 117/375 (~31%). L0A1 chạy sau trong cùng vòng lặp, rồi chấm BEST ban đầu/hai checkpoint và đóng ZIP. Đây là xác nhận khởi chạy, **chưa phải kết quả hoàn tất**. Log mới: `reports/simple_experiments_20261009/kaggle_retry_log.txt`; ảnh bằng chứng: `kaggle_retry_running.jpg`.


### 23.2 Kiểm tra kết quả chạy bù lúc khoảng 17:13 ngày 09/10/2026

L1A0_half_lr đã hoàn thành train 2 epoch. Log validation trong train: epoch 1 mAP50 ≈73,8%, mAP50–95 ≈52,3%; epoch 2 mAP50 ≈74,8%, mAP50–95 ≈53,0%. Lần validation cuối `best.pt`: precision ≈77,8%, recall ≈68,2%, mAP50 ≈74,9%, mAP50–95 ≈53,0%. Thời gian train báo 0,176 giờ (~10,56 phút); chưa gồm toàn bộ preflight và đánh giá sau.

AP50 từng lớp của lần chấm `best.pt` trong trainer: person 83,2%; table 75,7%; chair 62,3%; laptop 72,8%; cell phone 68,0%; backpack 81,7%; book 70,4%; bottle 84,8%.

Đối chứng cũ chấm `best.pt` khoảng 74,3%, nhánh giảm LR khoảng 74,9%: chênh lệch quan sát khoảng +0,6 điểm phần trăm từ log làm tròn. Đây chưa phải bảng chấm chung batch 8, chưa chứng minh cải thiện ổn định qua nhiều seed. So với BEST v0.5 lịch sử ~74,8%, mức 74,9% rất gần nhau; cần phép chấm lại cùng protocol để kết luận.

Nhánh augmentation nhẹ đã bắt đầu sau khi L1A0 hoàn tất; bảng so sánh cuối chưa hoàn tất ở thời điểm kiểm tra này. Chưa đạt mục tiêu 85% mAP50.


Đã Quick Save **Version 5 (`356722363`)** sau khi LR một nửa hoàn tất; Kaggle báo Successful và trang notebook xác nhận **35 output files**. Đây là snapshot giữa chừng, augmentation đang chạy; không phải kết quả cuối toàn bộ thử nghiệm.


CSV Version 5 đã đọc trực tiếp: L1A0 epoch 1 mAP50 **73,771%**, mAP50–95 **52,320%**; epoch 2 mAP50 **74,850%**, mAP50–95 **53,044%**. Bản preview được lưu ở `reports/simple_experiments_20261009/half_lr_saved_csv_preview.txt`. Kaggle Output chứa run, `results.csv`, `pilot_metadata.json` và thư mục weights. L0A1 log mới nhất đạt batch 371/375 (~98%) của epoch 1; chưa có mAP50 cho nhánh này. Công cụ chụp ảnh màn hình không trả ảnh ở lần kiểm tra này; bằng chứng dùng bảng CSV và URL Version 5, không dùng lại ảnh cũ làm bằng chứng kết quả mới.


## 24. Các hướng còn chưa làm và thứ tự ưu tiên

**Chưa có cơ sở kết luận YOLO26n không tăng mAP50 thêm được.** L1A0 tăng từ 73,771% lên 74,850% giữa hai epoch trong cùng lượt, nhưng so với BEST v0.5 ban đầu lịch sử ~74,8% thì gần ngang nhau. So sánh +0,6 điểm với đối chứng học thêm LR 0,001 không có nghĩa BEST hiện tại đã tăng thêm +0,6 điểm. Phép chấm chung đang chờ sau train. Mục tiêu 85% cách mức ~75% khoảng 10 điểm phần trăm; không hứa tinh chỉnh nhỏ đủ bù khoảng này.

### 24.1 Trạng thái đối chiếu

| Phần | Hướng | Đã làm | Còn chưa làm |
|---|---|---|---|
| 7, 18.5 | LR và học thêm | L0A0 2 epoch; L1A0 2 epoch | L2A0 LR 0,0001; xác nhận ứng viên 8–10 epoch với đối chứng cùng ngân sách; học tiếp 5 epoch khi validation còn tăng |
| 8 | Augmentation và tương tác LR | L0A1 đang epoch 2; epoch 1 AP50 ≈72,9% | L0A2 tắt mạnh augmentation; L1A1, L1A2, L2A1, L2A2. Cùng L2A0 là 6 nhánh chưa chạy |
| 9 | Fine-tune hai giai đoạn | Chưa | F0/F1 không freeze so freeze→unfreeze; R0/R1/R2 thích nghi→tinh chỉnh nhẹ/mạnh, theo stage và đối chứng đã mô tả |
| 10 | Resolution | Inference 960/1280 trên 300 val; smoke cũ thay cả batch và resolution | Inference 1536; nếu có tín hiệu, train 1280 và 1536 cùng batch 16, mỗi lượt 2 epoch; xác nhận trên đủ 2.500 val |
| 10 | SAHI | Tile 512/overlap 20% trên 300 val, AP50 giảm và chậm hơn | Tile 320/overlap 25%, và điều chỉnh cách ghép box; đây là inference, không tạo trọng số học thêm |
| 11 | Cân bằng độ khó | Thống kê lớp/kích thước dữ liệu | Xây pool train khó đã kiểm nhãn; H1 20%/H2 40%, giữ ngân sách 12.000 lượt/epoch, giới hạn lặp 2 lần và giữ nguồn/lớp |
| 12 | Audit FP/FN | Đối sánh tự động trên 300 val | Xem ảnh bị báo nhầm/bỏ sót, đặc biệt chair/phone; xác định nhãn thiếu/sai, che khuất, vật nhỏ, nhầm lớp hoặc box lệch; sửa theo version và lập lại baseline nếu cần |
| 13 | YOLO26s | Chưa | Cân nhắc sau các hướng ưu tiên; đo AP và độ trễ trên cùng thiết bị, không kết luận S kém chỉ sau 2 epoch |

### 24.2 Đề xuất tiếp theo

1. Hoàn tất L0A1 và chấm chung BEST ban đầu/L1A0/L0A1. Giữ checkpoint tốt nhất theo mAP50, tránh thay bằng last nếu kém.
2. Audit FP/FN bằng mắt trên các mẫu chair/phone để chọn hướng theo nguyên nhân. Dùng lỗi val xác định kiểu vấn đề; tìm mẫu tương ứng trong train, không đưa ảnh val sang train.
3. Thử L1A1 (LR 0,0005 + augmentation nhẹ) và L2A0 (LR 0,0001 + augmentation gốc), mỗi lượt 2 epoch. Đây là hai nhánh trong kế hoạch chưa chạy, chưa chắc tốt hơn.
4. Xác nhận nhánh có tín hiệu bằng 8–10 epoch với đối chứng cùng checkpoint ban đầu/split/seed/schedule/số epoch. Không mặc định nối thêm nhiều lượt khi chỉ loss train giảm.
5. Từ kết quả audit, ưu tiên H1/H2 hoặc R1/R2; freeze→unfreeze là nhánh phụ vì checkpoint đã fine-tune trên dữ liệu này. Resolution 1536/SAHI 320 ưu tiên khi lỗi vật nhỏ nổi trội; tile512 đã có kết quả không thuận lợi.
6. Nếu sau các lượt có đủ thời gian hội tụ và cải thiện dữ liệu vẫn không tăng, xem YOLO26s theo phần 13. Chưa mở rộng sang phần 14–17.

Đây là thứ tự đề xuất, **chưa tự khởi chạy thêm lượt GPU ngoài hai lượt chạy bù đã được yêu cầu**. Giữ yêu cầu của bạn: phần **14 distillation, 15 ensemble, 16 RF-DETR chưa cân nhắc; phần 17 đọc sau**.

Cơ sở chung: [Ultralytics model training tips](https://docs.ultralytics.com/guides/model-training-tips/) giải thích ảnh hưởng của dữ liệu, số epoch, LR schedule và quy trình fine-tune. Các mức LR, số epoch và thứ tự trên là thiết kế thử nghiệm cho dự án, không phải cam kết đạt 85%.


## 25. Thực hiện các bước ngắn, mục tiêu trước mắt 80% mAP50

### 25.1 Phạm vi đã thực hiện theo yêu cầu mới

Yêu cầu của bạn: làm phần đơn giản ngay, việc tốn nhiều thời gian để sau; 80% là mục tiêu trước mắt. Đã làm thêm:

1. Đọc bảng chấm chung cuối của hai lượt chạy bù trên toàn bộ 2.500 validation.
2. Thử tăng resolution inference lên 1536 trên cùng 300 ảnh của pilot cũ.
3. Thử lọc box trùng theo lớp, IoU 0,5/0,6/0,7/0,8, trên dự đoán 1280 đã lưu. Chọn một ứng viên để xác nhận trên đủ 2.500 val.
4. Xem bằng mắt 16 panel lỗi chair/phone của 15 ảnh val khác nhau; xem thêm 6 ảnh train từ ba nguồn để kiểm nguyên nhân dữ liệu.
5. Khởi chạy đúng hai cấu hình còn thiếu, mỗi lượt 2 epoch: L1A1 và L2A0. Không mở rộng toàn bộ ma trận hay train model khác.

Chưa có phép đo nào đạt 80% được xác nhận tại thời điểm khởi chạy hai nhánh này. Hai epoch là sàng lọc nhanh, không đủ kết luận năng lực tối đa hay hứa đạt mục tiêu.

### 25.2 Kết quả chấm chung của lượt chạy bù — đã hoàn tất

Nguồn: `pilot_comparison.json`, [Kaggle Version 6](https://www.kaggle.com/code/awm3636/project8-v05-fast-archive-from-best-v04/output?scriptVersionId=356731163). Version 6 đã Quick Save thành công với 95 output files, gồm bảng chấm chung và ZIP 97,4 MB. Snapshot Version 5 là giữa chừng; dùng Version 6 cho kết quả cuối.

Cùng 2.500 val, imgsz 1280, batch 8, GPU0, FP16/quantize=16, conf=0,001, nms=False, rect=True, max_det=300, augment=False. Cùng manifest và checkpoint ban đầu đã xác minh SHA256.

| Checkpoint | Epoch chọn | mAP50 | mAP50–95 | Precision | Recall | Chênh mAP50 với baseline |
|---|---:|---:|---:|---:|---:|---:|
| BEST v0.5 ban đầu | — | **74,710%** | **53,281%** | 80,633% | 66,312% | 0 |
| L1A0: LR một nửa | 2 | **74,829%** | 53,032% | 77,751% | **68,255%** | **+0,119 điểm** |
| L0A1: augmentation nhẹ | 2 | 74,318% | 51,965% | 77,172% | 67,110% | −0,392 điểm |

Giảm LR tăng mAP50 rất ít, đồng thời mAP50–95 và precision giảm, recall tăng. Giảm augmentation riêng chưa có lợi trong lượt 2 epoch này. Không lấy mức +0,6 điểm so với đối chứng cũ làm mức tăng của BEST ban đầu; mức tăng cùng protocol thực tế chỉ +0,119 điểm. Chưa thay checkpoint triển khai chỉ từ chênh nhỏ này.

Đối chứng L0A0 mất checkpoint sau lỗi phiên nên vẫn nằm trong `unavailable_runs`; không có phép chấm chung L0A0. Kết quả lịch sử của đối chứng chỉ dùng tham khảo.

Tệp local: `kaggle_common_comparison_summary.json` chứa số liệu trích từ preview; `kaggle_common_comparison_preview.txt` giữ nội dung UI đọc được. Summary không giả làm bản JSON gốc đầy đủ từng lớp.

### 25.3 Inference 1536 và lọc box trùng — đã thử trên 300 val

Giữ BEST v0.5 ban đầu, CPU FP32, cùng 300 ảnh/cùng GT/cùng COCOeval của phần 22. Vì protocol này khác Ultralytics GPU FP16 trên 2.500 ảnh, **không so trực tiếp các mức dưới đây với 74,710% ở trên**.

| Cấu hình | mAP50 | mAP50–95 | Chênh mAP50 so 1280 gốc |
|---|---:|---:|---:|
| 1280 gốc, end-to-end | 72,181% | 51,940% | 0 |
| 1536 gốc, end-to-end | 70,358% | 48,427% | −1,823 điểm |
| 1280 + lọc box trùng IoU 0,5 | 74,075% | 52,062% | +1,894 điểm |
| 1280 + lọc box trùng IoU 0,6 | **74,154%** | 52,238% | **+1,973 điểm** |
| 1280 + lọc box trùng IoU 0,7 | 73,987% | 52,369% | +1,806 điểm |
| 1280 + lọc box trùng IoU 0,8 | 73,729% | **52,407%** | +1,548 điểm |

Lọc box ở đây là **hậu xử lý thêm trên cùng đầu ra end-to-end đã lưu**, tách từng lớp, giữ box điểm cao trước và loại box cùng lớp chồng lấn vượt IoU đã chọn. Không đổi sang head one-to-many, không đặt `nms=True` trong YOLO, không học thêm và không tăng kích thước trọng số. Trong cảnh đông vẫn có thể loại nhầm hai vật cùng lớp chồng nhau; cần kiểm recall từng lớp.

Đã chọn IoU 0,6 theo mục tiêu mAP50 để xác nhận trên toàn bộ 2.500 val, dùng một lượt inference rồi chấm cả trước/sau trên đúng cùng dự đoán. Ngưỡng này được chọn từ một phần của val, vì vậy phép xác nhận đầy đủ vẫn là validation, chưa phải đánh giá độc lập trên test. Chưa thay mặc định ứng dụng.

Tăng resolution 1536 hiện không có tín hiệu tốt ở inference pilot; chưa dành GPU train 1536. Kết quả này không chứng minh train lại ở 1536 không thể cải thiện.

Bằng chứng: `quick_followups/results.json`, `quick_followups/imgsz1536_predictions.json.gz`; script `.temp/quick_followups_20261009.py`. Phép xác nhận đầy đủ: `.temp/fullval_nms_20261009.py`, thư mục `fullval_nms`.

### 25.4 Audit FP/FN đã xem bằng mắt và phát hiện nhãn thiếu trong train

**Validation:** đã xem 15 ảnh riêng, 16 panel từ các lỗi FP/FN chair/phone nhiều nhất; đây là mẫu có chủ đích, không đại diện toàn bộ dữ liệu.

- Ảnh val chair số 60: có ghế rõ nhưng GT chair bằng 0. Một phần box bị đếm FP thực chất nằm trên ghế chưa gắn nhãn; đồng thời có box nhầm vào bàn. Cần kiểm chính sách nhãn đủ tám lớp.
- Chair số 175: một ghế có ba box dự đoán chồng nhau. Đây là lý do thử hậu xử lý lọc box trùng.
- Các ảnh chair khác: che khuất bởi người/bàn, nhầm sofa, box chỉ lưng ghế so với toàn ghế. Cần chính sách box và truncation/occlusion thống nhất.
- Phone số 1: điện thoại cầm tay rõ có hai box dự đoán chồng với chiều dài khác nhau.
- Phone trong các ảnh lớp học: vật rất nhỏ, bị tay che, dễ lẫn sách/tài liệu; làm nét không tạo lại chi tiết vốn không có trong ảnh.

**Train:** đã xem 6 ảnh, hai ảnh mỗi nguồn `items_in_room`, `merged_3k_local`, `celulares`, chọn cố định seed42. Các box GT của tất cả lớp đều đã vẽ trên ảnh, không chỉ lớp đang xét.

| Panel train | Quan sát xác nhận | Việc cần sửa sau khi áp chính sách nhãn |
|---|---|---|
| 3, merged_3k_local | Rất nhiều người/bàn rõ, GT chỉ chair | Thêm person/table; kiểm bottle và box ghế |
| 5, celulares | Người và balô rõ, GT chỉ phone | Thêm person/backpack; kiểm book và người nền |
| 6, celulares | Hai người, bàn, sách/tài liệu, ghế rõ, GT chỉ phone | Thêm person/table/book/chair phù hợp phạm vi lớp |

Ba ảnh này đã đưa vào `visual_audit/confirmed_train_review_queue.csv`, kèm manifest ID và đường dẫn ảnh. Không tự sửa nhãn mơ hồ hoặc copy ảnh val sang train. Những ảnh train còn lại trong sáu ảnh chưa thấy lỗi chắc chắn; không coi là đã audit hết nguồn.

Kiểm đếm nguồn train cũng cho thấy:

- `merged_3k_local`: 2.426 ảnh, chỉ có nhãn chair/phone/book.
- `celulares`: 772 ảnh, chỉ có nhãn phone.
- `items_in_room`: 364 ảnh, nhãn sáu lớp; không có person/book.

Sự vắng lớp trong thống kê **tự nó không chứng minh thiếu nhãn**; các ảnh đã xem ở trên là bằng chứng cụ thể. Khi ghép dữ liệu chỉ gắn một vài lớp vào bài toán tám lớp, vật chưa gắn nhãn có thể bị học như nền, tạo tín hiệu xung đột. Đây là hướng dữ liệu cần ưu tiên hơn tiếp tục đoán LR.

Cách làm tiếp: rà các nguồn có bằng chứng nhãn thiếu, gắn đủ tám lớp trên **train**, thống nhất ghế/sofa, sách/giấy, cốc/chai và vật bị che; lưu version mới và hash mới. Nếu sửa lỗi GT val đã xác nhận thì đánh giá lại cả baseline và ứng viên trên cùng val đã sửa, đồng thời lưu bảng cũ. Không gọi chênh điểm do đổi nhãn là model học tốt hơn.

Bằng chứng: `visual_audit/findings.csv`, `findings.json`, `sheet_1.jpg`–`sheet_4.jpg`, `train_source_audit.jpg`, `train_source_audit_index.json`, `source_annotation_coverage.json`. Full visual review vẫn **False**.

### 25.5 Hai lượt GPU bổ sung — đã bắt đầu

| Nhánh | LR0 | Augmentation | Epoch | Trạng thái lúc khởi chạy |
|---|---:|---|---:|---|
| L1A1_half_lr_light_aug | 0,0005 | light như L0A1 | 2 | Đã bắt đầu epoch1 trên T4 ×2 |
| L2A0_low_lr_base_aug | 0,0001 | base như L0A0 | 2 | Chạy kế tiếp trong cùng cell |

Run timestamp `20261009_180315`. Cùng checkpoint ban đầu, 12.000 train/2.500 val, seed42, MuSGD, imgsz1280, batch32 và schedule ngắn của các pilot trước. Không tiếp tục từ L1A0. Lưu từng epoch; chọn checkpoint mAP50 tốt nhất rồi chấm cùng protocol 25.2; đóng ZIP sau cả hai. Bộ config qua `get_cfg`; cell Python compile thành công.

Output: `/kaggle/working/BTL-AI-Camera-v05-pilots/quick_followup_20261009_180315`. Cell thực thi lưu ở `.temp/kaggle_quick_followup_cell.py` và cuối notebook local `notebooks/kaggle/project8_v05_simple_pilots_2epochs.ipynb`.

### 25.6 Những phần để sau và tiêu chí dừng

- Lượt xác nhận 8–10 epoch có đối chứng cùng ngân sách, fine-tune hai giai đoạn, pool train khó/cân bằng mẫu: để sau.
- Rà/gắn lại hàng nghìn ảnh nguồn, train lại dataset đã sửa: cần thời gian riêng, không làm ngẫu nhiên để đổi điểm val.
- SAHI tile320, train1536, YOLO26s: để sau các kết quả nhanh và sửa dữ liệu. Tile512 hiện giảm điểm và chậm hơn.
- Phần 14 distillation, 15 ensemble, 16 RF-DETR vẫn chưa cân nhắc; phần17 đọc sau.
- Không chạy thêm nhánh ngoài hai lượt trên chỉ vì chưa tới80. Chọn ứng viên có bằng chứng cải thiện, rồi quyết định một lượt xác nhận dài hoặc sửa dữ liệu trước. Model lớn chỉ cân nhắc nếu các hướng này không đủ.

Nếu checkpoint tốt nhất chỉ quanh75%, còn thiếu khoảng5 điểm để80; vài phần mười từ LR không đủ bảo đảm đạt. Hậu xử lý lọc box trùng có tín hiệu tốt hơn ở pilot, nhưng phải dựa kết quả xác nhận đầy đủ và AP từng lớp trước khi dùng.

### 25.7 Xác nhận NMS đầy đủ trên 2.500 validation — đã hoàn tất local

Một lượt forward CPU FP32, batch1/rect/imgsz1280, đúng 2.500 val và 8.359 GT; cùng đầu ra trước/sau NMS, không đổi head. Đã kiểm hash từng ảnh/nhãn và giữ test riêng. Hai bộ metric cùng cho cải thiện:

| Bộ metric | Baseline | Thêm NMS theo lớp IoU0,6 | Chênh mAP50 |
|---|---:|---:|---:|
| COCOeval maxDets100 | 74,512% | 76,479% | +1,967 điểm |
| Ultralytics8.4.165 matching/AP từ cache | 74,690% | **76,715%** | **+2,025 điểm** |

mAP50–95 Ultralytics tăng 53,264% → 53,550%. Precision/recall ở ngưỡng F1 chọn bởi metric tăng 78,920%/67,183% → 80,060%/69,272%. Đây không phải recall vật nhỏ ở mọi ngưỡng: **COCO ARsmall50–95 giảm 40,338% → 36,627%**, dù APsmall tăng 16,747% → 17,365%. NMS bỏ box trùng nhưng cũng có thể loại vật thật chồng nhau; cần đánh giá tình huống đông và vật nhỏ.

| Lớp | AP50 baseline Ultralytics | AP50 sau NMS |
|---|---:|---:|
| person | 82,233% | 84,134% |
| table | 75,695% | 77,504% |
| chair | 59,238% | 61,342% |
| laptop | 74,097% | 75,804% |
| cell phone | 67,742% | 71,154% |
| backpack | 82,259% | 84,003% |
| book | 71,633% | 73,965% |
| bottle | 84,625% | 85,816% |

Thời gian local: cả vòng inference/chấm ~502 giây; median forward164,58ms/ảnh; median NMS0,098ms và p95 NMS9,546ms trên CPU này. Không suy ra FPS camera/GPU từ số đó. NMS không tăng số tham số hoặc kích thước model.

Protocol CPU khác FP16/batch8 Kaggle; không cộng mức +2,025 điểm vào checkpoint Kaggle để đoán kết quả. Muốn chốt trọng số mới + NMS phải chấm thật. Ngưỡng được chọn từ 300 ảnh thuộc cùng val; không coi đây là test độc lập. Chưa thay cấu hình mặc định ứng dụng.

Bằng chứng: `fullval_nms/results.json`, `ultralytics_cached_metrics.json`, `ground_truth.json`, `predictions.json.gz`. Validator NMS GPU chuẩn bị đã kiểm số box và cặp lớp/score khớp reference trên 300 ảnh (`adapter_verification.json`).

### 25.8 Kiểm tra lại lúc 22:03 ngày 09/10/2026

Hai lượt bổ sung đã hoàn thành; log draft có `ALL_QUICK_FOLLOWUPS_COMPLETE` và ZIP. Phiên GPU hiện đã off; bước xác nhận NMS trong cell sau **chưa được thực thi**. Không nhầm phiên off với train chưa xong.

| Nhánh | mAP50 chấm chung | Chênh với BEST ban đầu 74,710% |
|---|---:|---:|
| L1A1 LR0,0005 + augmentation nhẹ | **75,484%** | **+0,775 điểm** |
| L2A0 LR0,0001 + augmentation gốc | 75,423% | +0,713 điểm |

Nhánh L1A1 dẫn 0,062 điểm, rất nhỏ; chưa đủ xác nhận ổn định giữa seed. Cả hai đều có tín hiệu tốt hơn các lần giảm riêng lẻ trước. Chưa đạt80% bằng trọng số đơn lẻ. Log thực thi giữ ở `kaggle_quick_followups_log.txt`. Đang bảo toàn output và kiểm NMS trên checkpoint dẫn đầu; không chạy lại train nếu không cần.

### 25.9 Lưu phiên bản khi session đã off và câu hỏi về box

Kiểm tra lúc khoảng22:09: [Version7](https://www.kaggle.com/code/awm3636/project8-v05-fast-archive-from-best-v04?scriptVersionId=356791072) Quick Save Successful nhưng **Output0files**. Version này giữ notebook/log hai lượt hoàn tất, chưa bảo toàn trọng số mới. Không gọi QuickSave Successful là bằng chứng checkpoint đã tồn tại. Version6 vẫn giữ95files của các lượt trước. Đã bắt đầu phiên kiểm ổ/kết quả còn khôi phục được không; chưa tự train lại toàn bộ.

Mức **76,715%** là BESTv0.5 ban đầu + hậu xử lý class-awareNMSIoU0,6, cùng2.500val/CPUFP32/cùng dự đoán. Baseline theo đúng metric này là74,690%, tăng2,025điểm. Không phải model đã train tới76,715%, không phải điểm của checkpoint L1A1, không cộng số tăng NMS vào75,484% để suy ra kết quả mới.

Cách tận dụng box:

1. **Box dự đoán dư:** lọc box trùng khi inference, kiểm trường hợp nhiều vật cùng lớp chồng nhau. Cơ chế này giúp giảm FP, không tạo trọng số để học tiếp.
2. **Box nhãn thiếu:** bổ sung nhãn đúng cho các vật thuộc8lớp trong train; tránh để model học người/bàn/ghế chưa gắn nhãn thành nền. Dùng danh sách3ảnh train đã xác nhận làm điểm bắt đầu, sau đó rà nguồn có cùng kiểu lỗi.
3. **Box làm gợi ý annotation:** model hỗ trợ vẽ trước, người/kiểm duyệt xác nhận lớp/phạm vi trước khi học. Không dùng toàn bộ dự đoán làm groundtruth tự động, không chuyển val/test sang train.
4. **Học tiếp:** dùng checkpoint tốt + dataset train đã sửa, đo lại baseline/ứng viên nhất quán. Nếu đổi GTval do sửa nhãn thì lập lại cả baseline và ứng viên trên phiên bản mới; không coi tăng do đổi chuẩn chấm là model cải thiện.

Thêm càng nhiều box không tự bảo đảm tốt; box đúng vật/lớp/vị trí và chính sách annotation nhất quán mới tạo tín hiệu học hữu ích. Rà hàng nghìn ảnh vẫn thuộc công việc để sau theo yêu cầu thời gian của bạn.
