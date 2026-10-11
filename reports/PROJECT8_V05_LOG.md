# Project8 v05 — log thí nghiệm và kiểm thử

## Đổi tên ba trọng số hiện hành — 11/10/2026

R04 đổi thành `R04_YOLO26n_Project8_v05_finetuned_default.pt`; baseline đổi thành `Baseline_YOLO26n_Project8_v05_20261009.pt`; R01 đổi thành `R01_YOLO26n_Project8_v05_recovered_epoch0.pt`. Giữ nguyên ba nội dung và SHA256; cập nhật Streamlit/config app/config train/evaluator. Tên dropdown ghi vai trò dễ hiểu, không gắnNMS/size/sốlượt vì đó là cấu hình chạy. [Trọng số và đường dẫn hiện hành](ACTIVE_WEIGHTS_20261011.md), [manifest đổi tên](weights_renamed_20261011.json). Đường dẫn trong kết quả thí nghiệm cũ là tên tại thời điểm đo.

## Cập nhật Streamlit: kích thước, NMS và một/ba lượt — 11/10/2026

Đã thêm size512/640/768/960/1280, một hoặc ba lượt trên cùng checkpoint, NMS tắt/0,5/0,6; ba lượt bắt buộc NMS0,5 hoặc0,6. Mặc định R04/640/một lượt/NMS tắt/CPU. Có hướng dẫn sidebar, vô hiệu kết quả ảnh cũ khi đổi cấu hình, dừng webcam để áp dụng phiên mới, JSON ghi đúng cấu hình. 47test đạt, AppTest chuyển chế độ đạt, model thật và trình duyệt chạy ba lượt thành công. Khôngtrain/chấmtest; chưa benchmark webcam ba lượt hoặc5060. [Log triển khai chi tiết](STREAMLIT_INFERENCE_CONTROLS_20261011.md).

Lưu ý sửa mô tả preview lịch sử bên dưới: YAML có preview_fps10, nhưng giao diện webcam thực dùng fragment0,3giây, nên FPS hiển thị không suy ra từ giới hạn YAML hoặc FPS model T4. Các đoạn “chưa đổiNMS/app” bên dưới là trạng thái trước cập nhật này.

Theo yêu cầu mới11/10/2026, đã giữ đúng3PT trongweights(R04/baselinev05/R01), đặtR04 mặc địnhStreamlit/config vàbỏ7PTkhác≈56,38MiB. Đã kiểmhash/nạpmodel. Đây là3ứngviên hiệncó, khôngtop3toàncampaign; LASTv05 bịbỏ do giới hạn3file, khôngcókếtluận rankingchung. ChưađổiNMS/GPU haytrain. [Trọng số hiện hành vàmanifest](ACTIVE_WEIGHTS_20261011.md).

Mục tiêu bổ sung11/10/2026: nghiệm thu riêng webcam trênCPU máy hiện tại vàRTX5060Laptop8GB; chưa đổiapp. Cấu hình preview hiện10FPS nên không đồng nhấtFPS model vớiFPS UI. [Mục tiêu và việc còn thiếu chohai thiếtbị](WEBCAM_CPU_GPU_GOALS_20261011.md). Dọn7file tạm/trùng≈9,31MiB, giữcheckpoint/dataset/sốđo và mãbenchmark; [manifest dọnfile](cleanup_20261011/completed.json).

Tiến trình tổng hợp và phần còn thiếu cập nhật11/10/2026: [PROJECT8_PROGRESS_20261011.md](PROJECT8_PROGRESS_20261011.md).

## R04 640 trên một T4 — latency và FPS, 11/10/2026

Ước tính webcam bổ sung theo yêu cầu: giả định overhead chưa đo5/10/20ms trênT4 FP32 mean14,77ms thìFPS xử lý≈50,6/40,4/28,8; camera30FPS bị giới hạn≈30/30/28,8, camera60FPS≈50,6/40,4/28,8. Đây là kịch bản pipeline tuần tự khôngtíchhàngđợi, chưađocamera/tracker/UI và khônggiảlập5060Laptop. Xem bảng giả định trongbáocáo và `r04_single_t4_latency_20261011/real_webcam_estimates.json`.


Đã hoàn tất 6.000 phép đo: 200 ảnh validation từ 15 nguồn × 10 vòng × CPU FP32/T4 FP32/T4 FP16; chỉ GPU 0, batch 1, một forward, conf 0,25, NMS 0,6 và vẽ. Trên cùng VM CPU 2 threads: p50/p95/p99=71,09/79,13/84,96 ms, 7/2000 vượt100ms, khoảng14,48 FPS. T4 FP32: 14,00/20,78/25,49ms, max65,00ms, 0/2000 vượt100ms, khoảng67,70 FPS xử lý tuần tự. T4 FP16:14,97/21,39/26,36ms, bốn outlier vòng đầu lên2870,27ms; tổng khoảng47,08 FPS, các vòng2–10 khoảng63,97 FPS. Không loại outlier khỏi tổng; nguyên nhân chưa xác định. FP32 là lựa chọn ưu tiên về latency/ổn định trong phép đo này.

FPS là 1000/mean_ms trên ảnh nạp sẵn, chưa gồm capture/tracker/UI/network, không phải FPS webcam thực hay giả lập5060Laptop. Khôngtrain/chấmtest/đổiapp. Đã lưu đủ6000mẫu và ảnh chụp kết quả; dừngGPU. Chưa xác nhậnversionKaggle mới. [Báo cáo latency/FPS](r04_single_t4_latency_20261011/RESULTS.md).


## Yêu cầu đủ R01–R11 — 11/10/2026

Kết quả bước khôi phục mới: R01 đã chấm xong512/640/768/960 trênGPU T4, đủ2500val/8359GT mỗi lượt. CộngR04 dùng lại từcompletionJSON, hiện2/11 model và8/44 cặpmodel/size, cònR02/R03/R05/R06/R07/R08/R09/R10/R11 thiếucheckpoint. Runner ghi `partial_missing_checkpoints`, khôngghi complete. RuntimeCPU soR01/R04 xen kẽ cùng40ảnh×3lặp cũng xong; ở640 p95R01≈97,46ms/R04≈91,34ms, vẫn có5/120 và3/120frame vượt100ms, chưa cócapture/tracker/UI. Khôngtrain/chấmtest. [Báo cáo đủ11 đang bịchặn](webcam_all11_20261011/RESULTS.md).

Người dùng yêu cầu bật Kaggle và chấm đủ11 model, ở512/640/768/960 (44phép chấm; baseline không tính là một trong11). Đã bật phiên Kaggle và chuẩn bị runner validation riêng. R01 đã khôi phục, R04 giữ lại; chín checkpoint còn thiếu sau lượt dọn. Không đánh dấu đã đủ11. Vì chỉ thị trước đó là chưatrain, đang cần người dùng quyết định có cho phép tạo model mới bằng tái train chín cấu hình từ checkpoint dự án hay tiếp tục chỉ khôi phục model cũ. Xem [trạng thái đủ11](webcam_all11_20261011/STATUS.md).

## Ưu tiên webcam mới nhất — 10/10/2026

Cập nhật sau khi hoàn tất sang11/10/2026: **đã chấm R04 và baseline trên Kaggle ở đủ512/640/768/960, tổng8lượt, mỗi lượt2500val/8359GT**, chưa train/chấmtest. Ở640 GPUFP16 +NMS0,6: R04 mAP50/mAP50–95/F1conf0,25=71,603292%/49,596612%/66,458912%; baseline=71,176754%/50,382277%/66,142167%. R04 hơnAP50/F1 nhưng baseline hơnAP95; không có ứng viên thắng mọi metric. Giao diện trước đó đứng output; sau kết nối/khởi động kernel lại, JSON đã lưu trênworking xác nhận đủ8lượt và được lấy về `webcam_sizes_20261010/kaggle_observed_results.json`. Các protocol thực tế lưu nguyên trongrecord.

Kết quả đầy đủ đã lưu local và kiểm đủ8record/cache parity. Đã thử Quick Save kèm output chỉ cho phiên này, nhưng Version History vẫn chỉ xác nhậnversion8; **chưa xác nhậnversion9 được lưu**. PhiênGPU đã dừng sau khi lấy đủ kết quả để tránh tốn quota. Hai API screenshot trình duyệt báo không chụp được; bằng chứng hiện là completionJSON đọc trực tiếp từ output cell, không có ảnh chụp xác nhận mới.

RobustnessCPU đã xong299val ở5chế độ cho cả4size; tài nguyên/latency120samples/size cũng xong. Camera0 không mở được qua các backend nên **chưa đo được FPS/latency webcam thực**. Confidence sweep từcache đủ2500val:640 conf0,20 cóF1micro66,918124%, nhưngprecision73,311485% thấp hơnconf0,25=76,547638%; chưa benchmark runtime ởngưỡng0,20 nên không tự thayngưỡng triển khai. Xem [báo cáo bốn kích thước và10mục](webcam_sizes_20261010/RESULTS.md). Quyết định tạm thời vẫnR04/640, mộtforward,NMS0,6,conf0,25; không thayapp/alias. Chưa chạy hếtR01–R11 vì checkpoint thiếu, không gọi đây là model tốt nhất toàncampaign.

Người dùng chốt độ trễ ưu tiên số1, mAP50/mAP50–95 số2, đồng thời yêu cầu F1 cao và không chấp nhận giảm F1 quá nhiều để đổi lấy tốc độ. Vì vậy dùng độ trễ làm điều kiện đáp ứng, rồi chọn chất lượng tốt nhất trong các phương án đáp ứng; không xếp hạng chỉ theo số ms nhỏ nhất. Một forward/frame, không train. Chính sách này thay thế các đề xuất ưu tiên F1 đơn độc hoặc chọn512 chỉ vì nhanh nhất. Xem `webcam_sizes_20261010/selection_policy.json`.

R04 đã chấm đủ2500val ở512/640/768/960 trên CPU. Replay120 lần/size gồm predict+NMS+vẽ: p95 tương ứng75,79/82,48/110,65/147,17ms; F1micro conf0,25 tương ứng59,37/66,46/70,44/73,11%. **640 là ứng viên cân bằng tạm thời**: hơn512 7,09điểm F1 với p95 tăng6,69ms;768/960 vượt100ms ởp95. Replay chưa gồm capture/tracker/UI và vẫn có2/120frame vượt100ms ở640. Chưa gọi đây là nghiệm thu webcam.

Đã khôi phục R01 epoch0 từ Kaggle version8; SHA256 khớp bản đã chấm trong campaign: `3323fb2f72a91e2b827d4d7dc696fa66e57ed168d6e138e9e8af9a7c41ee5ece`. Lưu `weights/recovered_20261010/R01_epoch0.pt`. Không train để tái tạo checkpoint. R02/R07 vẫn chưa có artifact xác nhận. Đánh giá GPU bốn kích thước và robustness đang được hoàn tất; trạng thái và kết quả chi tiết lưu ở `webcam_sizes_20261010`.

Quyết định bổ sung10/10/2026 theo yêu cầu xét10tiêu chí: chọn phương án hiện tại **R04 raw, một forward640, extra class-awareNMS0.6, conf0.25**, doF1micro/AP tăng với chi phíNMS nhỏ. Không phảiR04 cộng3views. FPS thực, tài nguyên vàrobustness còn chưa benchmark; không tuyên bố model này thắngR01–R11 ở640. Xem [10 tiêu chí và quyết định](webcam640_20261010/SELECTION_10_CRITERIA.md). Chưa đổiapp hoặctrain.

## Kết quả webcam640 mới nhất — 10/10/2026

Đã chấm R04 một lượt đủ2500val/8359GT ở640CPU FP32: native mAP50=69,479293%, mAP50–95=49,239841%, P=77,908181%, R=61,549256%; thêm class-awareNMS0.6 từ cùng prediction đạt71,576838%/49,617590%, P=77,503701%, R=65,119777%. P/R trên là evaluator; cùngconf0.25microIoU0.5: nativeP/R/F1=70,523297%/58,846752%/64,158080%; NMS=76,547638%/58,727120%/66,463580%. NMS không thêm forward; giảm552FP, mất10TP. Native cache replay khớp validator trong1e-10.

Replay40ảnh batch1CPU sau5warmup: predict native p50/p95=51,81/71,40ms; đợt riêng predict+NMS=48,87/60,69ms, phầnNMS=0,42/1,44ms. Không kết luậnNMS làm model nhanh hơn vì dao động giữa đợt. Chưa đo camera capture/tracker/drawing/UI; chưa xác nhận FPS hoặc mọi frame dưới100ms.

Chưa khôi phục đượcR02/R07: local PT/ZIP đã bịdọn; Kaggle chỉ xác nhậnversion8 outputđếnR01, draftoff, chưa thấyartifactR07. Khôngtrain để tái tạo. **Chưa chọn được model tốt nhấtR01–R11 ở640**, chỉ đề xuấtR04+NMS0.6 là cấu hình cân bằng hơn trong hai cấu hìnhR04 đã đo. Khôngthayapp/alias, khôngchấmtest500. Mốc80,505213% cũ là3views1280, không phải webcam640. Chi tiết và hạn chế: [reports/webcam640_20261010/RESULTS.md](webcam640_20261010/RESULTS.md), JSON/cache giữ để đối soát.

## Chuyển tiêu chí sang webcam một lượt — quyết định người dùng 10/10/2026

- Mục tiêu sản phẩm là webcam: chỉ một lượt suy luận model cho mỗi frame được xử lý, imgsz=640 cho lần so sánh và thử tiếp theo. Không chọn pipeline ba views làm mặc định webcam. Người dùng nêu chi phí gần100ms mỗi lượt làm ba lượt quá chậm; đây là nhận định/giới hạn người dùng yêu cầu, chưa phải benchmark webcam được xác nhận trong lượt này.
- Mục tiêu dưới100ms/ảnh cần đo trên phần cứng thực tế, batch1, sau warmup; báo p50/p95 inference riêng và xử lý toàn frame, cùng FPS/frame age. Không suy FPS webcam từ validator GPU batch8 hay điểm TTA1280.
- Chỉ phát triển trên checkpoint dự án hiện có. Người dùng không muốn train lại từ pretrained gốc; không chạy nhánh fresh nano, YOLO26s hoặc RF-DETR. Phạm vi hiện được xác nhận là khôi phục checkpoint và so sánh, CHƯA train/học tiếp.
- Trong R01–R11 cùng protocol GPU1280: R07 native dẫn mAP50=76.969425%, mAP50–95=55.223987% và recall=69.547600%; R02 native dẫn precision=82.373006%. R04 native=76.893780%/54.623630%, precision80.900112%, recall68.460746%. Chênh R07/R04 mAP50 rất nhỏ, chưa xác nhận ổn định hoặc thứ hạng640.
- Một lượt + extra class-aware NMS0.6: R07 mAP50=79.012742%, recall72.699166%; R02 precision82.865152%; R05 recall71.726368% cao hơn R04=70.693508%. NMS là hậu xử lý, không thêm lượt model. P/R do evaluator báo không phải phép so camera ở cùng threshold triển khai; cần chốt threshold trên val và đo P/R tại cùng điều kiện.
- Vì mục tiêu đổi sang một lượt, R04 được chọn trước đây do thắng TTA không tự là checkpoint thích hợp nhất cho webcam. Cần khôi phục R07 từ artifact thật, kiểm hash rồi so với R04 tại640 trước khi quyết định; không train lại để giả khôi phục. Chưa thay alias/cấu hình ứng dụng, chưa chấm test500.
- Review bước đầu: kiểm cấu trúc/hashes/decode đủ12000 train; không nhãn thiếu/rỗng, 6 ảnh có box vượt biên cần kiểm. Đã xem thêm36 ảnh theo nguồn và recheck sheet cũ; ledger ghi27 ảnh tạm loại do lỗi/thiếu nhãn, tổng33 ảnh giữ lại để xử lý cùng6 lỗi biên. Danh sách train11967 là candidate chưa review đủ, KHÔNG training_ready và chưa kích hoạt. Xem reports/label_review_20261010/review_status.json và visual_review_ledger.csv. Không xóa ảnh gốc, không loại val/test hoặc ảnh chỉ vì khó.

Chỉ giữ một checkpoint tốt nhất của campaign; kết quả, cấu hình và kiểm thử ghi trong log Markdown này. Các checkpoint kém hơn, bản sao, cache và ZIP đã được xoá sau khi lượt so sánh CPU kết thúc.

Cập nhật: 10/10/2026 (Asia/Saigon). Kết quả tốt nhất mới nhất: **R04_three_views_nms_0.5 đạt 80,505213% mAP50 / 56,569637% mAP50–95**, đủ 2.500 validation / 8.359 GT, CPU FP32 1280, ba view `(1.0, không lật), (0.83, lật ngang), (0.67, không lật)` + class-aware NMS0.5. Chỉ giữ checkpoint R04 trong thư mục campaign; chưa chấm test độc lập. Các record cũ phía dưới là lịch sử.

Đã kiểm cả cách đóng gói pipeline này vào một PT rồi nạp bằng YOLO/default validator: full2500 ảnh/8359GT, AP50/AP50–95/P/R và tám lớp khớp trong1e-6 (thực tế các số AP/P/R khớp hoàn toàn). Artifact thử10.848.875bytes, SHA `e0200760df78c6719d700115f4077370882503e10836b5798237c370f4cf07f9`, đã xoá sau đo. Checkpoint công bố vẫn là bản raw R04 SHA `087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758`; chưa thay file công bố bằng wrapper trong lúc các ứng viên GPU còn đang train. Đây là kiểm chứng serialization, không phải tăng AP mới hay benchmark camera.

## Dữ liệu và tiêu chí

- Train 12.000 ảnh / 33.272 boxes; validation 2.500 ảnh / 8.359 boxes. Test 500 ảnh chưa chạy prediction/chấm điểm; metadata đã dùng kiểm nhóm cảnh và chuẩn bị lọc train.
- Classes: person, table, chair, laptop, cell phone, backpack, book, bottle (gộp cups + bottles).
- Manifest SHA256: `22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd`.
- Chọn epoch bằng mAP50, rồi mAP50-95 khi bằng điểm. CSV một-based, tên checkpoint zero-based.
- Full visual review: False. Không lấy prediction làm ground truth. Hash đúng không chứng minh annotation đầy đủ.
- Audit hiện hành phát hiện17 nhóm video dùng chung giữa split, tổng368 ảnh (255train/73val/40test). Đã xác nhận cùng cảnh bằng mắt ở hai nhóm train/val. Kết quả mAP trong log vẫn là số đo trên split hiện tại; chưa chứng minh độc lập giữa cảnh. Hai đối chứng fresh nano bỏ nhóm holdout/bỏ ngẫu nhiên255 train đã có code, chưa train. Không sửa split/nhãn của campaign đang chạy.
- Đích 80% ở đây là mAP50 trên validation 1280. Mục tiêu 85% theo protocol 640 trong kế hoạch cũ là protocol khác; chưa có bằng chứng đạt 95%.

## GPU Kaggle: kết quả thực đo

Ultralytics 8.4.165, Python 3.13.15, Torch 2.11.0+cu128, 2 Tesla T4. Protocol:

```json
{
  "data": "/tmp/project8_v05_15k_runtime/training_data.yaml",
  "split": "val",
  "imgsz": 1280,
  "batch": 8,
  "device": 0,
  "conf": 0.001,
  "iou": 0.7,
  "nms": false,
  "max_det": 300,
  "augment": false,
  "rect": true,
  "quantize": 16,
  "workers": 2,
  "plots": true
}
```

| Checkpoint | Pipeline | mAP50 (%) | mAP50-95 (%) | P (%) | R (%) |
|---|---|---:|---:|---:|---:|
| BASELINE | native | 74.709876 | 53.281233 | 80.632813 | 66.312232 |
| BASELINE | extra_nms_06 | 76.820560 | 53.629377 | 79.371946 | 70.536010 |
| R01_recover_L1A1 | native | 75.484442 | 53.775644 | 80.865141 | 66.782200 |
| R01_recover_L1A1 | extra_nms_06 | 77.567853 | 54.120272 | 80.600959 | 70.154018 |
| R02_light_longer | native | 76.400744 | 54.395666 | 82.373006 | 67.334471 |
| R02_light_longer | extra_nms_06 | 78.534371 | 54.764704 | 82.865152 | 69.747697 |
| R03_low_lr_base | native | 75.965756 | 54.577119 | 81.006841 | 67.657050 |
| R03_low_lr_base | extra_nms_06 | 78.016126 | 54.925144 | 80.223053 | 70.887880 |
| R04_class_weight_025 | native | 76.893780 | 54.623630 | 80.900112 | 68.460746 |
| R04_class_weight_025 | extra_nms_06 | 78.966129 | 54.956059 | 82.082540 | 70.693508 |
| R05_class_weight_05 | native | 76.722327 | 54.709159 | 80.586358 | 67.775412 |
| R05_class_weight_05 | extra_nms_06 | 78.846724 | 55.043306 | 80.530317 | 71.726368 |
| R06_no_mosaic | native | 76.326805 | 53.938839 | 80.039048 | 68.331306 |
| R06_no_mosaic | extra_nms_06 | 78.433050 | 54.317988 | 81.043875 | 71.065947 |
| R07_AdamW | native | 76.969425 | 55.223987 | 79.051814 | 69.547600 |
| R07_AdamW | extra_nms_06 | 79.012742 | 55.563895 | 80.526420 | 72.699166 |
| R08_freeze_backbone | native | 75.496356 | 53.690238 | 78.556404 | 67.485473 |
| R08_freeze_backbone | extra_nms_06 | 77.624530 | 54.092704 | 78.310883 | 71.272610 |
| R09_imgsz1536 | native | 74.582474 | 52.621882 | 78.479296 | 68.186811 |
| R09_imgsz1536 | extra_nms_06 | 76.717911 | 53.007364 | 79.475617 | 70.690851 |
| R10_mixup_cutmix | native | 74.975445 | 52.911243 | 77.437086 | 68.258791 |
| R10_mixup_cutmix | extra_nms_06 | 77.168843 | 53.257658 | 79.288246 | 70.171923 |
| R11_box_cls_loss | native | 75.816267 | 53.844932 | 79.812573 | 67.995783 |
| R11_box_cls_loss | extra_nms_06 | 77.887176 | 54.206796 | 79.562260 | 71.137810 |

GPU tốt nhất hiện đã đo: **R07_AdamW_extra_nms_06, 79.012742% mAP50**. Bảng CPU dùng riêng để so sánh TTA; R04 ba view +NMS0.5 đạt80,505213%, hai view +NMS0.6 đạt80,125017%. Không xếp trực tiếp điểm GPU một view với CPU nhiều view.

### Cấu hình và lịch sử epoch

#### R01_recover_L1A1

```json
{
  "model": "/kaggle/input/datasets/awm3636/project8-v05-best-checkpoint-pilots-20261009/BEST_Project8_v05_20261009.pt",
  "data": "/tmp/project8_v05_15k_runtime/training_data.yaml",
  "project": "/kaggle/working/BTL-AI-Camera-v05-campaign-20261010/batch_20261010_001917/train",
  "device": "0,1",
  "workers": 2,
  "cache": false,
  "fraction": 1.0,
  "patience": 0,
  "imgsz": 1280,
  "batch": 32,
  "optimizer": "MuSGD",
  "lr0": 0.0005,
  "lrf": 1.0,
  "cos_lr": false,
  "warmup_epochs": 0.25,
  "warmup_bias_lr": 0.0,
  "warmup_momentum": 0.8,
  "momentum": 0.937,
  "weight_decay": 0.0005,
  "seed": 42,
  "deterministic": true,
  "flipud": 0.0,
  "degrees": 0.0,
  "shear": 0.0,
  "perspective": 0.0,
  "bgr": 0.0,
  "mixup": 0.0,
  "cutmix": 0.0,
  "close_mosaic": 0,
  "multi_scale": false,
  "save": true,
  "save_period": 1,
  "resume": false,
  "cls_remap": false,
  "val": true,
  "nms": false,
  "max_det": 300,
  "conf": 0.001,
  "plots": true,
  "exist_ok": false,
  "mosaic": 0.2,
  "scale": 0.25,
  "translate": 0.05,
  "fliplr": 0.25,
  "hsv_h": 0.0075,
  "hsv_s": 0.35,
  "hsv_v": 0.2,
  "name": "R01_recover_L1A1",
  "epochs": 2
}
```

| Epoch | mAP50 (%) | mAP50-95 (%) |
|---:|---:|---:|
| 1 | 75.403000 | 53.732000 |
| 2 | 74.883000 | 53.340000 |

Epoch được chọn: 1; checkpoint SHA256 `3323fb2f72a91e2b827d4d7dc696fa66e57ed168d6e138e9e8af9a7c41ee5ece`.

#### R02_light_longer

```json
{
  "model": "/kaggle/input/datasets/awm3636/project8-v05-best-checkpoint-pilots-20261009/BEST_Project8_v05_20261009.pt",
  "data": "/tmp/project8_v05_15k_runtime/training_data.yaml",
  "project": "/kaggle/working/BTL-AI-Camera-v05-campaign-20261010/batch_20261010_001917/train",
  "device": "0,1",
  "workers": 2,
  "cache": false,
  "fraction": 1.0,
  "patience": 0,
  "imgsz": 1280,
  "batch": 32,
  "optimizer": "MuSGD",
  "lr0": 0.0005,
  "lrf": 1.0,
  "cos_lr": false,
  "warmup_epochs": 0.25,
  "warmup_bias_lr": 0.0,
  "warmup_momentum": 0.8,
  "momentum": 0.937,
  "weight_decay": 0.0005,
  "seed": 42,
  "deterministic": true,
  "flipud": 0.0,
  "degrees": 0.0,
  "shear": 0.0,
  "perspective": 0.0,
  "bgr": 0.0,
  "mixup": 0.0,
  "cutmix": 0.0,
  "close_mosaic": 0,
  "multi_scale": false,
  "save": true,
  "save_period": 1,
  "resume": false,
  "cls_remap": false,
  "val": true,
  "nms": false,
  "max_det": 300,
  "conf": 0.001,
  "plots": true,
  "exist_ok": false,
  "mosaic": 0.2,
  "scale": 0.25,
  "translate": 0.05,
  "fliplr": 0.25,
  "hsv_h": 0.0075,
  "hsv_s": 0.35,
  "hsv_v": 0.2,
  "name": "R02_light_longer",
  "epochs": 8
}
```

| Epoch | mAP50 (%) | mAP50-95 (%) |
|---:|---:|---:|
| 1 | 75.403000 | 53.732000 |
| 2 | 75.831000 | 54.211000 |
| 3 | 76.097000 | 54.153000 |
| 4 | 76.387000 | 54.378000 |
| 5 | 75.973000 | 53.957000 |
| 6 | 76.042000 | 53.839000 |
| 7 | 76.363000 | 54.386000 |
| 8 | 75.301000 | 54.150000 |

Epoch được chọn: 4; checkpoint SHA256 `c0b19c81af77842a59a596a057548641ca920175600edd61dd6bac620afaa0c1`.

#### R03_low_lr_base

```json
{
  "model": "/kaggle/input/datasets/awm3636/project8-v05-best-checkpoint-pilots-20261009/BEST_Project8_v05_20261009.pt",
  "data": "/tmp/project8_v05_15k_runtime/training_data.yaml",
  "project": "/kaggle/working/BTL-AI-Camera-v05-campaign-20261010/batch_20261010_001917/train",
  "device": "0,1",
  "workers": 2,
  "cache": false,
  "fraction": 1.0,
  "patience": 0,
  "imgsz": 1280,
  "batch": 32,
  "optimizer": "MuSGD",
  "lr0": 0.0001,
  "lrf": 1.0,
  "cos_lr": false,
  "warmup_epochs": 0.25,
  "warmup_bias_lr": 0.0,
  "warmup_momentum": 0.8,
  "momentum": 0.937,
  "weight_decay": 0.0005,
  "seed": 42,
  "deterministic": true,
  "flipud": 0.0,
  "degrees": 0.0,
  "shear": 0.0,
  "perspective": 0.0,
  "bgr": 0.0,
  "mixup": 0.0,
  "cutmix": 0.0,
  "close_mosaic": 0,
  "multi_scale": false,
  "save": true,
  "save_period": 1,
  "resume": false,
  "cls_remap": false,
  "val": true,
  "nms": false,
  "max_det": 300,
  "conf": 0.001,
  "plots": true,
  "exist_ok": false,
  "mosaic": 0.5,
  "scale": 0.5,
  "translate": 0.1,
  "fliplr": 0.5,
  "hsv_h": 0.015,
  "hsv_s": 0.7,
  "hsv_v": 0.4,
  "name": "R03_low_lr_base",
  "epochs": 4
}
```

| Epoch | mAP50 (%) | mAP50-95 (%) |
|---:|---:|---:|
| 1 | 75.400000 | 53.955000 |
| 2 | 75.052000 | 53.579000 |
| 3 | 75.490000 | 53.913000 |
| 4 | 75.971000 | 54.602000 |

Epoch được chọn: 4; checkpoint SHA256 `d5f828bff3d7fe18266bc3c3db11993410fa84f2dacb97f8fc8242255343ce0e`.

### AP50 từng lớp (%) trên GPU

| Pipeline | person | table | chair | laptop | cell phone | backpack | book | bottle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE_native | 82.386498 | 75.137150 | 61.123562 | 71.199682 | 68.270322 | 82.234011 | 72.419089 | 84.908696 |
| BASELINE_extra_nms_06 | 84.077974 | 77.465422 | 62.864463 | 73.533261 | 72.018622 | 83.684230 | 74.890824 | 86.029688 |
| R01_recover_L1A1_native | 83.323129 | 76.502287 | 63.558316 | 74.392400 | 68.610451 | 81.691544 | 71.614780 | 84.182631 |
| R01_recover_L1A1_extra_nms_06 | 85.059714 | 78.162678 | 65.645031 | 76.084886 | 71.580309 | 83.687728 | 74.557390 | 85.765087 |
| R02_light_longer_native | 82.695951 | 76.630993 | 66.720450 | 74.611321 | 70.793278 | 82.809443 | 73.223231 | 83.721285 |
| R02_light_longer_extra_nms_06 | 84.667884 | 78.862182 | 68.849764 | 77.169261 | 73.804234 | 83.947294 | 76.125104 | 84.849241 |
| R03_low_lr_base_native | 83.810453 | 76.815053 | 63.284260 | 73.112282 | 69.181972 | 82.846782 | 73.317064 | 85.358182 |
| R03_low_lr_base_extra_nms_06 | 85.021724 | 79.336628 | 65.182079 | 74.721495 | 72.733258 | 84.273394 | 76.114321 | 86.746107 |

## CPU FP32: kết quả thực đo riêng

Ultralytics 8.4.165, Torch 2.14.0+cpu, imgsz1280, batch8, rectTrue, conf0.001, max_det300, nmsFalse, quantizeNone. Toàn bộ 2.500 ảnh/nhãn val đã kiểm SHA256. Native cache replay khớp metric gốc trước NMS sweep. Các process có lúc chạy đồng thời nên timing không phải benchmark latency độc lập.

| Pipeline | mAP50 (%) | mAP50-95 (%) | P (%) | R (%) |
|---|---:|---:|---:|---:|
| BASELINE_native | 74.740711 | 53.289947 | 80.373597 | 66.509963 |
| BASELINE_nms_0.3 | 75.588802 | 52.620913 | 79.677569 | 70.190244 |
| BASELINE_nms_0.4 | 76.443151 | 53.074183 | 79.383262 | 70.687304 |
| BASELINE_nms_0.5 | 76.814995 | 53.412045 | 79.486865 | 70.521317 |
| BASELINE_nms_0.6 | 76.841585 | 53.644190 | 79.370472 | 70.470336 |
| BASELINE_nms_0.7 | 76.601290 | 53.770055 | 79.497920 | 69.852845 |
| BASELINE_nms_0.8 | 76.280856 | 53.843142 | 78.907753 | 69.715611 |
| R01_native | 75.480774 | 53.802078 | 80.812078 | 66.764913 |
| R01_nms_0.3 | 76.186639 | 53.116650 | 80.057401 | 70.267459 |
| R01_nms_0.4 | 77.154133 | 53.620859 | 80.370658 | 70.353075 |
| R01_nms_0.5 | 77.531812 | 53.947399 | 80.343553 | 70.507880 |
| R01_nms_0.6 | 77.533500 | 54.140237 | 80.591881 | 70.273503 |
| R01_nms_0.7 | 77.436577 | 54.339941 | 80.464773 | 69.847731 |
| R01_nms_0.8 | 77.082794 | 54.437152 | 80.385342 | 69.580289 |
| R01_flip | 78.916186 | 55.372631 | 82.356990 | 70.471383 |
| R02_native | 76.446012 | 54.411881 | 82.665702 | 67.166382 |
| R02_nms_0.3 | 77.156677 | 53.606193 | 82.246257 | 70.134632 |
| R02_nms_0.4 | 78.215656 | 54.210436 | 82.488683 | 70.672404 |
| R02_nms_0.5 | 78.533263 | 54.504420 | 82.365615 | 70.663629 |
| R02_nms_0.6 | 78.587581 | 54.799417 | 82.853609 | 69.802174 |
| R02_nms_0.7 | 78.500640 | 54.976003 | 82.746751 | 69.503550 |
| R02_nms_0.8 | 78.158209 | 55.045320 | 82.417419 | 69.439571 |
| R02_flip | 79.475396 | 55.737466 | 83.379581 | 70.563890 |
| R04_flip | 80.125017 | 56.081483 | 83.345557 | 71.645741 |
| R04_three_views | 80.426735 | 56.884799 | 83.836697 | 71.143103 |
| R04_three_views_nms_0.5 | 80.505213 | 56.569637 | 82.973544 | 72.129328 |
| R07_flip | 80.161299 | 56.584631 | 80.804144 | 72.907281 |
| R07_three_views | 80.174078 | 57.307680 | 82.845664 | 71.167107 |
| R07_three_views_nms_0.5 | 80.235977 | 57.043837 | 83.180264 | 71.494377 |

R01 và R02 flip đều đã chấm xong trên toàn bộ validation: hai view gốc + lật ngang, ghép class-aware NMS0.6. Đây là số đo thực tế của từng checkpoint.

R04 ba view đã quét thêm NMS0.3–0.8 từ raw cache trước merge/max_det; điểm replay NMS0.6 khớp hoàn toàn lượt trực tiếp trên2500 ảnh/8359GT. Chọn NMS0.5 theo mAP50: tăng0,078478 điểm phần trăm AP50 và giảm0,315162 điểm phần trăm AP50–95 so với NMS0.6. R07 hai/ba view và sáu ngưỡng NMS đã hoàn tất, full anchor parity PASS; tốt nhất AP50 là NMS0.5 đạt80,235977%, chưa vượt R04. Các record đầy đủ nằm ở cuối log; candidate R07 đã xoá sau so sánh.

## Các hướng đã chuẩn bị và tiến độ

- Batch GPU 12 nhánh / 52 epoch: R01-R11 hoàn tất; R12 YOLO26s pretrained đang train 10epoch, batch8 trên hai T4.
- GPU follow-up đã chuẩn bị10 pipeline khi có đủ top3: identity control, flip và ba view cho mỗi mô hình, ensemble top2/top3 và top2+flip. Mỗi pipeline quét NMS0.3–0.8 từ raw rows trước merge/max_det. Có real3-image direct/cache/replay smoke trên GPU trước mỗi full capture; full2500/8359, macro và AP từng lớp phải khớp anchor trong1e-6. Chưa chạy full GPU.
- Cache ensemble đã có helper và driver; chưa chạy thực nghiệm. Kiểm tra đúng manifest/protocol/classes, đúng thứ tự ảnh và exact GT canvas trước ghép.
- Sáu train subset và bảy config ablation (3epoch mỗi nhánh) đã kiểm hash/shape/config, chưa train.

| Subset | Train ảnh | Train boxes |
|---|---:|---:|
| D00 đủ dữ liệu | 12000 | 33272 |
| D01 bỏ merged3k | 9574 | 28496 |
| D02 bỏ ngẫu nhiên2426 seed42 | 9574 | 26549 |
| D03 bỏ celulares | 11228 | 32269 |
| D04 bỏ ngẫu nhiên772 seed42 | 11228 | 31181 |
| D05 bỏ cả hai nguồn | 8802 | 27493 |
| D06 bỏ ngẫu nhiên3198 seed42 | 8802 | 24545 |

Source/random có cùng số ảnh và bước/epoch; phân bố lớp/box khác nhau. Một số ảnh đã xác nhận thiếu nhãn person/table/book/backpack/chair, chưa kết luận toàn nguồn sai.

## Kiểm thử

**21 unit test PASS**, full suite 19.140 giây. Lệnh thực chạy:

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v
```

TDD: các module mới được chạy test khi chưa có module (RED), rồi toàn suite pass (GREEN).

| File test | SHA256 nguồn đã kiểm thử | Cases pass |
|---|---|---|
| test_cache_ensemble.py | `d40c788d5f14cc8af299876405246da9c85bb7b58ed8c8ac0642029aef4ea4ee` | test_complementary_detections_improve_recall_without_changing_targets<br>test_duplicate_model_keeps_nms_metric_identical<br>test_rejects_different_identity_mapping_or_canvas_targets<br>test_rejects_missing_duplicate_or_reordered_images |
| test_experiment_campaign.py | `572cfa03434c4e02ca70e752805796f1dc58bce7efcd74ef8635e7c5f7ed5bf3` | test_selects_ap50_epoch_instead_of_ap5095_epoch<br>test_rejects_incomplete_duplicate_and_nonfinite_epoch_records<br>test_ranking_rejects_different_ground_truth_or_protocol<br>test_bundle_contains_verified_payload_and_rejects_external_files |
| test_inference_variants.py | `b669639159b417fbf32ccfb5664ce80c9b495c1b7f541c185faade4f5f503451` | test_restores_asymmetric_flipped_box_after_integer_resize<br>test_class_aware_merge_suppresses_duplicates_before_global_limit<br>test_empty_views_and_rejects_invalid_shapes_or_nonfinite_values<br>test_half_precision_large_boxes_do_not_overflow_iou_areas<br>test_actual_backend_original_flip_and_scale_recover_one_box_per_image<br>test_single_view_has_native_prediction_parity_and_ensemble_retains_classes |
| test_training_subsets.py | `35296e11ec67e9198a0653a5a6ecdd26e70f58cfdf009314de6708c63b6f2025` | test_excludes_only_train_sources_and_preserves_input_manifest<br>test_random_control_is_reproducible_same_count_and_train_only<br>test_rejects_path_escape_or_mismatched_split_and_duplicate_ids<br>test_rejects_unknown_exclusions_holdout_ids_and_empty_subsets |
| test_validation_replay.py | `e32ee280c743c1d8364fd1f7d254c847633817c39ae03a59988b9003f011a6c0` | test_looser_nms_retains_two_true_overlapping_objects<br>test_missing_predictions_still_count_ground_truth_and_empty_images<br>test_rejects_duplicate_images_or_mismatched_ground_truth_shapes |

Integration checkpoint R01 thật trên 3 ảnh val ở320CPU: single-view và hai model trùng checkpoint khớp NMS tham chiếu; flip/scale chạy được. Replay native và NMS0.6 khớp metric tham chiếu. Smoke không dùng để xếp hạng full validation.

## Tính toàn vẹn và lưu trữ

Các ZIP tải về đã kiểm CRC, số byte và SHA256 từng payload; mã/config thực thi khớp nguồn local. Hash dưới đây giữ làm log sau khi xoá ZIP và checkpoint kém hơn.

| Gói đã kiểm | SHA256 | Payload |
|---|---|---:|
| baseline_verification.json | `b0720cf8d11b5517a08c5b3f7335a0315062f7fdf7478270eaf1c378c4598fd7` | 30 |
| R01_verification.json | `ac642148be93c370b47c3746da6b6cb3ed06ab2f23591d5bbef836537824658f` | 56 |
| R02_verification.json | `63a9dbe0d5a2bbccfbaa00d83a4fec57be1f2caccde92d8868a24f08068dda65` | 65 |
| R03_verification.json | `344eef326ea2aed603ffda8cce9d146ba09a2dcefacc49e3601f2c64df0b89b1` | 64 |
| inference_sources_verification.json | `7f4be857ce8033d165980580705718829980f323d102d60a1e3a107a98de6c1d` | nguồn |
| nms_sweep_sources_verification.json | `124025037d06d7df95b7ddb15b2e8562911a70a5a04af0a651954b931aa5ccbc` | nguồn |
| source_ablations_verification.json | `21b42ce22359e4eadd13e2056159fb7c3933d333fbfc27edfa26d383e4684199` | nguồn |
| source_ablations_train_sources_verification.json | `c9561bc81b5fa0a8af8347561a86edf0c71dddcc6fb1174600c68b0c85a93138` | nguồn |

R02_selection_verification.json:

```json
{
  "status": "pass",
  "selected_epoch": 4,
  "checkpoint_sha256": "c0b19c81af77842a59a596a057548641ca920175600edd61dd6bac620afaa0c1",
  "native_mAP50": 0.7640074402252824,
  "nms_mAP50": 0.7853437054155149,
  "gap_to_80_percentage_points": 1.465629458448514
}
```

R03_selection_verification.json:

```json
{
  "status": "pass",
  "selected_epoch": 4,
  "checkpoint_sha256": "d5f828bff3d7fe18266bc3c3db11993410fa84f2dacb97f8fc8242255343ce0e",
  "native_mAP50": 0.759657559984875,
  "nms_mAP50": 0.7801612567090301,
  "gap_to_80_percentage_points": 1.9838743290969907
}
```

Mã tái lập nằm trong `kaggle_tools/`, `notebooks/kaggle/` và `tests/`. Giữ mã nguồn và dataset cần chạy thí nghiệm; chỉ dọn artifact campaign.

Các lượt GPU còn lại tiếp tục trên Kaggle. Từ đây ghi kết quả mới vào log này, chỉ giữ checkpoint tốt nhất. File ứng viên chỉ giữ tạm khi cần chấm so sánh rồi xoá nếu kém hơn. Đường dẫn YAML trong các record là đường dẫn tại thời điểm đo, đã được dọn cùng artifact; dữ liệu và cấu hình nguồn `configs/data_v05.yaml` vẫn được giữ để tái lập validation.

## Bản tải Downloads đã dọn

Đã xoá bản tải trùng ở Downloads: C:\Users\aawm8\Downloads\campaign_20261010_001917_BASELINE.zip (6401350 bytes), SHA256 b0720cf8d11b5517a08c5b3f7335a0315062f7fdf7478270eaf1c378c4598fd7.

Đã xoá bản tải trùng ở Downloads: C:\Users\aawm8\Downloads\campaign_20261010_001917_R01_recover_L1A1_complete.zip (50181259 bytes), SHA256 ac642148be93c370b47c3746da6b6cb3ed06ab2f23591d5bbef836537824658f.

Đã xoá bản tải trùng ở Downloads: C:\Users\aawm8\Downloads\campaign_20261010_001917_R02_light_longer_complete.zip (135445584 bytes), SHA256 63a9dbe0d5a2bbccfbaa00d83a4fec57be1f2caccde92d8868a24f08068dda65.

Đã xoá bản tải trùng ở Downloads: C:\Users\aawm8\Downloads\campaign_20261010_001917_R03_low_lr_base_complete.zip (78677998 bytes), SHA256 344eef326ea2aed603ffda8cce9d146ba09a2dcefacc49e3601f2c64df0b89b1.

## Checkpoint duy nhất được giữ — hiện tại R04

- File: [BEST_Project8_v05.pt](C:/Users/aawm8/Downloads/BTL-AI-Camera-cd8/weights/campaign_20261010/BEST_Project8_v05.pt).
- Pipeline được chọn: **R04_three_views**; **80.426735% mAP50**, 56.884799% mAP50-95.
- SHA256: `087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758`. Bản copy giữ nguyên byte của epoch4 (`epoch3.pt`).
- So sánh trên cùng 2.500val / 8.359boxes, CPU FP32, không dùng test.
- Cần ba view `(1.0, không lật), (0.83, lật ngang), (0.67, không lật)` và class-aware NMS0.6 để tái lập điểm tốt nhất. Checkpoint một view native không mang điểm80,426735% này.

R04 full metric ở cuối log. JSON R02 bên dưới giữ làm lịch sử lựa chọn trước đó. Latency đo trong lúc có tải thí nghiệm, không phải benchmark camera/FPS.

```json
{
  "pipeline": "R02_flip",
  "checkpoint": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\weights\\campaign_20261010\\BEST_Project8_v05.pt",
  "checkpoint_sha256": "c0b19c81af77842a59a596a057548641ca920175600edd61dd6bac620afaa0c1",
  "mAP50": 0.794753964987905,
  "mAP50_95": 0.557374655074337,
  "precision": 0.8337958136953503,
  "recall": 0.7056388971518583,
  "per_class": {
    "person": {
      "AP50": 0.8526944395766974,
      "AP50_95": 0.5729806539332153
    },
    "table": {
      "AP50": 0.7889987871494266,
      "AP50_95": 0.5605936602009015
    },
    "chair": {
      "AP50": 0.6963265154467817,
      "AP50_95": 0.46122325242321427
    },
    "laptop": {
      "AP50": 0.7861993406017522,
      "AP50_95": 0.5058021838953096
    },
    "cell phone": {
      "AP50": 0.7530377060044638,
      "AP50_95": 0.4456479627864791
    },
    "backpack": {
      "AP50": 0.8436256796130664,
      "AP50_95": 0.6783578521282201
    },
    "book": {
      "AP50": 0.777880489663483,
      "AP50_95": 0.48975216202111393
    },
    "bottle": {
      "AP50": 0.8592687618475683,
      "AP50_95": 0.7446395132062424
    }
  },
  "speed": {
    "preprocess": 2.895719240009203,
    "inference": 307.8637395599944,
    "loss": 0.00024712000231374986,
    "postprocess": 0.058432839999295536
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      1.0,
      true
    ]
  ],
  "postprocessing": "explicit_flip_views_nms_06",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\reports\\campaign_20261010\\local_inference_20261009_180723\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "compared": [
    "BASELINE_native",
    "BASELINE_nms_0.3",
    "BASELINE_nms_0.4",
    "BASELINE_nms_0.5",
    "BASELINE_nms_0.6",
    "BASELINE_nms_0.7",
    "BASELINE_nms_0.8",
    "R01_native",
    "R01_nms_0.3",
    "R01_nms_0.4",
    "R01_nms_0.5",
    "R01_nms_0.6",
    "R01_nms_0.7",
    "R01_nms_0.8",
    "R01_flip",
    "R02_native",
    "R02_nms_0.3",
    "R02_nms_0.4",
    "R02_nms_0.5",
    "R02_nms_0.6",
    "R02_nms_0.7",
    "R02_nms_0.8",
    "R02_flip"
  ],
  "selection": "Highest measured AP50 under identical CPU FP32 validation protocol; GPU table separate."
}
```

## Dọn artifact campaign

Thư mục được dọn: C:\Users\aawm8\Downloads\BTL-AI-Camera-cd8\reports\campaign_20261010. Giữ checkpoint ở C:\Users\aawm8\Downloads\BTL-AI-Camera-cd8\weights\campaign_20261010\BEST_Project8_v05.pt và log này ở ngoài thư mục được xoá.

| ZIP đã xoá | Bytes | SHA256 |
|---|---:|---|
| campaign_20261010_001917_BASELINE.zip | 6401350 | b0720cf8d11b5517a08c5b3f7335a0315062f7fdf7478270eaf1c378c4598fd7 |
| campaign_20261010_001917_R01_recover_L1A1_complete.zip | 50181259 | ac642148be93c370b47c3746da6b6cb3ed06ab2f23591d5bbef836537824658f |
| campaign_20261010_001917_R02_light_longer_complete.zip | 135445584 | 63a9dbe0d5a2bbccfbaa00d83a4fec57be1f2caccde92d8868a24f08068dda65 |
| campaign_20261010_001917_R03_low_lr_base_complete.zip | 78677998 | 344eef326ea2aed603ffda8cce9d146ba09a2dcefacc49e3601f2c64df0b89b1 |
| inference_sources_20261010.zip | 22038 | 7f4be857ce8033d165980580705718829980f323d102d60a1e3a107a98de6c1d |
| local_inference_20261009_180723_BASELINE.zip | 2812712 | 6b979614b3729524a0c7fa4357a967692c4cc1918332d3f0b4d4f0d0d24fcc06 |
| local_inference_20261009_180723_FINAL.zip | 5606222 | 957eb2ca77e2e98170a8b9351c96c6005347e89db5cbec21bad00e1e939af8b6 |
| local_inference_20261009_180723_preflight.zip | 4282 | 7c86d1a5d5aad484d63e57fb61c1a4f6e3a628cb5fd85298a49b38f32aeb734b |
| local_inference_20261009_180723_R01.zip | 5604755 | b667d05c57c3cd165d48e62579d716ad728da28cdd7814a2cff9a67be4fa8f84 |
| local_R02_inference_20261009_183043_FINAL.zip | 2679690 | 17a74fcaaf1eee72587a868e078134fcb427cdc27a631451684de3997f7f2621 |
| local_R02_inference_20261009_183043_preflight.zip | 4425 | e43bbf3d0d047f0dc5ca5e2bd3aa257a13c66abcc617cc47b26fdea3b58aeea7 |
| local_R02_inference_20261009_183043_R02.zip | 2678235 | 43e214758a9bbec645d58395ca024a40928fbff8830419f87bfb0a165885c341 |
| nms_sweep_sources_20261010.zip | 23000 | 124025037d06d7df95b7ddb15b2e8562911a70a5a04af0a651954b931aa5ccbc |
| source_ablations_local_20261010.zip | 6905524 | 21b42ce22359e4eadd13e2056159fb7c3933d333fbfc27edfa26d383e4684199 |
| source_ablations_train_sources_20261010.zip | 29049 | c9561bc81b5fa0a8af8347561a86edf0c71dddcc6fb1174600c68b0c85a93138 |

Dọn thành công: 304 file (23 file .pt, 15 ZIP), 637056874 bytes. Đã xoá log campaign cũ sau khi gộp vào log này. Checkpoint duy nhất của campaign đã kiểm lại SHA256 sau dọn. Mã nguồn, tests và dataset được giữ để tiếp tục thí nghiệm.


### R04 CPU explicit views — bắt đầu

```json
{
  "timestamp_utc": "20261009_191354",
  "status": "running",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "source_sha256": "37693540af35abe433cb79c58ce4031bf497231b197b1b18b4bd6495ff717fad",
  "inference_helper_sha256": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
  "csv_sha256": "82dfd7626cbb34d18db12784986d0652b1278d2a4095cc917a1cf79e9258b1f6",
  "selected_epoch": 4,
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "validation_yaml_sha256": "f872aa9419cd633e271ed68ab7464dd0937bfd2652f653aacd7d4fe96bbc0f2c",
  "test_used": false,
  "views_to_measure": {
    "R04_flip": [
      [
        1.0,
        false
      ],
      [
        1.0,
        true
      ]
    ],
    "R04_three_views": [
      [
        1.0,
        false
      ],
      [
        0.83,
        true
      ],
      [
        0.67,
        false
      ]
    ]
  },
  "runtime": {
    "ultralytics": "8.4.165",
    "torch": "2.14.0+cpu",
    "device": "CPU",
    "torch_threads": 4
  }
}
```


### R04_flip — full validation hoàn tất

```json
{
  "pipeline": "R04_flip",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "mAP50": 0.8012501708226769,
  "mAP50_95": 0.5608148291231373,
  "precision": 0.8334555681380474,
  "recall": 0.7164574060266875,
  "views": [
    [
      1.0,
      false
    ],
    [
      1.0,
      true
    ]
  ],
  "postprocessing": "explicit_views_class_aware_nms_06",
  "extra_nms_iou": 0.6,
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "test_used": false,
  "validation_images": 2500,
  "validation_boxes": 8359,
  "speed": {
    "preprocess": 3.1147471199990835,
    "inference": 305.3654333199964,
    "loss": 0.0002551199977460783,
    "postprocess": 0.05994932000248809
  },
  "per_class": {
    "person": {
      "AP50": 0.8646910209684981,
      "AP50_95": 0.58019703535601
    },
    "table": {
      "AP50": 0.8045425065911477,
      "AP50_95": 0.5632383834750737
    },
    "chair": {
      "AP50": 0.6909724839280238,
      "AP50_95": 0.457066174658013
    },
    "laptop": {
      "AP50": 0.7963421469808983,
      "AP50_95": 0.5206641774573812
    },
    "cell phone": {
      "AP50": 0.7517855278678729,
      "AP50_95": 0.43848281561747077
    },
    "backpack": {
      "AP50": 0.8485069514764728,
      "AP50_95": 0.6894643618389933
    },
    "book": {
      "AP50": 0.7846477272454706,
      "AP50_95": 0.4978360372747409
    },
    "bottle": {
      "AP50": 0.8685130015230313,
      "AP50_95": 0.7395696473074156
    }
  },
  "replaces_pipeline": "R02_flip",
  "kept_as_best": true
}
```

- Đã xóa bản tải trùng: C:\Users\aawm8\Downloads\epoch3.pt; SHA256=087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758; bản workspace đã kiểm hash.
- Đã xóa bản tải trùng: C:\Users\aawm8\Downloads\results.csv; SHA256=82dfd7626cbb34d18db12784986d0652b1278d2a4095cc917a1cf79e9258b1f6; bản workspace đã kiểm hash.


## Đối chiếu các hướng còn lại — 2026-10-09T19:32:22.524783+00:00

Đã đọc lại các mục 4, 6, 8, 9, 10 của `tasks/plan.md`. Mốc 80% validation đã đạt bằng R04 hai view; không kết luận đã hoàn tất toàn bộ chiến dịch hoặc nghiệm thu sản phẩm.

| Hướng | Trạng thái thực tế | Việc tiếp theo |
|---|---|---|
| Class weighting 0.25 và 0.5 | R04/R05 train xong và full GPU validation xong | R04_flip CPU đã đạt 80,125017%; R05 NMS GPU 78,846724% |
| Mosaic=0, augmentation nhẹ | R06 đã train và full validation native/NMS xong | Native76,326805%; NMS78,433050%, chưa vượt R04 trên GPU |
| AdamW | R07 đã full validation native/NMS và TTA CPU | Native76,969425%; NMS79,012742%, đứng đầu GPU hiện tại; TTA CPU NMS0.5 đạt80,235977%, thấp hơn best R04 |
| Freeze backbone | R08 freeze10 đã full native/NMS | NMS77,624530%, kém best; chưa có thử freeze→unfreeze |
| Train 1536 | R09 đã full native/NMS common1280 | NMS76,717911%, kém best; không trộn với trainer/pilot1536 |
| Mixup/CutMix | R10 đã full native/NMS | Native74,975445%; NMS77,168843%, không vượt best GPU |
| Box/cls loss weights | R11 đã full native/NMS | Native75,816267%; NMS77,887176%, không vượt R07 trên GPU |
| YOLO26s pretrained | R12 đã bắt đầu train10epoch | Đây là ứng viên thực dụng; init khác nano warm-start, chưa phải đối chứng riêng architecture |
| Nano pretrained mới đối chứng với small pretrained | Source đã bổ sung, chưa train | Cùng recipe/split/seed/epoch như R12 để tách tác động kiến trúc khỏi warm-start |
| Fresh nano bỏ nhóm val/test trong train | Đã kiểm24.000 payload train; hai danh sách11745 ảnh đã tạo, đủ8 lớp; source đóng gói; chưa train | Bỏ255 train cùng17 nhóm video holdout, đối chứng bỏ ngẫu nhiên255; val8359GT giữ nguyên. Khởi tạo COCO mới để không kế thừa tiếp xúc từ warm-start |
| NMS sweep GPU | Có source và helper đã kiểm thử, chưa chạy full GPU | Top3, kiểm parity native/cache rồi IoU0.3–0.8 |
| TTA nhiều scale/flip | R04 và R07 CPU cùng sáu ngưỡng NMS đã xong; GPU source đã chuẩn bị | Best R04 ba view+NMS0.5 đạt80,505213%; R07 tốt nhất AP50 với NMS0.5 đạt80,235977%/57,043837%; chưa phải benchmark camera |
| Ensemble top2/top3 | Raw collector đã hỗ trợ nhiều mô hình; 7test PASS và real3-image/two-instance parity PASS; source fullGPU đóng gói; chưa đo ensemble full | Phép kỹ thuật dùng hai instance cùng R04, không tuyên bố tăng AP. FullGPU dùng checkpoint khác SHA, cùng GT/canvas/classes/protocol và quét6 NMS mỗi pipeline |
| Loại nguồn train nghi thiếu nhãn | Sáu subset đã kiểm, bảy cấu hình đã kiểm; chưa train | Full control + ba cặp source-removal/random-removal, 3 epoch mỗi nhánh |
| Freeze→unfreeze | Source và config-check đã chuẩn bị, chưa train | Cặp hai stage có đối chứng không freeze, cùng epoch budget và reset optimizer |
| Seed43/44 | Source và config-check đã chuẩn bị, chưa train | Lặp cấu hình thắng và ghi mọi seed, không chỉ seed cao nhất |
| Train dài hơn | Source và config-check đã chuẩn bị, chưa train | Theo dõi chuỗi val/loss, không kết luận overfit từ một epoch giảm |
| Distillation teacher→nano | Source và config-check đã chuẩn bị, chưa train | Chỉ thử nếu teacher R12 có lợi ích; cần kiểm head/classes và parity student khi lưu |
| Rà/sửa nhãn đủ 8 lớp, near-duplicate/scene | Full review chưa hoàn tất; đã có mẫu xác nhận thiếu nhãn | Giải quyết chất lượng dữ liệu để tiến tới 85/95; không dùng predictions làm GT, không gọi loại nguồn là đã sửa nhãn |
| Test500 độc lập | Chưa chấm model | Chỉ chấm sau khi khóa lựa chọn; báo giới hạn coverage, không tune bằng test |
| Benchmark/export/deploy camera | Chưa nghiệm thu | Đo CPU640, latency/FPS thật và parity; không lấy AP1280 TTA làm điểm vận hành640 |

Ưu tiên thực thi: hoàn tất batch đang chạy → NMS/TTA/ensemble trên GPU → source ablations → freeze/unfreeze và xác nhận seeds/lịch dài → distillation khi có teacher phù hợp. Thứ tự có thể đổi theo số đo và GPU/session còn lại. Không bảo đảm các hướng tăng điểm hoặc đạt 85/95. Checkpoint inferior/duplicate chỉ giữ tạm khi cần đo, sau đó dọn; kết quả và test giữ trong MD này.

### R04 provenance và CSV đầy đủ

- Parent: `92adfeb2516861a6958e8adc4a15cf764faa3956078da1ebc81db1fbb7014e31`.
- Selected epoch4, file zero-based `epoch3.pt`; checkpoint SHA256 `087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758`.
- CSV SHA256 `82dfd7626cbb34d18db12784986d0652b1278d2a4095cc917a1cf79e9258b1f6`; bốn dòng đã được `select_epoch(rows, 4)` kiểm tra.
- R04 LIGHT augmentation như R01, epochs4, cls_pw0.25, MuSGD lr0.0005, seed42, hai GPU. Full GPU native76,893780%, NMS0.6 78,966129%; CPU two-view80,125017%. Không đổi GT giữa các phép đo.
- R04 ZIP remote trained SHA256 `c576e1361f33147fc79d677190565d1c24b6bd8785d5d48a167d65648ecf3ac0`, complete `5947770f4b2c5149bdbf0872eb1a56a9273ef716dbcb0fac1cb060e104803dc3`; chỉ ghi hash Kaggle công bố, không tải/kiểm CRC hai ZIP này trên máy.
- R05 selected checkpoint SHA256 `cd47829a18719bbff09cbd8e02e73dd385d36a69824de594686e338062c1768c`; native AP50=0.76722326671923, AP50–95=0.54709159008715; NMS0.6 AP50=0.7884672360298326, AP50–95=0.5504330604463915. Trích full2500 validation từ output Kaggle; chưa tải checkpoint R05.

```csv
epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,lr/pg0,lr/pg1,lr/pg2,lr/pg3,lr/pg4,lr/pg5,lr/pg6,lr/pg7
1,329.421,1.08576,0.9094,0.01168,0.80357,0.6764,0.75399,0.53234,1.13913,1.04251,0.01565,0.00148404,0.000494681,0.00148404,0.000494681,0.00148404,0.000494681,0.00148404,0.000494681
2,622.66,1.0759,0.89769,0.01186,0.81796,0.66041,0.75298,0.53646,1.12293,1.04587,0.01483,0.0015,0.0005,0.0015,0.0005,0.0015,0.0005,0.0015,0.0005
3,914.661,1.08352,0.89485,0.01161,0.78122,0.68994,0.75892,0.539,1.1234,1.03527,0.01492,0.0015,0.0005,0.0015,0.0005,0.0015,0.0005,0.0015,0.0005
4,1208.21,1.06803,0.84994,0.01125,0.81485,0.6798,0.76938,0.54613,1.12234,0.99621,0.01478,0.0015,0.0005,0.0015,0.0005,0.0015,0.0005,0.0015,0.0005
```


### Các lượt xác nhận đã có source — chưa train

Bổ sung code seed43/44, lịch dài, freeze→unfreeze và KD có điều kiện. Hai nhánh freeze cùng 2+4 epoch và cùng reset optimizer; KD có student control cùng batch8/lr/3epoch. Số epoch thực phụ thuộc cấu hình nano thắng; ví dụ R04 cần34epoch nếu teacher tốt hơn,28 nếu bỏ KD. Không phải kết quả train. Dùng implementation distillation có sẵn của Ultralytics8.4.165, đã đọc source trainer/wrapper; chưa xác nhận chạy GPU hay parity artifact distill.

```json
{
  "status": "prepared_and_config_checked_not_trained",
  "synthetic_fixture_used_for_config_validation": true,
  "example_recipe_total_epochs": 34,
  "configs_checked": 9,
  "source_sha256": {
    "kaggle_tools\\experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
    "kaggle_tools\\inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
    "kaggle_tools\\confirmation_plan.py": "391cdcb880d02499c5b5ed36cc92047389e416c5cf6f5d59fda2227f70b01576",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.py": "95c57a568d0a288004652a575df953abdee2d3c07d70618f6d991ea673d2bac3",
    "notebooks\\kaggle\\project8_v05_confirmation_train_cell_20261010.py": "23582a4734454d87462d69592eed9a582bc075da45f34e637074d1eb6874c9b5",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.ipynb": "2ceccf561e081b903d4b0eafe9ec9e0a24dad9e2986c8abd10153af8ef906565"
  },
  "cell_source_sha256": "7228304c6c554981a9651012a03bc842d9c21fa15efaea0dc16de47718cfe5af",
  "installed_kd_source_sha256": "190311d996f9964a9f4302a41f43bd25f3b482516270e148df16f491b9d4fc9b",
  "tests": "Four semantic planning tests passed; full suite recorded separately."
}
```


### Các lượt xác nhận đã có source — chưa train

Bổ sung code seed43/44, lịch dài, freeze→unfreeze và KD có điều kiện. Hai nhánh freeze cùng 2+4 epoch và cùng reset optimizer; KD có student control cùng batch8/lr/3epoch. Số epoch thực phụ thuộc cấu hình nano thắng; ví dụ R04 cần34epoch nếu teacher tốt hơn,28 nếu bỏ KD. Không phải kết quả train. Dùng implementation distillation có sẵn của Ultralytics8.4.165, đã đọc source trainer/wrapper; chưa xác nhận chạy GPU hay parity artifact distill.

```json
{
  "status": "prepared_and_config_checked_not_trained",
  "synthetic_fixture_used_for_config_validation": true,
  "example_recipe_total_epochs": 34,
  "configs_checked": 9,
  "source_sha256": {
    "kaggle_tools\\experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
    "kaggle_tools\\inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
    "kaggle_tools\\confirmation_plan.py": "eef7c1238d27a1d33232e30d69e8d5885c2c6811336e92eae632a330bfcb41f0",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.py": "95c57a568d0a288004652a575df953abdee2d3c07d70618f6d991ea673d2bac3",
    "notebooks\\kaggle\\project8_v05_confirmation_train_cell_20261010.py": "f239354425e11ea912d461850a637cabc7d719e9ae01fe01f789c4e52a7b6af1",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.ipynb": "8e5dd91e77a682cdad08d894faef5325edd84bda047585163f7ce0354be75c5b"
  },
  "cell_source_sha256": "9fb18359c220e7285461c48a544b5e478a57aea6944d48306131f86f3d6d44c8",
  "installed_kd_source_sha256": "190311d996f9964a9f4302a41f43bd25f3b482516270e148df16f491b9d4fc9b",
  "tests": "Five semantic planning tests; full suite recorded separately."
}
```


### Kiểm thử bổ sung — toàn suite26 PASS

RED ban đầu: module confirmation_plan chưa có; GREEN:5 kiểm thử semantic mới và21 kiểm thử cũ đều pass. Kiểm đường dẫn portable đã sửa trước GREEN; baseline tham chiếu không có recipe không được chọn làm cấu hình train. Builder đã sửa đường dẫn manifest sang SOURCE_DATA theo kernel hiện hành; metadata source ở block mới nhất là bản sử dụng.

```json
{
  "timestamp_utc": "2026-10-09T19:38:54.319431+00:00",
  "status": "PASS",
  "tests_total": 26,
  "elapsed_seconds": 12.899,
  "command": ".venv\\Scripts\\python.exe -X utf8 -B -m unittest discover -s tests -v",
  "environment": {
    "PYTHONPATH": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8",
    "TEMP": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp",
    "TMP": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp"
  },
  "new_tests": [
    "test_baseline_reference_cannot_supply_a_missing_training_recipe",
    "test_seed_repeats_use_original_init_and_preserve_winning_recipe",
    "test_freeze_pair_has_equal_steps_and_same_stage_two_schedule",
    "test_distillation_is_conditional_and_has_matching_student_control",
    "test_rejects_incomplete_batch_or_mismatched_gt_and_protocol"
  ],
  "new_test_file_sha256": "e1ade127caeaea6c5fce2572d4811041f06ea9017a04cb93a7ed5f822226bc70",
  "planning_helper_sha256": "eef7c1238d27a1d33232e30d69e8d5885c2c6811336e92eae632a330bfcb41f0",
  "driver_sha256": "95c57a568d0a288004652a575df953abdee2d3c07d70618f6d991ea673d2bac3",
  "limitations": "Config/logic checks only; confirmation/freeze/distillation training has NOT run on GPU."
}
```


### R04_three_views — full validation hoàn tất

```json
{
  "pipeline": "R04_three_views",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "mAP50": 0.8042673469408561,
  "mAP50_95": 0.5688479918514012,
  "precision": 0.8383669666187495,
  "recall": 0.7114310282044596,
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "postprocessing": "explicit_views_class_aware_nms_06",
  "extra_nms_iou": 0.6,
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "test_used": false,
  "validation_images": 2500,
  "validation_boxes": 8359,
  "speed": {
    "preprocess": 2.9384457600084715,
    "inference": 348.4735738400035,
    "loss": 0.00023515999637311325,
    "postprocess": 0.060993759988923556
  },
  "per_class": {
    "person": {
      "AP50": 0.8625422449971606,
      "AP50_95": 0.5760641858935949
    },
    "table": {
      "AP50": 0.8033896878610551,
      "AP50_95": 0.5723947895142519
    },
    "chair": {
      "AP50": 0.7039366100000339,
      "AP50_95": 0.4756718250728035
    },
    "laptop": {
      "AP50": 0.8077202291135086,
      "AP50_95": 0.5466721148806155
    },
    "cell phone": {
      "AP50": 0.7464817614245698,
      "AP50_95": 0.43844475377410036
    },
    "backpack": {
      "AP50": 0.8595335705703024,
      "AP50_95": 0.7005709539154769
    },
    "book": {
      "AP50": 0.7852985345649273,
      "AP50_95": 0.4970856528406534
    },
    "bottle": {
      "AP50": 0.8652361369952909,
      "AP50_95": 0.7438796589197135
    }
  },
  "replaces_pipeline": "R04_flip",
  "kept_as_best": true
}
```


### R04 CPU explicit views — kết thúc

```json
{
  "status": "complete",
  "errors": {},
  "kept_checkpoint": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\weights\\campaign_20261010\\BEST_Project8_v05.pt",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758"
}
```

R04 CPU job exit0: flip và ba view full validation hoàn tất; không lỗi. Driver đã xóa bản checkpoint ứng viên tạm sau khi giữ R04 làm best; CSV tạm đã xóa sau khi toàn bộ CSV và SHA256 được ghi vào MD. Thư mục weights/campaign_20261010 chỉ còn BEST_Project8_v05.pt. Ba view tăng mAP tổng nhưng phone AP50=74,648176% (hai view75,178553%); chair70,393661%, vẫn dưới sàn75 của kế hoạch. Chưa nghiệm thu test/camera.


### Các lượt xác nhận đã có source — chưa train

Bổ sung code seed43/44, lịch dài, nano pretrained mới đối chứng small, freeze→unfreeze và KD có điều kiện. Hai nhánh freeze cùng 2+4 epoch và cùng reset optimizer; KD có student control cùng batch8/lr/3epoch. Số epoch thực phụ thuộc cấu hình nano thắng; ví dụ R04 và teacher10epoch cần44epoch nếu teacher tốt hơn,38 nếu bỏ KD. Không phải kết quả train. Dùng implementation distillation có sẵn của Ultralytics8.4.165, đã đọc source trainer/wrapper; chưa xác nhận chạy GPU hay parity artifact distill.

```json
{
  "status": "prepared_and_config_checked_not_trained",
  "synthetic_fixture_used_for_config_validation": true,
  "example_recipe_total_epochs": 44,
  "configs_checked": 10,
  "source_sha256": {
    "kaggle_tools\\experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
    "kaggle_tools\\inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
    "kaggle_tools\\confirmation_plan.py": "cf9a25afd0b4a2cc7b8ac9dd3aa25f6b95ba2a8b5d551bbb2fffac877d157bbe",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.py": "994232ea2998bba1fc49f52ebd5debf4b37fd3797837831f0d746485e128ef8f",
    "notebooks\\kaggle\\project8_v05_confirmation_train_cell_20261010.py": "64e27b0e4cfc2f2d4763166822731779fb4ca84ad19fdf96f97a3dc9e168a000",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.ipynb": "5c4703252170e735519b8ef8aea71d78ad57480c8f0e7ea9ddee9c1347ce64ac"
  },
  "cell_source_sha256": "52bead98b0e710bca10913ea5922cf38dd0d465e270608875849c70ea83c33fc",
  "installed_kd_source_sha256": "190311d996f9964a9f4302a41f43bd25f3b482516270e148df16f491b9d4fc9b",
  "tests": "Six semantic planning tests; full suite recorded separately."
}
```


### Kiểm thử bổ sung — toàn suite27 PASS

GREEN: 6 kiểm thử semantic của confirmation_plan và21 kiểm thử cũ đều pass. Kiểm đường dẫn portable đã sửa trước GREEN; baseline tham chiếu không có recipe không được chọn làm cấu hình train. Builder đã sửa đường dẫn manifest sang SOURCE_DATA theo kernel hiện hành; metadata source ở block mới nhất là bản sử dụng.

```json
{
  "timestamp_utc": "2026-10-09T19:48:49.155910+00:00",
  "status": "PASS",
  "tests_total": 27,
  "elapsed_seconds": 8.684,
  "command": ".venv\\Scripts\\python.exe -X utf8 -B -m unittest discover -s tests -v",
  "environment": {
    "PYTHONPATH": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8",
    "TEMP": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp",
    "TMP": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp"
  },
  "new_tests": [
    "test_fresh_nano_matches_small_pretrained_training_recipe",
    "test_baseline_reference_cannot_supply_a_missing_training_recipe",
    "test_seed_repeats_use_original_init_and_preserve_winning_recipe",
    "test_freeze_pair_has_equal_steps_and_same_stage_two_schedule",
    "test_distillation_is_conditional_and_has_matching_student_control",
    "test_rejects_incomplete_batch_or_mismatched_gt_and_protocol"
  ],
  "new_test_file_sha256": "714fd7ce074e8786ffd28dccbddfb00cd4fc240003911c923cc63388de6b5d6f",
  "planning_helper_sha256": "cf9a25afd0b4a2cc7b8ac9dd3aa25f6b95ba2a8b5d551bbb2fffac877d157bbe",
  "driver_sha256": "994232ea2998bba1fc49f52ebd5debf4b37fd3797837831f0d746485e128ef8f",
  "limitations": "Config/logic checks only; confirmation/freeze/distillation training has NOT run on GPU."
}
```


### R04 TTA × NMS sweep — bắt đầu

```json
{
  "status": "running",
  "timestamp_utc": "20261009_195822",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "source_sha256": "1a7de79820fff6cb77fbc4e3436570b4f727a566012bdc3ff1b74e278f18c1a6",
  "collector_sha256": "2499a70a8afa50856473957c62bd1f64a76cbd094ef4f961d92529ea6f5e6406",
  "replay_sha256": "b75e2237f159b1f03afab295012b93f08e799bdcd859de3c76f19c886b2d7a60",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "cache_note": "Unmerged restored predictions in RAM only, before class-aware NMS and global max_det."
}
```


### R04 TTA × NMS sweep — lỗi

```json
{
  "status": "error",
  "traceback": "Traceback (most recent call last):\n  File \"C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\notebooks\\kaggle\\project8_v05_local_TTA_nms_20261010.py\", line 103, in <module>\n    smoke_samples, captured = capture(smoke_yaml, 'smoke_capture', 3)\n                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\notebooks\\kaggle\\project8_v05_local_TTA_nms_20261010.py\", line 78, in capture\n    metrics = model.val(validator=factory, project=str(output), name=label, exist_ok=False,\n              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.venv\\Lib\\site-packages\\ultralytics\\engine\\model.py\", line 641, in val\n    validator(model=self.model)\n  File \"C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.venv\\Lib\\site-packages\\torch\\utils\\_contextlib.py\", line 124, in decorate_context\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.venv\\Lib\\site-packages\\ultralytics\\engine\\validator.py\", line 271, in __call__\n    self.update_metrics(preds, batch)\n  File \"C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\kaggle_tools\\view_replay.py\", line 61, in update_metrics\n    raise ValueError('Raw capture batch order/count differs')\nValueError: Raw capture batch order/count differs\n"
}
```


### R04 TTA × NMS sweep — bắt đầu

```json
{
  "status": "running",
  "timestamp_utc": "20261009_200751",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "source_sha256": "1a7de79820fff6cb77fbc4e3436570b4f727a566012bdc3ff1b74e278f18c1a6",
  "collector_sha256": "edfe3ca9f20db16f7ab374c55525a2f1fccaa9989f3ba71bd180b8c2e215d04b",
  "replay_sha256": "b75e2237f159b1f03afab295012b93f08e799bdcd859de3c76f19c886b2d7a60",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "cache_note": "Unmerged restored predictions in RAM only, before class-aware NMS and global max_det."
}
```


### R04 raw TTA cache — real3-image integration PASS

```json
{
  "status": "PASS",
  "images": 3,
  "raw_rows": 2700,
  "ground_truth_boxes": 14,
  "metrics_equal_within": 1e-06,
  "used_for_full_AP_claim": false
}
```

### Kiểm thử raw TTA cache — 31 PASS sau sửa backend copy

Lượt smoke đầu đã dừng với `Raw capture batch order/count differs`, chưa bắt đầu đo đủ 2.500 ảnh. Nguyên nhân: AutoBackend sao chép wrapper có tensor tạo trong inference_mode, bộ thu cũ trỏ vào bản không được chạy. CacheViewsValidator.init_metrics hiện gắn với model thực của backend; không thay đổi prediction, ground truth hoặc cơ chế fusion của Ultralytics.

RED: kiểm thử bản sao backend thất bại trước sửa. GREEN: bốn kiểm thử view_replay pass (8,588 giây); toàn suite **31/31 PASS trong 9,098 giây**, exit0. Lệnh: `.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v`, PYTHONPATH=workspace, TEMP/TMP=.temp.

Các kiểm thử mới:

- test_capture_does_not_change_inference_or_retain_previous_batch
- test_capture_is_before_merge_limit_and_restores_each_image_canvas
- test_raw_capture_can_recover_objects_removed_by_stricter_merge_and_cap
- test_validator_uses_the_actual_inference_backend_copy_and_preserves_targets

SHA256 sau sửa:

- kaggle_tools/view_replay.py: `edfe3ca9f20db16f7ab374c55525a2f1fccaa9989f3ba71bd180b8c2e215d04b`
- tests/test_view_replay.py: `290ddfe1906b20e726e09bd3cf335bc7a3de0335d176887bf75f2b425971dc5b`
- driver TTA/NMS: `1a7de79820fff6cb77fbc4e3436570b4f727a566012bdc3ff1b74e278f18c1a6`

Lượt TTA/NMS mới đã khởi động sau GREEN; chỉ kết quả thực đo đủ validation và parity đạt yêu cầu mới được dùng chọn best.

### R06_no_mosaic — đối chiếu đầy đủ từ output Kaggle

Đủ 2.500 validation / 8.359 GT; protocol GPU FP16 ở đầu log. Checkpoint chọn epoch4, file epoch3.pt, SHA256 `4a313895e0dd2faa8ecce24d3105609d090ebacdc6ccc686a069354c7b54c024`. Không tải checkpoint này về máy vì kém hơn best.

| Pipeline | mAP50 | mAP50–95 | Precision | Recall |
|---|---:|---:|---:|---:|
| native | 0.7632680506379141 | 0.5393883924039047 | 0.8003904767026827 | 0.6833130635804314 |
| extra_nms_06 | 0.784330502587374 | 0.5431798790499489 | 0.8104387477877172 | 0.7106594706963714 |

| Class | Native AP50 | Native AP50–95 | NMS0.6 AP50 | NMS0.6 AP50–95 |
|---|---:|---:|---:|---:|
| person | 0.8333267333168934 | 0.5612284539476227 | 0.8432715167164002 | 0.5616709511159927 |
| table | 0.7601735264673173 | 0.5385963222313189 | 0.7812124867065089 | 0.541481214910444 |
| chair | 0.6595137840703696 | 0.4394781121389655 | 0.6815116138036929 | 0.44276538533166765 |
| laptop | 0.7667740382959178 | 0.5016766077581333 | 0.781530609000741 | 0.5053821293919589 |
| cell phone | 0.7012840791863858 | 0.4194911819440395 | 0.7431420453759867 | 0.42802822655993183 |
| backpack | 0.8192955895322448 | 0.6351603549964012 | 0.8340542601559973 | 0.6348071113697373 |
| book | 0.720865051990868 | 0.4761898332829704 | 0.7517928816673755 | 0.48538046585610434 |
| bottle | 0.8449116022433163 | 0.7432862729317858 | 0.8581286072722895 | 0.7459235478637544 |

Speed ms/image native: preprocess3.162033547208557 / inference8.042573235603777 / postprocess0.16154117759579092. NMS0.6: preprocess2.878807491594489 / inference7.854872206800793 / postprocess9.903247488397028. Không suy ra FPS camera từ phép validation này.

### R07 AdamW — full validation Kaggle và ứng viên TTA cục bộ

Native AP50=0.76969424603517, AP50–95=0.5522398703716809, P=0.7905181358959329, R=0.6954760030677113.

NMS0.6 AP50=0.7901274169734709, AP50–95=0.5556389514732439, P=0.8052641972618061, R=0.7269916647889907. Đủ2.500 validation /8.359 GT, cùng manifest và protocol GPU ở đầu log. Native tốt hơn R04 native; NMS tốt hơn R04 NMS0.6 khoảng0,046612 điểm phần trăm. Chưa kết luận thắng R04 ba view CPU.

| Class | Native AP50 | Native AP50–95 | NMS0.6 AP50 | NMS0.6 AP50–95 |
|---|---:|---:|---:|---:|
| person | 0.8428340029365692 | 0.5764956072978176 | 0.8574409751374708 | 0.5788377655923631 |
| table | 0.7651457523583195 | 0.5526179651963987 | 0.789623139992446 | 0.5579023954680513 |
| chair | 0.6647740513934619 | 0.44924917340240134 | 0.6809564334027383 | 0.4499334906425328 |
| laptop | 0.7563469027598189 | 0.49865659983066024 | 0.771858589729574 | 0.5034758321484077 |
| cell phone | 0.7173449005399486 | 0.43953000716745727 | 0.7479314902879883 | 0.44315032459291137 |
| backpack | 0.8260448659621582 | 0.6700267382947571 | 0.8425856777502857 | 0.6702264311377077 |
| book | 0.7322370889642243 | 0.4795210437110672 | 0.7657817700456669 | 0.48920243419772974 |
| bottle | 0.8528264033668598 | 0.7518218280728883 | 0.8648412594415977 | 0.7523829380062479 |

GPU speed ms/image native: pre3.094861132407823/infer7.973813304401118/post0.15359179678853252; NMS0.6: pre2.7864526236007805/infer7.909264674403676/post9.999878332397929. Đây là validation timing.

Checkpoint epoch4/fileepoch3.pt SHA256 `ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785`;20.942.661bytes. CSV SHA256 `7d13a9ce9bc19fd3414c89033ddaf3c553805e3237f3b0fcbe623f93dbf7ad6e`. Trained ZIP đã được kernel kiểm SHA `8d362addcaef8dfe15583df37865686fd18120cde23c7241e66999f9057a1fa6`; không tải ZIP.

Đã tải đúng hai file qua UI, kiểm SHA256 và lưu tạm trong .temp/project8_v05_candidates. Hai bản tải trùng trong Downloads đã xoá sau kiểm hash cả bản nguồn và bản workspace. Chỉ một PT công bố trong weights/campaign_20261010 vẫn là R04; R07 là ứng viên so sánh, chưa promoted.

CSV nguyên văn:

```csv
epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,lr/pg0,lr/pg1,lr/pg2
1,320.482,1.09708,1.009,0.01189,0.79382,0.66946,0.74872,0.52849,1.15057,1.14636,0.01576,9.89362e-05,9.89362e-05,9.89362e-05
2,600.394,1.10152,0.98758,0.01228,0.80125,0.67152,0.75644,0.5353,1.13537,1.11463,0.01524,0.0001,0.0001,0.0001
3,881.962,1.10196,0.96173,0.01197,0.79275,0.68117,0.76173,0.54415,1.11498,1.09995,0.01466,0.0001,0.0001,0.0001
4,1166.82,1.0678,0.9084,0.01128,0.79307,0.69583,0.7703,0.55223,1.09417,1.05544,0.01406,0.0001,0.0001,0.0001
```

### Trạng thái nguồn thực thi tiếp theo

R07 dùng common LIGHT recipe của batch, đổi optimizer=AdamW và lr0=0.0001, 4epoch, cls_pw=0; init92adfeb2516861a6958e8adc4a15cf764faa3956078da1ebc81db1fbb7014e31. R08 freeze=10,3epoch hiện đang chạy.

Driver R07_views sao chép flow R04 đã đo thành công, đổi ứng viên/hash/prefix; kiểm AST thành công và select_epoch trên CSV thật chọn4. Dùng cùng ExplicitViewsValidator đã nằm trong suite31 PASS, cùng CPU FP32 protocol/GT với best. Chỉ chạy sau khi R04 TTA/NMS đã kết thúc để tránh thay PT công bố lúc đang đo. Hai view và ba view được đo đủ2500; ứng viên PT/CSV sẽ được xoá nếu cả hai lượt kết thúc không lỗi, CSV đầy đủ đã có ở log này.

Quick Save 'Campaign R01-R06 full metrics' đã được yêu cầu qua UI với Save Output cho phiên này. UI hiện vẫn báo Committing..., lịch sử mới nhất vẫn Version8; chưa xác nhận Version9 thành công. Không chạy Save & Run All hoặc dừng batch.

Driver R07_views sau bổ sung dọn CSV: SHA256 9e3c954e6c34753b508f350adea4d63f456557537356e74a87fad35ea8cf3b16. Chưa chạy, không phải số đo TTA R07.


### R04 full raw capture — parity PASS

```json
{
  "status": "PASS",
  "images": 2500,
  "boxes": 8359,
  "raw_rows": 2250000,
  "exact_reference_pipeline": "R04_three_views",
  "reference": {
    "mAP50": 0.8042673469408561,
    "mAP50_95": 0.5688479918514012,
    "precision": 0.8383669666187495,
    "recall": 0.7114310282044596,
    "per_class": {
      "person": {
        "AP50": 0.8625422449971606,
        "AP50_95": 0.5760641858935949
      },
      "table": {
        "AP50": 0.8033896878610551,
        "AP50_95": 0.5723947895142519
      },
      "chair": {
        "AP50": 0.7039366100000339,
        "AP50_95": 0.4756718250728035
      },
      "laptop": {
        "AP50": 0.8077202291135086,
        "AP50_95": 0.5466721148806155
      },
      "cell phone": {
        "AP50": 0.7464817614245698,
        "AP50_95": 0.43844475377410036
      },
      "backpack": {
        "AP50": 0.8595335705703024,
        "AP50_95": 0.7005709539154769
      },
      "book": {
        "AP50": 0.7852985345649273,
        "AP50_95": 0.4970856528406534
      },
      "bottle": {
        "AP50": 0.8652361369952909,
        "AP50_95": 0.7438796589197135
      }
    }
  },
  "metrics_equal_within": 1e-06
}
```


### R04 three-view NMS IoU0.3 — full replay hoàn tất

```json
{
  "mAP50": 0.7930983264042198,
  "mAP50_95": 0.5572498587181537,
  "precision": 0.8323759106437923,
  "recall": 0.7148012047885888,
  "per_class": {
    "person": {
      "AP50": 0.8460977818945872,
      "AP50_95": 0.5639270841095544
    },
    "table": {
      "AP50": 0.7768747253552373,
      "AP50_95": 0.5463997277648438
    },
    "chair": {
      "AP50": 0.6855821409671815,
      "AP50_95": 0.4576507921655451
    },
    "laptop": {
      "AP50": 0.8132161158514399,
      "AP50_95": 0.5430992863094093
    },
    "cell phone": {
      "AP50": 0.737338722648226,
      "AP50_95": 0.4305923563606395
    },
    "backpack": {
      "AP50": 0.853662351749841,
      "AP50_95": 0.6933875558027317
    },
    "book": {
      "AP50": 0.7733548284803603,
      "AP50_95": 0.48568487898812285
    },
    "bottle": {
      "AP50": 0.8586599442868857,
      "AP50_95": 0.7372571882443826
    }
  },
  "pipeline": "R04_three_views_nms_0.3",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.3,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R04 three-view NMS IoU0.4 — full replay hoàn tất

```json
{
  "mAP50": 0.8016116844170003,
  "mAP50_95": 0.5629777786975373,
  "precision": 0.8303724950506267,
  "recall": 0.7212291686999195,
  "per_class": {
    "person": {
      "AP50": 0.8595750285322449,
      "AP50_95": 0.5722705712842824
    },
    "table": {
      "AP50": 0.7979489962058388,
      "AP50_95": 0.5640602255769446
    },
    "chair": {
      "AP50": 0.6994788527948377,
      "AP50_95": 0.4660646836376632
    },
    "laptop": {
      "AP50": 0.8135056287494061,
      "AP50_95": 0.5447857912911307
    },
    "cell phone": {
      "AP50": 0.7427262263047769,
      "AP50_95": 0.43272789696883845
    },
    "backpack": {
      "AP50": 0.857730288179922,
      "AP50_95": 0.6958618629113327
    },
    "book": {
      "AP50": 0.7807758328149559,
      "AP50_95": 0.48947504748219517
    },
    "bottle": {
      "AP50": 0.8611526217540195,
      "AP50_95": 0.7385761504279115
    }
  },
  "pipeline": "R04_three_views_nms_0.4",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.4,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R04 three-view NMS IoU0.5 — full replay hoàn tất

```json
{
  "mAP50": 0.8050521265694426,
  "mAP50_95": 0.5656963669995626,
  "precision": 0.8297354359894933,
  "recall": 0.721293281303833,
  "per_class": {
    "person": {
      "AP50": 0.8640113786810067,
      "AP50_95": 0.5744499650057329
    },
    "table": {
      "AP50": 0.8041683119998588,
      "AP50_95": 0.5692793461060456
    },
    "chair": {
      "AP50": 0.7026889774797049,
      "AP50_95": 0.4690993507605502
    },
    "laptop": {
      "AP50": 0.8113582632785037,
      "AP50_95": 0.5433629357526237
    },
    "cell phone": {
      "AP50": 0.7458552926594973,
      "AP50_95": 0.43502508454379274
    },
    "backpack": {
      "AP50": 0.8599994010428751,
      "AP50_95": 0.6985161673784781
    },
    "book": {
      "AP50": 0.7864815854726537,
      "AP50_95": 0.4943149930652407
    },
    "bottle": {
      "AP50": 0.8658538019414398,
      "AP50_95": 0.7415230933840367
    }
  },
  "pipeline": "R04_three_views_nms_0.5",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.5,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R04 three-view NMS IoU0.6 — full replay hoàn tất

```json
{
  "mAP50": 0.8042673469408561,
  "mAP50_95": 0.5688479918514012,
  "precision": 0.8383669666187495,
  "recall": 0.7114310282044596,
  "per_class": {
    "person": {
      "AP50": 0.8625422449971606,
      "AP50_95": 0.5760641858935949
    },
    "table": {
      "AP50": 0.8033896878610551,
      "AP50_95": 0.5723947895142519
    },
    "chair": {
      "AP50": 0.7039366100000339,
      "AP50_95": 0.4756718250728035
    },
    "laptop": {
      "AP50": 0.8077202291135086,
      "AP50_95": 0.5466721148806155
    },
    "cell phone": {
      "AP50": 0.7464817614245698,
      "AP50_95": 0.43844475377410036
    },
    "backpack": {
      "AP50": 0.8595335705703024,
      "AP50_95": 0.7005709539154769
    },
    "book": {
      "AP50": 0.7852985345649273,
      "AP50_95": 0.4970856528406534
    },
    "bottle": {
      "AP50": 0.8652361369952909,
      "AP50_95": 0.7438796589197135
    }
  },
  "pipeline": "R04_three_views_nms_0.6",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.6,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R04 three-view NMS IoU0.7 — full replay hoàn tất

```json
{
  "mAP50": 0.7986351377135756,
  "mAP50_95": 0.5701480101791307,
  "precision": 0.8265139567405038,
  "recall": 0.7091475994542042,
  "per_class": {
    "person": {
      "AP50": 0.8558967602983544,
      "AP50_95": 0.5777446705465528
    },
    "table": {
      "AP50": 0.7979443602381275,
      "AP50_95": 0.5735783006614745
    },
    "chair": {
      "AP50": 0.6973538918204167,
      "AP50_95": 0.478290441944894
    },
    "laptop": {
      "AP50": 0.8010731140267697,
      "AP50_95": 0.5448762301136337
    },
    "cell phone": {
      "AP50": 0.739116384424569,
      "AP50_95": 0.43996394701492425
    },
    "backpack": {
      "AP50": 0.8560781619488259,
      "AP50_95": 0.7025571674115926
    },
    "book": {
      "AP50": 0.7789207343676179,
      "AP50_95": 0.4980805132733753
    },
    "bottle": {
      "AP50": 0.8626976945839244,
      "AP50_95": 0.7460928104665991
    }
  },
  "pipeline": "R04_three_views_nms_0.7",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.7,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R04 three-view NMS IoU0.8 — full replay hoàn tất

```json
{
  "mAP50": 0.7838295212843198,
  "mAP50_95": 0.5673625737210615,
  "precision": 0.8229522652820294,
  "recall": 0.6875258190225935,
  "per_class": {
    "person": {
      "AP50": 0.8426016535911133,
      "AP50_95": 0.5755931140969127
    },
    "table": {
      "AP50": 0.7830804167726115,
      "AP50_95": 0.5699175951976336
    },
    "chair": {
      "AP50": 0.6820887540375724,
      "AP50_95": 0.47406316211712457
    },
    "laptop": {
      "AP50": 0.7891901152825445,
      "AP50_95": 0.5405588973045715
    },
    "cell phone": {
      "AP50": 0.7126015258111378,
      "AP50_95": 0.43506925086350945
    },
    "backpack": {
      "AP50": 0.8503661282192646,
      "AP50_95": 0.7039489872200214
    },
    "book": {
      "AP50": 0.7546016852541118,
      "AP50_95": 0.49103372443725846
    },
    "bottle": {
      "AP50": 0.8561058913062025,
      "AP50_95": 0.7487158585314605
    }
  },
  "pipeline": "R04_three_views_nms_0.8",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.8,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R04 TTA × NMS sweep — kết thúc

```json
{
  "status": "complete",
  "results_count": 6,
  "winner": {
    "mAP50": 0.8050521265694426,
    "mAP50_95": 0.5656963669995626,
    "precision": 0.8297354359894933,
    "recall": 0.721293281303833,
    "per_class": {
      "person": {
        "AP50": 0.8640113786810067,
        "AP50_95": 0.5744499650057329
      },
      "table": {
        "AP50": 0.8041683119998588,
        "AP50_95": 0.5692793461060456
      },
      "chair": {
        "AP50": 0.7026889774797049,
        "AP50_95": 0.4690993507605502
      },
      "laptop": {
        "AP50": 0.8113582632785037,
        "AP50_95": 0.5433629357526237
      },
      "cell phone": {
        "AP50": 0.7458552926594973,
        "AP50_95": 0.43502508454379274
      },
      "backpack": {
        "AP50": 0.8599994010428751,
        "AP50_95": 0.6985161673784781
      },
      "book": {
        "AP50": 0.7864815854726537,
        "AP50_95": 0.4943149930652407
      },
      "bottle": {
        "AP50": 0.8658538019414398,
        "AP50_95": 0.7415230933840367
      }
    },
    "pipeline": "R04_three_views_nms_0.5",
    "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
    "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
    "protocol": {
      "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
      "split": "val",
      "imgsz": 1280,
      "batch": 8,
      "device": "cpu",
      "workers": 0,
      "conf": 0.001,
      "iou": 0.7,
      "max_det": 300,
      "rect": true,
      "quantize": null,
      "nms": false,
      "augment": false,
      "plots": false
    },
    "views": [
      [
        1.0,
        false
      ],
      [
        0.83,
        true
      ],
      [
        0.67,
        false
      ]
    ],
    "extra_nms_iou": 0.5,
    "postprocessing": "explicit_views_class_aware_nms",
    "validation_images": 2500,
    "validation_boxes": 8359,
    "test_used": false,
    "timing_note": "AP replay only; no camera latency/FPS measurement.",
    "kept_as_best": true,
    "replaces_pipeline": "R04_three_views"
  },
  "checkpoint_file_count": 1,
  "raw_cache_persisted": false
}
```


### R07 CPU explicit views — bắt đầu

```json
{
  "timestamp_utc": "20261009_202641",
  "status": "running",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "source_sha256": "9e3c954e6c34753b508f350adea4d63f456557537356e74a87fad35ea8cf3b16",
  "inference_helper_sha256": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
  "csv_sha256": "7d13a9ce9bc19fd3414c89033ddaf3c553805e3237f3b0fcbe623f93dbf7ad6e",
  "selected_epoch": 4,
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "validation_yaml_sha256": "f872aa9419cd633e271ed68ab7464dd0937bfd2652f653aacd7d4fe96bbc0f2c",
  "test_used": false,
  "views_to_measure": {
    "R07_flip": [
      [
        1.0,
        false
      ],
      [
        1.0,
        true
      ]
    ],
    "R07_three_views": [
      [
        1.0,
        false
      ],
      [
        0.83,
        true
      ],
      [
        0.67,
        false
      ]
    ]
  },
  "runtime": {
    "ultralytics": "8.4.165",
    "torch": "2.14.0+cpu",
    "device": "CPU",
    "torch_threads": 4
  }
}
```

R04 TTA/NMS driver exit0, sáu ngưỡng full replay hoàn tất. Sau cập nhật selection đã kiểm thư mục công bố đúng1 PT và SHA khớp metadata; cache raw chỉ RAM, smoke TXT/YAML đã dọn. R07_views bắt đầu đủ payload preflight2500 và đang đo R07_flip; không công nhận điểm TTA R07 khi chưa hoàn tất. Bằng chứng UI hiện hành lưu .temp/PROJECT8_V05_KAGGLE_CURRENT.jpg, R08 đang chấm epoch3; Quick Save vẫn Committing..., chưa có phiên9 được xác nhận.


### Kiểm nhóm video giữa train/val/test — audit thực

```json
{
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "rows": 15000,
  "split_counts": {
    "train": 12000,
    "val": 2500,
    "test": 500
  },
  "exact_image_sha_duplicates": 0,
  "exact_pixel_sha_duplicates": 0,
  "cross_split_group_count": 17,
  "cross_split_images": 368,
  "cross_split_groups": {
    "video:items_in_room__vcompress_2_mov": {
      "splits": {
        "test": 1,
        "train": 15,
        "val": 2
      },
      "source": "items_in_room"
    },
    "video:backpacks_v7__IMG_8016_MOV": {
      "splits": {
        "val": 2,
        "train": 4
      },
      "source": "backpacks_v7"
    },
    "video:backpacks_v7__IMG_8017_MOV": {
      "splits": {
        "test": 1,
        "train": 1
      },
      "source": "backpacks_v7"
    },
    "video:backpacks_v7__IMG_8018_MOV": {
      "splits": {
        "test": 5,
        "train": 8,
        "val": 2
      },
      "source": "backpacks_v7"
    },
    "video:backpacks_v7__IMG_8019_MOV": {
      "splits": {
        "val": 5,
        "train": 12
      },
      "source": "backpacks_v7"
    },
    "video:backpacks_v7__IMG_8020_MOV": {
      "splits": {
        "test": 1,
        "train": 10,
        "val": 4
      },
      "source": "backpacks_v7"
    },
    "video:backpacks_v7__IMG_8021_MOV": {
      "splits": {
        "test": 4,
        "train": 15,
        "val": 3
      },
      "source": "backpacks_v7"
    },
    "video:backpacks_v7__IMG_8022_MOV": {
      "splits": {
        "test": 2,
        "train": 7,
        "val": 1
      },
      "source": "backpacks_v7"
    },
    "video:backpacks_v7__IMG_8023_MOV": {
      "splits": {
        "test": 2,
        "train": 5,
        "val": 3
      },
      "source": "backpacks_v7"
    },
    "video:classroom_phone_v18__Class_1_Video_1_mp4": {
      "splits": {
        "test": 2,
        "train": 17,
        "val": 6
      },
      "source": "classroom_phone_v18"
    },
    "video:classroom_phone_v18__Class_1_Video_2_mp4": {
      "splits": {
        "test": 1,
        "train": 11,
        "val": 3
      },
      "source": "classroom_phone_v18"
    },
    "video:classroom_phone_v18__Class_2_Video_1_mp4": {
      "splits": {
        "test": 5,
        "train": 28,
        "val": 7
      },
      "source": "classroom_phone_v18"
    },
    "video:classroom_phone_v18__Class_2_Video_2_mp4": {
      "splits": {
        "test": 8,
        "train": 40,
        "val": 8
      },
      "source": "classroom_phone_v18"
    },
    "video:classroom_phone_v18__Class_3_Video_1_mp4": {
      "splits": {
        "test": 2,
        "train": 51,
        "val": 19
      },
      "source": "classroom_phone_v18"
    },
    "video:classroom_phone_v18__Class_3_Video_2_mp4": {
      "splits": {
        "test": 1,
        "train": 12,
        "val": 2
      },
      "source": "classroom_phone_v18"
    },
    "video:classroom_phone_v18__Class_4_Video_1_mp4": {
      "splits": {
        "test": 3,
        "train": 7,
        "val": 5
      },
      "source": "classroom_phone_v18"
    },
    "video:classroom_phone_v18__Class_4_Video_2_mp4": {
      "splits": {
        "test": 2,
        "train": 12,
        "val": 1
      },
      "source": "classroom_phone_v18"
    }
  },
  "visual_pairs_reviewed": [
    "video:backpacks_v7__IMG_8016_MOV",
    "video:classroom_phone_v18__Class_3_Video_2_mp4"
  ],
  "visual_finding": "Same backpack/background and same classroom video across train/val, confirmed on four original images. Other fifteen groups identified from source video names; not all visually reviewed.",
  "interpretation": "Current scores remain measurements on the existing split; they do not prove independence across scenes. Near-scene audit remains incomplete. No images or labels changed."
}
```


### Các lượt xác nhận đã có source — chưa train

Bổ sung code seed43/44, lịch dài, nano pretrained mới đối chứng small, hai nhánh fresh nano bỏ nhóm holdout/bỏ ngẫu nhiên, freeze→unfreeze và KD có điều kiện. Hai nhánh freeze cùng 2+4 epoch và cùng reset optimizer; KD có student control cùng batch8/lr/3epoch. Số epoch thực phụ thuộc cấu hình nano thắng; ví dụ R04 và teacher10epoch cần64epoch nếu teacher tốt hơn,58 nếu bỏ KD. Không phải kết quả train. Dùng implementation distillation có sẵn của Ultralytics8.4.165, đã đọc source trainer/wrapper; chưa xác nhận chạy GPU hay parity artifact distill.

```json
{
  "status": "prepared_and_config_checked_not_trained",
  "synthetic_fixture_used_for_config_validation": true,
  "example_recipe_total_epochs": 64,
  "configs_checked": 12,
  "source_sha256": {
    "kaggle_tools\\experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
    "kaggle_tools\\training_subsets.py": "07250bf66aab73b3c347fbcbc33e48f4b988f5443bf101d8c547768387b8cb50",
    "kaggle_tools\\group_controls.py": "2aa9d18fff31ec946751592a9291da663519db12b12e4ab44c279b0c9c455e76",
    "kaggle_tools\\inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
    "kaggle_tools\\confirmation_plan.py": "5c196e75cd2e3adeb7d4f298c70cdb73ee5cfffa1d011d663869515c90c6e484",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.py": "3f5ecdf44330edab5d0f95414bce7a364d4a5e3a06924f404e7f41c6f7848f6f",
    "notebooks\\kaggle\\project8_v05_confirmation_train_cell_20261010.py": "23295c983c74067d091ea4a9ffce3c023d4e847ae4a2ca79cea52a29b644ffb2",
    "notebooks\\kaggle\\project8_v05_confirmation_train_20261010.ipynb": "0ffa1dfa36e236472efa7924eb8e462eda1468dddea7981c312ca6d46de6f32f"
  },
  "cell_source_sha256": "f3e330c3b586b64f5ee4e4d40e116c07b86f5329cab68e040183d2584ae3a165",
  "installed_kd_source_sha256": "190311d996f9964a9f4302a41f43bd25f3b482516270e148df16f491b9d4fc9b",
  "tests": "Eight semantic planning tests and four group-control tests; full suite recorded separately."
}
```


### R07_flip — full validation hoàn tất

```json
{
  "pipeline": "R07_flip",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "mAP50": 0.8016129943356763,
  "mAP50_95": 0.5658463131753598,
  "precision": 0.8080414449133656,
  "recall": 0.7290728101097517,
  "views": [
    [
      1.0,
      false
    ],
    [
      1.0,
      true
    ]
  ],
  "postprocessing": "explicit_views_class_aware_nms_06",
  "extra_nms_iou": 0.6,
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "test_used": false,
  "validation_images": 2500,
  "validation_boxes": 8359,
  "speed": {
    "preprocess": 3.0681391200138024,
    "inference": 322.6317347999953,
    "loss": 0.00029728000954492017,
    "postprocess": 0.06439840000530239
  },
  "per_class": {
    "person": {
      "AP50": 0.8691406732438633,
      "AP50_95": 0.5830825478933457
    },
    "table": {
      "AP50": 0.8001318256718033,
      "AP50_95": 0.5677828974214647
    },
    "chair": {
      "AP50": 0.6967997325912539,
      "AP50_95": 0.46222406731880383
    },
    "laptop": {
      "AP50": 0.7930702366201527,
      "AP50_95": 0.5192936019558654
    },
    "cell phone": {
      "AP50": 0.760509577566908,
      "AP50_95": 0.45617935295416884
    },
    "backpack": {
      "AP50": 0.8449100291424394,
      "AP50_95": 0.6799174547920731
    },
    "book": {
      "AP50": 0.7787519435106015,
      "AP50_95": 0.4990321881939238
    },
    "bottle": {
      "AP50": 0.8695899363383888,
      "AP50_95": 0.7592583948732334
    }
  },
  "kept_as_best": false
}
```

### R08 freeze10 — số đo đủ validation từ Kaggle

Epoch3/fileepoch2.pt SHA256 `4ed05d5fb66d15cf094ef68460015b1de4e7ab467065cad51082c54733fa9c32`. Common LIGHT recipe, freeze=10,3epoch, init92adfeb...; cùng2500 validation/8359GT và protocolGPU ở đầu log. Không tải checkpoint kém hơn về máy.

Native AP50=0.7549635599276877; AP50–95=0.5369023774899933; P=0.7855640370048028; R=0.6748547325546423.

NMS0.6 AP50=0.7762453013897859; AP50–95=0.5409270392539918; P=0.7831088295544685; R=0.7127260984230426.

| Class | Native AP50 | Native AP50–95 | NMS0.6 AP50 | NMS0.6 AP50–95 |
|---|---:|---:|---:|---:|
| person | 0.8401409705203262 | 0.5625440623920775 | 0.8516719998696975 | 0.5616591114174175 |
| table | 0.7720080880232096 | 0.5508120421688197 | 0.7920682541389186 | 0.5540722337169541 |
| chair | 0.6385869810614192 | 0.4334845643766966 | 0.6600570522774254 | 0.43584128061456556 |
| laptop | 0.7227809704539421 | 0.4714582240531956 | 0.7412140054448858 | 0.47606186534785444 |
| cell phone | 0.6854129759939962 | 0.4139322822612345 | 0.7214535027252726 | 0.42159392380927596 |
| backpack | 0.809587775353098 | 0.6525466909524343 | 0.8309667856447902 | 0.6584547200588199 |
| book | 0.7308795692664523 | 0.47411217522415433 | 0.7591682456381441 | 0.4815168143181768 |
| bottle | 0.8403111487490583 | 0.7363289784913328 | 0.8533625653791523 | 0.7382163647488703 |

GPU timing ms/image: native pre3.0988322512064768/infer7.9524908183972/post0.15725878039374946; NMS pre2.8193642659985927/infer7.932366386796639/post9.91678702359932. Không phải phép đo camera.


### Đối chứng nhóm cảnh — kiểm đủ payload train và tạo danh sách

```json
{
  "status": "prepared_not_trained",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "original_train_images": 12000,
  "removed_train_images": 255,
  "removed_train_boxes": 1656,
  "removed_group_keys": [
    "video:backpacks_v7__IMG_8016_MOV",
    "video:backpacks_v7__IMG_8017_MOV",
    "video:backpacks_v7__IMG_8018_MOV",
    "video:backpacks_v7__IMG_8019_MOV",
    "video:backpacks_v7__IMG_8020_MOV",
    "video:backpacks_v7__IMG_8021_MOV",
    "video:backpacks_v7__IMG_8022_MOV",
    "video:backpacks_v7__IMG_8023_MOV",
    "video:classroom_phone_v18__Class_1_Video_1_mp4",
    "video:classroom_phone_v18__Class_1_Video_2_mp4",
    "video:classroom_phone_v18__Class_2_Video_1_mp4",
    "video:classroom_phone_v18__Class_2_Video_2_mp4",
    "video:classroom_phone_v18__Class_3_Video_1_mp4",
    "video:classroom_phone_v18__Class_3_Video_2_mp4",
    "video:classroom_phone_v18__Class_4_Video_1_mp4",
    "video:classroom_phone_v18__Class_4_Video_2_mp4",
    "video:items_in_room__vcompress_2_mov"
  ],
  "validation_images": 2500,
  "validation_boxes": 8359,
  "labels_changed": false,
  "holdout_roles_changed": false,
  "test_metadata_used_for_group_exclusion": true,
  "test_predictions_used": false,
  "test_labels_read": false,
  "full_visual_review_complete": false,
  "note": "Fresh COCO initializer required: removing rows cannot undo warm-start exposure. Known groups only; not a complete near-scene audit. Random control matches images/steps, not box distribution.",
  "subsets": {
    "group_disjoint": {
      "train_images": 11745,
      "train_boxes": 31616,
      "class_boxes": {
        "person": 6673,
        "table": 2352,
        "chair": 5008,
        "laptop": 1266,
        "cell phone": 5059,
        "backpack": 1577,
        "book": 6416,
        "bottle": 3265
      },
      "shared_holdout_groups": 0,
      "source_counts": {
        "study_desk": 368,
        "bag": 669,
        "celulares": 772,
        "classroom_objects": 255,
        "items_in_room": 349,
        "smartphone_v1": 1319,
        "ron88": 1200,
        "cup_v2_v3": 330,
        "backpacks_v7": 196,
        "coco2017_train": 819,
        "laptop_roboflow_v1": 881,
        "classroom_roboflow_v1": 1774,
        "student_behaviour_detection_v6": 353,
        "merged_3k_local": 2426,
        "roboflow_mobile_phone_v1": 34
      },
      "train_identity_sha256": "31c4f6f61c5971bc96b287b1f1b97362e5664e4b5ab489ec09d0b774b5abeec5",
      "train_list": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\group_controls_20261009_203952\\group_disjoint_train.txt",
      "train_list_sha256": "4cfca1c3be3da601c93ac4cb33953e65fe507cb2c5f9b5eb685cd897ceb90e3b",
      "data_yaml": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\group_controls_20261009_203952\\group_disjoint.yaml",
      "data_yaml_sha256": "b5c1d468541936904854ab387bf0cddefadb743092dd8e80352842e7c3e3e0f8"
    },
    "random_removal": {
      "train_images": 11745,
      "train_boxes": 32585,
      "class_boxes": {
        "person": 7344,
        "table": 2285,
        "chair": 4914,
        "laptop": 1248,
        "cell phone": 5724,
        "backpack": 1599,
        "book": 6275,
        "bottle": 3196
      },
      "shared_holdout_groups": 17,
      "source_counts": {
        "study_desk": 361,
        "bag": 649,
        "celulares": 753,
        "classroom_objects": 250,
        "items_in_room": 360,
        "smartphone_v1": 1287,
        "ron88": 1172,
        "classroom_phone_v18": 173,
        "cup_v2_v3": 326,
        "backpacks_v7": 255,
        "coco2017_train": 802,
        "laptop_roboflow_v1": 864,
        "classroom_roboflow_v1": 1742,
        "student_behaviour_detection_v6": 349,
        "merged_3k_local": 2369,
        "roboflow_mobile_phone_v1": 33
      },
      "train_identity_sha256": "9a05818ab8d2583a3a8abd20d3a6d01bd4eefca5ca8df2b48c6f2535bdfe87e0",
      "train_list": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\group_controls_20261009_203952\\random_removal_train.txt",
      "train_list_sha256": "d75750f9a346d40d15000644a789207bf5ab96c9144d410763079fe3a781d322",
      "data_yaml": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\group_controls_20261009_203952\\random_removal.yaml",
      "data_yaml_sha256": "fbe0ea8585e48eb462bf48f30c5ff9c9b07ab672815c5e6dd76562b7a2403b7f"
    }
  }
}
```

### Kiểm thử mới về nhóm holdout — suite37 PASS

RED group_controls: thiếu module trước triển khai. GREEN:4 kiểm thử group_controls pass1,204giây. RED confirmation_plan: hai kiểm thử mới báo thiếu tham số group_controls; GREEN:8 kiểm thử plan pass0,007giây. Toàn suite **37/37 PASS,16,580giây,exit0** với lệnh `.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v`, PYTHONPATH=workspace,TEMP/TMP=.temp.

Các kiểm thử mới bảo đảm: chỉ loại train thuộc nhóm val/test; không sửa record/nhãn/holdout; từ chối nhóm thiếu, ID trùng, role lạ và train rỗng; tạo danh sách thật với hash payload, đủ lớp và validation gốc; từ chối payload đã thay trước khi ghi; hai plan fresh nano khớp recipe đối chứng và protocol evaluation; từ chối GT khác, nhóm còn giao hoặc số ảnh đối chứng khác nhau.

Builder mới kiểm12 cấu hình trên fixture giả lập, không phải kết quả train. Ví dụ nano thắng4epoch/teacher10epoch:64epoch nếu chạyKD,58epoch nếu bỏKD. Hai nhánh nhóm cảnh bổ sung20epoch; phải tính thời gian/quota/session còn lại khi thực thi, chia phiên nếu cần bảo toàn checkpoint/provenance. Bộ cell/ipynb được đóng gói lại cùng helper chọn train và helper nhóm; hash nguồn ở record mới nhất. Chưa nạp/chạy các cell này trên GPU.


### Packed confirmation definitions — portable smoke PASS

```json
{
  "status": "PASS",
  "scope": "Packed definitions and actual group selection, no GPU training",
  "packed_definitions_require_local_kaggle_tools": false,
  "synthetic_plan_fixture_imported_locally": true,
  "actual_selected_train_images": 11745,
  "canonical_train_identity_sha256": "31c4f6f61c5971bc96b287b1f1b97362e5664e4b5ab489ec09d0b774b5abeec5",
  "synthetic_plan_configs": 12,
  "synthetic_plan_epochs": 64,
  "cell_source_sha256": "f3e330c3b586b64f5ee4e4d40e116c07b86f5329cab68e040183d2584ae3a165"
}
```

Hash kiểm thử mới: tests/test_group_controls.py SHA256 `d28ed80aca9c7f8ce20c2a9820cf5bbe805277a768968562d86d1153a1b03f92`; tests/test_confirmation_plan.py SHA256 `f783a29e47ae111d1fad2c1c42835dbbde2d17ecd0db268d9d2f127265fae3b5`.

Chuẩn bị dữ liệu thật đã exit0: group_disjoint11745ảnh/31616box, random_removal11745ảnh/32585box; cả8 lớp còn hiện diện. Nhánh group_disjoint không còn source_group_key giao với val/test theo metadata. Đây là kiểm các nhóm đã biết, không chứng minh toàn bộ cảnh gần trùng đã được rà. Validation2500/8359GT không bị loại hoặc sửa.

Đối soát cuối lượt: best công bố vẫn R04_three_views_nms_0.5, mAP50=0.8050521265694426, đúng1PT và SHA087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758. R07_flip đủ2500/8359GT đạt0.8016129943356763, không promoted; R07_three_views đang thực thi. KaggleR09 train1536 đã epoch3/3 khoảng85%, chưa có full common1280 native/NMS. Quick Save vẫnCommitting..., không xác nhậnversion9. Group-controls payload preparation và packed-definitions smoke đều exit0;37testsPASS đã ghi ở trên. Các thử GPU mới, source ablations, seeds/freeze/KD và đối chứng nhóm cảnh còn chưa train.

### R09 train1536 — full common1280 validation

Checkpoint epoch3/fileepoch2.pt SHA256 `0c383267793750625842bad4b448a94edf7e245059ed8df58ab69d4263df772b`. Train1536/batch16/3epoch, cùng LIGHT recipe/init/LR còn lại; đây là cấu hình đổi cả resolution và batch. Trainer chọn epoch bằng mAP50 ở1536; phép chấm common sau train dùng1280 theo protocolGPU ở đầu log, đủ2500ảnh/8359GT. Không trộn điểm trainer1536 với bảngcommon1280.

Native AP50=0.7458247382497158, AP50–95=0.5262188231327589, P=0.7847929582140958, R=0.6818681091192573.

NMS0.6 AP50=0.7671791073259336, AP50–95=0.530073644714327, P=0.7947561673428932, R=0.7069085067816925. Kém best, không tải PT R09.

| Class | Native AP50 | Native AP50–95 | NMS0.6 AP50 | NMS0.6 AP50–95 |
|---|---:|---:|---:|---:|
| person | 0.8301646351283173 | 0.562907809867296 | 0.8443130166196208 | 0.5652779921065753 |
| table | 0.7615519951678328 | 0.5268009251191075 | 0.7823912716413246 | 0.5295897479729124 |
| chair | 0.640561409050611 | 0.4216301190704062 | 0.6574720605108308 | 0.4209630763083204 |
| laptop | 0.6638207589652563 | 0.42328879971676026 | 0.6908610533881767 | 0.4340465949554006 |
| cell phone | 0.6838635948008112 | 0.4150040166280703 | 0.7205905410745619 | 0.4232357478882425 |
| backpack | 0.8134965311051606 | 0.6498142447273132 | 0.8312448781008523 | 0.6499927575742215 |
| book | 0.7192504690817536 | 0.4553526211193534 | 0.7465005374937272 | 0.46252270226309566 |
| bottle | 0.853888512697983 | 0.7549520488137648 | 0.8640594997783749 | 0.7549605386458482 |

Speed ms/image native pre3.088620777604956/infer8.030844695997075/post0.17515516239727733; NMS pre2.7729652047979467/infer8.017461256792739/post10.5590844440012. Chưa phải benchmark camera.

Đã giữ tạm thêm bản R07_for_nms.pt (SHAecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785) để quét NMS sau khi R07_views hoàn tất; bản tạm chỉ phục vụ so sánh và sẽ xoá sau sáu ngưỡng, không thêm PT công bố. Driver quét R07 SHA256 `e7562fab79688245662499985d549b7b9e7bb483cf05a87948ef8fecd8540f97`: đọc reference đủ2500 từ MD, kiểm parity trước sweep, chỉ promote nếu tuple(AP50,AP50–95) thắng best hiện hành và protocol/GT khớp. Source tạo từ flow R04 đã chạy thành công, cùng helper trong suite37PASS; kiểm AST thành công; chưa chạy sweep R07 ở thời điểm ghi dòng này.


### R07_three_views — full validation hoàn tất

```json
{
  "pipeline": "R07_three_views",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "mAP50": 0.8017407825336353,
  "mAP50_95": 0.5730767976702114,
  "precision": 0.8284566429903828,
  "recall": 0.7116710749043234,
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "postprocessing": "explicit_views_class_aware_nms_06",
  "extra_nms_iou": 0.6,
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "test_used": false,
  "validation_images": 2500,
  "validation_boxes": 8359,
  "speed": {
    "preprocess": 2.9392289999945205,
    "inference": 351.5939240399806,
    "loss": 0.0002584399961051531,
    "postprocess": 0.05972476001770701
  },
  "per_class": {
    "person": {
      "AP50": 0.8577317718644061,
      "AP50_95": 0.5772592620498125
    },
    "table": {
      "AP50": 0.7948183727150674,
      "AP50_95": 0.5670884293475085
    },
    "chair": {
      "AP50": 0.7049174771712111,
      "AP50_95": 0.48125136450360434
    },
    "laptop": {
      "AP50": 0.80949113089093,
      "AP50_95": 0.5462547467874592
    },
    "cell phone": {
      "AP50": 0.7489860228041731,
      "AP50_95": 0.44738582434768864
    },
    "backpack": {
      "AP50": 0.8530688810686449,
      "AP50_95": 0.7081100950313959
    },
    "book": {
      "AP50": 0.7790243491984411,
      "AP50_95": 0.49853469695082336
    },
    "bottle": {
      "AP50": 0.8658882545562092,
      "AP50_95": 0.7587299623433976
    }
  },
  "kept_as_best": false
}
```


### R07 CPU explicit views — kết thúc

```json
{
  "status": "complete",
  "errors": {},
  "kept_checkpoint": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\weights\\campaign_20261010\\BEST_Project8_v05.pt",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758"
}
```

R07_views exit0 sau hai lượt đủ2500/8359GT: R07_flip80,161299%/56,584631%; R07_three_views80,174078%/57,307680%. Cả hai không thắng best theo tuple(AP50,AP50–95); R07_three_views có AP50–95 cao hơn R04_three_views_nms_0.5 nhưng AP50 thấp hơn. Giữ tiêu chí chọn đã công bố, không đổi tiêu chí sau khi thấy điểm. R07_epoch3.pt và R07_results.csv tạm đã dọn tự động; chỉ R07_for_nms.pt còn giữ để quét ngưỡng. FullCSV/hashes/per-class đều đã ghiMD. Lượt R07 raw TTA/NMS mới bắt đầu sau khi handle R07_views terminalexit0, không chạy chồng hai lượt hay thay PT đang được đo.


### R07 TTA × NMS sweep — bắt đầu

```json
{
  "status": "running",
  "timestamp_utc": "20261009_205831",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "source_sha256": "e7562fab79688245662499985d549b7b9e7bb483cf05a87948ef8fecd8540f97",
  "collector_sha256": "edfe3ca9f20db16f7ab374c55525a2f1fccaa9989f3ba71bd180b8c2e215d04b",
  "replay_sha256": "b75e2237f159b1f03afab295012b93f08e799bdcd859de3c76f19c886b2d7a60",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "cache_note": "Unmerged restored predictions in RAM only, before class-aware NMS and global max_det."
}
```


### R07 raw TTA cache — real3-image integration PASS

```json
{
  "status": "PASS",
  "images": 3,
  "raw_rows": 2700,
  "ground_truth_boxes": 14,
  "metrics_equal_within": 1e-06,
  "used_for_full_AP_claim": false
}
```

Xác minh lượt hiện hành: R07 TTA/NMS đã kiểm lại5000payload val, real3-image parityPASS và bắt đầu full capture2500. Không thay trọng số công bố trong lúc capture đang chạy. Tạm chỉ cònR07_for_nms.pt; thư mục campaign vẫn đúng1PT và SHA khớp selection. KaggleR10 mixup/cutmix epoch1/3 đang chạy; QuickSave vẫnCommitting..., chưa xác nhậnversion9. Bằng chứng UI hiện hành đã lưu .temp/PROJECT8_V05_KAGGLE_CURRENT.jpg.


### One-file pipeline — kiểm kỹ thuật serialization/predict

```json
{
  "scope": "Two real native models in one inference artifact; technical tensor only, no AP claim",
  "component_sha256": [
    "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
    "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785"
  ],
  "views": [
    [
      1.0,
      false
    ]
  ],
  "extra_nms_iou": 0.6,
  "backend_roundtrip": "PASS",
  "public_predict": "PASS",
  "source_sha256": "5f58725f2bf9c50a539b03d6e9c87d57e49595504db4df30afbda4e4cfddc204",
  "helper_sha256": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
  "artifact_bytes": 21722501,
  "artifact_sha256": "1b32e1eeb668c6abd20d27369eb64fbf135581ed8a80ec243d2dcfd76eaa9353",
  "temporary_model_removed": true
}
```


### R07 full raw capture — parity PASS

```json
{
  "status": "PASS",
  "images": 2500,
  "boxes": 8359,
  "raw_rows": 2250000,
  "exact_reference_pipeline": "R07_three_views",
  "reference": {
    "mAP50": 0.8017407825336353,
    "mAP50_95": 0.5730767976702114,
    "precision": 0.8284566429903828,
    "recall": 0.7116710749043234,
    "per_class": {
      "person": {
        "AP50": 0.8577317718644061,
        "AP50_95": 0.5772592620498125
      },
      "table": {
        "AP50": 0.7948183727150674,
        "AP50_95": 0.5670884293475085
      },
      "chair": {
        "AP50": 0.7049174771712111,
        "AP50_95": 0.48125136450360434
      },
      "laptop": {
        "AP50": 0.80949113089093,
        "AP50_95": 0.5462547467874592
      },
      "cell phone": {
        "AP50": 0.7489860228041731,
        "AP50_95": 0.44738582434768864
      },
      "backpack": {
        "AP50": 0.8530688810686449,
        "AP50_95": 0.7081100950313959
      },
      "book": {
        "AP50": 0.7790243491984411,
        "AP50_95": 0.49853469695082336
      },
      "bottle": {
        "AP50": 0.8658882545562092,
        "AP50_95": 0.7587299623433976
      }
    }
  },
  "metrics_equal_within": 1e-06
}
```


### R07 three-view NMS IoU0.3 — full replay hoàn tất

```json
{
  "mAP50": 0.7872609329511747,
  "mAP50_95": 0.5602685720031138,
  "precision": 0.8357998710063532,
  "recall": 0.7066932605201883,
  "per_class": {
    "person": {
      "AP50": 0.8391179028040843,
      "AP50_95": 0.5632368767464394
    },
    "table": {
      "AP50": 0.7602146129033724,
      "AP50_95": 0.5396945021866026
    },
    "chair": {
      "AP50": 0.6774390389011276,
      "AP50_95": 0.4580763581634524
    },
    "laptop": {
      "AP50": 0.8041455801792887,
      "AP50_95": 0.5431591868115387
    },
    "cell phone": {
      "AP50": 0.7431793087857932,
      "AP50_95": 0.44227755493900195
    },
    "backpack": {
      "AP50": 0.8486571946583703,
      "AP50_95": 0.6961064253925204
    },
    "book": {
      "AP50": 0.7668541474428576,
      "AP50_95": 0.48796291282298354
    },
    "bottle": {
      "AP50": 0.8584796779345039,
      "AP50_95": 0.7516347589623711
    }
  },
  "pipeline": "R07_three_views_nms_0.3",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.3,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R07 three-view NMS IoU0.4 — full replay hoàn tất

```json
{
  "mAP50": 0.7973406182537831,
  "mAP50_95": 0.5663843581268919,
  "precision": 0.8212967104884493,
  "recall": 0.7190085281720713,
  "per_class": {
    "person": {
      "AP50": 0.8521893115250756,
      "AP50_95": 0.5691591296166841
    },
    "table": {
      "AP50": 0.7893091647858448,
      "AP50_95": 0.5611063820856299
    },
    "chair": {
      "AP50": 0.6930192107424009,
      "AP50_95": 0.46649140660133376
    },
    "laptop": {
      "AP50": 0.8072452309203936,
      "AP50_95": 0.5436585826740437
    },
    "cell phone": {
      "AP50": 0.7463839615129719,
      "AP50_95": 0.4434662771572273
    },
    "backpack": {
      "AP50": 0.8518757426832881,
      "AP50_95": 0.6999006160440948
    },
    "book": {
      "AP50": 0.7743440369624752,
      "AP50_95": 0.4915068699160674
    },
    "bottle": {
      "AP50": 0.8643582868978146,
      "AP50_95": 0.7557856009200549
    }
  },
  "pipeline": "R07_three_views_nms_0.4",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.4,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R07 three-view NMS IoU0.5 — full replay hoàn tất

```json
{
  "mAP50": 0.8023597697118917,
  "mAP50_95": 0.5704383713179386,
  "precision": 0.831802636521848,
  "recall": 0.7149437684992677,
  "per_class": {
    "person": {
      "AP50": 0.8569682389308231,
      "AP50_95": 0.573677943539509
    },
    "table": {
      "AP50": 0.7992857679823149,
      "AP50_95": 0.5662208276727216
    },
    "chair": {
      "AP50": 0.7036105085874644,
      "AP50_95": 0.4758239415378015
    },
    "laptop": {
      "AP50": 0.8093655582952434,
      "AP50_95": 0.5449856765433452
    },
    "cell phone": {
      "AP50": 0.7489932048486692,
      "AP50_95": 0.44592046223644155
    },
    "backpack": {
      "AP50": 0.8548116576221855,
      "AP50_95": 0.7029373190558801
    },
    "book": {
      "AP50": 0.7802962338430284,
      "AP50_95": 0.4959462040397642
    },
    "bottle": {
      "AP50": 0.8655469875854049,
      "AP50_95": 0.7579945959180459
    }
  },
  "pipeline": "R07_three_views_nms_0.5",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.5,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R07 three-view NMS IoU0.6 — full replay hoàn tất

```json
{
  "mAP50": 0.8017407825336353,
  "mAP50_95": 0.5730767976702114,
  "precision": 0.8284566429903828,
  "recall": 0.7116710749043234,
  "per_class": {
    "person": {
      "AP50": 0.8577317718644061,
      "AP50_95": 0.5772592620498125
    },
    "table": {
      "AP50": 0.7948183727150674,
      "AP50_95": 0.5670884293475085
    },
    "chair": {
      "AP50": 0.7049174771712111,
      "AP50_95": 0.48125136450360434
    },
    "laptop": {
      "AP50": 0.80949113089093,
      "AP50_95": 0.5462547467874592
    },
    "cell phone": {
      "AP50": 0.7489860228041731,
      "AP50_95": 0.44738582434768864
    },
    "backpack": {
      "AP50": 0.8530688810686449,
      "AP50_95": 0.7081100950313959
    },
    "book": {
      "AP50": 0.7790243491984411,
      "AP50_95": 0.49853469695082336
    },
    "bottle": {
      "AP50": 0.8658882545562092,
      "AP50_95": 0.7587299623433976
    }
  },
  "pipeline": "R07_three_views_nms_0.6",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.6,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R07 three-view NMS IoU0.7 — full replay hoàn tất

```json
{
  "mAP50": 0.7972441652981391,
  "mAP50_95": 0.5747672777734157,
  "precision": 0.8206969463290734,
  "recall": 0.7085714889512095,
  "per_class": {
    "person": {
      "AP50": 0.8527607209028882,
      "AP50_95": 0.5793297418498058
    },
    "table": {
      "AP50": 0.7913933726131901,
      "AP50_95": 0.5704018424283569
    },
    "chair": {
      "AP50": 0.7008222080167539,
      "AP50_95": 0.4840379740300598
    },
    "laptop": {
      "AP50": 0.8036897143850672,
      "AP50_95": 0.5458410345052107
    },
    "cell phone": {
      "AP50": 0.7404744712011635,
      "AP50_95": 0.44847598595200455
    },
    "backpack": {
      "AP50": 0.8526730527605295,
      "AP50_95": 0.7102023526372755
    },
    "book": {
      "AP50": 0.7725033368002483,
      "AP50_95": 0.4994839828654481
    },
    "bottle": {
      "AP50": 0.8636364457052725,
      "AP50_95": 0.7603653079191642
    }
  },
  "pipeline": "R07_three_views_nms_0.7",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.7,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R07 three-view NMS IoU0.8 — full replay hoàn tất

```json
{
  "mAP50": 0.7831568353968072,
  "mAP50_95": 0.5716124191765812,
  "precision": 0.797876648518562,
  "recall": 0.7015524247944787,
  "per_class": {
    "person": {
      "AP50": 0.8391292335533532,
      "AP50_95": 0.5768906109263469
    },
    "table": {
      "AP50": 0.7768276234203482,
      "AP50_95": 0.5632572789751034
    },
    "chair": {
      "AP50": 0.6892781086383507,
      "AP50_95": 0.4821796927768694
    },
    "laptop": {
      "AP50": 0.7885311672333482,
      "AP50_95": 0.5409737833767154
    },
    "cell phone": {
      "AP50": 0.717046656656227,
      "AP50_95": 0.44384209618638126
    },
    "backpack": {
      "AP50": 0.8451202562469845,
      "AP50_95": 0.7106716229163298
    },
    "book": {
      "AP50": 0.7522029326500492,
      "AP50_95": 0.4938507203274802
    },
    "bottle": {
      "AP50": 0.8571187047757962,
      "AP50_95": 0.761233547927423
    }
  },
  "pipeline": "R07_three_views_nms_0.8",
  "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.8,
  "postprocessing": "explicit_views_class_aware_nms",
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "timing_note": "AP replay only; no camera latency/FPS measurement."
}
```


### R07 TTA × NMS sweep — kết thúc

```json
{
  "status": "complete",
  "results_count": 6,
  "winner": {
    "mAP50": 0.8023597697118917,
    "mAP50_95": 0.5704383713179386,
    "precision": 0.831802636521848,
    "recall": 0.7149437684992677,
    "per_class": {
      "person": {
        "AP50": 0.8569682389308231,
        "AP50_95": 0.573677943539509
      },
      "table": {
        "AP50": 0.7992857679823149,
        "AP50_95": 0.5662208276727216
      },
      "chair": {
        "AP50": 0.7036105085874644,
        "AP50_95": 0.4758239415378015
      },
      "laptop": {
        "AP50": 0.8093655582952434,
        "AP50_95": 0.5449856765433452
      },
      "cell phone": {
        "AP50": 0.7489932048486692,
        "AP50_95": 0.44592046223644155
      },
      "backpack": {
        "AP50": 0.8548116576221855,
        "AP50_95": 0.7029373190558801
      },
      "book": {
        "AP50": 0.7802962338430284,
        "AP50_95": 0.4959462040397642
      },
      "bottle": {
        "AP50": 0.8655469875854049,
        "AP50_95": 0.7579945959180459
      }
    },
    "pipeline": "R07_three_views_nms_0.5",
    "checkpoint_sha256": "ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785",
    "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
    "protocol": {
      "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R07_views_20261009_202641\\validation_data.yaml",
      "split": "val",
      "imgsz": 1280,
      "batch": 8,
      "device": "cpu",
      "workers": 0,
      "conf": 0.001,
      "iou": 0.7,
      "max_det": 300,
      "rect": true,
      "quantize": null,
      "nms": false,
      "augment": false,
      "plots": false
    },
    "views": [
      [
        1.0,
        false
      ],
      [
        0.83,
        true
      ],
      [
        0.67,
        false
      ]
    ],
    "extra_nms_iou": 0.5,
    "postprocessing": "explicit_views_class_aware_nms",
    "validation_images": 2500,
    "validation_boxes": 8359,
    "test_used": false,
    "timing_note": "AP replay only; no camera latency/FPS measurement.",
    "kept_as_best": false
  },
  "checkpoint_file_count": 1,
  "raw_cache_persisted": false
}
```


### R10 mixup/cutmix — full GPU validation hoàn tất

```json
{
  "status": "complete",
  "checkpoint": "/kaggle/working/BTL-AI-Camera-v05-campaign-20261010/batch_20261010_001917/train/R10_mixup_cutmix/weights/epoch2.pt",
  "checkpoint_sha256": "6a1825e6ec818aedb1c3ac572d2daabce961a49942a6c2db44e206599498253a",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "/tmp/project8_v05_15k_runtime/training_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": 0,
    "conf": 0.001,
    "iou": 0.7,
    "nms": false,
    "max_det": 300,
    "augment": false,
    "rect": true,
    "quantize": 16,
    "workers": 2,
    "plots": true
  },
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "native": {
    "mAP50": 0.749754454572953,
    "mAP50_95": 0.529112430341649,
    "precision": 0.7743708554602097,
    "recall": 0.682587906034699,
    "speed": {
      "preprocess": 3.0921876460015483,
      "inference": 8.015502877599646,
      "loss": 0.0015736007997475098,
      "postprocess": 0.14847942799897282
    },
    "per_class": {
      "person": {
        "AP50": 0.8343837886690708,
        "AP50_95": 0.5672038838772148
      },
      "table": {
        "AP50": 0.7532877692982411,
        "AP50_95": 0.5365598843562569
      },
      "chair": {
        "AP50": 0.6453689429391202,
        "AP50_95": 0.4282332422757797
      },
      "laptop": {
        "AP50": 0.7026568434206241,
        "AP50_95": 0.457343420576038
      },
      "cell phone": {
        "AP50": 0.6755818836319873,
        "AP50_95": 0.4039975756387652
      },
      "backpack": {
        "AP50": 0.8210283797599397,
        "AP50_95": 0.6348227638441188
      },
      "book": {
        "AP50": 0.7135132573350751,
        "AP50_95": 0.4574562285127901
      },
      "bottle": {
        "AP50": 0.8522147715295657,
        "AP50_95": 0.7472824436522286
      }
    }
  },
  "extra_nms_06": {
    "mAP50": 0.7716884274724101,
    "mAP50_95": 0.5325765759260295,
    "precision": 0.7928824567913855,
    "recall": 0.7017192343695685,
    "speed": {
      "preprocess": 2.786009751205711,
      "inference": 7.913351275199966,
      "loss": 0.0011064131984312553,
      "postprocess": 11.343980159602507
    },
    "per_class": {
      "person": {
        "AP50": 0.8456512076355778,
        "AP50_95": 0.5659832803373652
      },
      "table": {
        "AP50": 0.7751734316433787,
        "AP50_95": 0.5383077383949197
      },
      "chair": {
        "AP50": 0.6688338322048494,
        "AP50_95": 0.43151830812050757
      },
      "laptop": {
        "AP50": 0.7181325656455075,
        "AP50_95": 0.46181097503989593
      },
      "cell phone": {
        "AP50": 0.7140497155537452,
        "AP50_95": 0.4098776959527163
      },
      "backpack": {
        "AP50": 0.8391709366032416,
        "AP50_95": 0.6366781306526476
      },
      "book": {
        "AP50": 0.7483534199783688,
        "AP50_95": 0.46764862231420484
      },
      "bottle": {
        "AP50": 0.8641423105146124,
        "AP50_95": 0.7487878565959789
      }
    }
  },
  "extra_nms_iou": 0.6,
  "inferior_checkpoint_downloaded": false,
  "evidence": "Visible CAMPAIGN_METRIC records read with computer use; R11 epoch1 started."
}
```

R10 không vượt R07 trong protocol GPU chung, không thay best CPU R04. Kaggle đã bắt đầu R11_box_cls_loss epoch1/3; R12 vẫn chờ. Không tải checkpoint R10 kém hơn về máy.


### Best one-file pipeline — real-image serialization parity

```json
{
  "status": "PASS",
  "scope": "Three real validation images; serialization parity only, not full AP",
  "source_sha256": "d005e7e8b32dcb667ff68214b97733689392bbf91d127add544ca394f771b9b8",
  "helper_sha256": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "image_ids": [
    "base__12f78045aa495f8c",
    "base__aff5be491a975045",
    "base__0ed87ce450768a9e"
  ],
  "boxes": 14,
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.5,
  "test_used": false,
  "artifact_sha256": "e0200760df78c6719d700115f4077370882503e10836b5798237c370f4cf07f9",
  "artifact_bytes": 10848875,
  "control": {
    "mAP50": 0.4851169371608179,
    "mAP50_95": 0.23762359175122388,
    "precision": 0.7306218681513749,
    "recall": 0.525,
    "ground_truth_boxes": 14
  },
  "reloaded": {
    "mAP50": 0.4851169371608179,
    "mAP50_95": 0.23762359175122388,
    "precision": 0.7306218681513749,
    "recall": 0.525,
    "ground_truth_boxes": 14
  },
  "temporary_artifact_removed": true
}
```


Real-image serialization probe cleanup: {"generated_smoke_label_cache_removed": true, "labels": 3, "annotation_files_changed": false}


### Best one-file pipeline — full reload validation bắt đầu

```json
{
  "status": "running",
  "scope": "Full 2500-image validation through serialized pipeline and default YOLO validator",
  "source_sha256": "7ea7361ae586bab3d4595a1e87309bc46c43322f3d6f4d5bea4cad1e4fd7fc4c",
  "helper_sha256": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
  "component_checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "validation_images": 2500,
  "validation_boxes": 8359,
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.5,
  "test_used": false,
  "reference_pipeline": "R04_three_views_nms_0.5",
  "published_model_replaced": false,
  "artifact_sha256": "e0200760df78c6719d700115f4077370882503e10836b5798237c370f4cf07f9",
  "artifact_bytes": 10848875
}
```


### GPU follow-up cells — rebuilt without deleted archive dependencies

```json
{
  "inference_followup": {
    "execution_status": "prepared_not_GPU_executed",
    "compilation": "PASS",
    "compression_roundtrip": "PASS",
    "sources": {
      "kaggle_tools/experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
      "kaggle_tools/inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
      "notebooks/kaggle/project8_v05_inference_followup_20261010.py": "adfb71f09b1bbd58aa55c5f06a34a3dc4a007c06109adfc4d1d1ae798abb6c02"
    },
    "embedded_source_sha256": "7aaa569a7eda47ffea4886080ee48a619ac41c3e59857f33baacd5946f090472",
    "embedded_source_bytes": 17125,
    "cell": "notebooks/kaggle/project8_v05_inference_followup_cell_20261010.py",
    "cell_sha256": "669b04e9a9d8a6e91459ef8f6274c194b523390173f6149bc7c927f291f27129",
    "notebook": "notebooks/kaggle/project8_v05_inference_followup_20261010.ipynb",
    "notebook_sha256": "9f06ed0989737cc1b491a3e85323c9272a30b05e9d955765de2dcb4580bb119a"
  },
  "nms_sweep": {
    "execution_status": "prepared_not_GPU_executed",
    "compilation": "PASS",
    "compression_roundtrip": "PASS",
    "sources": {
      "kaggle_tools/experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
      "kaggle_tools/inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
      "kaggle_tools/validation_replay.py": "b75e2237f159b1f03afab295012b93f08e799bdcd859de3c76f19c886b2d7a60",
      "notebooks/kaggle/project8_v05_nms_sweep_20261010.py": "b621f7025a7436a2f1c1e101a985c98b2ec4a8de963e20637cb0c8cf985aca2f"
    },
    "embedded_source_sha256": "b6a7d3fa0e396dc830d38f30ab32e6f2ae04fe7084300ed22b4d1fdb9519a330",
    "embedded_source_bytes": 19923,
    "cell": "notebooks/kaggle/project8_v05_nms_sweep_cell_20261010.py",
    "cell_sha256": "8103a330b157c1cc68155ed07e34e7380343ad6998211101c9f946191aac25d0",
    "notebook": "notebooks/kaggle/project8_v05_nms_sweep_20261010.ipynb",
    "notebook_sha256": "eb6c8f5a5223d4bc3114fb6b424f2254296568ab3dcd6d68defa22503ff187e4"
  },
  "source_ablations_train": {
    "execution_status": "prepared_not_GPU_executed",
    "compilation": "PASS",
    "compression_roundtrip": "PASS",
    "sources": {
      "kaggle_tools/experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
      "kaggle_tools/inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
      "kaggle_tools/training_subsets.py": "07250bf66aab73b3c347fbcbc33e48f4b988f5443bf101d8c547768387b8cb50",
      "notebooks/kaggle/project8_v05_prepare_source_ablations_20261010.py": "9d61c90d1480af145125af289d78cdf21b2a56dc5bc743ccff400e2537c40b1e",
      "notebooks/kaggle/project8_v05_source_ablations_train_20261010.py": "fc7236c072e0b7757cda8f9b024919f481e61ddc4c0d571d11114348cb3bb1e7"
    },
    "embedded_source_sha256": "af4dbec2f55c781a680edb3fe891de84440d88c5e7b805b6ba73017170e279f2",
    "embedded_source_bytes": 26327,
    "cell": "notebooks/kaggle/project8_v05_source_ablations_train_cell_20261010.py",
    "cell_sha256": "c9409997898c50d1670f37955fa209b7aa669500910b3bd2a7c039c8d201a24f",
    "notebook": "notebooks/kaggle/project8_v05_source_ablations_train_20261010.ipynb",
    "notebook_sha256": "42477609fe1f2562473f4f8d6e124b5100d6dcfc4afb99695d043186081e73f0"
  }
}
```


### Rebuilt GPU follow-ups — definitions and packaging tests

```json
{
  "packed_definitions": {
    "inference_followup": {
      "status": "PASS",
      "embedded_source_sha256": "7aaa569a7eda47ffea4886080ee48a619ac41c3e59857f33baacd5946f090472",
      "definitions_executed": true,
      "requires_local_kaggle_tools": false,
      "GPU_jobs_submitted": false,
      "validation_AP_claim": false
    },
    "nms_sweep": {
      "status": "PASS",
      "embedded_source_sha256": "b6a7d3fa0e396dc830d38f30ab32e6f2ae04fe7084300ed22b4d1fdb9519a330",
      "definitions_executed": true,
      "requires_local_kaggle_tools": false,
      "GPU_jobs_submitted": false,
      "validation_AP_claim": false
    },
    "source_ablations_train": {
      "status": "PASS",
      "embedded_source_sha256": "af4dbec2f55c781a680edb3fe891de84440d88c5e7b805b6ba73017170e279f2",
      "definitions_executed": true,
      "requires_local_kaggle_tools": false,
      "GPU_jobs_submitted": false,
      "validation_AP_claim": false
    }
  },
  "packaging_tests": {
    "count": 2,
    "status": "PASS",
    "seconds": 0.155,
    "red": "ModuleNotFoundError before builder existed",
    "intermediate": "Default Windows sandbox temp permission error; rerun with approved workspace temp",
    "coverage": [
      "Current source decoding/hash and notebook parity without deleted logs/archives",
      "Malformed source rejects all writes before packaging"
    ]
  },
  "builder_sha256": "b4458d83de6a0424f74622e6bde80cf41c9320768a88a4ba1920c6d1a3707029",
  "tests_sha256": "5e8db5d5871876328c94d2fc3243491d293deee3a9716125b92e980879b83d44"
}
```


### R11 box/cls loss — full GPU validation hoàn tất

```json
{
  "status": "complete",
  "checkpoint": "/kaggle/working/BTL-AI-Camera-v05-campaign-20261010/batch_20261010_001917/train/R11_box_cls_loss/weights/epoch2.pt",
  "checkpoint_sha256": "c6cb3d1d58c00b5b0076ea392e127631c22c6400c32b94b77552c4998e3a83e3",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "/tmp/project8_v05_15k_runtime/training_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": 0,
    "conf": 0.001,
    "iou": 0.7,
    "nms": false,
    "max_det": 300,
    "augment": false,
    "rect": true,
    "quantize": 16,
    "workers": 2,
    "plots": true
  },
  "validation_images": 2500,
  "validation_boxes": 8359,
  "test_used": false,
  "native": {
    "mAP50": 0.7581626721183428,
    "mAP50_95": 0.5384493153152468,
    "precision": 0.7981257317005045,
    "recall": 0.6799578262539434,
    "speed": {
      "preprocess": 3.080282981594064,
      "inference": 7.972913726805563,
      "loss": 0.0012780336051946507,
      "postprocess": 0.1606225375981012
    },
    "per_class": {
      "person": {
        "AP50": 0.8419768406339656,
        "AP50_95": 0.5682759986871104
      },
      "table": {
        "AP50": 0.7597378335168876,
        "AP50_95": 0.548469708491523
      },
      "chair": {
        "AP50": 0.6566358388572904,
        "AP50_95": 0.4421962479584283
      },
      "laptop": {
        "AP50": 0.7209392952151814,
        "AP50_95": 0.46498836788545467
      },
      "cell phone": {
        "AP50": 0.7015945505257777,
        "AP50_95": 0.42117434917807894
      },
      "backpack": {
        "AP50": 0.8181188051598569,
        "AP50_95": 0.6372722891077427
      },
      "book": {
        "AP50": 0.7210465337686767,
        "AP50_95": 0.4765559989201389
      },
      "bottle": {
        "AP50": 0.8452516792691058,
        "AP50_95": 0.7486615622934977
      }
    }
  },
  "extra_nms_06": {
    "mAP50": 0.7788717620144407,
    "mAP50_95": 0.5420679578522519,
    "precision": 0.7956225962810733,
    "recall": 0.7113781023836554,
    "speed": {
      "preprocess": 2.7601565111988746,
      "inference": 7.93584323719915,
      "loss": 0.0011740440000721718,
      "postprocess": 9.863377662390848
    },
    "per_class": {
      "person": {
        "AP50": 0.856939442982873,
        "AP50_95": 0.5681085214739828
      },
      "table": {
        "AP50": 0.7823296491120146,
        "AP50_95": 0.5512738278442039
      },
      "chair": {
        "AP50": 0.6742886785134826,
        "AP50_95": 0.44282847509594064
      },
      "laptop": {
        "AP50": 0.7366924068979135,
        "AP50_95": 0.47032077500472685
      },
      "cell phone": {
        "AP50": 0.7356101962912436,
        "AP50_95": 0.4271603259137847
      },
      "backpack": {
        "AP50": 0.835745569513083,
        "AP50_95": 0.6405918975419576
      },
      "book": {
        "AP50": 0.750309901751155,
        "AP50_95": 0.4850839550583528
      },
      "bottle": {
        "AP50": 0.8590582510537612,
        "AP50_95": 0.7511758848850663
      }
    }
  },
  "extra_nms_iou": 0.6,
  "inferior_checkpoint_downloaded": false,
  "evidence": "Visible CAMPAIGN_METRIC records read with computer use; R12 epoch1 started."
}
```

R11 không vượt R07 trong protocol GPU chung và không thay best CPU R04. R12_yolo26s đã khởi chạy10epoch (1500step/epoch với batch8), chưa có AP đầy đủ. Không tải checkpoint R11 kém hơn về máy.


### Best one-file pipeline — full reload validation kết thúc

```json
{
  "status": "PASS",
  "scope": "Full 2500-image validation through serialized pipeline and default YOLO validator",
  "source_sha256": "7ea7361ae586bab3d4595a1e87309bc46c43322f3d6f4d5bea4cad1e4fd7fc4c",
  "helper_sha256": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
  "component_checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "validation_images": 2500,
  "validation_boxes": 8359,
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.5,
  "test_used": false,
  "reference_pipeline": "R04_three_views_nms_0.5",
  "published_model_replaced": false,
  "artifact_sha256": "e0200760df78c6719d700115f4077370882503e10836b5798237c370f4cf07f9",
  "artifact_bytes": 10848875,
  "measured": {
    "mAP50": 0.8050521265694426,
    "mAP50_95": 0.5656963669995626,
    "precision": 0.8297354359894933,
    "recall": 0.721293281303833,
    "speed": {
      "preprocess": 2.9357026000056066,
      "inference": 339.28300895998694,
      "loss": 0.0002499600028386339,
      "postprocess": 0.056004920006671455
    },
    "per_class": {
      "person": {
        "AP50": 0.8640113786810067,
        "AP50_95": 0.5744499650057329
      },
      "table": {
        "AP50": 0.8041683119998588,
        "AP50_95": 0.5692793461060456
      },
      "chair": {
        "AP50": 0.7026889774797049,
        "AP50_95": 0.4690993507605502
      },
      "laptop": {
        "AP50": 0.8113582632785037,
        "AP50_95": 0.5433629357526237
      },
      "cell phone": {
        "AP50": 0.7458552926594973,
        "AP50_95": 0.43502508454379274
      },
      "backpack": {
        "AP50": 0.8599994010428751,
        "AP50_95": 0.6985161673784781
      },
      "book": {
        "AP50": 0.7864815854726537,
        "AP50_95": 0.4943149930652407
      },
      "bottle": {
        "AP50": 0.8658538019414398,
        "AP50_95": 0.7415230933840367
      }
    }
  },
  "parity_tolerance": 1e-06,
  "temporary_artifact_removed": true
}
```


### Multimodel raw-view cache — real-checkpoint integration

```json
{
  "status": "PASS",
  "scope": "Two independent instances of the same real R04 checkpoint; raw capture integration only, no ensemble AP gain claim",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "images": 3,
  "boxes": 14,
  "views": [
    [
      1.0,
      false
    ],
    [
      0.83,
      true
    ],
    [
      0.67,
      false
    ]
  ],
  "extra_nms_iou": 0.6,
  "test_used": false,
  "source_sha256": "a67b99a054215653ca8353fbf7e852ad3d9023b99252fe03f9fed793f7bef4ef",
  "collector_sha256": "e705d11b55083968f282c299a147d55d873436ee87136b9a98eb0a520dfcaec3",
  "tests": {
    "count": 7,
    "status": "PASS",
    "seconds": 8.437,
    "red": "Three new multimodel cases failed against the single-checkpoint collector"
  },
  "captured": {
    "mAP50": 0.47697240875442765,
    "mAP50_95": 0.25319300838632997,
    "precision": 0.7143476203831737,
    "recall": 0.525
  },
  "control": {
    "mAP50": 0.47697240875442765,
    "mAP50_95": 0.25319300838632997,
    "precision": 0.7143476203831737,
    "recall": 0.525
  },
  "replayed": {
    "mAP50": 0.47697240875442765,
    "mAP50_95": 0.25319300838632997,
    "precision": 0.7143476203831737,
    "recall": 0.525
  },
  "raw_rows": 5400,
  "parity_tolerance": 1e-06
}
```


Smoke cleanup: {"scope": "multimodel raw integration", "generated_smoke_cache_removed": true, "labels": 3, "annotation_files_changed": false}


### GPU follow-up cells — rebuilt without deleted archive dependencies

```json
{
  "inference_followup": {
    "execution_status": "prepared_not_GPU_executed",
    "compilation": "PASS",
    "compression_roundtrip": "PASS",
    "sources": {
      "kaggle_tools/experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
      "kaggle_tools/inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
      "kaggle_tools/validation_replay.py": "b75e2237f159b1f03afab295012b93f08e799bdcd859de3c76f19c886b2d7a60",
      "kaggle_tools/view_replay.py": "e705d11b55083968f282c299a147d55d873436ee87136b9a98eb0a520dfcaec3",
      "notebooks/kaggle/project8_v05_inference_followup_20261010.py": "fff1fef91e815e059a2f8df8a640925edc95cf627352637a99d5988d818a6389"
    },
    "embedded_source_sha256": "f94053b6fb6d85ca93aba7ffec34048a4e071848f2dcbaecaeca737c986b8f47",
    "embedded_source_bytes": 30773,
    "cell": "notebooks/kaggle/project8_v05_inference_followup_cell_20261010.py",
    "cell_sha256": "cfbb8f8a8a049ee01f912f6ec08b062cd3ca070b5503353f234f687eb989f5ad",
    "notebook": "notebooks/kaggle/project8_v05_inference_followup_20261010.ipynb",
    "notebook_sha256": "2e62b87e4c19c3a3ebc88f5430e689434cc38a11e4ff4a0538c4ebdb2b1db18d"
  },
  "nms_sweep": {
    "execution_status": "prepared_not_GPU_executed",
    "compilation": "PASS",
    "compression_roundtrip": "PASS",
    "sources": {
      "kaggle_tools/experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
      "kaggle_tools/inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
      "kaggle_tools/validation_replay.py": "b75e2237f159b1f03afab295012b93f08e799bdcd859de3c76f19c886b2d7a60",
      "notebooks/kaggle/project8_v05_nms_sweep_20261010.py": "b621f7025a7436a2f1c1e101a985c98b2ec4a8de963e20637cb0c8cf985aca2f"
    },
    "embedded_source_sha256": "b6a7d3fa0e396dc830d38f30ab32e6f2ae04fe7084300ed22b4d1fdb9519a330",
    "embedded_source_bytes": 19923,
    "cell": "notebooks/kaggle/project8_v05_nms_sweep_cell_20261010.py",
    "cell_sha256": "8103a330b157c1cc68155ed07e34e7380343ad6998211101c9f946191aac25d0",
    "notebook": "notebooks/kaggle/project8_v05_nms_sweep_20261010.ipynb",
    "notebook_sha256": "eb6c8f5a5223d4bc3114fb6b424f2254296568ab3dcd6d68defa22503ff187e4"
  },
  "source_ablations_train": {
    "execution_status": "prepared_not_GPU_executed",
    "compilation": "PASS",
    "compression_roundtrip": "PASS",
    "sources": {
      "kaggle_tools/experiment_campaign.py": "591a66f145b2aa652f9b0a06d4167d076286edc2a8e037c19cdd6054d9fca2b4",
      "kaggle_tools/inference_variants.py": "096da2364e2d3236bc008047efa06f7cb14aa415467d38bd150e3a042a771781",
      "kaggle_tools/training_subsets.py": "07250bf66aab73b3c347fbcbc33e48f4b988f5443bf101d8c547768387b8cb50",
      "notebooks/kaggle/project8_v05_prepare_source_ablations_20261010.py": "9d61c90d1480af145125af289d78cdf21b2a56dc5bc743ccff400e2537c40b1e",
      "notebooks/kaggle/project8_v05_source_ablations_train_20261010.py": "fc7236c072e0b7757cda8f9b024919f481e61ddc4c0d571d11114348cb3bb1e7"
    },
    "embedded_source_sha256": "af4dbec2f55c781a680edb3fe891de84440d88c5e7b805b6ba73017170e279f2",
    "embedded_source_bytes": 26327,
    "cell": "notebooks/kaggle/project8_v05_source_ablations_train_cell_20261010.py",
    "cell_sha256": "c9409997898c50d1670f37955fa209b7aa669500910b3bd2a7c039c8d201a24f",
    "notebook": "notebooks/kaggle/project8_v05_source_ablations_train_20261010.ipynb",
    "notebook_sha256": "42477609fe1f2562473f4f8d6e124b5100d6dcfc4afb99695d043186081e73f0"
  }
}
```


### Rebuilt GPU follow-ups — definitions and packaging tests

```json
{
  "packed_definitions": {
    "inference_followup": {
      "status": "PASS",
      "embedded_source_sha256": "f94053b6fb6d85ca93aba7ffec34048a4e071848f2dcbaecaeca737c986b8f47",
      "definitions_executed": true,
      "requires_local_kaggle_tools": false,
      "GPU_jobs_submitted": false,
      "validation_AP_claim": false
    },
    "nms_sweep": {
      "status": "PASS",
      "embedded_source_sha256": "b6a7d3fa0e396dc830d38f30ab32e6f2ae04fe7084300ed22b4d1fdb9519a330",
      "definitions_executed": true,
      "requires_local_kaggle_tools": false,
      "GPU_jobs_submitted": false,
      "validation_AP_claim": false
    },
    "source_ablations_train": {
      "status": "PASS",
      "embedded_source_sha256": "af4dbec2f55c781a680edb3fe891de84440d88c5e7b805b6ba73017170e279f2",
      "definitions_executed": true,
      "requires_local_kaggle_tools": false,
      "GPU_jobs_submitted": false,
      "validation_AP_claim": false
    }
  },
  "packaging_tests": {
    "count": 2,
    "status": "PASS",
    "seconds": 0.155,
    "red": "ModuleNotFoundError before builder existed",
    "intermediate": "Default Windows sandbox temp permission error; rerun with approved workspace temp",
    "coverage": [
      "Current source decoding/hash and notebook parity without deleted logs/archives",
      "Malformed source rejects all writes before packaging"
    ]
  },
  "builder_sha256": "9cb133b33fc54fdd2b412f9d1ce994d9fc899633757e214425375054cc8f74b7",
  "tests_sha256": "5e8db5d5871876328c94d2fc3243491d293deee3a9716125b92e980879b83d44"
}
```


### Multimodel TTA/NMS follow-up — current combined tests

```json
{
  "status": "PASS",
  "count": 11,
  "seconds": 8.619,
  "modules": [
    "tests.test_view_replay",
    "tests.test_inference_followup",
    "tests.test_followup_packaging"
  ],
  "note": "The preceding packed-definitions record retained historical packaging-only timing 0.155s; the current combined test run is 11 tests in 8.619s.",
  "source_sha256": {
    "kaggle_tools\\view_replay.py": "e705d11b55083968f282c299a147d55d873436ee87136b9a98eb0a520dfcaec3",
    "notebooks\\kaggle\\project8_v05_inference_followup_20261010.py": "fff1fef91e815e059a2f8df8a640925edc95cf627352637a99d5988d818a6389",
    "notebooks\\kaggle\\build_project8_followups.py": "9cb133b33fc54fdd2b412f9d1ce994d9fc899633757e214425375054cc8f74b7",
    "tests\\test_view_replay.py": "77009e66d8c4a609780fcd339b43fb3566d616c122a0935b7c54a2fbe3c557d1",
    "tests\\test_inference_followup.py": "0cd8a03ad84d762b9f70f22aa8908828636b77d485b0d9a64188e90e4daf1932"
  }
}
```


### Identity clipping — full validation bắt đầu

```json
{
  "status": "running",
  "scope": "Full val identity clipping audit; CPU FP32, no test",
  "source_sha256": "462743460c354570355399a4420dce61b8966a62c3e9d973eca12d041bb49b6e",
  "checkpoint_sha256": "087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758",
  "manifest_sha256": "22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd",
  "protocol": {
    "data": "C:\\Users\\aawm8\\Downloads\\BTL-AI-Camera-cd8\\.temp\\R04_views_20261009_191354\\validation_data.yaml",
    "split": "val",
    "imgsz": 1280,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": true,
    "quantize": null,
    "nms": false,
    "augment": false,
    "plots": false
  },
  "validation_images": 2500,
  "validation_boxes": 8359,
  "published_model_replaced": false
}
```

### Kaggle R12 — kiểm tra tài nguyên khi output chậm cập nhật

Quan sát qua computer use: Draft Session 5h16m/12h, CPU 297%, RAM 16.9/30 GiB; hai GPU T4 đang dùng 83% và 85%, bộ nhớ 5.9 và 7/15 GiB. Quota hiển thị 19:25/30 giờ. Đây là bằng chứng tài nguyên còn hoạt động, chưa phải bằng chứng epoch hoặc metric R12 mới. Không huỷ, chạy lại hay Run All. Show Versions vẫn là 8, chưa xác nhận Quick Save mới thành công.

Đã bắt đầu kiểm tra full 2500 validation /8359 GT về clipping tọa độ identity view trước NMS .6 trên CPU FP32. Source: notebooks/kaggle/project8_v05_identity_clip_probe_20261010.py; SHA256 462743460c354570355399a4420dce61b8966a62c3e9d973eca12d041bb49b6e. Session local 9733 còn chạy; chưa có kết quả cuối. Không tạo model PT khác, không dùng test, không thay model đã chọn. Kết quả cuối sẽ được script ghi trực tiếp vào chính MD này.

### RF-DETR — kiểm tra khả thi từ tài liệu chính thức, chưa train

Đã đọc ngày 2026-10-10: https://rfdetr.roboflow.com/learn/train/ và https://github.com/roboflow/rf-detr/blob/develop/docs/learn/train/dataset-formats.md cùng training-parameters.md. Hướng dẫn công khai nêu fine-tune pretrained COCO và cấu hình T4 batch 4, grad_accum_steps 4; đây là gợi ý tài liệu, chưa đo trên dữ liệu Project8 và chưa chứng minh mức mAP nào.

Có khác biệt phiên bản cần giải quyết: trang hướng dẫn công khai mô tả COCO train/valid/test; tài liệu develop hỗ trợ COCO và YOLO tự phát hiện, COCO ưu tiên nếu cùng tồn tại. Không suy ra API develop giống bản PyPI đang cài. Chưa cài rfdetr, chưa đổi Torch/CUDA của phiên Kaggle đang train. Cần pin release/commit cụ thể, kiểm tra source về loader/class mapping/test hooks và chạy smoke trước full train. Không tự tạo test giả hoặc dùng 500 test trong quá trình chọn model. RF-DETR giữ trong phạm vi các hướng thử còn lại; chưa có kết quả training hay inference và chưa được so sánh với best YOLO.

### Chốt báo cáo và dừng theo yêu cầu người dùng — 2026-10-10

Người dùng yêu cầu báo cáo đến đây và dừng. Đã dừng công việc, không khởi chạy thử nghiệm mới. Session local identity clipping 9733 đã terminal exit 1; output cuối quan sát được 148/313 batch, chưa hoàn thành full validation và không có metric cuối hợp lệ. Không coi đây là PASS hay bằng chứng clipping tương đương. Yêu cầu tải wheel RF-DETR ngoài sandbox bị người dùng từ chối; không tải lại bằng cách khác và chưa cài RF-DETR.

Best hiện tại giữ nguyên: R04 + ba views + extra NMS IoU .5, mAP50 0.8050521265694426, mAP50-95 0.5656963669995626 trên 2500 val/8359 GT CPU FP32 tại 1280. Một checkpoint campaign giữ tại weights/campaign_20261010/BEST_Project8_v05.pt; checkpoint raw không tự chứa cấu hình TTA/NMS. Chưa có kết quả 85% hoặc 95%, chưa đánh giá 500 test. Các hướng TTA/ensemble GPU, confirmation, source/group controls, distillation, RF-DETR còn chưa hoàn tất.

Phiên Kaggle không bị huỷ; quan sát cuối trước khi dừng là R12 đang thực thi, hai GPU còn hoạt động, nhưng stdout chưa cập nhật thêm. Không khẳng định trạng thái hiện tại sau thời điểm quan sát. Gọi cập nhật mục tiêu paused trả về không có goal của thread; không tạo goal mới.
Ngày 2026-10-10, theo yêu cầu kết thúc các tác vụ: đã Cancel Run rồi Run > Stop session trên Kaggle. UI xác nhận Draft Session off (run a cell to start), menu đổi thành Start session. Ảnh xác nhận: .temp/PROJECT8_KAGGLE_STOPPED.jpg. Không khởi chạy lại. Local validation 9733 đã terminal exit 1 trước đó; không có kết quả hoàn chỉnh. Handle tải wheel 13444 không còn tồn tại khi kiểm tra.
