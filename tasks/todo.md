# Checklist triển khai — BTL-AI-Camera-cd8

> **Bổ sung 12/10/2026:** các mục R0–R7 cuối file là kế hoạch mở rộng được đề xuất, **chưa triển khai**. Người dùng đã xác nhận cần tránh trùng đề tài nhận biết gian lận của nhóm độc lập. Chưa coi lựa chọn tối ưu YOLO theo tài nguyên là đã được người dùng chốt. T0–T10 và phần Kaggle 08/10 bên dưới giữ trạng thái lịch sử; chiến dịch checkpoint07 ngày 12/10 có [checklist riêng](experiments/kaggle_classroom_07_todo.md).

Ngày08/10/2026. Đi cùng [kế hoạch chi tiết](plan.md). Đã ghép v0.5 trong chat chính theo ủy quyền: **15.000 ảnh (12.000/2.500/500), đã đủ số ảnh, còn thiếu0. Mục tiêu15.000/12.000 train/2.500 val/500 test giữ nguyên. Tất cả 15.000 ảnh full_review_pending; đã phát hiệnmẫu pizza/collage cóchairID2, nên **training_ready=False**. Các dấu[x] dướiđây chỉ ghi phầnđã thựchiện, không chứngnhận full review. Lượt này cậpnhậttàiliệu theoartifact, chưa sửa nhãn hoặctrain.

## T0 — Đối chiếu và bàn giao tài liệu (M0)

- [x] Đọc report/audit/comparison/CSV/config và code liên quan; kiểm manifest, nhãn, hashes và CRC.
- [x] Lập plan tiếng Việt với số đo/giả thuyết/mục tiêu, phương án split mới và chẩn đoán lỗi.
- [x] Lưu biên bản đối chiếu; kiểm liên kết và phép tính; không thay dữ liệu/code/checkpoint/backend.
- [ ] Người dùng review các đề xuất còn mở trong plan; không hỏi lại các target đã chốt.

Phụ thuộc: không. Tệp: plan.md, todo.md, verification_20261008.json. Kiểm chứng: link tồn tại, tổng số liệu khớp artifacts; AP không được mô tả là phép chấm lại trong phiên này.

## T1 — Chốt rubric và thiết kế review queue chỉ đọc

- [ ] Tạo rubric 8 ID, cup+bottle, box/occlusion/ambiguous và negative thật; mẫu ledger có reviewer/reason/provenance/version.
- [ ] Tạo queue theo nguồn/nhóm/split/size và phương pháp sampling có seed; giữ test khỏi prediction của model phát triển.
- [ ] Phân công reviewer/hòa giải; đo được thời gian và disagreement trong pilot.

Phụ thuộc: T0 và giao triển khai. Tệp dự kiến: rubric, queue, review ledger (3 artifact dữ liệu/tài liệu, chưa sửa source). Kiểm chứng: uniqueID, lookup đúng ảnh/nhãn; sample không gọi reviewed trước khi người rà hoàn tất. Quy mô: một phiên chuẩn bị, rồi các đơn vị review nhỏ bên dưới.

## T2 — Pilot và hiệu chỉnh effort

- [ ] Chọn400 flagged ngẫu nhiên phân tầng +100 targeted train/val, không trùng; thêm 100 base approved để kiểm ghi chú pending và nhóm còn lại.
- [ ] Rà đủ 8 lớp từng ảnh; double-review 20%, hòa giải bất đồng theo rubric.
- [ ] Báo lỗi/nguồn, median/p90 phút/ảnh và tỷ lệ mẫu; phần targeted không dùng để ước lượng tỷ lệ lỗi toàn dataset.

Phụ thuộc: T1. Artifact: pilot ledger, QC report, effort estimate. Kiểm chứng: đủID, không cộng pilot hai lần vào15.000 ảnh; giới hạn khoảng 25–40 ảnh/phiên reviewer, chia nhỏ hơn với cảnh đông. Đây là mẫu thử quy trình, không thay full review.

### Cổng sau T1–T2

- [ ] Rubric và disagreement đã được xử lý; estimate cập nhật bằng số đo thật.
- [ ] Review test không tiết lộ lỗi model trên test để hướng training.

## T3 — Full review 15.000 ảnh hiện có, bao gồm34topup đãthêm

- [ ] Ưu tiên truymapping nguồn cellphone_train_3class_filtered,3 peerảnh củaissue pizza→chair; xác minh labels/YAML trước/sau remap, không tựđảo ID hàng loạt; giữexclusion ledger cho3 mẫu đãloại.
- [ ] Rà đủ 8 lớp từngảnh:12.000 ảnh base v04, 2.966 ảnh merged_3k và 34 ảnh phone topup; pilot chỉchuẩn hóaquytrình, khôngmiễn ràapproved cũ.
- [ ] Mỗi phiên25–40 ảnh ghi reviewer/reason/provenance/version; QC 10% mỗi lô vàmọi bấtđồng; full reviewpending chỉđổi sauthựcrà.

Phụ thuộc: T1/T2. Kiểm chứng: ledger 15.000/15.000 sau khi thực hiện, unresolved=0, semantic QA đạt. Topup đã hoàn tất; kiểm tra format/hash không thay thế visual review.

## T4 — Topup số lượng đã hoàn tất

- [x] Merged_3k3.000 đầu vào:31 trùng,3pizzađãloại; giữ2.966unique mới.
- [x] Đãthêm34 ảnhphoneunique vào train, đạt15.000=12.000/2.500/500; thiếu0.
- [x] Manifest/sốfile và30.000payloadhash khớp; checkpointgiữbestv04.
- [ ] Fullreviewđủ8lớp cho34topup cùngtoàn bộ15.000, provenance/QC/mapping vàsceneaudit.

PhụthuộcqualitygateT6; khôngcòn nhiệm vụthu34ảnh. Khôngđánh dấu fullreviewđãđạt bằngformat/hash.

## T5 — Kiểm lại migration đã thực hiện và holdout500

- [x] V05 giữ 500 test cũ, mọi11ảnhbottle; chuyển 206 test cũ→val; khôngv03/v04holdout→train.
- [x] Source class remap0book→6,1chair→2,2mobilephone→4; source filename/clip grouping được giữheuristically.
- [ ] Kiểm semantic mapping theo issue đã xác nhận, dù3 mẫu đãloại; full scene/near-duplicate/exposure audit chưa hoàn thành.
- [ ] Báocoverage test 500 theo8 lớp/nhóm/background; bottle 11 ảnh/38 box chưađủnghiệm thu. Khôngdùngprediction test đểsửanhãn/chọntrain/tuning.

Phụ thuộc:T3 vàhồsơ lịch sử. Artifact:split_moves.json629 chuyển, exposure/near-duplicate report, test data card. Source-test/valid ảnhmới chuyểntrain phảiđược giải thích theo provenance; khôngnhầm vớiholdout cũ v03/v04. MọiđổiGT/split vềsau tạo versionmới, khônghỏi lạiphươngánA/B cho mergeđãđược ủy quyền.

## T6 — Khóa release15.000 ảnh và quality gate

- [ ] Full review 15.000/15.000 và kiểm tự động 100%; tranh chấp giải quyết/QC đạt, source/session/provenance đầy đủ.
- [ ] Tạo release mới đúng 12.000/2.500/500 cùng migration/hash; giữ mọi release/split/report cũ; kiểm holdout không nhiễm.
- [ ] Khóa protocolA/B, coverage, annotation version, FPS/P/R/budget còn mở; test niêm phong trước chọn model.

Phụ thuộc: T3,T4,T5. Artifact: release manifest, hashes/data card, QA signoff. Kiểm chứng: checksum thật, số ảnh duy nhất=15.000, không trùng nhóm qua split; không chạy training nếu full review chưa hoàn tất.

### Cổng trước GPU

- [ ] T6 đạt. Giữ baseline lịch sử70,706%/50,694% đúng GT cũ.
- [ ] Nhận rõ budget GPU/lượt/seed; luồng training chọn đúng release v05 sau full review, không vô tình chạy exporterlegacy cup.

## T7 — Tạo baseline mới và nano corrected-data

- [ ] E0: chấm checkpoint hiện tại trênval2.500/GT mới, A/B riêng; không lấy điểm cũ so như cùng GT.
- [ ] E1: smoke riêng rồi20 epoch nano khi được giao, init/seed/config/hash/log rõ, epochs 20/close_mosaic 5 là giả thuyết.
- [ ] Báo đủ 8 lớp, P/R/AP/source/size/occlusion, PR/confusion/thư viện FN/FP; đối chiếu overfit/underfit/domainshift bằng bảng mục 4.1.

Phụ thuộc: T6. Artifact: run folder + evaluation bundle + comparison. Kiểm chứng: finite, mapping, actualhead; không tự quy gain E0→E1 cho sửa nhãn riêng vì có thêm training.

## T8 — Ablation hoặc small và độ lặp

- [ ] Chỉ chọn một thay đổi chính theo evidence; không gridsearch không giới hạn.
- [ ] Nếu cầnsmall: dùngyolo26s.pt pretrained; n-fresh/s-fresh cùng recipe để tách ảnh hưởnginit; không đổi tênnano thành small.
- [ ] Kiểmseed42/43/44 cho đối chứng/ứng viên nếu budget cho phép; báovariance/CI, dừng theo ngân sách và guardrails.

Phụ thuộc: T7. Artifact: run matrix, decision record, seed summary. Kiểm chứng: mục 9 củaplan; không gọi1 seed là kết quả ổn định, không chọn theo test.

## T9 — CPU 640/NMS-free và nghiệm thu test

- [ ] ĐoA/B đúnghead; CPU p50/p95, processed/displayedFPS, frameage/droppedframes, RAM/bytes theo protocol.
- [ ] Khóa checkpoint/threshold bằngval; sau đó mởtest500 một đợt nghiệm thu, đủ 8 class/subtypes/background/CI.
- [ ] Báo đạt/chưa đạt 85/60, sàn 75 và ngưỡng vận hành đã chốt; nếu thiếu chứng cứ thì ghi chưa đủ, không làm dễholdout.

Phụ thuộc: T8 hoặc dừng ởT7 nếu nano đủ. Artifact: benchmark, finaltest report, modelcard. Kiểm chứng: không gọimAP50–95 làAP95; phản hồi test không dùng chọn train/tuning; lưu kết quả thất bại trung thực.

## T10 — Promote và rollback khi được giao triển khai

- [ ] Candidate vượt cổng và có mapping/hash/config/manifest đúng; bản deploy tối ưu parity với gốc.
- [ ] Giữ checkpoint gốc, lưu alias/config trước chuyển; smoke đủ ảnh/video/camera vàhealth sau chuyển.
- [ ] Có rollback vềv04 hiện tại; v03 giữ để đối chứng, không nhầm phạm viID7.

Phụ thuộc: T9 đạt và được giao deploy. Artifact: deployment record, artifact deploy, rollback record. Kiểm chứng: không xóa/stripbản gốc; nếu lỗi vận hành thì rollback có log.

### Definition of Done nhánh dữ liệu/model

- [ ] Target15.000 ảnh/split/full review và hợp đồng8 ID đạt.
- [ ] Testđộc lập đủ chứng cứ, protocol/GT/hash cố định, không tuning bằng test.
- [ ] Chất lượng/lớp/camera/old-domain/latency/size theo tiêu chí đã chốt.
- [ ] Hồ sơ tái lập vàrollback đầy đủ; nếu không đạt thì bàn giao evidence cùng lựa chọn tradeoff.

## Kaggle — cập nhật 08/10/2026

- [x] Dọn ZIP trùng, staging cũ, cache Python và 5 Markdown thừa; giữ bằng chứng cleanup.
- [x] Tạo gói 14.500 train/val và best v0.4, giữ riêng 500 test; CRC/hash và syntax notebook đạt.
- [x] Người dùng cho phép smoke trước khi full review xong: 1 epoch, fraction=0.05; full run vẫn chặn.
- [x] Notebook v0.5 đã nhập vào Kaggle; gói đã upload, đang được Kaggle xử lý.
- [ ] Smoke GPU chạy xong, kiểm log/loss/checkpoint và lưu kết quả.

## Kế hoạch mở rộng R — cấu hình YOLO theo giới hạn tài nguyên

Đọc mục 0 của [plan.md](plan.md). Mọi checkbox R đều để mở: đây là đề xuất cho bước sau, không ghi nhận chức năng đã có. Phạm vi loại trừ nhận biết gian lận; GAN tùy chọn. Những đường dẫn chưa tồn tại dưới đây là **tệp dự kiến**, cần đối chiếu kiến trúc repo khi triển khai.

### R0 — Chốt mục tiêu và hợp đồng thí nghiệm

- [ ] Chọn hướng nghiên cứu chính, người dùng, input/output; đối soát phạm vi BTL/recognition cũ và ranh giới với nhóm khác.
- [ ] Ghi GPU thực, VRAM, ngân sách search/huấn luyện cuối, máy inference và quy tắc chọn model trên validation.
- [ ] Khóa câu hỏi nghiên cứu, không gian tìm kiếm, đối chứng và cách tính chi phí/thất bại.

Phụ thuộc: không. Tệp dự kiến: `tasks/plan.md`, `tasks/todo.md`, `docs/RESOURCE_AWARE_EXPERIMENT_PROTOCOL.md` (3 tệp). Kiểm chứng: truy được từng mục tiêu đến metric/đối chứng; không có ngưỡng thiếu đơn vị hay lẫn VRAM với RAM. Cổng: người dùng chọn phạm vi trước khi chuyển từ plan sang thực hiện.

### R1 — Tạo baseline và hồ sơ dữ liệu dùng chung

- [ ] Dùng release đạt quality gate T6 cho kết luận chất lượng; khóa train/val/test và lịch sử exposure. Ngoại lệ thử phương pháp phải ghi đúng phạm vi riêng.
- [ ] Chạy B0 cấu hình cố định và B1 AutoBatch theo protocol; lưu quality, VRAM, thời gian, seed và hash.
- [ ] Kiểm mọi model dùng cùng nhãn/evaluator, test chưa tham gia chọn cấu hình.

Phụ thuộc: R0, T6 cho nhánh nghiệm thu chính. Tệp dự kiến: protocol, baseline config, evaluation adapter, bài kiểm class map/protocol (4 tệp, chưa kể artifact run). Kiểm chứng: chấm lại cùng predictions cho kết quả nhất quán; các trường hợp trống/sai lớp/box trùng được xử lý đúng. Baseline chỉ hoàn tất khi có artifact thật.

### Cổng R0–R1

- [ ] Mục tiêu/phần cứng/ngân sách đã chốt và baseline tái lập được; trạng thái dữ liệu được báo đúng.

### R2 — Đo chi phí và nhận biết cấu hình không khả thi

- [ ] Đo bộ nhớ đỉnh và thời gian qua train/validation, ghi chi phí theo từng GPU và phần dự phòng.
- [ ] Phân biệt OOM với lỗi nhãn/runtime; lưu trial thất bại rồi thu hồi tài nguyên trước trial kế tiếp.
- [ ] Xuất bảng khả thi cho model/kích thước ảnh/batch mà không mặc định batch lớn nhất là tốt nhất.

Phụ thuộc: R1. Tệp dự kiến: `src/training/resource_profile.py`, cấu hình profiling, `tests/test_resource_profile.py` (3 tệp). Kiểm chứng: unit test đường thất bại/giới hạn; số đo trên GPU thật qua ít nhất một cấu hình vừa và một cấu hình không phù hợp được kiểm soát; không dùng mô phỏng làm bằng chứng GPU nhỏ thực.

### R3 — Chạy random search có ngân sách

- [ ] B2 lấy mẫu có seed từ không gian đã khóa, mỗi trial có init và thư mục kết quả độc lập.
- [ ] Tính cả profiling/evaluation/run lỗi vào budget, dừng launch đúng hạn và giữ checkpoint/log khi gián đoạn.
- [ ] Xuất lịch sử trial và lựa chọn bằng validation; không dùng test để chọn model.

Phụ thuộc: R2. Tệp dự kiến: `src/training/search.py`, `src/training/trial_store.py`, search config, `tests/test_training_search.py` (4 tệp). Kiểm chứng: mô phỏng clock, OOM và restart; không chạy lặp trial hoàn tất, không tự nới ngân sách; pilot thật đối chiếu thời gian với ledger.

### Cổng R2–R3

- [ ] Profiling và bộ chạy có ngân sách đã được kiểm; B2 tạo được một bảng kết quả hợp lệ trước khi thêm phương pháp phức tạp hơn.

### R4a — Thử chính sách lựa chọn có xét tài nguyên

- [ ] Triển khai P với bước lọc khả thi và phân bổ tài nguyên theo quy tắc R0; log lý do giữ/dừng từng cấu hình.
- [ ] Kiểm việc dừng sớm ở các mốc so sánh tương thích; ghi rõ cách xử lý lịch học khi cấp thêm tài nguyên.
- [ ] Có biến thể bỏ từng thành phần để đo lợi ích riêng, không đổi đồng thời dữ liệu và phương pháp tìm kiếm.

Phụ thuộc: R3. Tệp dự kiến: search policy, search runner, policy config, bài kiểm policy (4 tệp). Kiểm chứng: dùng các đường cong mô phỏng có cấu hình học chậm; chạy pilot và lưu quyết định để replay được.

### R4b — Đánh giá giả thuyết dưới cùng ngân sách

- [ ] So B0/B1/B2/P và ablation; báo chất lượng theo thời gian, chi phí đầy đủ, OOM, AP từng lớp và recall vật nhỏ.
- [ ] Lặp seed ghép cặp khi đủ budget; nếu chỉ một lượt, ghi kết quả sàng lọc sơ bộ.
- [ ] Lưu kết luận cả khi P không hơn B2; không gọi tích hợp công cụ có sẵn là tính mới đã chứng minh.

Phụ thuộc: R4a. Tệp dự kiến: comparison script, `docs/RESOURCE_AWARE_EXPERIMENT_PROTOCOL.md`, báo cáo đối chứng (3 tệp). Kiểm chứng: tính lại bảng từ trial ledger; cùng ngân sách và quy tắc lựa chọn, không chọn seed đẹp nhất. Khảo sát công trình liên quan trước khi tuyên bố đóng góp mới.

### R5a — Bàn giao gói model có thể kiểm tra

- [ ] Xuất weights/config/class map/hash/model card và trạng thái đạt/chưa đạt theo quy tắc đã khóa.
- [ ] Đo trên máy camera đích; tách chất lượng, FPS xử lý, FPS hiển thị, p95 độ trễ và RAM.
- [ ] Có bản đối chứng và cách rollback; chưa tự chuyển alias ứng dụng.

Phụ thuộc: R4b. Tệp dự kiến: model bundle writer, benchmark adapter, bài kiểm bundle, hướng dẫn bàn giao (4 tệp). Kiểm chứng: nạp lại gói đúng hash, đúng mapping; benchmark lặp cùng input/máy và có ca ứng viên không đạt.

### R5b — Hoàn thiện luồng người dùng

- [ ] UI nhận dataset/thiết bị/ngân sách và hiển thị điều kiện chưa đạt trước khi chạy.
- [ ] Hiển thị trạng thái trial, chi phí đã dùng, lựa chọn chất lượng–chi phí và lý do đề xuất.
- [ ] Cho xem gói kết quả trên camera, xử lý được không có GPU hoặc không tìm thấy ứng viên đạt.

Phụ thuộc: R5a. Tệp dự kiến: `src/classroom_ai/interfaces/web/training.py`, `src/classroom_ai/training/manager.py`, bài kiểm luồng job, hướng dẫn UI (4 tệp). Kiểm chứng: một lượt từ input đến report; Stop/restart không tạo job trùng; đối chiếu số UI với ledger. Không đưa toàn bộ máy tìm kiếm vào file UI.

### Cổng R4–R5

- [ ] Đối chứng đủ để đánh giá giả thuyết; người khác mở được gói và kiểm đúng model trên ứng dụng.

### R6 — Nghiệm thu và tái lập

- [ ] Khóa phương pháp/model/threshold trước đánh giá test; báo giới hạn về dataset, thiết bị, seed và điều kiện camera.
- [ ] Thành viên khác chạy được quy trình từ gói bàn giao; số đo và cấu hình truy nguồn được.
- [ ] Báo riêng hoàn thành phần mềm, đạt chất lượng camera và bằng chứng cải thiện của phương pháp.

Phụ thuộc: R5b, các cổng chất lượng liên quan T9/T10. Tệp dự kiến: final report, model card, hướng dẫn tái lập, script kiểm gói (4 tệp). Kiểm chứng: kiểm thử luồng thực, kết quả test sau khóa và biên bản chạy máy khác. Hoàn tất báo cáo thất bại không đồng nghĩa đã chứng minh phương pháp tốt hơn.

### R7 — GAN tùy chọn sau tuyến chính

- [ ] Chọn một giả thuyết dữ liệu cụ thể và protocol riêng trước khi làm; nếu chuyển sang sinh trắc học phải tách dataset và metric.
- [ ] So ảnh thật + augmentation thông thường với nhánh thêm GAN trên cùng test ảnh thật, kiểm nhãn/phân nhóm và lưu provenance.
- [ ] Báo lợi ích, thất bại và tổng chi phí tạo/kiểm ảnh; không biến ảnh sinh thành ảnh gốc độc lập hoặc tự thay release 15.000 ảnh.

Phụ thuộc: R6, người dùng chọn nhánh và ngân sách riêng. Đây là epic tùy chọn chưa đủ đặc tả triển khai; phải tách thành các task tối đa khoảng 3–5 tệp sau khi chọn loại GAN/bài toán. Kiểm chứng dự kiến: ablation có/không GAN và đánh giá dữ liệu thật giữ riêng; không bắt đầu bằng việc thêm GAN vào camera runtime.
