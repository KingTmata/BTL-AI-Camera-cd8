# Checklist triển khai — BTL-AI-Camera-cd8

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

### Definition of Done cuối

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
