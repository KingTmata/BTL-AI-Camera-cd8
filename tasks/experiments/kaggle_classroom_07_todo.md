# Tasklist — Kaggle fine-tune từ checkpoint 07 — YOLO26n Project8 v0.4

Ngày 12/10/2026. [Plan chi tiết](kaggle_classroom_07_plan.md).

Tên đầy đủ: **checkpoint 07 — YOLO26n Project8 v0.4, phiên Kaggle ngày 08/10/2026, best epoch 19**. 07 là thứ tự dropdown Streamlit, không phải R07 của chiến dịch v0.5. Nguồn K3 Project8 v0.3 → fine-tune v0.4 12.000 ảnh, reset optimizer/scheduler, dừng thủ công 20/50 epoch.

Trọng số: `runs/kaggle/project8_v04_12k_from_best50_20261008_151936/weights/best.pt`.
SHA256: `d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb`.

6 cấu hình, tối đa 10 epoch/cấu hình, patience=3; chưa launch. Checklist này riêng với tasks/todo.md của dataset v0.5.

## T0 — Đối chiếu và lập plan (S)

- [x] Kiểm hash checkpoint, report/config/source early-stop local và tài khoản Kaggle awm3636.
- [x] Lưu plan 6 configs, 2 eval controls, data gates, protocol và ngân sách đề xuất.
- [x] Không ghi đè plan/todo khác hoặc source chat CPU/GPU.

Phụ thuộc: không. Tệp: hai tài liệu này. Kiểm chứng: đọc lại Markdown, links tồn tại, 6×10=60 epoch chính, chưa train.

## T1 — Chốt metric và budget (S)

- [ ] Review plan; chốt threshold classroom/legacy/FP và imgsz/mode ứng dụng.
- [ ] Chốt wall-clock budget, quota dự phòng và ngoại lệ zero-fitness/ties.

Phụ thuộc: T0. Tệp: plan, experiment_spec.json. Kiểm chứng: epochs=10, patience=3, seed=42; proposed/approved rõ, không test để chọn config.

## T2a — Rubric và nguồn lớp học (S)

- [ ] Khóa rubric 8 lớp, ID7 cup+bottle, box/occlusion/negative.
- [ ] Có ảnh gốc, scene/session IDs, provenance và reviewer; auto-label chưa review không được dùng.

Phụ thuộc: T0. Tệp: rubric, intake manifest, review ledger. Kiểm chứng: visual review mẫu từng nguồn và class map.

## T2b — Review từng lô 25–40 ảnh (M/lô)

- [ ] Rà mọi vật thuộc 8 lớp, QC/double-review, hòa giải; unresolved=0 trước approved.
- [ ] Review cả dữ liệu cũ được dùng; không tự cho qua full_review_pending v0.5.

Phụ thuộc: T2a. Tệp: ledger, label delta, QC report. Kiểm chứng: overlays, IDs/bounds/missing objects; truy ngược reviewer/reason/version.

## T2c — Khóa dataset/split (M)

- [ ] Split theo buổi/cảnh/clip; kiểm near-duplicates và exposure xuyên split.
- [ ] Hash/decode/format/semantic QA đạt, coverage từng lớp rõ; training_ready có bằng chứng.
- [ ] Cùng train/sampling cho 6 configs; classroom val và legacy val riêng, test giữ kín.

Phụ thuộc: T2b. Tệp: data.yaml, manifest, split audit, data card. Kiểm chứng: nhóm cảnh độc lập, không holdout cũ vào train, ledger khớp.

### Cổng A

- [ ] T1 và T2c đạt; nhánh phương pháp trên miền cũ không được tuyên bố giải quyết lớp học.

## T3 — Notebook draft chưa launch (M)

- [ ] Cells gates/baselines/smoke/runner/archive; launch=False mặc định.
- [x] Đã tạo và compile notebook; người dùng yêu cầu thực hiện nên START_TRAINING=True cho bản launch riêng. Matrix unit tests 3/3 đạt; Version 2 đã gửi và UI Kaggle báo Running với GPU T4×2. Version 1 dùng draft cũ đã Cancelled.
- [ ] Matrix đúng; mỗi model mới từ checkpoint đầy đủ ở đầu tasklist, resume=False, không nối chuỗi weights.
- [ ] Xác minh architecture/layer list freeze; output riêng, exist_ok=False.

Phụ thuộc: T1; được draft trước T2c nhưng không train. Tệp: notebook, config matrix, runner. Kiểm chứng: parse/compile không execute, review config diff và gates.

## T4 — Kaggle preflight/stopper (S)

- [ ] GPU/VRAM/quota/session/storage/runtime/input hash có bằng chứng.
- [ ] Kiểm source fitness/stopper, chuỗi tăng/tie/giảm/zero, MuSGD/AdamW; log giới hạn.
- [ ] Khóa batch/device/AMP và scheduler 10 epoch, inputs private đúng manifest.

Phụ thuộc: T3, T2c khi stage dữ liệu. Tệp: preflight, stopper verification. Kiểm chứng: không train trong preflight, source/version thực được lưu.

### Cổng B — Trước training

- [ ] Plan được chốt, dataset ready, runtime pass, quota đủ; chưa đạt thì không train.

## T5 — Smoke 1 epoch riêng (S)

- [x] Log Version 2 xác nhận smoke hoàn tất 1 epoch, final validation và output tại train/SMOKE_checkpoint07; fitness CSV log=0.54093. Nhãn chưa full review; không coi đây là nghiệm thu lớp học. Native mAP50/mAP50–95 log làm tròn=0.768/0.541.

- [ ] Smoke hữu hạn, train/val overlays đúng, đo thời gian/VRAM/AMP.
- [ ] Cập nhật estimate/budget; archive riêng; E1 vẫn nạp checkpoint gốc.

Phụ thuộc: cổng B. Tệp: smoke report, estimate. Kiểm chứng: review train_batch, weights hữu hạn, đủ ngân sách job kế.

## T6a — Hai đối chứng chỉ eval (M)

- [ ] Checkpoint gốc đầy đủ ở đầu tasklist và YOLO26n pretrained gốc chấm cùng val.
- [ ] Mapping COCO80→8 đúng, cup+bottle gộp ID7; tách AP/application protocol; lưu GT/predictions/hash.

Phụ thuộc: T5. Tệp: baseline metrics, predictions, protocol. Kiểm chứng: class IDs/sample count/matching nhất quán, chưa test.

## T6b — Tuần tự từng cấu hình (S/run)

- [ ] E1: MuSGD, lr0=1e-4.
- [ ] E2: MuSGD, lr0=3e-4.
- [ ] E3: AdamW, lr0=1e-4.
- [ ] E4: AdamW, lr0=3e-4.
- [ ] E5: như E1, chỉ scale=0.2.
- [ ] E6: như E1, chỉ freeze backbone đã xác minh.

Phụ thuộc: T6a; run trước archive xong. Tệp mỗi run: args, CSV, weights, log/metadata/archive. Kiểm chứng: initial hash đúng, <=10 epoch, patience=3, stop reason đúng, best/last hữu hạn, archive đọc được; kiểm quota/OOM trước job kế; retry không âm thầm đổi protocol hoặc vượt tổng 10 epoch/run.

## T7 — Chọn bằng validation (M)

- [ ] Bảng 8 models, per-class AP/PR/FN/FP/duplicate, domain/size/occlusion và timing.
- [ ] Áp threshold đã chốt; không đạt thì no_promote; khóa config trước test, nêu giới hạn một seed.

Phụ thuộc: T6b. Tệp: comparison, validation report, selection. Kiểm chứng: không trộn AP640/NMS với trainer1280/NMS-free; chỉ val để chọn.

### Cổng C

- [ ] Selection khóa, test không lộ để tuning, archive/checksum đủ.

## T8 — Test và bàn giao (M)

- [ ] Test một lần checkpoint gốc và candidate đã chọn; không candidate đạt thì báo no_promote theo phạm vi chốt.
- [ ] Báo coverage 8 lớp và giới hạn; không đổi config theo test.
- [ ] Archive/download khôi phục được, hash verified; không tự thay model ứng dụng.

Phụ thuộc: cổng C. Tệp: test metrics, REPORT.md, artifact manifest. Kiểm chứng: split/GT/protocol khớp, weights load được, phân biệt early_stop/budget_stop/failed/completed.
