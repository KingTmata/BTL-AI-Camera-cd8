# BTL AI Camera — danh mục tài liệu

**Dataset chính hiện tại:** [v0.5](data/dataset/project8_v0.5_merged_20261008/README.md), **15.000 ảnh duy nhất:12.000 train /2.500 validation /500 test**, đủ số lượng mục tiêu. Đã giữ2.966 ảnh mới từ merged_3k sau loại31 trùng và3 pizza sai nhãn, thêm34 ảnhphone unique. Toàn bộ15.000 ảnh đang chờ full visual review; chưa train hoặc cómetricv05. Recognition dùng `configs/dataset_current.json`; checkpoint vẫn bestv04, Traininglocal v03 giữ hợp đồng riêng. [Báo cáo hiện hành](reports/results/dataset_merge_v05_20261008/REPORT.md) · [Kế hoạch](tasks/plan.md).

**Chạy chương trình:** mở `run-demo.cmd`; lần đầu chạy `setup-demo.cmd`. Điểm khởi động Python là `src/app.py`. Sau đợt dọn ngày 08/10/2026, dự án giữ 20 file Python phục vụ ứng dụng, cài đặt và xuất Kaggle; 42 script tạm, bản sao, CLI riêng và file kiểm thử đã được sao lưu rồi loại khỏi cây dự án. Xem [báo cáo dọn Python](reports/results/python_cleanup_20261008/REPORT.md). Các hướng dẫn cũ gọi `scripts/data/*` hoặc `src.inference.demo` là hồ sơ lịch sử; muốn dùng lại cần khôi phục từ bản sao.

**Dataset ghép mới:** [project8 v0.4](data/dataset/project8_v0.4_merged_20261008/README.md) có 12.000 ảnh, 37.453 box sau khi loại 138 ảnh thuộc nhóm độ sáng thấp; ID 7 `bottle` gồm cả cup và bottle theo yêu cầu. [Tải ZIP](data/dataset/project8_v0.4_merged_20261008). Bộ v0.4 đã train 20 epoch mới; Recognition dùng best epoch 19 với tên nhãn bottle. Cấu hình Training local v0.3 được giữ riêng.

Tài liệu được sắp xếp lại ngày **05/10/2026** theo trạng thái công việc đã có bằng chứng.

| Nhóm | Nội dung | Mở danh mục |
|---|---|---|
| README | 30 tài liệu còn việc cần làm hoặc tiếp tục dùng làm hướng dẫn | [Kế hoạch và hướng dẫn đang dùng](README/README.md) |
| README_DA_HOAN_THANH | Hồ sơ của các đợt nhập/gộp dữ liệu, kiểm kê, dọn và train đã kết thúc; bổ sung báo cáo Kaggle và bảng 30 epoch ngày 06/10/2026 | [Hồ sơ đã hoàn thành](README_DA_HOAN_THANH/README.md) |
| Hướng dẫn tuần 3 tiếp tục | Giữ nguyên nội dung và vị trí theo yêu cầu | [Triển khai tuần ba tiếp tục](TRIEN_KHAI_TUAN_3_TIEP_TUC.md) |

Hướng dẫn cài đặt và chạy đầy đủ trước đây nằm tại [README dự án](README/README_DU_AN.md). Các lệnh trong tài liệu chạy từ **thư mục gốc dự án**, dù file hướng dẫn đã chuyển vào thư mục README.

**Đối soát 08/10/2026:** đã thực thi **50 epoch trong 6 run** (1+4+1+10+14+20). Xem [bảng đủ 50 epoch](README_DA_HOAN_THANH/reports/results/training_50_epochs_20261008/REPORT.md) và [Kaggle 08/10 — kiểm trọng số, kết quả và 20 epoch mới](README_DA_HOAN_THANH/reports/results/kaggle/project8_26n_kaggle_20261008_040051/REPORT.md). Best v0.3 `weights/BEST_Project8_20261008.pt` vẫn được giữ; alias `weights/BEST_Project8.pt` hiện trỏ đến best v0.4, xem cập nhật mới bên dưới. Best thuộc epoch 19 K3. Evaluator chung mAP50/mAP50–95 **56,727%/35,896%**, tăng **2,879/2,049 điểm %** so K2. Chưa kiểm test độc lập/FPS hoặc nghiệm thu. [Trạng thái theo tuần](README/README.md#trạng-thái-theo-tuần).

Bảy file metadata Markdown đi cùng dataset/artifact giữ tại vị trí cũ để bảo toàn hồ sơ và checksum; xem [danh sách ngoại lệ](README/README.md#metadata-giữ-tại-vị-trí-gốc). Đã bỏ hai file điều hướng thừa; hướng dẫn tuần 3 liên kết trực tiếp đến tài liệu chính.

**Kaggle v0.4 lịch sử:** [Notebook 12k từ best 50 epoch](notebooks/kaggle/project8_v04_12k_from_best50.ipynb); draft Kaggle đã nhập, START_TRAINING=False. Xem [báo cáo chuẩn bị](reports/results/kaggle_prepare_v04_20261008/REPORT.md). Dataset/checkpoint đã gắn Input Kaggle; full run mới có 20 epoch được tải và đối soát.

**Cập nhật v0.4 — 08/10/2026:** đã nhập ZIP full train 20/50 epoch; dùng `weights/BEST_Project8_v04_20261008.pt` (best epoch 19), alias `weights/BEST_Project8.pt`. Trainer 1280/NMS-free: 75,356% / 53,488%; cùng protocol COCO 640/NMS=True trên val v0.4: trước 47,593% / 31,047% → mới 70,706% / 50,694%. Trên val v0.3: 56,727% / 35,896% → 55,279% / 35,049%, nên giữ best cũ để đối chứng. Dataset mới 12.000 ảnh (9.540/1.754/706); ID 7 bottle gồm cup+bottle. Recognition đã đổi mapping; local Training/export v0.3 vẫn giữ riêng. [Báo cáo đầy đủ](README_DA_HOAN_THANH/reports/results/kaggle/project8_v04_12k_from_best50_20261008_151936/REPORT.md).


## Chuẩn bị Kaggle v0.5

Dữ liệu hiện tại: **15.000 ảnh = 12.000 train + 2.500 validation + 500 test**. Đã thêm 34 ảnh phone; không còn thiếu ảnh. 30.000 payload là 15.000 ảnh và 15.000 file nhãn. Full visual review chưa hoàn tất; training_ready=false.

[Notebook v0.5 từ best v0.4](notebooks/kaggle/project8_v05_15k_from_best_v04.ipynb) đã chuẩn bị; người dùng cho phép smoke 1 epoch trên 5% train trước khi review xong, full run vẫn bị chặn. Gói data/exports/project8_v05_trainval_best_v04_20261008.zip có train/val và best v0.4; giữ 500 test riêng ở local. Xem [hướng dẫn Kaggle](README/docs/KAGGLE_TRAINING_HANDOFF.md), [kiểm chứng](reports/results/kaggle_v05_preparation_20261008/preparation.json) và [danh sách file đã dọn](reports/results/workspace_cleanup_20261008/removed.json).
