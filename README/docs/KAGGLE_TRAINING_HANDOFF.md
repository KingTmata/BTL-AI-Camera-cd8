> **Hiện tại — v0.5, 08/10/2026:** 15.000 ảnh = 12.000 train + 2.500 validation + 500 test. Full visual review chưa hoàn tất, training_ready=false.
>
> Notebook mới: [project8_v05_15k_from_best_v04.ipynb](../../notebooks/kaggle/project8_v05_15k_from_best_v04.ipynb). Gói upload nhanh: data/exports/project8_v05_trainval_best_v04_20261008.zip.bin (14.500 ảnh train/val, không chứa ảnh/nhãn test), kèm BEST_Project8_v04_20261008.pt. Đây là ZIP nguyên byte, giữ đuôi .bin để Kaggle không tự bung. [Biên bản chuẩn bị và checksum](../../reports/results/kaggle_v05_preparation_20261008/preparation.json).
>
> Fine-tune từ best v0.4 với resume=False, 25 epoch mới, imgsz 1536, patience 10, close_mosaic=5, cache=False; tổng batch 16 (8/GPU trên 2×T4). 1536/batch32 đã CUDA OOM; người dùng chọn 1536/batch16, cho phép thử 1280/batch32 nếu vẫn OOM. Smoke luôn 1 epoch/5% train, close_mosaic=0. Full review chưa xong nên full run vẫn bị chặn. Sau review cần cập nhật release/hash và gói upload. Nội dung v0.3/v0.4 dưới đây là lịch sử.
>
> **Kiểm thử archive thật trên Kaggle — 09/10/2026:** [Dataset Private một file](https://www.kaggle.com/datasets/awm3636/project8-v05-single-archive-and-best-v04), 1.440.034.912 byte, SHA256 `236972ca3e6142d977a29068cf217dc5312c8fb870c8094a98c4960386a0eb5b`. Không gắn thêm dataset 29.005 file vào phiên mới. Preflight CPU giải nén đủ 29.005 file trong 28 giây, kiểm 14.500 cặp ảnh–nhãn, 41.631 box, đúng BEST v0.4. [Kết quả PASS](../../reports/kaggle_archive_preflight_20261009.json) chỉ xác nhận archive/dữ liệu/trọng số, không phải train GPU. [Notebook hiện hành](https://www.kaggle.com/code/awm3636/project8-v05-fast-archive-from-best-v04/edit) đã khởi động được 2×T4; đang kiểm smoke. Phiên cũ tải Input bị kẹt đã dừng theo yêu cầu chuyển sang cách mới.
> **Cập nhật v0.4 — 08/10/2026:** Recognition mặc định dùng `weights/BEST_Project8.pt`, alias của best v0.4 epoch 19. Tám nhãn hiện hành kết thúc bằng `bottle` (gộp cup+bottle). Dataset 12k: 9.540 train / 1.754 val / 706 test. Training local và exporter v0.3 vẫn dùng hợp đồng cũ `cup`; notebook v0.4 và `configs/data_v04.yaml` dành cho bộ mới. Các phần v0.3 bên dưới là lịch sử hoặc hướng dẫn cho bộ cũ.

# Handoff huấn luyện Kaggle — project8_v0.3

Cập nhật 05/10/2026. Quy trình này tạo một **Kaggle Dataset riêng tư** từ gói export và chạy notebook trên tài khoản Kaggle của người dùng. Ứng dụng local chỉ chuẩn bị/tải gói và nhập kết quả; nó không điều khiển Kaggle hay GPU cloud.

## Nguồn dữ liệu và checkpoint

- Release gốc `project8_v0.3-approved-20261004` vẫn nằm tại `data/dataset/project8_v0.3` và không bị sửa. Gói export chỉ có ảnh/nhãn **train 3.883** và **val 798**; test 321 ảnh không được chép sang cloud.
- Gói giữ `RELEASE.json`, full `checksums.json`, DATA_CARD, mapping, validation/statistics, manifest export đã đổi đường dẫn sang tương đối, runtime YAML riêng, SHA-256 cho file gói và provenance checkpoint.
- Checkpoint gợi ý sau khi R3 dừng: `runs/train/project8_v03_26n_cpu_r2best_e5_20261005_134221_540fdcbf/weights/best.pt`, SHA-256 `4034b0ec824b2e9a5e5b945f4626dbbdf2bf506ac26e5575c73e240f4a52f5d3` (5.375.493 byte). Run dừng sau epoch 1/5 theo yêu cầu, exit 0, final validation và checkpoint verification đạt; train/val 1.770,219/124,094 giây, final val 68,406 giây. Điều kiện thật: CPU, imgsz 768, batch 4. Final trainer NMS-free P=50,4715%, R=33,3978%, mAP50=29,5330%, mAP50–95=18,3409%. Các số khác imgsz/batch so với R1/R2 nên không kết luận cải thiện do Kaggle hay do cấu hình.
- Gói gồm pretrained `weights/yolo26n.pt` làm baseline và checkpoint project8 được chọn. Notebook chấm cả hai trên cùng validation protocol NMS, imgsz 640/conf .001/IoU .70/max_det 300. Đây mới là phép so sánh chung; chưa có kết quả cho đến khi notebook thực sự chạy.

## Tạo gói và tải lên Kaggle

1. Trong Training chọn **Kaggle Notebook**, chọn R3 best (được chọn mặc định nếu run đã kiểm), rồi bấm **Tạo gói Kaggle**. Nút này chờ training local kết thúc, kiểm run/checkpoint/hash nguồn, tạo ZIP và cung cấp nút tải ZIP cùng notebook. Không có tác vụ cloud nào được tự khởi động.
2. Xem bảng byte/MB/MiB trong app, lưu SHA-256 ZIP để đối chiếu sau tải.
3. Trên Kaggle vào Datasets → New Dataset, tải ZIP và giữ **Private**; tạo Notebook, chọn **Add Input** và gắn Dataset vừa tạo. Kaggle tự bung ZIP upload thành cây file và giữ cấu trúc; notebook hỗ trợ cả cây đã bung lẫn ZIP còn nguyên trong `/kaggle/input`.
4. Mở `project8_training.ipynb` trong Notebook editor. Giữ `BUNDLE_INPUT='/kaggle/input'` nếu chỉ gắn một bundle; nếu có nhiều bundle, đổi sang đường dẫn Input riêng của dataset cần dùng. Trên thanh đầu trang mở **Settings → Accelerator → GPU**; chọn **None** để dùng CPU. Chọn accelerator sẽ restart session. Bật Internet trong Settings nếu cần cài package. Notebook sao chép gói sang `/tmp`, kiểm kích thước/SHA-256 từng file rồi mới dùng. Ảnh được đọc từ scratch, không stream từng ảnh từ `/kaggle/input`.
   Nếu GPU bị mờ hoặc không xuất hiện, Kaggle yêu cầu xác minh số điện thoại; sau khi xác minh, tùy chọn còn phụ thuộc quota và máy GPU khả dụng tại thời điểm đó. Trong panel Session options, mở **Get phone verified** để xem luồng xác minh.
5. Chạy setup/smoke cells và kiểm log `nvidia-smi`, `torch.cuda.is_available()` nếu chọn GPU. Nếu chọn GPU nhưng CUDA không có, notebook dừng và yêu cầu đổi accelerator hoặc chọn CPU. Cài `ultralytics==8.4.165` cùng `pycocotools`, không cài lock Torch CPU local; notebook đối chiếu CUDA trước/sau pip.
   Notebook cập nhật tự tìm bundle trong `/kaggle/input` nếu chỉ gắn một bundle. Sau khi xác minh SHA-256, smoke cell tạo `/tmp/project8_kaggle_runtime/training_data.yaml` với đường dẫn tuyệt đối và dùng chính `check_det_dataset(..., autodownload=False)` của Ultralytics để kiểm train/val. YAML trong bundle đã hash giữ nguyên; `path: .` của nó không được truyền trực tiếp cho trainer vì Ultralytics phân giải dấu chấm từ thư mục làm việc. Dataset đã tải lên không cần tải lại khi chỉ cập nhật notebook này.
6. Huấn luyện mặc định tắt (`START_TRAINING=False`). Khi sẵn sàng, đặt True. Gợi ý một run mới tối đa 5 epoch từ R3 best; GPU `imgsz=768`, `batch=4` nếu đủ VRAM. Theo yêu cầu chỉ train bằng GPU, cell train kiểm lại `EXECUTION_DEVICE`, ID GPU và CUDA ngay trước khi tạo run/nạp model: chọn CPU, mất CUDA, thiếu GPU hoặc trùng ID đều dừng rõ, không fallback CPU. Muốn train CPU thì dùng app local. Cấu hình remote phải được ghi như điều kiện riêng, không gọi app local là GPU remote.
7. Checkpoint `last.pt`, `best.pt`, epoch files và result ZIP được ghi dưới `/kaggle/working/BTL-AI-Camera/kaggle_runs`. Lấy ZIP về máy hoặc dùng Save Version → Save & Run All để tạo Kaggle version có output. Không coi file trong session là đã persist trước khi tải/lưu thành công. Nếu notebook đang chạy tương tác, Save & Run All sẽ chạy notebook từ đầu trong session riêng; tránh bấm khi không muốn train lại.
8. Tải `<run_id>_result.zip` từ output và trong app chọn **Xác minh và nhập checkpoint**. App kiểm manifest/hash, protocol validation và model detect đúng 8 lớp; sau đó ghi weights vào `weights/kaggle/`. Metrics được ghi nhận là dữ liệu import từ result ZIP, không phải app local tự chạy.

## Kiểm tra trên draft ngày 05/10/2026

Đã nhập notebook Project8 vào draft `btn ai cd8` và chạy các cell setup/preflight trên Kaggle sau khi người dùng cho phép. Runtime có **GPU T4 x2**, Internet bật, Python **3.13.15**, Torch **2.11.0+cu128**, CUDA Torch **12.8**, `torch.cuda.is_available()=True`; cấu hình chọn `device=0`, dùng một GPU. Đã kiểm checksum/kích thước **9.377 file**, xác nhận **3.883 ảnh train / 798 ảnh val**, nạp checkpoint R3 best đúng tám lớp và xác minh đường dẫn mà trainer sử dụng. Chưa đo tốc độ hay chất lượng sau train trên Kaggle.

Lần kiểm tra đầu phát hiện smoke cell dùng biến `WORK` chưa khai báo. Bản sửa lấy thư mục runtime từ `BUNDLE.parent`, đã tái hiện lỗi trong kiểm thử rồi vượt qua **49/49 kiểm thử local** và smoke check thực tế trên kernel Kaggle sạch. YAML nguồn trong bundle, release gốc và trọng số nguồn giữ nguyên. Draft đã lưu; **`START_TRAINING=False`**, các cell train, baseline và đóng gói kết quả chưa thực thi. Bước tiếp theo cần người dùng cho phép lượt train mới trước khi bật guard.

## Hạn mức, tiếp tục và khôi phục

Kaggle hiện mô tả CPU/GPU notebook session tối đa 12 giờ và tối đa 20 GB output trong `/kaggle/working`. Tài liệu GPU mô tả quota thường khoảng 30 giờ/tuần, đôi khi cao hơn theo nhu cầu/tài nguyên. Mức khả dụng, model GPU và quota thực tế thay đổi; kiểm tra trang Settings/quota của tài khoản trước khi dự trù. Không có phép đo trên Kaggle của dự án này nên chưa đưa hệ số tăng tốc hay cam kết hoàn thành.

Notebook dành `RUN_MODE='RESUME_KAGGLE'` cho `last.pt` còn optimizer của run Kaggle đã lưu. Sau khi run hoàn tất, `best.pt` có thể đã strip optimizer; dùng nó để khởi tạo fine-tune **run mới**, không gọi là resume. Nếu OOM, giữ run cũ rồi mở run mới từ best với batch nhỏ hơn (4→2) hoặc imgsz 768→640; ghi cấu hình thay đổi. Nếu runtime bị ngắt, chỉ resume khi `last.pt` còn tồn tại trong input/output version đã lưu hoặc người dùng đã tải về. Checkpoint chỉ nằm trong session bị mất thì không khôi phục được.

Ngân sách đề xuất: một lượt cloud tối đa 5 epoch, sau đó chấm pretrained và best cùng validation protocol, xem AP từng lớp/ảnh lỗi và mới quyết định lượt kế. Không dùng test để chọn model, cấu hình hay ngưỡng. Không tự chạy lượt này trên tài khoản người dùng.

## Số đo gói

Điền bảng này sau thao tác export thật trong app; ZIP được đo ở đúng lần tạo. MB dùng 1.000.000 byte; MiB dùng 1.048.576 byte.

| Thành phần | Byte | MB | MiB |
|---|---:|---:|---:|
| Ảnh train | 450.783.702 | 450,784 | 429,901 |
| Nhãn train | 1.186.243 | 1,186 | 1,131 |
| Ảnh val | 71.964.376 | 71,964 | 68,631 |
| Nhãn val | 280.261 | 0,280 | 0,267 |
| Metadata + YAML | 6.220.013 | 6,220 | 5,932 |
| Checkpoint project8 + pretrained baseline | 10.919.946 | 10,920 | 10,414 |
| Code + notebook | 23.122 | 0,023 | 0,022 |
| FILES.json (hash + provenance) | 2.519.554 | 2,520 | 2,403 |
| Tổng gói đã bung (9.378 file) | 543.897.217 | 543,897 | 518,701 |
| ZIP tải lên Kaggle | 520.957.322 | 520,957 | 496,824 |

SHA-256 ZIP đo thực: `fa4fd64cac2256079fa49156bd38104ae8a6783411396722a0d6732a7625f099`. Tổng các file đã bung là 543.897.217 byte, gồm `FILES.json`; các byte payload trước đây không tính file manifest này là 541.377.663. Kaggle giải nén archive upload nên dự trù khoảng 521 MB upload/Private Dataset và 544 MB cho bản staged trong `/tmp`; nên để ít nhất 1 GB scratch trống cho staging/temporary. Run output gồm checkpoint từng epoch, best/last, log, metric và result ZIP; R3 epoch0 là 15.537.443 byte, nên dành khoảng 250 MB output cho lượt 5 epoch (ước lượng, kích thước thực phụ thuộc checkpoint). Tài liệu Kaggle nêu trần output hiện tại lớn hơn nhiều, nhưng cần kiểm phần dung lượng còn lại trong tài khoản/runtime.

Full dataset backup khác gói huấn luyện: full release folder được đo trước export là 559.921.993 byte (10.020 file), gồm cả 321 ảnh/nhãn test và metadata/cache dẫn xuất. Train+val media/labels là 524.214.582 byte; hai checkpoint là 10.919.946 byte. Gói này giữ metadata release/checksum cần truy vết nhưng bỏ test media và cache. Nó được đánh dấu export trong `FILES.json`; không được coi là release gốc đầy đủ.

## Tham khảo chính thức

- [Kaggle Datasets: archive upload, private datasets, Add Input](https://www.kaggle.com/docs/datasets)
- [Kaggle Notebooks: session, Save Version, output](https://www.kaggle.com/docs/notebooks)
- [Kaggle Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage)

## Kaggle — hoàn thành 10 epoch và nhập kết quả — 05/10/2026

Run `project8_26n_kaggle_20261005_092653` dùng R3 best, T4 x2 (`device=0,1`), imgsz yêu cầu 1080/thực 1088, batch tổng 4, workers 2/rank, cache False, close_mosaic=5, MuSGD lr0=.001, seed42, resume=False. Dataset 3.883 train/798 val giữ nguyên; không dùng test. CSV đủ 10 epoch hữu hạn, trung bình train+val 156,520 giây/epoch. Best sau epoch 9, last sau epoch 10; đã kiểm tensor đối chiếu với epoch8/epoch9.

Đánh giá COCO bbox cùng protocol 640/conf=.001/IoU=.70/NMS=True trên val: pretrained mAP50/mAP50–95 **28,18%/18,89%**, best **47,78%/28,62%** (tăng **19,60/9,72 điểm %**). Tách khỏi final trainer 1088/NMS-free **43,8%/26,5%**. Person/table/book/cell phone tăng rõ; laptop/backpack/cup giảm so baseline ở protocol chung, cần kiểm ảnh lỗi. Chưa kết luận model hoàn thiện hay đo FPS.

Đã tải full archive 166.119.984 byte và result ZIP 4.858.177 byte, xác minh hash cloud, 44 file payload và đủ best/last/epoch0–epoch9 đúng 8 lớp. Model đã nhập `weights/kaggle/project8_26n_kaggle_20261005_092653_fd9cb282f05f.pt`. Best/last đã strip optimizer; epoch files còn optimizer. Kaggle draft vẫn có ConcurrencyViolation; dữ liệu local an toàn nhưng chưa xác nhận version cloud persist. Không train thêm, đổi dữ liệu hoặc dùng test.

Báo cáo, bảng đủ epoch và hướng đi tiếp: [REPORT.md](../../reports/results/kaggle/project8_26n_kaggle_20261005_092653/REPORT.md). Checkpoint và full ZIP tại `runs/kaggle/project8_26n_kaggle_20261005_092653/`; metrics/plots tại `reports/results/kaggle/project8_26n_kaggle_20261005_092653/`.

## Kaggle — fine-tune hoàn tất 14 epoch — 06/10/2026

Run project8_26n_kaggle_20261006_083226 fine-tune từ checkpoint best của run 05/10 (SHA-256 nguồn fd9cb282f05f0053166174e6c8d13221f7bfff54e8f06673b21bf8134951a3a8), resume=False. Runtime ghi GPU 2× T4, imgsz 1280, batch tổng 16, 14 epoch, workers 2/rank, cache False, MuSGD lr0=.001, seed 42, close_mosaic=5. Augmentation: mosaic=.5, fliplr=.5, flipud=0, degrees=0, translate=.1, scale=.5, mixup=0, cutmix=0. Dataset 3.883 train/798 val; không dùng test.

Result ZIP đã tải, SHA-256 1bd9b9ffa191f68dcf70c7ee6391b5685549962149ef49dbab0fdc2cee9e375d; các file trong FILES.json đều khớp size/hash. Best mới SHA-256 ad136a17611ff29fd5b4dc39bed7826706a08c8d5283f63f7af8fefc8860eddf, đã nhập vào weights/kaggle/ và nạp thử local thành công với task detect/đủ 8 lớp.

Trên cùng protocol bbox validation 640/conf=.001/IoU=.70/NMS=True/max_det=300: pretrained mAP50/mAP50–95 28,18%/18,89%; fine-tuned 53,85%/33,85% (tăng 25,67/14,95 điểm phần trăm). Đây khác metrics trainer ở 1280/NMS-free. Backpack và cup giảm nhẹ theo evaluator chung; laptop AP50 tăng nhưng AP50–95 giảm. Báo cáo run và bảng 14 epoch: [REPORT.md](../../README_DA_HOAN_THANH/reports/results/kaggle/project8_26n_kaggle_20261006_083226/REPORT.md). ZIP nguyên vẹn được lưu dưới runs/kaggle/project8_26n_kaggle_20261006_083226/; giữ ZIP là đủ để lưu JSON/YAML/CSV phụ trợ, còn riêng best.pt là đủ để inference. ZIP không có dataset hoặc last.pt; chưa xem ảnh lỗi, chạy test độc lập, đo FPS hay nghiệm thu triển khai.
Trên cùng protocol bbox validation 640/conf=.001/IoU=.70/NMS=True/max_det=300: pretrained mAP50/mAP50–95 28,18%/18,89%; fine-tuned 53,85%/33,85% (tăng 25,67/14,95 điểm phần trăm). Đây khác metrics trainer ở 1280/NMS-free. Backpack và cup giảm nhẹ theo evaluator chung; laptop AP50 tăng nhưng AP50–95 giảm. Báo cáo run và bảng 14 epoch: [REPORT.md](../../README_DA_HOAN_THANH/reports/results/kaggle/project8_26n_kaggle_20261006_083226/REPORT.md). ZIP nguyên vẹn được lưu dưới runs/kaggle/project8_26n_kaggle_20261006_083226/; giữ ZIP là đủ để lưu JSON/YAML/CSV phụ trợ, còn riêng best.pt là đủ để inference. ZIP không có dataset hoặc last.pt; chưa xem ảnh lỗi, chạy test độc lập, đo FPS hay nghiệm thu triển khai.
