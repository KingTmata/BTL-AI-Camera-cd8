# Tiến trình Project8 — cập nhật 11/10/2026

**Cập nhật Streamlit mới nhất:** đã thêm kích thước512–1280, NMS tắt/0,5/0,6 và một/ba lượt trên cùng checkpoint; ba lượt bắt buộc NMS. Mặc định R04/640/một lượt/NMS tắt/CPU. Có hướng dẫn, JSON đúng cấu hình và kiểm tra47test/AppTest/model thật/trình duyệt. Ba lượt là tùy chọn chất lượng mới được yêu cầu; webcam vẫn ưu tiên một lượt. [Chi tiết và giới hạn chưa nghiệm thu](../webcam/streamlit_inference_controls_20261011.md). Preview UI thực dùng fragment0,3giây, chưa đạt mục tiêu hiển thị webcam nhanh.

**Cập nhật mới nhất về trọng số:** theo yêu cầu giữ ba file, `weights/` chỉ còn R04, baselinev05 và R01; R04 đã được đặt mặc định cấu hình và Streamlit. Đã xóa7file khác trongweights≈56,38MiB, có manifest SHA256. Các đoạn dưới ghi “chưa đổi mặc định” hoặc “giữ toàn bộ checkpoint” mô tả giai đoạn trước yêu cầu này; xem [trạng thái hiện hành](active_weights_20261011.md). Chưa đổiCPU/NMS hay chạytrain.

**Mục tiêu bổ sung:** phải kiểm và chọn cấu hình chạy ổn trên cả CPU máy hiện tại và RTX 5060 Laptop 8 GB. Xem [mục tiêu hai môi trường, các bước nghiệm thu và giới hạn preview hiện tại](../webcam/webcam_cpu_gpu_goals_20261011.md). Chưa đổi ứng dụng; `runtime.preview_fps` hiện là10, nên FPS xử lý T4 không phải FPS hiển thị thực.

**Dọn file đã hoàn tất:** xóa7 file tạm/trùng, tổng9.760.086byte (≈9,31MiB). Hai mã benchmark đã được giữ ở thư mục báo cáo; từng nội dung hai ZIP upload đã kiểm bằng bản nguồn giữ lại trước khi xóa. Giữ toàn bộ checkpoint, dataset, cache đánh giá và6.000 số đo. Xem [danh sách và SHA256](cleanup_20261011/completed.json).

## Mục tiêu và phạm vi đã chốt

Ứng dụng live webcam phải vừa đúng vừa nhanh: độ trễ ưu tiên số 1, mAP50/mAP50–95 ưu tiên số 2; F1 phải cân bằng, không đánh đổi nhiều F1 chỉ để tiết kiệm vài ms. Mục tiêu dưới 100 ms mỗi ảnh xử lý; một forward/frame, không dùng R04 cộng ba lượt suy luận. Chưa có con số ngưỡng giảm F1 chấp nhận được. Các kích thước yêu cầu so là 512, 640, 768, 960.

Chỉ dùng checkpoint dự án hiện có để khôi phục/chấm; chưa train mới, chưa quay lại pretrained gốc, chưa đổi alias/config ứng dụng. YOLO26s (R12), RF-DETR và distillation hoãn. Không có cơ sở định lượng để gọi toàn dự án đã đạt bao nhiêu phần trăm; số đếm công việc có bằng chứng bên dưới thay cho ước lượng “hơn 80%”.

## Những việc đã hoàn tất

1. Đọc log campaign R01–R11: có kết quả lịch sử ở 1280, gồm precision, recall và mAP. R07 đứng đầu AP/recall một lượt trong protocol lịch sử, R02 precision cao nhất; không dùng thứ hạng đó để tuyên bố thắng ở640. R04 cộng có AP cao hơn nhờ ba forward và NMS, không phù hợp hướng webcam một lượt.
2. Giữ checkpoint R04 đúng nguồn; khôi phục checkpoint cũ R01 epoch0 từ Kaggle, kiểm SHA256. Compact R04 phục vụ inference đã kiểm từng tensor bằng bản gốc. Không thay một checkpoint khác vào tên R02–R11.
3. Chấm R04 CPU đủ bốn kích thước, và R04/baseline GPU đủ bốn kích thước trên 2.500 validation/8.359 GT mỗi phép. R01 GPU cũng đã chấm đủ bốn kích thước; cache/replay AP parity được kiểm. Baseline là model riêng, không tính thành một trong11.
4. Review tự động toàn bộ12.000train: đọc/giải mã, SHA256, cú pháp nhãn, class, tọa độ/count. Review ảnh mẫu36 và review trước đó:27ảnh xác nhận vấn đề nhãn,6ảnh cấu trúc cần kiểm thêm;33ảnh tạm giữ khỏi tập đề xuất train. Không xóa ảnh gốc, không sửaval/test. Full visual review chưa xong.
5. Kiểm scene/video leakage:17nhóm chung giữa các split,368ảnh (255train/73val/40test). Đây là vấn đề đánh giá cần xử lý, chưa phải bằng chứng model tổng quát tốt.
6. R04 robustness có phép đo trên299ảnh từ15nguồn: clean/dark/blur/JPEG/noise; giữ tách số đo subset với fullval. Confidence sweep từcache đã xong;640 conf0,20 cóF1≈66,918%, nhưng ứng viên vẫnconf0,25 vì chưa đo runtime cấu hình mới và threshold chưa được kiểm độc lập.
7. CPU runtime R04 bốn size và R01/R04 so xen kẽ đã xong. Webcam0 probe không đọc được camera, nên chưa có FPS/latency webcam thực.
8. Mới nhất: mộtTeslaT4, R04/640/batch1, 200ảnh validation từ15nguồn×10vòng×3chế độ =6.000phép đo. CPU/GPU cùngVM;CUDA chỉ thấyGPU0. Đã lưu đầy đủ6.000mẫu, thống kê từngvòng, JSON và screenshot. GPU đã dừng, khôngtrain/chấmtest.

## Kết quả mới nhất về GPU, FPS và webcam ước tính

| Chế độ | Trung vị ms | p95 ms | p99 ms | Max ms | >100ms | FPS xử lý từmean |
|---|---:|---:|---:|---:|---:|---:|
| cpu_fp32 | 71.09 | 79.13 | 84.96 | 108.79 | 7/2000 | 14.5 |
| t4_fp32 | 14.00 | 20.78 | 25.49 | 65.00 | 0/2000 | 67.7 |
| t4_fp16 | 14.97 | 21.39 | 26.36 | 2870.27 | 4/2000 | 47.1 |

FP16 có4lượt≈2,8giây ởvòng1; khôngxóaoutlier, chưaxácđịnhnguyênnhân. Vòng2–10 FP16 đạt≈64FPS,0/1800vượt100ms;FP32≈68,1FPS,0/1800vượt100ms. FP32 đang tốt hơn vềlatency/ổnđịnh trongphépđonày. Đo gồm predict+NMS+vẽ; khôngcapture/tracker/UI/network. Mãbenchmark, nguồncheckpoint và protocol nằm trong [báo cáo T4](../performance/r04_single_t4_latency_20261011/results.md).

Webcamước tính, giả định overheadthêm5/10/20ms: camera30FPS≈30/30/28,8FPS;camera60FPS≈50,6/40,4/28,8FPS. Đây là ba kịch bản, không phải số đo webcam hoặc khoảng tin cậy. Không tích hàng đợi frame là giả định cho công thức, chưa xác nhận ứng dụng thực hiện. Latency đầu-cuối còn phụ thuộc exposure/capture/buffering/presentation. T4 không giả lập RTX5060Laptop8GB; giới hạnVRAM không tái tạo kiến trúc/TensorCore/TGP. Điểm chất lượng không được chấm lại trong benchmarklatency.

R04 GPUFP16 ở640, mộtforward+NMS0,6, conf0,25: **mAP50 71,6033%; mAP50–95 49,5966%; F1 66,4589%**. Không nhầm với khoảng76,9%mAP50 mộtlượt1280 hoặc80,5%ba lượt1280. Baseline640 hơnR04 vềAP95 nhưng kémAP50/F1; không có model thắng mọi metric đã kiểm.

## Đối chiếu 10 tiêu chí

| Tiêu chí | Bằng chứng hiện có | Còn thiếu |
|---|---|---|
| Precision | Fullval ởcácsize, cảmacro evaluator vàmicro conf0,25 | Đủ11checkpoint vàmôi trường mục tiêu |
| Recall | Fullval/cached matching ởcácsize | Đủ11 và lỗi bỏ sót thực tế |
| mAP50 | R01/R04/baseline bốnsize, lịch sử11lượt1280 | 9checkpoint còn thiếu ở512–960 |
| mAP50–95 | Cùngfullvalprotocol, cócacheparity | Sođủ11, kiểm boundingboxtrongcảnh thực |
| F1 | Fixedconf0,25, sweepthreshold từcache | Chốtconf độc lập vàsođủ11 |
| Latency | CPUcácsize, mộtT4 640×2000/chếđộ | Laptop5060 vàpipelinewebcam đầu-cuối |
| FPS | FPS xử lý đo từmean;webcamchỉước tính | FPS camera/UI thật,dropframe/hàngđợi |
| Robustness | Subset299ảnh,5biếnthể | Cảnhmới/chekhuất/video thật vàsođủmodel |
| Tài nguyên | RSSCPU,CUDA allocated/reserved,protocoldevice | VRAM/RAM/CPU/nhiệt/côngsuất trênlaptop chạy lâu |
| Chất lượng từnglớp/nguồn/cảnh | PerclassAP vànguồnsplit/leakage đãkiểm | Reviewnhãntoànbộ,phân tích lỗi video/cảnh độc lập |

## Những việc chưa hoàn tất và thứ tự tiếp theo

1. **Chấm đủR01–R11: hiện2/11model,8/44cặpmodel/size.** R02,R03,R05,R06,R07,R08,R09,R10,R11 thiếucheckpoint. Nguồnlocal/Kaggle đãkiểm chưa cung cấp9file đó. Không dùng metrics1280 thay kết quả640, không tuyên bố đãchạyđủ11. Xem [trạng thái44phép chấm](../webcam/webcam_all11_20261011/status.md) và [kết quả hiệncó](../webcam/webcam_all11_20261011/results.md).
2. Khôi phục checkpoint nếu tìm được backup. Tái train9cấu hình sẽ tạo thí nghiệm mới, không khôi phục đúngcheckpointcũ; chưa được người dùng cho phép. Không tựtrain do yêu cầu dọnfile.
3. Reviewnhãn toàn bộ và kiểm6ảnh ngoài biên;33ảnh tạm giữ,11967ảnh ứng viên chưa được review trực quan hết. Ảnh nền nhãn rỗng cần review nội dung, không mặc định là lỗi. Không xóa dữ liệu gốc.
4. Đối chứng nguồn và nhóm cảnh: kế hoạch/tooling đã có, các phép train/so sánh nhân quả chưa được hoàn tất. Học tiếp từcheckpoint cũ không loại được ảnh hưởng dữ liệu/leakage màcheckpoint đã học. Không gọi dữ liệu là độc lập trước khi giải quyết vấn đề này.
5. Freeze/unfreeze, xác nhận lặp và học lâu hơn đều chưa triển khai thành chuỗi train mới trong phạm vi hiện tại. Distillation cầnteacher/phương án phù hợp;RF-DETR vàYOLO26s đãhoãn.
6. Đo trực tiếp laptopRTX5060 vàwebcam30/60FPS: capture→predict→tracker→UI,timestamp/dropframe/hàngđợi, chạy lâu vànguồnđiện. Hiện localtorchCPU-only/nvidia-smi không được tìm thấy trongPATH, chưa xác nhận cóGPU khảdụng trênmáy đang chạy.
7. Chấm đủcácmodel cóthểkhôiphục vàchọn theo latency+AP+F1; chỉ sau đó mới chốt cấu hình ứng dụng. R04/640/FP32/NMS0,6/conf0,25 là hướng GPU tạmthời, chưanghiệmthu/đổiapp.
8. Kiểmđịnh cuối trênsplit độc lập saukhi chốtprotocol;500test vẫn được giữriêng,chưachấm. KaggleQuickSave/versionmới chưađược xácnhận; localJSON vàscreenshot làbằngchứng đãgiữ.

## File và nguồn cần giữ

- Checkpoint R04: `weights/campaign_20261010/BEST_Project8_v05.pt`; R01: `weights/recovered_20261010/R01_epoch0.pt`. Không xóa checkpoint,compactcopy,dataset hoặc validationcache trong lần dọn này.
- [Log đầy đủ](project8_v05_log.md), [báo cáo bốnsize](../webcam/webcam_sizes_20261010/results.md), [benchmarkT4/FPS/webcamước tính](../performance/r04_single_t4_latency_20261011/results.md), JSON sốđo và ảnh xácnhận.
- Mãbenchmark đãchuyển từ.temp vào `r04_single_t4_latency_20261011/scripts/`; có hướngdẫn chạyselectedcell, khôngRunAll.
- Dọn chỉ bảnZIPupload trùng đãso từngmember với filegiữlại, bản mãbenchmark tạm sau khi copy/hashkhớp, hai generator báo cáo dùngmộtlần và bytecodecache.temp. Danhsách/hashes vàkếtquả nằm trong `cleanup_20261011/`. Không đụng các thayđổi/xóa file cósẵn tronggitstatus.
