# E3 gốc: confidence, ảnh cắt và augmentation từng nhóm

Người dùng yêu cầu làm tất cả các hướng còn lại, trì hoãn sửa nhãn và loại bỏ model lớn hơn. Ngày14/10 xác nhận chọn E3gốc, không kế thừa E3augmentation mới.

Nguồn E3original bestEpoch10, AdamWLR0.0001, SHA2569474d7f2b4c869e2101cf9f4f64b0f7f0411b859a8d31d3b2425aba48f479a13. Giữ datasetv05:12000train,2500val,8359valboxes;500test không sử dụng; nhãn giữ nguyên, chưareview. Không đổi modeldefaultStreamlit hoặc publishảnh.

## Phép thử

1. Chấm native E3gốc1280,batch8,device0,halfFalse,rectTrue,conf.001,nmsFalse,maxdet300.
2. Suy luận singleimage1280 trên2500val; lưu dự đoán raw tạiconf.001. Cắt512/768px,chồng25%,kèm dự đoán toànảnh để giữ vật thể lớn. Ghép theo lớp bằngNMS IoU.6/.7/.8; đối chứng fullimage có/không cùng bướcNMS. YOLO26native end2end giữ nguyên; đây làNMS bổ sung để ghép tile, không nhầm vớiNMShead hoặcIoU chấmmAP.
3. Confidence.01,.03,.05,.1,.15,.2,.25,.3,.4,.5: rematchGT ởIoU.5 tại từngngưỡng. BáoP/R/F1macro,micro,TP/FP/FN từnglớp. AP50/AP50–95 dùng toàn bộ dự đoán từfloor.001, không báoAP tănggiả vìlọcconfidence. Ghi độtrễmean/p95 sauwarmup,FP32. Nativebatch8 vàsingleimage được tách rõ, không trộnP/R tạingưỡngcốđịnh vớioperatingpointvalidator.
4. Nămrun độc lập từE3gốc, mỗi7epoch1280,batch32,device0,1T4x2,AdamWlr.0001,patience0,resumeFalse,seed42. CONTROL giữbase HSV.015/.7/.4,fliplr.5,translate.1,scale.5,mosaic.5close2. Tất cả dùng danh sáchAlbumentations rõràng, danh sáchrỗng chođốichứng/hìnhhọc; transforms=[] khôngdùngdefaults trong8.4.165. LIGHT_CONTRAST thêmbrightness±.15,contrast±.2p.30,gamma85–115p.15,CLAHEp.05; GEOMETRY chỉthêmdegrees8,shear1,perspective.0001; GRAY chỉthêmToGrayp.15; INVERT chỉthêmInvertImgp.05. Không chain trọngsố giữa cácrun.

Run tuần tự,cap60mintraining/run và5hcampaign kể từpreflighttrọngsố, tốiđa35epoch mới; khôngOOMretry,khôngbatchchange. Đây là lựa chọntriểnkhai cho yêu cầu mới, không kếthừa giới hạn2.5h của phiên7epoch cũ. UI phải xácnhận khôngjobactive trướcSubmit. Ghi logPROGRESS_FULL từresults.csv mỗi epoch, nativebest/last táchtrainerfinal, checksum ZIP vàbest/last.

Kết quả chỉcho phươngpháp trênnhãnchưareview. Ưutiên giảmFN/recallvàF1, kèmprecision vàlatency; chưachọn mộtprecisionfloor thayngười dùng. Xuất tradeofffrontier, không tựpromote model.

## Trạng thái chuẩn bị

Notebooklocal notebooks/kaggle/e3_original_improvement_20261014.ipynb kiểmcúpháp5codecells. Ba kiểmtra evaluator đãpass: phủmépảnh,mergetheolớp,confidenceTP/FP/FN. Kaggle14/10:Version7Successful,0ActiveEvents,Draftoff,T4x2,Interneton; notebook mới đãImport, chưaSubmit.

## Đính chính kết quả phiên cũ

Tronglần báo trước đãlấy hàng cuối last.pt làm sốaftertrain chung. Codephiêncũ chấm best.pt trước,last.pt sau. Hàngnativebestquan sát3071.5s:P.784R.683AP50.756AP95.456; hàngnativelast3119.7s:P.800R.675AP50.762AP95.450. Cần thu hồiJSON để lấy số chínhxác vàF1macro; không dùng phiênaugmentationmới làmnguồn.

Version8 script358107241 failed sau104.7s tại preflightNoOp(A.DualTransform),0epoch. Đã sửa dùng list rỗng; kiểmtra regression đủ5nhóm và3evaluatorchecks tổng4tests PASS vớiUltralytics8.4.165/Albumentations2.0.8. Thưviệntest cài vào visualization/e3-test-deps,khôngđổivenvStreamlit. Đangimportbảnsửa,khôngjobactive.

Version9 script358109182 đã xác nhận Running nền Save & Run All lúc02:55ICT14/10, một Active Event, T4x2. Log CONFIG_VERIFIED và ARCHIVE_READY; chưa xác nhận epoch. Notebook SHA58c80e83b28793e21426db6446ab7d3295c0aaa37a34cdf3e45701b1f7ea53dc. Giữ một tab viewer, không launch lại.

Người dùng yêu cầu bổ sung thời gian đã chạy và còn lại. Ước tính ban đầu còn4–5h dựa phiên7epoch trước52m10s; chưa đo tốc độ tile nên không cam kết đủ5run trước budget5h. Khi có log, dùng tốc độ thực tế để cập nhật ETA run và toàn bộ, báo theo khoảng và phân biệt budget với thời gian hoàn tất. Người dùng có thể tắt máy vì đây là phiên nền Kaggle; monitor Codex local không được hứa chạy khi máy tắt.

## Trạng thái cuối Version 9 và sửa cảnh báo

Kaggle Logs xác nhận Version9 hoàn tất thành công sau 11795.1 giây (3 giờ 16 phút 35 giây), dùng GPU T4 x2, output 207.14 MB. Kiểm tra giao diện hiện tại cho thấy Draft Session off và 0 Active Events; hiện không có run nào đang chạy. Chưa tải outputs theo lựa chọn trước đó của người dùng, nên chưa xác minh được epoch cuối theo từng run hoặc các chỉ số chi tiết từ artifacts.

Cảnh báo lặp lại là deprecation warning của tham số suy luận `half=False`, không phải lỗi train. Ultralytics 8.4.165 ánh xạ cờ cũ sang chế độ FP32; `quantize=32` là cách tường minh mới để giữ FP32. Đã thay lời gọi trong evaluator và nguồn dựng notebook, rồi tạo lại `notebooks/kaggle/e3_original_improvement_20261014.ipynb`; kiểm tra tĩnh xác nhận notebook sinh ra không còn `half=False`/`halfFalse`.

Cấu hình chiến dịch Version9: E3 original best, SHA256 `9474d7f2b4c869e2101cf9f4f64b0f7f0411b859a8d31d3b2425aba48f479a13`; 5 run độc lập tuần tự CONTROL/LIGHT_CONTRAST/GEOMETRY/GRAY/INVERT, mỗi run 7 epoch, `imgsz=1280`, `batch=32`, `device=0,1` (T4 x2), AdamW `lr0=0.0001`, seed42, patience0, resume=False. Giới hạn là 60 phút train/run và 5 giờ tổng chiến dịch. Dữ liệu train/val v0.5 gồm 12,000/2,500 ảnh và 8,359 val boxes; test500 không dùng, nhãn chưa review. CONTROL giữ HSV .015/.7/.4, fliplr .5, translate .1, scale .5, mosaic .5 và close_mosaic=2. LIGHT_CONTRAST thêm brightness±.15/contrast±.20 (p=.30), gamma85–115 (p=.15), CLAHE(p=.05); GEOMETRY thêm rotate8°, shear1°, perspective .0001; GRAY ToGray(p=.15); INVERT InvertImg(p=.05). Mọi chỉ số val phải được diễn giải như thử phương pháp trên nhãn chưa review.

## Chạy lại Version 10

Người dùng nhắc cần chạy thật thay vì chỉ đọc log. Trước khi submit, Active Events xác nhận 0; đã sửa 5 chỗ `half=False` trong Kaggle editor thành `quantize=32`, tìm kiếm xác nhận đúng 5 kết quả và 0 kết quả cũ; sửa comment thành 5 independent method runs. Kaggle Settings hiển thị GPU T4 x2 đang được chọn; Internet đang bật. Ngày14/10/2026 lúc14:12 ICT đã submit **E3 Original 5 ablations 7e 1280 B32 T4x2 FP32** dưới Version10. Active Events hiện1; phiên Version10 được UI ghi `Running`, accelerator GPU T4 x2. Chưa có epoch/progress log xác nhận ở lần kiểm tra vừa submit; cần đợi CONFIG/INITIAL/epoch log trước khi báo training epoch thực tế.

Version10 vẫn dùng E3 AdamW lr0.0001 SHA256 `9474d7f2b4c869e2101cf9f4f64b0f7f0411b859a8d31d3b2425aba48f479a13`. Các run: CONTROL, LIGHT_CONTRAST, GEOMETRY, GRAY, INVERT, 7 epoch mỗi run; 1280, batch tổng32, GPU0,1 hai T4, seed42, patience0, resume=False; cap 60 phút mỗi run, ngân sách chiến dịch5 giờ, không tự retry OOM/đổi batch/thêm epoch. Version9 chạy xong trong11795.1s (~3h16m); ETA lần này chỉ khoảng3–5h nếu hạ tầng tương tự, không phải bảo đảm. Vẫn chưa tải artifacts theo lựa chọn trước; hiện không có metrics kết quả mới.

Kiểm tra 14:16 ICT: Version10 còn Running, một Active Event, GPU T4x2. Kaggle log đã tiến từ `INFERENCE_PROGRESS 700 / 2500` ở phút3 lên `1700 / 2500` ở phút5; đây là đánh giá suy luận trước training, nên số epoch train hoàn tất hiện là0/35. Chưa có metrics training; ETA thô còn khoảng3h12m–4h55m, lấy thời lượng Version9 là3h16m trừ5 phút đã chạy và mức trần ngân sách5h làm căn cứ, không phải cam kết. Cập nhật lại khi epoch1 bắt đầu và có tốc độ thực đo.

Kiểm tra trực tiếp log Version10 `scriptVersionId=358249763` lúc14:22 ICT: UI báo `Running for 684.5s`, một Active Event, GPU T4x2, Output 0 B. Đây chưa phải trạng thái lỗi; log không có traceback, CUDA OOM, NaN loss hoặc lỗi fatal. Có debugger warning về frozen Python modules (chỉ có thể làm debugger bỏ lỡ breakpoint; log ghi debugging vẫn tiếp tục). Dataset đã được giải nén/hash/label-check, preflight qua, 2,500 ảnh val baseline và thử tile 512/768 đã xong. Log ghi `TRAIN_LAUNCHED E3_AdamW_lr1e4_CONTROL_1280_E7_B32 PID 100 ...` ở elapsed600.3s; lúc quan sát elapsed684.5s chưa có dòng epoch hoàn tất. Vì vậy đây là lúc CONTROL vừa vào khởi tạo/train; training tiến độ xác nhận vẫn0/35 epoch, dù tiến trình Kaggle còn chạy. Output 0 B trong lúc job chạy chưa đủ để kết luận thất bại.

Metrics đã xuất trong log đều là **baseline native/inference trước train**, không phải kết quả các run. Dòng native validator `all` ở conf mặc định .001: P.798, R.697, mAP50.772, mAP50–95.557. Inference evaluator thêm macro F1 và độ trễ; toàn ảnh không merge: mAP50.79143, mAP50–95.56817, macro P.82450, macro R.71515, macro F1.76328, mean18.2ms, p95 20.3ms. Toàn ảnh merge IoU .6/.7/.8 lần lượt có (P,R,mAP50,mAP50–95,F1macro): (.82651,.71744,.79550,.56554,.76537), (.82469,.71515,.79166,.56821,.76334), (.82450,.71515,.79144,.56817,.76328). Tile512 cùng IoU .6/.7/.8: (.71702,.66489,.72245,.50841,.68607), (.70492,.64805,.70414,.50312,.67076), (.62701,.64717,.66731,.48416,.62402). Tile768 cùng IoU .6/.7/.8: (.77280,.69015,.75724,.53959,.72369), (.76407,.68688,.74762,.53788,.71696), (.74685,.68658,.73836,.53342,.70694). Đây là nhãn v0.5 chưa review; chỉ dùng thử phương pháp.

Tốc độ train/epoch chưa đo vì chưa có epoch hoàn tất. ETA còn khoảng3.1–4.8 giờ theo thời lượng Version9 và ngân sách5h; đây là khoảng ước lượng chưa được epoch hiện tại hiệu chỉnh, không phải cam kết.
