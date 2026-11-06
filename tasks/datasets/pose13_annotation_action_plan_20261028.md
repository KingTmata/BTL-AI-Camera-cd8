# Kế hoạch gán bbox + pose13 và nhận dạng hành vi

Ngày: 28/10/2026. Yêu cầu chính thức vẫn nằm ở [plan.md](../plan.md); tài liệu này mô tả cách triển khai, không thay mục tiêu nghiên cứu.

Kế hoạch mở rộng theo yêu cầu02/11 về một web demo, Pose/tracking và nhãn hành vi theo chuỗi thời gian nằm tại [unified_demo_action_plan_20261102.md](../video/unified_demo_action_plan_20261102.md). Đây là thiết kế đề xuất, chưa triển khai; các kết quả đã chạy bên dưới giữ phạm vi và ngày của chúng.

**Goal:** Hôm nay mở lại Label Studio và backend L để sửa nhãn ảnh; sau này chuyển dữ liệu đã duyệt thành dataset và nghiên cứu hành vi theo thời gian.

**Architecture:** Label Studio 8080 lưu ảnh/annotation. Backend 9090 chạy YOLO26l Pose pretrained và chuyển 17 điểm sang head13 để tạo prediction. Người sửa và Submit trước khi đưa annotation vào dataset. Nhận dạng hành vi là giai đoạn riêng, không tự bật khi chạy backend.

**Tech stack:** Label Studio 1.23.2, SDK 2.1.2, Ultralytics 8.4.183, PyTorch CPU hiện có; FFmpeg/ffprobe trên Kaggle cho tách video sau này.

**Spec:** Các quyết định và hợp đồng trong chính tài liệu này, dựa trên yêu cầu hội thoại đã duyệt. Người dùng yêu cầu triển khai ngay phần gán nhãn; phần hành vi chỉ lập kế hoạch. Thực hiện trực tiếp trên bộ cài hiện có, không tạo worktree cho dữ liệu runtime.

## 1. Phạm vi và các phương án

- Chọn hôm nay: giữ model pretrained L17, chuyển đầu ra sang13 trong backend. Không train hoặc thay checkpoint gốc.
- Phương án train model13: để sau, cần annotation13 đúng và bộ test độc lập; đầu dự đoán phải đổi kích thước. Không thể đổi tên file weights17 thành13.
- Phương án tiếp tục gán17: không dùng cho project13 vì người dùng chỉ muốn một đầu, không mũi/mắt/tai riêng.
- Mục tiêu train hiện tại là Detect: xuất bbox person. Pose13 vẫn lưu để kiểm tra và dùng về sau.
- Hai video Downloads: Thành.MOV (1920x1080,1429 frame, khoảng47,65s) và Thành (1).MOV (989 frame,khoảng32,98s). Chưa tách frame trong phiên thoại. Người dùng chọn tách trên Kaggle, không tự trích tất cả frame local hôm nay.
- Không tự Submit nhãn AI thành GT, không ghi đè annotation người đã sửa, không gọi train từ webhook.
- Giữ bộ gán17 và dữ liệu lịch sử. Bản chuyển máy ngoài Downloads là bản độc lập; sửa local không đồng nghĩa đã phát hành bản đó.

## 2. Hợp đồng pose13 và chuyển đổi

Thứ tự cố định (trái/phải theo cơ thể người, không theo người nhìn ảnh):

| Index | Nhãn | Nguồn COCO17 |
|---|---|---|
|0|head|gộp điểm mặt0..4|
|1,2|left_shoulder,right_shoulder|5,6|
|3,4|left_elbow,right_elbow|7,8|
|5,6|left_wrist,right_wrist|9,10|
|7,8|left_hip,right_hip|11,12|
|9,10|left_knee,right_knee|13,14|
|11,12|left_ankle,right_ankle|15,16|

- Head gợi ý: median x/y các điểm mặt có score>=0,5; nếu không có thì median cả5. Score head là trung bình score các điểm được chọn. Đây là heuristic, phải sửa về vị trí tâm đầu trong ảnh khi lệch; đầu nghiêng/quay lưng cần review riêng. Các lần gán mới dùng cùng quy ước tâm đầu.
- Lưu raw17 và score riêng để truy vết; JSON hiển thị chỉ có head13. Đầu đã gộp không thể khôi phục chính xác5 điểm mặt cũ từ JSON13.
- Hiện điểm score>=0,05; ẩn dưới0,05. Giữ các điểm chồng nhau. Đây là ngưỡng gợi ý, không phải visibility GT.
- Mỗi người có1 bbox, tối đa13 điểm; parentID liên kết điểm với bbox. Không hứa luôn đủ13 điểm khi có ngưỡng. Người sửa bổ sung điểm có thể xác định, không bịa điểm ngoài ảnh hoặc bị che hoàn toàn.
- Bbox conf0,25, NMS IoU0,55, imgsz1280; không đổi ngưỡng bbox thành5% chỉ vì ngưỡng điểm5%.
- Vai trái–vai phải–hông phải–hông trái–vai trái tạo khung thân. Mỗi tay nối vai–khuỷu–cổ tay; mỗi chân nối hông–gối–cổ chân. Đầu nối hai vai theo lựa chọn hiện hành. Đủ13 điểm có14 cạnh.
- Chỉ nối các điểm còn hiện, cùng người. Skeleton là các đoạn thẳng không mũi tên, không nối xuyên sang người khác. Renderer local đã sửa cho Label Studio1.23.2; nâng cấp phải kiểm lại.

## 3. Quy trình vẽ và duyệt nhãn hôm nay

1. Mở project9 “Pose 13 điểm - đầu và thân”. Giữ12 task ảnh đang có và annotation cũ.
2. Kết nối backend9090. Chạy thử1 ảnh bằng yêu cầu predict thật; kiểm response chứa bbox,13 tên khớp và relation đúng person. Không rerun cả bộ nếu prediction cũ đã đúng.
3. Với ảnh mới: upload ảnh gốc; yêu cầu prediction từ backend rồi mở task để sửa. Không upload ảnh đã vẽ skeleton làm ảnh train.
4. Kiểm bbox trước: bổ sung người thiếu, xóa box giả/trùng, sửa box bao người nhìn thấy theo cùng quy tắc. Không tự mở rộng bbox cho phần cơ thể không có bằng chứng.
5. Kiểm từng người: head, vai, khuỷu,cổ tay,hông,gối,cổ chân. Trái/phải theo giải phẫu; phóng to khi đông người. Kéo từng điểm về đúng vị trí, tránh chỉ kéo đường nối.
6. Kiểm parentID và cạnh sau sửa: thêm/đổi điểm không mặc định có cạnh mới. Nếu UI không tự cập nhật, dùng công cụ relation hoặc bộ dựng lại cạnh theo13 tên trong cùng person; không nối bằng khoảng cách gần nhất.
7. Điểm bị che: giữ quy ước absent/occluded/visible riêng. Nếu chưa cấu hình được trạng thái trong UI thì đánh dấu cần review; không suy v=2 từ score cao. Head13 hiện tại chưa có quy trình visibility hoàn chỉnh để tự xuất train Pose.
8. Submit sau khi sửa cả ảnh. Chưa gán xong không đánh dấu là ảnh nền. Nhãn AI chưa duyệt vẫn ở predictions.
9. Xuất JSON gốc để giữ provenance, relation và lịch sử. Preview bbox+skeleton chỉ là ảnh xem kiểm tra.

## 4. Tách video trên Kaggle (giai đoạn tiếp theo)

- Quyết định đã xác nhận ngày28/10: upload cảhai video Thành lên Kaggle, tách đều **10 ảnh/s**, tải ảnh về máy rồi gán cảbbox+pose13 trong Label Studio local. Không import nguyênvideo hoặc tách video người dùng trên máy này trong lượt hiện tại.
- FFmpeg/ffprobe đọc video gốc. Notebook hiện có chỉ triển khai lấy đều10ảnh/s. Hỗ trợ2,3,6,8,10 ảnh/s theo từng đoạn là mở rộng sau này; không nói là đã có chọn đoạn tự động.
- Cấu hình đoạn có video_id,start_s,end_s,sample_fps; khoảng thời gian dạng[start,end), không chồng đoạn. Đoạn cần dày hơn do người chọn, không giả vờ hệ thống đã tự phát hiện đoạn khó.
- Lấy frame nguồn gần mốc lấy mẫu, khử trùng bằng video_hash+source_frame_index; không tạo frame bằng nội suy hình ảnh. Giữ1920x1080 và hướng xoay hiển thị đúng; transform xoay phải áp dụng cả tọa độ nhãn.
- Manifest bắt buộc: image_id,filename,source_video_sha256,source_frame_index (0-based),source_pts,time_base,timestamp_seconds,width,height,sampling_fps,segment_id. Ghi source PTS trước bước fps/setpts có thể sửa timestamp.
- ffprobe tạo bảng PTS và FFmpeg trích đúng source frame đã chọn; số JPG phải khớp manifest. Không suy thời gian bằng index/30; video nguồn gần29,987FPS và có thể VFR.
- Task Label Studio giữ metadata trong data; tên ảnh duy nhất giữa hai video. Nếu chỉ upload ảnh, giữ manifest ngoài và join bằng image_id/hash, không dựa task order.
- Nhãn video cũ: đọc sequence,keyframe,enabled,trackID. Chỉ nội suy bbox trong khoảng track hoạt động, cùng cảnh và có hai đầu phù hợp; ghi provenance interpolated để người duyệt. Không tự suy pose từ bbox.
- Kiểm: frame đầu/cuối, mốc đổi6/8/10FPS, khoảng trống track, video xoay, timestamp không đều, upload đổi tên. Tổng số ảnh là kết quả chọn mẫu thực tế, không cam kết bằng duration*fps làm tròn.

## 5. Bộ chuyển JSON sang YOLO (sau khi có annotation duyệt)

Một CLI dự kiến `scripts/datasets/label_studio_to_yolo.py` nhận JSON,ảnh gốc,manifest và chế độ detect/pose13. Một test file tương ứng, không tách nhiều helper dư thừa.

- Detect: mỗi ảnh cóTXT cùng stem; mỗi người1 dòng `0 cx cy w h`, bbox từ phần trăm LS/100, tâm=(x+w/2,y+h/2). FileYOLO dùng tọa độ0..1.
- Pose13: mỗi người1 dòng `0 cx cy w h x0 y0 v0 ... x12 y12 v12` (44 giá trị). `kpt_shape: [13,3]`, `flip_idx: [0,2,1,4,3,6,5,8,7,10,9,12,11]`. Thứ tự ở mục2.
- Visibility:0 không có nhãn,1 có nhãn nhưng che,2 có nhãn nhìn thấy. Điểm thiếu xuất0,0,0; không chuyển confidence0,05 thành visibility. Từ chối xuất Pose nếu dữ liệu không đủ phân biệt trạng thái theo hợp đồng đã chốt.
- Chỉ lấy annotation đã duyệt, không tự fallback predictions. Nhiều annotation khác nhau phải chọn bản được duyệt rõ ràng. Bbox xoay/nhãn lạ/điểm trùng tên trong cùng người phải báo lỗi có taskID; không âm thầm lấy một bản.
- Liên kết bằng parentID; điểm mồ côi báo lỗi. Relation không xuất vàoTXT; skeleton là cấu hình hiển thị. Bbox và điểm phải dùng cùng ảnh/kích thước.
- Có báo cáo ảnh thiếu/không duyệt,người/khớp thiếu,ngoài biên,NaN,stem trùng. EmptyTXT chỉ cho ảnh nền đã duyệt.
- Chia train/val/test theo buổi/video, không random frame sát nhau. Hai video là pilot, chưa đủ bằng chứng tổng quát; ưu tiên thêm buổi quay độc lập trước công bố metric.
- Test trước code: bbox mẫu biết đáp án; nhiều người; thiếu điểm; sai parent;13 tên/44 cột;flip2lần;annotation vs prediction;trùngstem;vượtbiên;leak video;roundtrip vẽ overlay khớp ảnh gốc. Dùng loader YOLO kiểm shape mà không chạy train trái yêu cầu.

## 6. Nhận dạng hành vi — kế hoạch nghiên cứu để làm sau

Đầu vào là chuỗi có thời gian: video→person/pose→track ID cục bộ→đặc trưng theo thời gian→ứng viên hành vi→review/đánh giá. Task ảnh độc lập không tự cung cấp track ID ổn định.

### A. Định nghĩa và nhãn sự kiện

- Bám các hành vi chính thức: giơ tay,đứng lên,di chuyển lên bảng,viết bảng. Lưu actor_track_id,video_id,start_s,end_s,label,evidence,uncertain,reviewer; cho phép đồng thời nhiều hành vi.
- Giơ tay: bằng chứng cổ tay/khuỷu/vai và diễn tiến; phân biệt chạm đầu,vươn vai,che khuất. Không dùng duy nhất wrist_y<shoulder_y trên1frame để kết luận.
- Đứng lên: chuyển tư thế theo thời gian,thân/hông/đầu và bbox; camera nhìn xiên/che chân cần trạng thái không chắc chắn.
- Di chuyển lên bảng: quỹ đạo người và vùng bảng/đường đi do người định nghĩa theo camera; chỉ thấy người di chuyển không đủ kết luận lên bảng.
- Viết bảng: cần ngữ cảnh gần bảng và bằng chứng tay/ảnh; pose13 đơn lẻ thường không đủ phân biệt chỉ bảng,đứng cạnh bảng hoặc viết. Dùng ảnh crop/đối tượng bảng bổ sung khi cần.
- Không suy ngủ từ cúi đầu1–3s; không suy mức năng nổ từ một sự kiện. Ranh giới sự kiện mơ hồ ghi uncertainty, không ép nhãn chắc chắn.

### B. Baseline trước model chuỗi

- Tách track khi đổi cảnh; đo ID switch/mất track/che khuất, không nối người chỉ vì bbox gần nhất. Chuẩn hóa tọa độ theo tâm/thân,giữmask thiếu vàtimestamp; không thay missing bằng0 mà bỏmask.
- Baseline quy tắc có hysteresis và thời lượng tối thiểu được hiệu chỉnh trên development set. Ghi thời điểm bắt đầu,kết thúc,độ tin cậy,lý do và nguồn frame để kiểm chứng.
- Không chốt trước ST-GCN/PoseC3D/RTMPose/Jev; chỉ thử một model chuỗi khi baseline cho thấy lỗi có hệ thống và có dữ liệu đủ. So với quy tắc trên cùng test, cùng budget.
- 3FPS có thể phù hợp gán Detect nhưng chưa chắc đủ bắt động tác tay nhanh; đánh giá riêng2/3/6/8/10FPS trên các đoạn đã gán thời gian. Không coi tăng frame là tăng số mẫu độc lập.

### C. Kiểm chứng và tổng hợp

- Pilot nhãn chéo ít nhất2người cho một phần dữ liệu; xử lý bất đồng trước đánh giá. Số lượng pilot do nguồn lực quyết định, không coi số ít video là đủ.
- Tách buổi/lớp/camera giữa train/dev/test khi dữ liệu cho phép. Không dùng test chọn threshold,không dùng prediction tự sinh làmGT.
- Báo bbox precision/recall;pose sai số vị trí theo scale/thân và missing;tracking IDF1/ID switches;hành vi precision/recall/F1 từng lớp và temporal IoU,đếm sự kiện/thời lượng sai lệch. Nêu protocol/tolerance trước chạy test.
- Đo coldstart,latency xử lý p50/p95,RAM/VRAM riêng theo thiết bị; dưới3s là mục tiêu xử lý, không giới hạn lịch sử quan sát3s. Chưa có benchmark thì không khẳng định đạt.
- Tổng hợp tần suất/thời lượng và mốc video làm thông tin tham khảo cho giảng viên; giữ uncertain và khả năng xem bằng chứng,không tự chấm mức tương tác thành sự thật.

## 7. Các bước thực hiện và nghiệm thu hôm nay

- [x] Snapshot SQLite trước cấu hình; ghi số task/annotation để đảm bảo không mất dữ liệu.
- [x] Kiểm adapter pose13 bằng test threshold0,05/overlap/head/parent/14edges; quan sát lỗi trước khi cài vào backend local.
- [x] Cài adapter vào `tools/label_studio/work/label-studio-ml-backend/label_studio_ml/examples/yolo/control_models/pose13.py`; đăng ký trong model.py, giữ VideoRectangle.
- [x] Bật Studio8080 và backend9090 loopback bằng venv hiện có; MODEL_ROOT trỏ weights gốc,YOLO_POSE_MODEL=L; không nạpX hoặc train.
- [x] TestHTTPpredict một task ảnh thật, so số box/điểm/cạnh,parentID và tên13. Health200 không thay cho predict thật.
- [x] Gắn backend vào project9; bật gợi ý,giữ prediction và annotation đã có. Ảnh mới yêu cầu backend chạy khi cần.
- [ ] Mở giao diện project9; kiểm skeleton đoạn thẳng,không mũi tên,di chuyển điểm và lưu nháp/Submit theo thao tác người dùng. Nếu không kiểm được thao tác lưu phải ghi giới hạn.
- [x] Ghi kết quả thực tế vào cuối tài liệu và project_memory; dọn file tạm phát sinh hôm nay, giữ dữ liệu/prediction/raw dùng truy vết.

## Kết quả thực tế

- Notebook: [thanh_frames_10fps_label_studio.ipynb](../../notebooks/kaggle/thanh_frames_10fps_label_studio.ipynb). Đã kiểm JSON/cú pháp và sampling CFR/VFR; đã chạy FFmpeg/ffprobe trên video tổng hợp30frame/1s→10JPG, đúng source indices,PTS,kích thước. Chưa thực thi trên Kaggle hoặc hai video người dùng. Notebook xuất ZIP ảnh+manifest+XML, không xuất nhãn AI/TXT/train.
- Thực hiện Kaggle: tạo Dataset chứa haiMOV gốc→mở notebook bằng Import Notebook→Add Input dataset đó→Run All→tải ZIP từOutput. Nếu Kaggle đổi tên file, điền đúng2đường dẫn trongVIDEO_PATHS; notebook dừng và liệt kê file thay vì đoán.
- Backend L đã chạyHTTPpredict task22:200,16box/162điểm/177cạnh,19,45s cold-first-request trong lượtkiểm. Không coi đây là benchmark tốc độ ổn định hay metric chất lượng. Kết quả chỉlưu riêng ở `runs/label_studio_pose_20261028/`,không ghi prediction vào DB.
- Đã sửa lỗi SDK2.1.2: labels_attrs trả LabelTag,đọc model_index qua attr. Khóa nạp/suy luận model dùng chung để tránh request song song sửa predictor. Đã bỏ eager-load modelvideoX khiimport; nhánhvideo chỉnạp khi có requestvideo.
- 8test puremapping qua; kiểm adapter với SDKLabelInterface thật qua; test renderer hiệncài qua: center-to-center,không mũitên,relationrectanglegiữnguyên. Chưa kiểm kéođiểm/lưu trên UI vì tab đang yêu cầu đăngnhập.
- Đối chiếu SQLite trước/sau:toànbộ29task,3annotation,28prediction giữ nguyên từngrow. Project9 có12task,0annotation;không tựSubmit.
- Kết nối project9 đang chờ giải quyết cờML_BLOCK_LOCAL_IP: API từchối127.0.0.1. Auto-review banđầu từchối cách đặtfalse vìcho phép mọiURLMLmạngnộibộ; đãhoàntác rồi hỏi rõ. Người dùng sauđó cho phép chỉbộcài local này; đãápdụng ML_BLOCK_LOCAL_IP=false và khởiđộng lạiStudio loopback. Webhook vẫn bịbảo vệ riêng,không nới cờURLtoàncục;đãxácnhận quaAPI backend8 trạngtháiCO/Connected,modelversion yolo26l-pose-head13-point0.05. Khôngtạo webhook gửi nhãn tựđộng;backendkhôngtrain.
- Bộchuyển JSON→YOLO TXT và pipelinehànhvi là kếhoạch,chưa viết/chạy. 13điểm JSON đangcó khôngcần chuyển17lầnnữa;chỉcần đổi serialization nếu xuất TXT và chốtvisibilitytrướcxuấtPose.
- Lượt tiếp: theo xác nhận người dùng, đã tạo Dataset Kaggle **Private** chứa hai video, notebook cũng **Private**. Kaggle đổi tên thành `Thnh.MOV`/`Thnh (1).MOV`; đã sửa notebook theo tên thực tế và chạy: 477+330=807 ảnh. Đã tải ZIP, kiểm hash tất cả ảnh và hash hai video khớp nguồn; đã import807 task ảnh vào project9 với image_id,hash,frame,PTS,time_base và timestamp.
- Test nhãn đầu tiên task30: 4bbox/47điểm/51relation; UI đã thấy vùng person và pose. Do project đang lọc phiên bản prediction cũ, đã chuyển model_version của project sang rỗng; kiểm serializer cho thấy prediction cũ và mới đều hiện. Chưa Submit; tiến trình backend đang xử lý các ảnh còn lại, theo dõi `tools/label_studio/outputs/thanh_10fps_20261028/prediction_progress.json` và số Predictions trong project. Không coi tiến trình đã chạy xong khi state chưa complete.

## Cập nhật sau khi duyệt và chuẩn bị Kaggle

- Batch dự đoán đã hoàn thành 807/807 ảnh, 0 lỗi. Người dùng xác nhận duyệt toàn bộ 807 ảnh hai video và yêu cầu lưu prediction thành annotation; đã lưu đúng 807 annotation, giữ các prediction và không đụng 12 ảnh cũ. Export JSON giữ bbox, pose13 và relations; manifest giữ frame, PTS và timestamp nguồn.
- Chia video ngẫu nhiên theo yêu cầu: `Thành (1).MOV` cho 330 ảnh train/1.387 bbox và `Thành.MOV` cho 477 ảnh validation/1.799 bbox. Đây là pilot hai video, không có test độc lập; không rút ra kết luận khái quát chỉ từ validation này.
- **Cập nhật 31/10/2026, thay split trên theo yêu cầu người dùng:** `Thành.MOV` làm **train477ảnh/1.799bbox**, `Thành (1).MOV` làm **validation330ảnh/1.387bbox** để train bằng bộ nhiều ảnh hơn. Bộ Detect và notebook local đã cập nhật; bản Kaggle cũ chưa thay đổi trong lượt đổi split. Tiếp tục chia theo video, không random frame và chưa có test độc lập.
- Converter `scripts/datasets/label_studio_to_yolo.py` đã xuất Detect person; 6 test converter và loader Ultralytics đã qua. JSON Pose13 vẫn được giữ, nhưng không mặc định confidence điểm thành visibility, nên chưa có TXT Pose13 sẵn sàng train.
- Dataset Kaggle Private `awm3636/thanh-detect-person-and-pose13-reviewed-20261028` chứa bộ Detect, checkpoint gốc, JSON nhãn và manifest, schema, converter và quyết định chia. Notebook Kaggle Private `awm3636/thanh-detect-person-20e-1280-t4x2` đã nhập, gắn dataset, chọn 20 epoch/imgsz1280/T4x2, batch16 mặc định; chưa thực thi train theo yêu cầu mới nhất của người dùng.
- Checkpoint LIGHT_CONTRAST hiện có là bản inference-only, không thấy adapter LoRA riêng. Người dùng chọn tìm checkpoint LoRA tiếp tục trước, nếu không có thì dùng các trọng số tương thích để train Detect bình thường một lớp. Chưa tìm thấy bản được xác nhận ở các vị trí đã kiểm tra; notebook chuẩn bị phương án dự phòng, giữ nguyên checkpoint 8 lớp. Chưa có metric hoặc kiểm chứng chạy train trên Kaggle.
