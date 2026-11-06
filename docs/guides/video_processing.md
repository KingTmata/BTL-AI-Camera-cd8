# Video local: web thống nhất và lõi dùng chung

## Điểm mở chính — web video

Bấm đúp **`run_demo.cmd`**, giữ cửa sổ backend đang chạy và mở **http://127.0.0.1:8090**. Máy mới chạy `setup_demo.cmd` trước; môi trường có sẵn cài thêm `requirements/web.txt`. Web xử lý trên máy này: full dùng cùng `run_video()` với CLI, replay dùng lõi `video.online.run_replay()`; không cần Label Studio8080 hoặc backend9090.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/web.txt
.venv\Scripts\python.exe -m classroom_ai.interfaces.web --port 8090
```

Chọn video/file trên máy, chọn **Detect/Pose**, model, thiết bị và ByteTrack rồi bắt đầu. Detect chỉ xuất người trong web này. Pose pretrained tự trả hộp người và17khớp; app giữ raw17, chuyển13khớp đầu/thân, không coi đây là nhãn hành vi. Model N/L là ứng viên để so trên dữ liệu thực, chưa bảo đảm phát hiện đủ người ở xa hoặc mờ.

Web nhận CFR/VFR, giữ PTS thực. **Full** xử lý đủ frame trước khi xem/tua/gán nhãn; tua và bật/tắt hộp/ID/skeleton dùng dữ liệu đã lưu, không suy luận lại. **Replay** chọn Pose lấy mẫu, phát nguồn ghi sẵn theo PTS1× và công bố từng mẫu khi có, không dùng ảnh lấy mẫu để gán GT. Source/weights/nhãn gốc giữ nguyên. Bản xem full có thể mã hóa lại để trình duyệt phát; provenance nằm trong manifest. Một lượt đang chạy; **Dừng** yêu cầu hủy và chờ thao tác AI/codec hiện tại kết thúc. Đọc trạng thái `complete/failed/cancelled`, không suy hoàn tất từ việc trang còn mở hoặc có file partial.

Phần **Gán nhãn theo thời gian** dùng người/đoạn track cục bộ, start/end theo giây nguồn, nhãn tư thế và hành động độc lập. Đứng và giơ tay có thể cùng thời gian. Coverage ghi riêng khoảng đã kiểm cho từng nhãn; không có sự kiện không đồng nghĩa đã xem và xác nhận âm. Ghi ranh giới bị cắt, uncertainty và bằng chứng; chỉ chọn đã duyệt sau khi xem. Gợi ý giơ tay dùng luật chưa hiệu chỉnh, luôn là nháp cần người kiểm; chưa có model tự nhận đủ đứng/ngồi/lên bảng/viết bảng.

### Gán nhãn từ video gốc, không cần chạy AI trước

Ưu tiên hiện hành03/11 là hoàn thiện nhãn hành vi theo chuỗi; **tích hợp CMC để sau**. Registry [classroom_sources.json](../../data/videos/classroom_sources.json) ghi tám nguồn Thành/Đô/ba clip Khánh/Sơn/Phong, hash, metadata và lượt chuẩn bị. Khác tên hoặc khác hash chưa chứng minh khác buổi quay; chưa tự gán train/val/test hành vi. Bộ bbox ảnh đã duyệt giữ nguyên split477train/330val của Thành.

Chọn **Cách xử lý → Chuẩn bị gán nhãn · không chạy AI**, chọn video có sẵn rồi bấm bắt đầu. Hệ thống chuẩn bị MP4 xem được trên browser và ledger nativePTS cho mọi frame; không nạp model, không tạo prediction, người, event hoặc coverage. Sau đó dùng **Nguồn đã chuẩn bị để gán nhãn → Mở nguồn đã chọn** để chuyển video ngay trong tab hiện tại. Mỗi nguồn giữ nhánh bản nhãn riêng. Nguồn chuẩn bị không dùng nút gợi ý quy tắc hay xuất video AI vì chưa có Pose/ID model.

1. Đọc **Hướng dẫn gán nhãn hành vi theo chuỗi thời gian** trên web. Định nghĩa duy nhất ở [action_rubric.yaml](../../configs/action_rubric.yaml), phiên bản `classroom-actions-v1-draft-20261103`; đây là bản nháp để nhóm thống nhất với giảng viên, chưa tự được phê duyệt. Các khớp Pose13 là tọa độ cơ thể, khác các lớp hành vi. Gán hành vi trực tiếp trên diễn biến RGB không bắt buộc phải sửa đủ13khớp cho từng ảnh trước.
2. Tạo người quan sát độc lập: chọn đoạn nguồn và khoảng thật sự nhìn thấy cùng người. Không gán danh tính theo tên video; tên Khánh/Sơn/Phong chỉ nhận diện file. Khi mất quan sát dài hoặc cắt cảnh, dùng khoảng riêng và chỉ nối sau khi người duyệt xác nhận.
3. Tạm dừng ở một frame rõ trong khoảng người đó. Mở **Khung tham chiếu để nhận ra người do nhóm khai báo**, chọn khung trên video bằng hai góc hoặc kéo chuột, ghi mô tả vị trí/áo, rồi **thêm khung vào bản nháp**. Khung gắn đúng frame/index/nativePTS/timebase và SHA nguồn; nó chỉ giúp nhận ra người tại mốc đó, không giả thành tracking liên tục. Mỗi khoảng người của nguồn chuẩn bị cần ít nhất một khung trước khi GT-ready.
4. Xem liên tục cả trước và sau hành động. Gán từng episode với onset/end theo thời gian video, không suy từ ảnh rời hoặc số frame/FPS. “Đứng” là posture; “đứng lên” cần chuỗi ngồi→chuyển→đứng. Cho phép đứng và giơ tay đồng thời. Hiện diện ở bảng chưa chứng minh đi lên bảng hay viết bảng.
5. Ghi **coverage riêng cho từng nhãn** trong khoảng đã xem, kể cả âm đã kiểm. Nội dung hành vi đã rõ nhưng thiếu onset/end thì giữ `uncertain=false` và đánh dấu biên tương ứng bị cắt; dùng `uncertain=true` khi chính hành vi còn chưa chắc. Khoảng chưa xem không phải âm. Ngồi/đứng/chuyển/không xác định tư thế không chồng nhau cho cùng người, còn các hành động độc lập có thể chồng nhau. Nếu chân bị che nhưng tay vẫn rõ, không tự xóa bằng chứng giơ tay đã thấy.
6. Hoàn tất thao tác **thêm/lưu sửa** trong biểu mẫu và thêm khung tham chiếu vào bản nháp trước khi **Lưu bản nhãn trên máy chủ**. Chưa chắc thì giữ nháp; không tự chọn đã duyệt. Có thể gán/lưu/duyệt pilot với split rỗng hoặc chưa phân chia. Muốn xuất GT còn cần provenance, nhóm nguồn và xác nhận duyệt thực tế theo cổng readiness; đánh giá test sau này phải tuân theo source-group/split và protocol đã khóa.

`classroom-engagement (1).zip` được dùng để tham khảo cue cổ tay trên vai và hiện diện vùng bảng; code/weights trong ZIP không được chạy để sinh GT. JSON Pose đi kèm là mẫu tổng hợp của generator trong ZIP. COMBO/điểm tương tác/cooldown của app cũ không thành nhãn nghiên cứu. Phiên bản luật demo hiện tại và cue tham khảo được ghi riêng trong rubric để nhóm kiểm chứng sau khi có nhãn thật.

API local bổ sung: `POST /api/annotation-sources {"source":"<đường tương đối trong project>"}`, `GET /api/annotation-sources`, `GET /api/action-rubric`. Kết quả có kind `annotation_source_preparation`, `frames_inferred=0`; các mảng model rỗng có nghĩa chưa chạy AI, không phải âm đã được kiểm tra. Metadata/report/manifest và ledger nằm trong thư mục output riêng của từng lượt.

### Lưu MP4 có nhãn từ kết quả Pose

Với **Full → Pose → ByteTrack**, tùy chọn **Tự lưu video có nhãn** được bật sẵn cho lượt chạy mới; có thể bỏ chọn trước khi chạy. Sau khi nhận diện hoàn tất, máy chủ xếp bước xuất MP4 bằng kết quả đã lưu, không chạy lại mô hình. Lượt phân tích vẫn `complete` và phần rà soát tiếp tục dùng được trong lúc xuất. Detect, tắt tracking hoặc Replay lấy mẫu không dùng tùy chọn này.

Với một lượt Full/Pose đã hoàn tất, dùng bảng **Xuất video có nhãn**: nhập bắt đầu/kết thúc theo giây nguồn rồi bấm **Lưu MP4 có nhãn**. Để trống kết thúc để xuất tới cuối nguồn. ROI dùng vùng sàn trước bảng đã chọn và gắn đúng SHA nguồn; cấu hình được giữ cố định khi bấm xuất. Giơ tay, đứng lên và đến vùng bảng là ba gợi ý quy tắc nháp độc lập; khớp thiếu, ID không chắc hoặc chưa chọn vùng không tự trở thành hành động dương. Luật đọc lịch sử raw từ đầu nguồn dù chỉ xuất một đoạn giữa video.

**Dừng xuất** chỉ hủy bước xuất hiện tại, giữ kết quả nhận diện và bản nhãn. Khi hoàn tất, **Xem video đã lưu** phát bản MP4 ngay trong cùng tab; Back trở về workspace. **Tải MP4** và báo cáo đi cùng từng bản xuất. Mỗi lần xuất có thư mục riêng `video_exports/<id>/` với `video.mp4`, `report.json` và timeline; không ghi đè source, preview, annotated hoặc bản xuất trước. Chỉ bản hoàn tất cả kiểm media và công bố backend mới có liên kết tải/phát.

MP4 gắn trực tiếp bbox/Pose/ID/chữ hành động vào ảnh. Khoảng cắt được lượng tử hóa theo clock nguồn; nếu bắt đầu giữa hai PTS, giữ hình đang hiển thị và cắt thời lượng đầu/cuối, báo cả requested/actual interval. Giữ timing VFR và căn âm thanh theo mốc nguồn; đoạn không giao với stream âm thanh được bù im lặng, nguồn không có audio vẫn không tạo audio. Báo cáo lưu hash, cấu hình, căn PTS, thời lượng và phạm vi nháp; video demo không phải GT hoặc bằng chứng accuracy.

API local: POST `/api/jobs/{id}/video-export` nhận `start_s`, `end_s`, `action_rules` (`basic_pose` mặc định), `board_roi`, `board_roi_source_sha256`; trả202. GET job giữ `video_export` và `video_export_eligible`; POST `/video-export/cancel` hủy riêng bước xuất. Artifact của bản hoàn tất nằm ở `/api/jobs/{id}/video-export/{export_id}/video.mp4?inline=true` và `report.json`. API tạo job mới nhận `save_action_video=true` khi dùng Full/Pose/ByteTrack; mặc định API là false để giữ tương thích với consumer cũ, còn web gửi lựa chọn đã tích.

Mỗi lượt nằm dưới `runs/demo/jobs/<id>/output/<run>/`:

| Tệp | Dùng cho |
|---|---|
| `manifest.json` | Schema2, hash nguồn/model, cấu hình, trạng thái và coverage thực có |
| `frames.jsonl` | Full: frame/PTS/time_base, detection, raw track/Pose; `display_tracks` giữ lớp EMA/nội suy riêng |
| `frames.times`, `frames.idx` | Full: chỉ mục tua theo timestamp/offset; không phải nhãn |
| `events.jsonl`, `proposals.json` | Trạng thái sự kiện/gợi ý của run; rỗng không phải negative |
| `review-<id>.json` | Snapshot người lưu: nguồn, events, coverage, phạm vi actor và tổng hợp |
| `summary.json`, `report.json` | Report hiệu năng/lỗi và phạm vi đo; summary của full khác tổng hợp hành vi đã lưu trong review snapshot |
| `preview.mp4`, `annotated.mp4` | Full: bản xem với overlay tùy chọn và bản vẽ sẵn |
| `updates.jsonl`, `latencies.jsonl` | Replay: toàn bộ mẫu đã công bố và latency server; giữ sequence/source index/PTS, gap và trạng thái chỉ là model prediction |
| `actions.jsonl` | Replay: ledger gợi ý quy tắc đã đóng, `origin=rule`, `review=draft`; gồm cả gợi ý flush lúc kết thúc không đi cùng JPEG mới. Tắt quy tắc thì ledger rỗng, không phải xác nhận âm |

Consumer/bot local đọc manifest trước, kiểm `schema_version` và trạng thái, rồi đọc frame và snapshot đã chọn. Giữ `origin`, `review`, `uncertain` và coverage; không lấy prediction nháp làm GT hoặc tự gửi dữ liệu ra ngoài. `count` hành vi là số onset quan sát được; episode đã có trước đầu đoạn không tự thành một lần bắt đầu mới. Tổng hợp ở cấp lớp, chưa có điểm đánh giá cá nhân hay phân mức tương tác tự đặt.

Máy Core5 120U đã thử IntelGraphics iGPU qua OpenVINO: Pose N1280 cùng head/input,20frame, median predict39,63ms trên GPU so348,96ms PyTorchCPU, chưa gồm decode/tracking/encode/browser. Cold compile/warmup gần30giây, chưa phải nghiệm thu end-to-end dưới3giây hoặc chạy dài. [Bằng chứng lượt thử](../../runs/demo/igpu_trial_20261103/results.json). Pilot, dữ liệu/thiết bị còn thiếu và bằng chứng cập nhật ở [kế hoạch hiện hành](../../tasks/video/unified_demo_action_plan_20261102.md#13-thực-hiện-kiểm-chứng-và-phần-phải-để-riêng).

### Chọn Intel iGPU cho Pose

Cài tùy chọn một lần rồi khởi động lại backend:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/intel_gpu.txt
```

Trong web chọn **Pose → yolo26n-pose.pt → Intel iGPU · OpenVINO (Pose)**. Giữ imgsz1280 khi đối chiếu clip Đô hiện có;640 chạy nhanh nhưng bỏ rất nhiều người trên lượt thử. Detect vẫn dùng CPU/CUDA. “Tự động” giữ quy tắc CUDA/CPU; cần chọn Intel tường minh. Lần đầu với model/kích thước mới sẽ export/compile nên phải chờ; cache dùng lại sau đó ở `runs/runtime/openvino_pose/`, có hash nguồn, phiên bản, shape và hash IR. Không tự tải weights, cài driver hoặc âm thầm fallback CPU nếu đã chọn GPU. `manifest.json`/`metadata.json` lưu runtime, thiết bị thực và precision hint; FP16 có thể làm confidence qua ngưỡng và khác một vài box/khớp so với CPU.

CLI dùng cùng lõi: thêm `--analysis-task pose --weights weights/yolo26n-pose.pt --device intel:gpu`. Lọc box trùng NMS là lựa chọn riêng trước tracking; có thể giảm ID trùng nhưng cũng loại nhầm người che nhau. Lớp hiển thị có `display_keypoints13` lọc EMA theo timestamp, τ80ms; raw17/raw13 và missing vẫn giữ nguyên, không dùng khớp đã lọc làm nhãn GT. Canvas chỉ thu nhỏ lớp vẽ theo kích thước đang xem, tọa độ JSON/video xuất vẫn theo ảnh nguồn. Có thể mở `?diagnostics=1` để đo callback/native dropped frames; số đo tối đa600mẫu nằm ở data-playback-stats của phần tử video sau pause/hết clip.

## Pilot hành vi và tổng hợp cả lớp

Pilot Thành đã có bản xem đủ1.429frame tại [lượt pilot](http://127.0.0.1:8090/?job=3636dc928fc84544b809c47d19238ed9). Nguồn train47,651667giây giữ nguyên hash/PTS; sáu khoảng ứng viên trong `runs/demo/pilot_thanh_20261102/pilot_manifest.json` cùng một nguồn, không chia sang val/test. Năm video mới đã có trên máy ở cập nhật03/11 và được ghi trong registry nguồn bên trên; tiếp tục pilot Thành để thống nhất cách gán trước khi mở rộng. Đây là model hints, **chưa có nhãn hành vi/coverage GT**. Nhóm duyệt và chốt rubric với giảng viên trong tuần03–08/11; chưa đánh dấu đã hoàn tất việc người duyệt và không tự gán hành vi khi ảnh/diễn biến không rõ.

Nhóm dùng mã người duyệt cục bộ như `group-a`/`group-b`, mỗi người mở nhánh riêng qua `?job=<id>&reviewer=group-a` để gán độc lập. Nhánh mới không sao chép nhãn người khác. Đây là tách quy trình trên máy local, không phải hệ thống tài khoản/phân quyền. Khi đối chiếu, giữ hai snapshot trước adjudication, không ghi đè chúng bằng một kết quả được gọi là “đã thống nhất”.

Link pilot hiện hành: [group-a](http://127.0.0.1:8090/?job=3636dc928fc84544b809c47d19238ed9&reviewer=group-a), [group-b](http://127.0.0.1:8090/?job=3636dc928fc84544b809c47d19238ed9&reviewer=group-b). [Kiểm readiness thực](../../runs/demo/pilot_thanh_20261102/readiness_verification_20261103.json) ghi cả hai nhánh đang trống, evaluation-ready=false và GTexport409. Đây là điểm bắt đầu để nhóm duyệt, không phải hai GT snapshot đã hoàn thành.

1. Mở “Người duyệt, rubric và nguồn nhãn”, điền mã người duyệt và phiên bản rubric thực sự dùng. Source group của pilot là `thanh_mov_941205c3ad5f6d30`, split đã duyệt là `train`; metadata không thay bộ ảnh train/val đang có.
2. Tạo “Người quan sát độc lập với AI”, chọn cảnh và khoảng thực sự nhìn thấy người đó. Dùng mã cục bộ do công cụ cấp; không cần tên sinh viên. Người bị AI bỏ sót vẫn phải có thể gán nhãn. Ghi riêng span khi mất quan sát; không tự nối qua cảnh.
3. Nếu dùng gợi ý track AI, xác nhận từng liên kết AI→người quan sát theo khoảng thời gian. Nhiều track AI trùng có thể được người duyệt liên kết tới cùng một người; không sửa ID/model raw hoặc tự gộp chúng như tracking đã đúng.
4. Xem diễn biến trước/sau và gán tư thế/hành vi riêng, coverage từng nhãn, uncertainty và biên bị cắt. Chưa xem không phải âm; “đứng” khác “đứng lên”; một người có thể đứng và giơ tay cùng lúc.
5. Lưu snapshot. Chỉ xác nhận trạng thái duyệt khi đã làm thật. “Tải bundle GT đã duyệt” chỉ mở khi có người/rubric/nguồn, actor độc lập và coverage đủ điều kiện; bản cũ/nháp vẫn đọc được nhưng thiếu provenance không tự thành GT. Nếu không có ví dụ dương một hành vi, ghi thiếu để thu bổ sung.

Trong phần **Các khoảng pilot cần rà soát**, sáu nút đưa tới các khoảng của cùng nguồn Thành. Máy chủ chỉ nạp manifest khi hash, kích thước, số frame, duration, native PTS và biên thời gian khớp ledger nguồn đã lưu; không dùng FPS để đoán lại mốc. Không có pilot phù hợp thì ẩn phần này, manifest mơ hồ hoặc sai nguồn không được tự chọn.

Chọn một khoảng để tua bản xem đã lưu, không chạy AI hoặc cắt thêm video. Thời gian gợi ý của nhãn là giao **khoảng pilot ∩ phạm vi người đang chọn ∩ cảnh nguồn**; thời gian thêm người mới nằm trong giao khoảng pilot với cảnh. Không có giao hợp lệ thì để trống thời gian nhãn, cần chọn người phù hợp. Khi đang sửa một nhãn, chọn khoảng chỉ tua và giữ nguyên biểu mẫu đang sửa; lưu/hủy sửa rồi chọn lại nếu muốn đổi thời gian gợi ý. Đọc lại tiến độ không tua video, không thay thời gian đang nhập hoặc nhãn chưa lưu.

Audit chỉ đọc **snapshot đã lưu của nhánh hiện tại**, theo từng khoảng và bốn nhãn hành động. Bảng tách người·giây đã rà soát, phạm vi người đã khai báo, coverage, người·giây còn thiếu, nhãn dương đã duyệt, nháp và uncertainty. Coverage chồng nhau được hợp trước khi tính; số khoảng nhãn dương không phải số onset của lớp. Nhánh trống hiện sáu khoảng **Chưa bắt đầu**. Chưa khai báo người thì denominator chưa biết: `coverage_fraction=null`, `unreviewed_actor_seconds=null`, không đổi thành0 hoặc100%. Có phạm vi người nhưng chưa rà một nhãn thì coverage0 là chưa phủ phạm vi đã khai báo, không phải kết luận không có hành vi đó.

Trạng thái **Đã phủ người đã khai báo** chỉ nói đủ coverage của những người nhóm đã tạo trong các khoảng đó. `source_inventory_confirmed=false` vẫn giữ nguyên; chưa xác nhận đã liệt kê toàn bộ người trong video, hoàn tất pilot hoặc hoàn tất nghiên cứu. GT của một phần đã rà soát vẫn có thể hợp lệ để xuất trong khi các khoảng/nhãn khác còn thiếu. “Đọc lại tiến độ đã lưu” và việc chọn khoảng không thêm người, event, coverage hoặc xác nhận duyệt; tiến độ không lấy nhãn của nhánh còn lại.

“Tổng hợp quan sát của lớp” đọc **bản đã lưu**, gồm count onset, thời lượng quan sát theo người·giây, coverage và episode bị cắt theo cửa sổ/cả buổi. Nháp đang sửa chưa được tính. Chưa rà soát hiển thị khác số0; không có điểm/phân mức tự đặt. Bấm minh chứng để về actor/mốc nguồn đã lưu. Cửa sổ phải hữu hạn, lớn hơn0 và tạo tối đa500khoảng; chọn cửa sổ lớn hơn nếu video dài. Các API local:

```text
GET /api/jobs/{id}/review-context?reviewer_id=group-a
GET /api/jobs/{id}/readiness?reviewer_id=group-a
GET /api/jobs/{id}/pilot-context?reviewer_id=group-a
GET /api/jobs/{id}/gt-bundle?reviewer_id=group-a
GET /api/jobs/{id}/class-summary?window_seconds=60&reviewer_id=group-a
GET /api/jobs/{id}/class-export?window_seconds=60&reviewer_id=group-a
```

`class-export`/“Tải JSON cấp lớp” là hợp đồng chia sẻ tối thiểu `schema_version=1`, `kind=class_observation_summary`: chỉ số cấp lớp/cửa sổ và bất định tổng hợp; bỏ event/actor/ghi chú/người duyệt/đường dẫn/model/pixel. SHA nguồn vẫn có thể liên kết với video đã biết, chưa phải ẩn danh hoàn toàn. Consumer phải kiểm version/kind, giữ `null` là chưa có coverage, không thay bằng0 và không tự suy điểm tương tác. Chưa tự gửi dữ liệu sang CMC University; đích/API cụ thể còn cần chốt.

[Ví dụ tải thật từ UI trước khi duyệt](../../runs/demo/pilot_thanh_20261102/class_observations_unreviewed_20261103.json) có `saved_snapshot_available=false`, count/duration=null và reviewedcoverage0. Ví dụ này kiểm đường xuất và giữ trạng thái chưa biết; không phải thống kê lớp đã được giảng viên xác nhận.

## Chấm hành vi theo protocol và corpus

Runner dùng event F1 ghép một-một theo actor/label/tIoU, khác AP/mAP xếp hạng score. Protocol phải ghi rõ `protocol_id`, `rubric_version`, `phase=pilot|dev|test`, `status=draft|locked`, `iou_threshold`, `boundary_tolerance_s` (có thể `null`), `development_source_groups`, `test_source_groups`, `schema_version=1`. Chọn ngưỡng/dung sai bằng pilot/dev rồi khóa trước test; công cụ không tự chọn hoặc cung cấp ngưỡng nghiên cứu mặc định. Phase test chỉ nhận nguồn test được khai báo và protocol locked; nguồn held-out bị chặn khỏi cả pilot/dev dù ai đổi trường split chữ trong GT.

```powershell
.venv\Scripts\python.exe -m classroom_ai.video.action_evaluation --prediction prediction.json --reference reviewed-gt.json --protocol protocol.json --output evaluation-new.json
```

Các tên JSON trong lệnh là đầu vào do nhóm xuất/chọn, chưa phải bộ GT đã có. Prediction tải từ gợi ý quy tắc giữ source SHA/actor namespace/model provenance; GT phải qua cổng duyệt. Nếu khác namespace hoặc dùng ID AI thay actor độc lập, cung cấp `--actor-map mapping.json` có `source_sha256`, `from_namespace`, `to_namespace`, `reviewer_id` và `links` được duyệt. Không đoán danh tính từ ID gần nhau. Actor chưa ánh xạ sẽ báo lỗi thay vì âm thầm bỏ người để làm F1 đẹp hơn. Actor GT bị model bỏ sót vẫn tính FN trong coverage hợp lệ.

Chấm nhiều nguồn: manifest `schema_version=1`, `items=[{"prediction":"p.json","reference":"gt.json","actor_map":"map.json"}]`; đường dẫn tính từ manifest, actor_map tùy trường hợp. Chỉ dùng **một GT snapshot cho mỗi source hash**; các cửa sổ cùng video không phải nhiều nguồn độc lập.

```powershell
.venv\Scripts\python.exe -m classroom_ai.video.action_evaluation --corpus corpus.json --protocol protocol.json --output corpus-evaluation-new.json
```

Report giữ hash đầu vào/protocol, P/R/F1 từng hành vi và micro, thiếu lớp dương/ignored/coverage, sai số onset/end chỉ ở biên có bằng chứng, sai số đếm và union-duration theo actor. Duplicate prediction tính FP; unknown/uncertain không thành negative. Báo MAE/absolute error cùng signed error khi tổng hợp nhiều nguồn để tránh lỗi dương/âm triệt tiêu. Tệp đầu ra mới được tạo độc quyền, không ghi đè nhãn hoặc báo cáo đã có. Công cụ bundle/corpus và các cổng protocol/mapping đã kiểm bằng fixture; corpus hiện **chưa có GT hành vi thật** để công bố metric lớp học.

### Đối chiếu hai người duyệt độc lập

Sau khi cả hai nhánh đã được người duyệt xác nhận đủ điều kiện xuất, tải riêng bundle schema2, giữ nguyên hai file trước đối chiếu. Hàm `compare_review_bundles(review_a, review_b, actor_map=..., protocol=...)` và CLI dưới đây mô tả mức đồng thuận giữa hai bản người gán. Đây là **descriptive agreement**, không phải độ chính xác mô hình, GT adjudicated hoặc bằng chứng hai người thực sự duyệt độc lập chỉ vì có hai mã khác nhau. Hai bundle phải cùng SHA nguồn, duration, cảnh, rubric, source group và split; mã người duyệt phải khác nhau. Cả hai phải qua cổng provenance/coverage đã duyệt. Không tự sao chép một nhánh rồi đổi mã để tạo đối chứng.

Người kiểm tra phải xác nhận các cặp **người do hai nhánh khai báo** trong từng khoảng và tạo `human-pairs.json` theo mẫu sau. Đây là mẫu cấu trúc cần thay toàn bộ giá trị giữ chỗ bằng dữ liệu và xác nhận thực, chưa phải mapping đã duyệt của pilot:

```json
{
  "schema_version": 1,
  "kind": "human_actor_pairs",
  "source_sha256": "<SHA256 nguồn chung của hai bundle>",
  "mapping_reviewer_id": "<mã người đã kiểm các cặp>",
  "review": "reviewed",
  "a": {
    "snapshot_sha256": "<canonical SHA256 của toàn bundle A>",
    "reviewer_id": "group-a",
    "actor_namespace": "<namespace đúng trong bundle A>"
  },
  "b": {
    "snapshot_sha256": "<canonical SHA256 của toàn bundle B>",
    "reviewer_id": "group-b",
    "actor_namespace": "<namespace đúng trong bundle B>"
  },
  "pairs": [
    {
      "segment_id": 0,
      "a_track_id": "human:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "b_track_id": "human:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "start_s": 0,
      "end_s": 5
    }
  ]
}
```

`review="reviewed"` và `mapping_reviewer_id` ở cấp trên cùng là xác nhận **mapping**, tách với metadata người duyệt của hai bundle. Không dùng ID AI hoặc đoán cùng người vì UUID/namespace trùng nhau. Khoảng cặp phải nằm trong phạm vi của cả hai người trong cùng cảnh; không ghép chồng mâu thuẫn. Cùng namespace cũng vẫn phải khai báo cặp rõ ràng. `snapshot_sha256` là digest của JSON canonical toàn bundle, không phải SHA byte của file có thụt dòng. Tính bằng helper hiện có rồi điền vào mapping; sửa bundle sau đó làm mapping không còn khớp:

```powershell
.venv\Scripts\python.exe -c "import json; from classroom_ai.video.action_evaluation import review_snapshot_sha256; print(review_snapshot_sha256(json.load(open('review-a.json', encoding='utf-8'))))"
.venv\Scripts\python.exe -c "import json; from classroom_ai.video.action_evaluation import review_snapshot_sha256; print(review_snapshot_sha256(json.load(open('review-b.json', encoding='utf-8'))))"
.venv\Scripts\python.exe -m classroom_ai.video.action_evaluation compare-reviews --review-a review-a.json --review-b review-b.json --actor-map human-pairs.json --protocol protocol.json --output agreement-new.json
```

Protocol dùng ngưỡng tIoU do nhóm chọn trên pilot/dev, không tự coi ví dụ/ngưỡng fixture là tiêu chí nghiên cứu. Đối chiếu người–người yêu cầu `boundary_tolerance_s=null` để ghép đối xứng theo khoảng quan sát. Chỉ so trên cặp người tường minh và **giao coverage cùng nhãn mà cả hai đã rà soát**. Mask unknown/uncertain từ một trong hai phía được áp dụng theo loại nhãn; bản còn nhãn/coverage nháp chưa qua cổng duyệt để đối chiếu. Người chưa ánh xạ hoặc phần coverage không chung là chưa giải quyết, không thành negative. Episode vượt khe coverage hoặc ranh giới cặp bị loại và ghi lý do, không cắt để tạo onset mới.

Report có `agreement_f1`, số matched/A-only/B-only theo nhãn, coverage chung và audit phần bị loại. Sai khác biên B trừ A chỉ có khi cả hai xác nhận biên không bị cắt. Không có episode hợp lệ ở cả hai phía thì agreement là `null`, không tự thành100%. CLI giữ SHA byte các file đầu vào riêng với canonical digest và tạo file output mới, không ghi đè snapshot. Nhóm đọc disagreement và quay lại bằng chứng nguồn để adjudicate thủ công; công cụ không tự đồng ý, duyệt nhãn hay chọn một nhánh làm chân lý. Hai bản cùng video vẫn là một nguồn khi đánh giá corpus, không phải hai video test độc lập.

## Replay cuốn chiếu nguồn đã ghi

Trong cùng form chọn **Replay · nguồn ghi sẵn1×, Pose lấy mẫu**, tần suất AI ban đầu5Hz (giới hạn0<Hz≤10) và model/device đã kiểm. Replay chỉ hỗ trợ Pose; Detect vẫn ở full. Source release theo PTS thật; một inference đang chạy và tối đa một mẫu pending mới nhất, không xếp FIFO vô hạn. Nhịp AI5Hz không phải video30FPS và không phải camera trực tiếp. Nếu mất mẫu/quá tải, giữ source index/PTS và ghi gap/drop, không tự tạo quan sát hành vi trong khoảng trống.

Đường replay mặc định dùng PyAV16.1.0 với `AUTO`,4luồng, một decoder phần mềm trả ảnh và nativePTS cùng lượt decode. Việc này tránh đường OpenCV cộng FFprobe cùng giải mã HEVC hai lần; full OpenCV vẫn giữ như trước. Không đổi codec/file gốc hoặc tự chuyển VFR thành CFR.

Preflight toàn nguồn, hash và export/compile/warmup vẫn diễn ra trước source clock, đo riêng trong cold timing. Kết quả Pose/ID được công bố ngay khi có; không đợi video xuất xong. JPEG và JSON phải cùng sequence; ảnh cũ bị409 khi mẫu mới đã thay. Ảnh chỉ giữ trong RAM cho lượt mới nhất, còn manifest/updates/latencies/actions/report được lưu. Replay chưa có GT timeline hoặc bộ nhận dạng bốn hành vi. Mặc định quy tắc tắt, `action_rules=off` và `action_status=not_run`; bật tùy chọn chỉ chạy gợi ý giơ tay chưa hiệu chuẩn.

Muốn thử, chọn **Quy tắc gợi ý trong replay → Gợi ý giơ tay · nháp** trước khi bắt đầu. Lựa chọn này bật ByteTrack và khóa ô tracking; tắt quy tắc để đổi tracking. Backend cũng từ chối rule nếu không dùng ByteTrack. Full không dùng tùy chọn này và luôn gửi `action_rules=off`; khi lượt đang chạy không đổi cấu hình. Quy tắc đọc quan sát Pose raw được ghép với track xác nhận, không dùng khớp hiển thị đã lọc làm bằng chứng mới. Đây chưa phải nhận dạng tự động đủ bốn hành vi, không phải kết quả đã hiệu chuẩn hoặc GT.

Panel **Gợi ý giơ tay · nháp** chỉ xuất hiện khi lượt replay thực sự bật rule. Số active/pending thuộc mẫu đang hiển thị, không phải tổng người hoặc tổng hành vi của lớp. Số0 đọc là **chưa nhận gợi ý**, không phải không có giơ tay. Chỉ sau khi JPEG giải mã và cùng sequence được commit, UI mới nhận counters/episode của mẫu đó. Danh sách giữ tối đa20 episode đã đóng mà trình duyệt nhận được, loại ID trùng/cũ trong phiên; không tự thêm vào events, coverage hoặc bản người duyệt. Đổi lượt/hủy thì xóa panel cũ, tải lại trang không khôi phục lịch sử nhận thành một ledger đầy đủ.

`actions.jsonl` là ledger bền vững của **toàn bộ gợi ý đã đóng**: mỗi envelope giữ `event_id`, `event` với `origin=rule`/`review=draft`, availability và lịch sử quyết định. Có episode được flush ở EOF/hủy mà không có JPEG mới, nên số `action_summary.emitted_draft_events` trong report có thể lớn hơn danh sách UI. Tải **toàn bộ gợi ý đã lưu · actions.jsonl** để kiểm ledger; không coi20episode hoặc các gói UI nhận được là toàn bộ phiên. Nguồn lấy mẫu quá thưa so với `max_gap_s` có cảnh báo `nominal_cadence_below_rule_max_gap`: có thể không tích lũy đủ lịch sử để phát gợi ý. Cảnh báo và ledger rỗng không chứng minh có/không có hành vi hay độ chính xác.

Tách thời gian **episode quan sát** (`event.start_s/end_s`), **lịch sử quyết định** (`decision_history_s`, từ đầu episode đến quan sát dùng để quyết định) và **availability nội bộ**. `source_to_available_ms` kết thúc lúc rule tính xong, trước ghi bền vững, công bố frame và trình duyệt; terminal flush không có release tương ứng thì trường này là `null`. Không dùng số đó thay cho release→server callback, release→client DOMcommit, browser paint hoặc latency suy ra hành vi đầu-cuối. Bật rule có chi phí riêng; các số đo Thành trước đó không tự trở thành benchmark có rule.

```text
GET /api/jobs/{id}/latest
GET /api/jobs/{id}/live-frame?sequence=N
GET /api/clock
```

Ảnh live được trình duyệt giải mã trước, rồi commit cùng overlay của đúng sequence. UI poll nối tiếp khoảng5Hz, bỏ kết quả/ảnh cũ khi đổi job và thu hồi objectURL; một mẫu đã bị thay thì chờ nhịp kế tiếp, không chạy AI lại. Khi replay kết thúc, ảnh chỉ còn trong RAM của lượt mới nhất; sau restart có thể chỉ đọc artifact đã lưu.

Có hai phạm vi đo riêng:

- **Server:** scheduled source release→đầu callback công bố máy chủ. `latencies.jsonl` giữ toàn bộ mẫu đã công bố; report tính p50/p95/max/số≥3giây trên cả lượt, cùng cold/first-update và chi phí callback. Đây chưa gồm transport/client và không phải độ trễ suy ra hành vi.
- **Client:** `/api/clock` hiệu chuẩn khoảng offset bằng ba lần trao đổi, chọn RTT nhỏ nhất, làm lại sau60giây; clock_id đổi làm mất hiệu lực hiệu chuẩn cũ. Sau JPEGdecode và image/overlay DOMcommit, ghi lower/upper cho release→commit và publish→commit. Mốc này là **ảnh đã giải mã/gán vào giao diện**, chưa phải browser paint, photon hay camera capture; hai trường paint/camera vẫn `null`. Clock/timestamp thiếu hoặc không hợp lệ là unknown, không dùng0 thay thế.

Bấm **Tải số đo commit giao diện JSON** để tải dữ liệu đo vào máy; không POST sang dịch vụ khác hoặc tự thêm vào artifact server. JSON giữ sequence/PTS/RTT/khoảng offset, click→firstcommit nếu trang quan sát được thao tác click và clock hợp lệ. Reload lượt cũ không tự có số cold từ click. Tối đa10.000rawrecord, có `records_truncated/records_dropped`; lifetime count/sum/max/số upper≥3s vẫn riêng theo clockepoch. Không gọi quantile tính từ phần raw đã cắt là quantile toàn lượt. Mã kiểm local có thể đọc summary ở `data-replay-timing` của phần tử `liveImage`.

### Bằng chứng replay Thành ngày03/11

Lượt **`6e49502f236a4bd3b425fe441262f131`** dùng đúng SHA Thành `941205c3ad5f6d300b399be5102d6191227b0b9dd7397e76732252a39a5c19a4`,1.429frame/47,651667giây, N1280/conf0,05/ByteTrack/NMS tắt, GPU.0 thực. PyAVnativePTS/AUTO4 giải mã một lần; sourceclock47,812s. Có239mẫu được chọn,236mẫu công bố,3pending replacements và3gaps. Không coi những khoảng mất mẫu là đã có hành vi quan sát được.

| Mốc đo | Kết quả trong lượt này |
|---|---|
|Release nguồn→server callback,236publication|p50142,833ms; p95244,750ms; max565,667ms;0mẫu≥3s|
|Release nguồn→client decode/DOMcommit,172commit hợp lệ|Mean bounds295,118–299,418ms; maxupper663,667ms;0upper≥3s|
|Server publish→client decode/DOMcommit,172commit hợp lệ|Mean bounds143,209–147,509ms; maxupper384,300ms|
|Clock client/server|RTT4,3ms trong summary;0clockreset;172record,không truncation; paint/camera latencynull|
|Cold trước sourceclock|64.047ms, trong đó preflight28.797ms và export/compile/warmup33.860ms|
|Bấm bắt đầu→client commit đầu tiên|64.423,5ms, firstcommitclockvalid=true|

172clientcommits không đồng nghĩa trình duyệt đã commit đủ236publication. “0upper≥3s” mô tả cận trên của các phép đo hợp lệ, không thay cho kiểm video/camera/photon hoặc độ trễ của bốn hành vi. **Warm của lượt ngắn này dưới3s; cold vẫn khoảng64s.** Chưa có soak30phút, nguồn test độc lập, metric hành vi lớp học hoặc nghiệm thu giảng viên.

[Bản tóm tắt browser](../../runs/demo/replay_client_commit_20261103.json), [rawclientrecords từ nút tải UI](../../runs/demo/replay_client_commit_records_20261103.json), [report server](../../runs/demo/jobs/6e49502f236a4bd3b425fe441262f131/output/replay_719cd3d1458b43c2a256795ee6c54f8a/report.json) và [manifest](../../runs/demo/jobs/6e49502f236a4bd3b425fe441262f131/output/replay_719cd3d1458b43c2a256795ee6c54f8a/manifest.json) giữ phạm vi/cấu hình. Rawdownload SHA256 `ca8f46fc59074bf99e2446ced328e7b0f9ec023cae86451ebc1993e1a6a01453`;236serverPTS đã đối chiếu fullsource,172clientframe/PTS đã đối chiếu serverupdates trong [evidence tổng hợp](../../runs/demo/replay_verification_20261103.json). File này vẫn giữ lượt lỗi trước: Thành p95server10,671s, max11,299s,190mẫu≥3s trên đường giải mã HEVC hai lần. Decoder và cấu hình luồng đã đổi cùng nhau, không gọi so sánh này là ablation chỉ riêng số lần decode và không xóa lượt lỗi để chỉ giữ kết quả tốt.

Bổ sung kiểm chức năng rule trên **clip Đô2giây**, lượt `a9487a3ddb74401e8f18746fa6b1b143`:60frame release,10mẫu chọn/10công bố,0replacement/0gap, PyAV và GPU.0 thực; rule `raise_hand` đã bật. [Report smoke](../../runs/demo/jobs/a9487a3ddb74401e8f18746fa6b1b143/output/replay_97dbb8de20b04ed9b3a95ccd5c73e0fb/report.json) ghi10frame rule,32track rows nhưng0quan sát được rule hỗ trợ/32unsupported,0event. Kết quả0 không xác nhận không có giơ tay và chưa kiểm được ví dụ dương hoặc chất lượng rule. Serverp95150,29ms/max188ms trên10publication, coldprep38,844s; đây chỉ là smoke chức năng ngắn, không phải benchmark Thành mới, ablation hoặc nghiệm thu latency hành vi. Browser đã đối chiếu ảnh/overlay của mẫu9,frame54,PTS54054,time_base1/30000; không suy từ việc căn đúng một mẫu thành đo đầy đủ browser latency. Các số47giây Thành và lượt A/B lỗi phía trên giữ nguyên phạm vi.

Mục tiêu nghiên cứu còn mở: nhóm duyệt GT/rubric03–08/11; dữ liệu và baseline đủ giơ tay/đứng lên/lên bảng/viết bảng, protocol/dev-test khóa và nguồn độc lập; giảng viên đối chiếu tổng hợp; camera/soak/tải dài và chính sách dữ liệu. Consumer local và class-export đã có; tích hợp CMC University được người dùng để sang giai đoạn sau, chưa chọn hệ thống/API đích. Theo dõi tại [mục14 kế hoạch](../../tasks/video/unified_demo_action_plan_20261102.md#14-hoàn-thiện-theo-mục-tiêu-chính-thức-trong-tháng112026).

Kiểm hồi quy phần mềm ngày03/11 sau pilot audit, đối chiếu hai người duyệt và tùy chọn replay rule: **464pytest qua,1skip,1warning trong110,18s;85Node qua trong3.174ms; Ruff kiểm102file sạch**. Đây là kiểm mã và fixture, tách với GT hành vi, số đo GPU thật và nghiệm thu lớp học. Chưa có đối chiếu hai GT thật của pilot; các giới hạn cold, lượt A/B lỗi và cổng nghiên cứu bên trên vẫn giữ nguyên.

## CLI/TUI và công cụ còn được giữ

Streamlit ảnh/webcam/Training cũ chạy riêng bằng `python -m classroom_ai.interfaces.streamlit_ui`; phần dưới mô tả CLI/TUI dùng cùng backend. Web thống nhất là hướng chính cho video/timeline; `docs/project_site/` là web thuyết minh, không chạy model.

Chạy từ thư mục gốc repository. Cần môi trường `.venv` hiện có và **ffmpeg, ffprobe trong PATH**. CLI video dùng môi trường hiện có; TUI nút bấm dùng thêm Textual tùy chọn. Không cài `opencv-python-headless`: bản này cần HighGUI để mở cửa sổ.

## Giao diện bấm trong terminal

**Bấm đúp `run_video.cmd`**, hoặc chạy `.\run_video.cmd` không tham số. Giao diện tối giản có ô đường dẫn, nút **Chọn…**, chế độ **Nhanh/Đầy đủ**, **Chạy/Dừng**, thanh tiến độ và khung log. Dùng chuột hoặc Tab/Enter; Ctrl+R chạy, Ctrl+D dừng, Ctrl+Q thoát. Nút Chọn mở cây thư mục ngay trong terminal; có thể dán đường dẫn thư mục khác, Enter rồi chọn file video.

Đã cài Textual vào `.venv` trên máy triển khai. Máy mới cài phần tùy chọn bằng:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/video_tui.txt
.\run_video.cmd
# Hoặc chọn sẵn thiết bị/cấu hình khi mở TUI:
.\run_video.cmd --tui --device cuda:0 --imgsz 1280
```

Mặc định video dùng LIGHT_CONTRAST/1280/confidence 0,05/device auto (checkpoint đầy đủ bên dưới). TUI không tự bắt đầu xử lý khi mở. Cửa sổ video OpenCV xuất hiện khi bấm Chạy; Space trong cửa sổ video để pause/resume. Tiến độ và log trong terminal cập nhật một lần/giây, giữ tối đa 200 dòng hiển thị. Tắt animation và con trỏ nhấp nháy. Trạng thái nạp model có chữ/thời gian, không phần trăm giả.

Mỗi lượt có thư mục mới dưới `runs/video_tui`, gồm `console.log` và thư mục kết quả backend. Dừng gửi cờ cho backend kiểm tra tối đa mỗi 200 ms, đợi tác vụ native đang chạy hoàn tất. Thoát TUI yêu cầu dừng trước; sau 10 giây mới buộc dừng cây process do TUI tạo để tránh bỏ process chạy nền. Ép thoát có thể để lại file partial và không có báo cáo cuối; không coi đó là completed.

### Các phần liên quan thế nào?

```text
Terminal: Windows Terminal / cửa sổ console
  └─ TUI Textual: ô nhập, nút bấm, tiến độ
       ├─ gửi cấu hình và lệnh dừng
       └─ nhận log văn bản, không nhận hình ảnh
            ↕
       Backend Python ở process riêng
         ├─ OpenCV: giải mã video thành frame trong RAM
         ├─ YOLO: nhận diện đối tượng trên frame được chọn
         ├─ OpenCV: vẽ box và mở cửa sổ phát video
         └─ FFmpeg: xuất MP4/âm thanh trong chế độ full
```

**Terminal** là nơi hiển thị ứng dụng chữ và nhận thao tác. **PowerShell/cmd** là shell dùng để khởi chạy lệnh. **TUI** (Text User Interface) là giao diện tương tác nằm trong terminal. **Backend** ở đây là process Python local thực hiện công việc, không cần web server. **OpenCV** là thư viện xử lý ảnh/video mà backend sử dụng; YOLO thực hiện nhận diện, còn FFmpeg thực hiện xuất video.

Process TUI không import torch, cv2 hoặc Streamlit; frontend không giữ model hay frame. Vì vậy nhịp cập nhật chữ 1 Hz độc lập với nhịp phát video 30 FPS. TUI vẫn có chi phí CPU/RAM riêng. Bằng chứng thử ngắn và giới hạn đo nằm ở [báo cáo TUI](../../reports/video/tui_20261015/results.md); không cam kết không có overhead trên mọi thiết bị.

## Chạy

```powershell
# Nhanh: phát theo tốc độ gốc, lấy mẫu 5 frame/giây cho YOLO.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast

# Đầy đủ: nhận diện mọi frame, xuất xong mới phát kết quả.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode full

# Demo trên laptop GPU: chỉ rõ GPU, không âm thầm chuyển CPU nếu lỗi.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast --device cuda:0 --imgsz 1280

# Đo riêng đường phát, không nạp YOLO.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast --baseline

# Đo nhận diện/xuất video không có cửa sổ.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode full --device cpu --no-display

# Chọn checkpoint và sampling tường minh.
.\run_video.cmd --source "C:\Videos\lop hoc.mp4" --mode fast --weights "weights\BEST_Project8.pt" --sample-fps 5 --buffer-ms 250
```

Tương đương: `.venv\Scripts\python.exe -m src.video ...`. `--help` liệt kê tham số. `Space` tạm dừng/tiếp tục **phát**; `Q`, `Esc` hoặc đóng cửa sổ để dừng. Đang xử lý toàn bộ có thể dừng bằng Q/Esc hoặc Ctrl+C, không tạm dừng inference. Khi phát kết quả **full**, kéo thanh `Frame` hoặc dùng **A/D** để lùi/tiến 5 giây, kể cả khi pause; tua chỉ giải mã MP4 đã xuất, không chạy lại YOLO. Hết video giữ frame cuối: Space phát lại từ đầu, Q đóng và hoàn tất phiên. Chế độ fast chưa hỗ trợ tua. Cửa sổ xem trước **không phát âm thanh**; MP4 chế độ full giữ track âm thanh đầu tiên, mã hóa AAC.

## Bám người và giảm nháy trong Full

Tính năng tùy chọn; cài thêm `pip install -r requirements/video_tracking.txt` trong `.venv`. TUI chọn **Đầy đủ** rồi bật **Bám người**. Hoặc mở sẵn cấu hình demo:

```powershell
.\run_video.cmd --tui --mode full --weights "weights/seed_repeats_20261014/E3_LIGHT_CONTRAST_seed123_E2_best.pt" --confidence 0.05 --imgsz 1280 --tracking bytetrack
```

ByteTrack chỉ theo dõi person, đợi hai frame liên tiếp trước khi hiện P{id}. Mỗi ID có màu giả ngẫu nhiên riêng, giữ nguyên qua các frame và khoảng nội suy. Đây là ID trong video, không phải danh tính sinh viên. Bảy lớp khác vẫn vẽ detection gốc.

Lượt một chạy YOLO/tracking và ghi JSONL; lượt hai đọc lại nguồn, làm mượt/nội suy và xuất MP4, không chạy lại YOLO. Có detection thì nét liền; chỉ lấp khoảng <=0,3s có hai đầu xác nhận cùng ID/cảnh, nét đứt và nhãn `noi suy`. Mất dấu lâu thì ẩn, dù còn nhớ track khoảng3s để thử ghép lại. Không giữ box đứng yên3s hoặc suy ra YOLO đã phát hiện trên frame nội suy.

CLI thêm `--tracking off|bytetrack` (mặc định off), `--track-max-lost-seconds 3`, `--track-gap-max-ms 300` (0 tắt), `--track-smooth-alpha 0.65` (1 tắt EMA). Chỉ full; fast/baseline không hỗ trợ tracking. Có thể chỉ định checkpoint khác như bình thường, không tự thay mặc định model.

Ngoài detection gốc có `tracks_raw.jsonl` và `tracks.jsonl`; trạng thái observed/interpolated, anchors, cấu hình/ngưỡng/màu và chi phí đều ghi riêng. Tiến độ chia nhận diện/tracking và dựng video. Chưa ReID/GMC, chưa bảo đảm ID ổn định khi che lâu/camera di chuyển mạnh; heuristic cắt cảnh không hoàn hảo. Xem [kế hoạch và tiêu chí](../../tasks/video/video_tracking_plan_20261015.md).

## Hai chế độ

- `full`: mỗi frame được nhận diện một lần, ghi dần qua FFmpeg; kiểm lại số frame, FPS và âm thanh trước khi đổi `annotated.partial.mp4` thành `annotated.mp4`. Giữ tốc độ nguồn (25, 30000/1001, 30...), không có cam kết hoàn thành toàn bộ dưới ba giây.
- `fast`: phát theo thời gian nguồn, YOLO lấy mẫu theo timestamp. Một tác vụ chạy và một tác vụ chờ; frame chờ cũ bị thay khi xử lý chậm. Chỉ vẽ box lên đúng frame đã nhận diện. Footer giữ số lượng/timestamp/tuổi kết quả gần nhất, không coi frame chưa nhận diện là đã được YOLO kiểm tra. Vì không tracking, box không xuất hiện liên tục giữa các frame lấy mẫu.

Bộ đệm mặc định 250 ms, tối đa 12 frame/128 MiB ảnh thô. Ngưỡng thực dùng giá trị nhỏ hơn khi độ phân giải/FPS cao. Decoder ngừng đọc khi đầy; pause không tiếp tục tích hàng đợi. Có thể có một lần đọc đang chạy khi bấm pause. Hàng đợi không đại diện toàn bộ RAM: còn model, bộ nhớ decoder/encoder và frame đang xử lý.

Mặc định video dùng `LIGHT_CONTRAST_rsLoRA_best_E1.pt`, `imgsz=1280`, confidence **0,05**, một lượt, không thêm NMS. Đây là mức thấp nhất của slider UI hiện có, không phải giới hạn toán học của YOLO; có thể tăng nhận diện yếu và báo nhầm. `--weights`/`--confidence` ghi đè tường minh. Dòng khởi chạy báo tên checkpoint, imgsz và confidence; metadata giữ đường dẫn/hash thực. Các lượt cũ giữ cấu hình cũ; không thay mặc định Streamlit/Training. `--nms-iou` bật NMS bổ sung. `--device auto` kiểm CUDA thực tế, nếu không dùng được sẽ ghi lý do chọn CPU trong báo cáo và bản tổng kết. Không dùng số đo 120 ms từ một máy làm chuẩn máy khác.

## Độ phân giải và chất lượng

Decoder đọc frame ở kích thước nguồn; full vẽ box trên frame đó và xuất cùng FPS/số frame. Kích thước lẻ được pad tối đa một pixel để tương thích H.264/yuv420p. Cửa sổ xem trước có thể thu nhỏ tới 1280×800 để vừa màn hình; việc này không thu nhỏ video xuất.

YOLO đổi kích thước bản ảnh đầu vào theo `imgsz` và padding, sau đó tọa độ được quy về ảnh gốc. Vì vậy video 1920×1080 vẫn xuất 1920×1080 khi suy luận ở 1280. Muốn suy luận gần kích thước nguồn Full HD, chọn rõ `--imgsz 1920`; tốc độ/RAM/VRAM cần đo lại, không bảo đảm nhanh hoặc chính xác hơn trên mọi nguồn. Frame ở trong RAM, không ghi từng ảnh JPEG trung gian.

Giữ độ phân giải/FPS không đồng nghĩa giữ nguyên từng pixel. Hiện xuất **H.264, CRF 20, yuv420p** và AAC: có nén mất dữ liệu, chưa có tùy chọn lossless/HDR/10-bit. File gốc được giữ nguyên. Chưa tự thay codec/chất lượng nén vì sẽ thay đổi dung lượng, thời gian mã hóa và khả năng phát tương thích.

## Tiến độ và chi phí

Terminal báo bước hiện tại, thời gian bước/tổng, tiến độ xử lý full, ETA, RAM và VRAM khi khả dụng; cập nhật nhiều nhất một nhịp/giây trong vòng lặp. Nạp model/warm-up không có phần trăm giả. Sau khi sẵn sàng, nhịp tiến độ tiếp theo báo `MODEL READY` kèm thời gian load/warm-up và RAM; báo cáo giữ `model_ready_rss_bytes`. Cuối phiên in chi phí từng công đoạn và đường dẫn báo cáo.

Kết quả nằm trong một thư mục mới dưới `runs/video/` hoặc `--output-dir`:

| File | Nội dung |
|---|---|
| `metadata.json` | Nguồn/FPS/phân giải, cấu hình, hash checkpoint, runtime, thiết bị thực |
| `detections.jsonl` | Frame, timestamp, box, confidence, chi phí mỗi lần nhận diện |
| `report.json` | Trạng thái, lỗi, counters, latency cold/warm, chi phí, bộ đệm, RAM/VRAM |
| `annotated.mp4` | Video full đã hoàn tất và kiểm tra |
| `encoder.log` | Chi tiết FFmpeg trong chế độ full |

`failed`/`stopped` không được báo là hoàn thành. Giữ file `.partial.mp4` khi bị ngắt để phục vụ chẩn đoán; không tự coi đó là video hoàn chỉnh. Mỗi lượt có tên riêng, không ghi đè nguồn hoặc lượt cũ.

Các mốc khác nhau:

- `first_result_available_*`: có kết quả trong bộ nhớ, chưa chắc đã hiển thị.
- `first_result_display_warm_ms`: kết quả đầu đã qua `imshow` và xử lý sự kiện HighGUI, tính từ sau warm-up. Đây là phép đo gần điểm trình bày, không phải đo photon trên màn hình.
- `first_result_display_cold_ms`: cùng sự kiện nhưng gồm toàn bộ khởi động, import runtime, kiểm nguồn và nạp model.
- `--no-display` và baseline không được báo đạt latency hiển thị; các trường tương ứng là `null`.
- FPS callback YOLO khác FPS hiển thị. Percentile từng công đoạn dùng tối đa 2.000 mẫu gần nhất; count/total/max tính cả phiên. RAM peak là đỉnh **lấy mẫu**, không phải bảo đảm đỉnh tuyệt đối. Tối đa 3.600 mẫu tài nguyên được giữ trong RAM.
- `display_fps_wall` gồm thời gian người dùng pause; `display_fps_excluding_user_pause` chỉ trừ thời gian pause tường minh, vẫn tính decoder buffering. Cả hai được giữ để không che giật/đệm. `user_pause_seconds` ghi riêng.
- Khi tua/phát lại full, hai trường FPS trung bình được để `null` và có ghi chú phiên tương tác; vẫn ghi chi phí từng lần hiển thị/tua. Không dùng phiên tua làm benchmark phát tuyến tính. Full không ghi mốc first displayed của fast; `None` ở mốc đó không phải lỗi phát.

Kiểm CFR quét timestamp cả nguồn bằng ffprobe, streaming và giới hạn bộ nhớ. Chi phí kiểm nguồn được báo riêng và tính vào cold start; video dài có thể mất thời gian đáng kể trước khi nạp model. Nguồn VFR/gián đoạn/hỏng bị từ chối rõ ràng. Bản đầu không tự chuyển đổi VFR. Chiều ảnh lẻ được pad thêm tối đa một pixel khi xuất H.264/yuv420p.

## Kiểm chứng

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_video*.py" -v
```

Thử cùng video/checkpoint/settings: baseline hiển thị, full không hiển thị, fast có hiển thị. Chạy riêng từng phiên để tránh tranh tài nguyên. Nguồn test tổng hợp chỉ kiểm pipeline/thời gian, không đánh giá chất lượng nhận diện lớp học. Không suy ra tốc độ RTX 5060 Laptop từ CPU/T4.

Chưa xác nhận mục tiêu 28,5+ FPS trên mọi máy, độ ổn định 30 phút/sáu giờ hoặc hiệu năng RTX 5060 Laptop nếu chưa có báo cáo chạy thực tế tương ứng. Không thay đổi webcam/Training UI. Kế hoạch chi tiết: [video_processing_plan_20261015.md](../../tasks/video/video_processing_plan_20261015.md).

Nguồn API: [OpenCV HighGUI](https://docs.opencv.org/4.10.0/d7/dfc/group__highgui.html), [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/), [FFmpeg](https://ffmpeg.org/ffmpeg.html), [ffprobe](https://ffmpeg.org/ffprobe.html).

Chỉ hiện và lưu detection người: `run_video.cmd --source "C:\video\lop_hoc.mp4" --mode full --person-only`. Cờ `--person-only` dùng được với full/fast và tracking; JSON detection cũng chỉ chứa person (ID 0). Không đổi trọng số hoặc confidence.

Trong giao diện mở bằng `run_video.cmd`, tích **Chỉ hiện người** trước khi bấm **Chạy** để chỉ vẽ/lưu detection person; bỏ tích để hiện đủ lớp. Dùng được với Nhanh và Đầy đủ.

**Kiểu box người** trong TUI là lựa chọn duy nhất: **Box gốc** (theo từng lần nhận diện, tracking off) hoặc **Box bám người** (ByteTrack, làm mượt và nội suy khoảng mất ngắn, chỉ Full). Chế độ Nhanh dùng Box gốc. “Chỉ hiện người” độc lập, dùng để ẩn lớp đồ vật. Tracking không giữ box vô hạn.

TUI có ô **Conf (0–1)** và mục **Lọc box trùng** (NMS tắt hoặc IoU 0,50/0,60/0,70). Conf cao hơn loại nhận diện điểm thấp nhưng có thể bỏ sót người. NMS lọc box chồng lấn; có thể gộp nhầm người đứng sát nhau. Đổi trước khi bấm Chạy; không tác động video đã xuất.
