# Bàn giao hiện trạng: đã làm, dùng được và còn thiếu

**Cập nhật cuối: 04/11/2026 — Asia/Saigon.**

### Bổ sung demo vùng bảng cho các video — 04/11

Đã xuất đủ 7/7 bản MP4 mới, giữ làm mượt box/Pose và logic giơ tay, đứng lên hiện có. Khánh.MOV và Khánh (1).MOV dùng hướng người dùng xác nhận: **mép dưới, phía camera**; vùng đích chuẩn hóa `[0, 0.8, 1, 1]`. Chỉ gợi ý đi lên bảng khi có chuyển động về hướng này; không phát nhãn viết bảng khi bảng nằm ngoài hình. Đô và Thành dùng vùng bảng/vùng sàn riêng theo nguồn; ở cạnh bảng được hiển thị viết bảng theo quy ước demo người dùng.

| Bản demo mới | Thời lượng |
|---|---:|
| [Khánh góc 2 — đã lọc, nên dùng trước](../../runs/demo/smooth_all_20261103/khanh_02_filtered_board_demo.mp4) | 10,711 s |
| [Khánh góc 1](../../runs/demo/smooth_all_20261103/khanh_01_board_demo.mp4) | 8,008 s |
| [Khánh góc 2 — cấu hình cũ](../../runs/demo/smooth_all_20261103/khanh_02_board_demo.mp4) | 10,711 s |
| [Đô — có vùng bảng trắng](../../runs/demo/smooth_all_20261103/do_main_board_demo.mp4) | 7,975 s |
| [Thành 22 giây](../../runs/demo/smooth_all_20261103/thanh_22s_board_demo.mp4) | 22,000 s |
| [Thành 20 giây](../../runs/demo/smooth_all_20261103/thanh_20s_board_demo.mp4) | 20,008 s |
| [Thành 10 giây](../../runs/demo/smooth_all_20261103/thanh_10s_board_demo.mp4) | 10,000 s |

Cấu hình, link phát web và mốc thời gian ở [board_all_manifest.json](../../runs/demo/smooth_all_20261103/board_all_manifest.json). Bảy bản đều xác nhận nativePTS hợp lệ; không chạy model lại, không đổi raw/GT. Đợt này kiểm 41 test board/renderer/export, 9 test API xuất video và Ruff đều qua (API có một warning Starlette). Chưa chạy lại toàn bộ cây.

**Lỗi cơ bản còn thấy trên ảnh trích video thật:** chữ có thể chồng nhau, có box/ID trùng, Pose rung hoặc bbox thay đổi do bàn che có thể gây gợi ý đi lên bảng sai. Bộ làm mượt chỉ giảm rung hiển thị, chưa giải quyết hết các lỗi này. Viết bảng theo vị trí vẫn là quy ước demo, chưa là nhận dạng động tác viết thật hoặc nhãn GT đã duyệt. Tùy chọn vùng bảng ngoài hình đã có qua API xuất; chưa thêm điều khiển riêng cho tùy chọn này trên web.

Đây là bản tổng hợp hiện trạng để nhóm tiếp tục công việc, không thay nguồn mục tiêu chính thức tại [tasks/plan.md](../plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026). Kế hoạch triển khai chi tiết ở [unified_demo_action_plan_20261102.md](unified_demo_action_plan_20261102.md), ngữ cảnh ở [project_memory.md](../project_memory.md).

**Kết luận hiện tại:** có thể demo phát hiện người, Pose, ID cục bộ, gợi ý một số hành vi theo thời gian và xuất video có nhãn. Chưa hoàn tất nhãn hành vi do người duyệt, nhận dạng viết bảng, giữ đúng ID qua mọi che khuất, đánh giá chất lượng trên test độc lập hoặc nghiệm thu toàn bộ đề tài.

Người dùng đã yêu cầu ưu tiên demo sớm, chấp nhận ghi lại lỗi cơ bản; tích hợp CMC để sau. Sau đó mục tiêu tự động đã được tạm dừng và xóa. File này chỉ bàn giao, không tự khởi động lại mục tiêu hay tiếp tục chạy thử nghiệm.

## 1. Những gì dùng được ngay

| Hạng mục | Dùng được hiện tại | Giới hạn cần nói khi demo |
|---|---|---|
| Web local thống nhất | Một web tại `http://127.0.0.1:8090/`; xem/tua video, Detect/Pose, tracking, gán nhãn và xuất kết quả | Chưa phải hệ thống CMC đã tích hợp |
| Detect người | Bbox người trên ảnh/video | Chưa có đánh giá mới đủ để cam kết recall trong lớp thật |
| Pose | Dùng model pretrained; raw COCO17 và lớp hiển thị Pose13 | Thiếu khớp khi quay lưng, che khuất, người xa hoặc ảnh mờ |
| Tracking | ByteTrack, ID cục bộ theo cảnh/phiên, buffer mất quan sát và cờ tái bắt | ID không phải danh tính sinh viên; còn tạo thêm ID hoặc đổi ID |
| Hành vi | Gợi ý giơ tay, đứng lên, di chuyển đến vùng bảng | Là luật nháp chưa hiệu chỉnh; chưa đủ bốn hành vi nghiên cứu |
| Ngữ cảnh | Có logic riêng giơ tay rồi đến bảng khi đủ bằng chứng cùng người liên tục | Không suy ý định; thiếu bằng chứng thì không tạo quan hệ |
| Video đầu ra | MP4 có bbox, ID, Pose và chữ hành động ngay cạnh người | Nhãn ghi trên video vẫn là gợi ý, không phải GT |
| JSON | Raw detections/Pose/tracks, thời gian nguồn, event/coverage và tổng hợp cấp lớp | Tổng hợp chỉ kết luận trên phần thực sự được rà soát |
| Công cụ gán nhãn | Xem video gốc, tạo người cục bộ, khoanh box tham chiếu, gán episode và coverage theo nhánh | Công cụ đã có không có nghĩa dữ liệu đã được người duyệt xong |
| Intel iGPU | Có tùy chọn Pose qua OpenVINO, đã chạy thật với `execution_devices=[GPU.0]` | Chưa có bảo đảm nhanh hơn CPU cho mọi cấu hình; Detect qua đường này chưa hỗ trợ |

Cách mở web khi cần: chạy `run_demo.cmd` ở gốc dự án. Không chạy thêm một máy chủ nếu cổng 8090 đang được phiên demo sử dụng.

## 2. Dữ liệu và file nhãn đang có

### 2.1. Video gốc

Đã kiểm lại sự tồn tại của cả tám nguồn trong registry:

| Nguồn | Frame nguồn đã chuẩn bị | Trạng thái |
|---|---:|---|
| Thành.MOV | 1.429 | Có video, preview và ledger thời gian |
| Thành (1).MOV | 989 | Có video, preview và ledger thời gian |
| Đô.MOV | 2.676 | Có video, preview và ledger thời gian |
| Khánh.MOV | 877 | Có video, preview và ledger thời gian |
| Khánh (1).MOV | 323 | Có video, preview và ledger thời gian |
| Khánh (2).MOV | 398 | Có video, preview và ledger thời gian |
| Sơn.MOV | 5.554 | Có video, preview và ledger thời gian |
| Phong.MOV | 5.451 | Có video, preview và ledger thời gian |
| **Tổng** | **17.697** | **8/8 nguồn; 78 khoảng rà soát** |

Đường dẫn, hash, metadata và link từng nguồn ở [classroom_sources.json](../../data/videos/classroom_sources.json). Lượt chuẩn bị chỉ giải mã/chuyển preview và lưu PTS, không tự chạy AI hoặc sinh GT. Bằng chứng ở [annotation_preparation_verification_20261103.json](../../runs/demo/annotation_preparation_verification_20261103.json).

**17.697 là số frame video đã chuẩn bị, không phải số ảnh JPG đã chọn để train.** Bộ khoảng 5.900 ảnh từng thảo luận chưa được xác nhận là một dataset mới đã trích, gán nhãn và kiểm đủ.

### 2.2. Ảnh và nhãn Detect/Pose cũ

| Bộ | Đang có | Lưu ý |
|---|---|---|
| Thành | 807 ảnh và JSON nhãn bbox/Pose | Có export; người dùng từng xác nhận duyệt bộ Thành |
| Đô | 893 ảnh và JSON nhãn bbox/Pose | Export hiện có 893 annotation; kết quả trùng prediction, bản thân export không chứng minh từng ảnh đã được sửa/duyệt thủ công |
| Detect train — Thành.MOV | 477 ảnh, 477 file nhãn, 1.799 bbox | Giữ quyết định dùng bộ lớn hơn làm train |
| Detect validation — Thành (1).MOV | 330 ảnh, 330 file nhãn, 1.387 bbox | Giữ split đã thống nhất |
| Detect test độc lập | Chưa có trong bộ Detect này | Không lấy frame gần nhau từ cùng video làm test độc lập |

File nhãn đã có trên máy:

- [JSON Thành](../../tools/label_studio/outputs/thanh_10fps_20261028/label_studio_annotations_20261028.json).
- [Bộ YOLO Detect Thành](../../tools/label_studio/outputs/thanh_10fps_20261028/yolo_detect_v1/).
- [JSON Đô](../../tools/label_studio/outputs/do_10fps_20261029/do_893_label_studio.json).

**Chưa sẵn sàng train Pose13 từ các export hiện tại:** audit gần nhất thấy 54.892 keypoint Thành/Đô thiếu trạng thái visibility tường minh. Confidence của prediction không thay visibility GT. Chưa có bằng chứng nghiệm thu một bộ Pose13 mới đã sửa đủ và xuất TXT hợp lệ.

Đã có notebook/công cụ chuẩn bị Detect và các checkpoint từ công việc trước. Bản bàn giao này không xác nhận chiến dịch train person mới hoặc model hành vi mới đã huấn luyện xong chỉ dựa vào việc có notebook/weights.

### 2.3. Nhãn hành vi theo thời gian

- Có [rubric bản nháp](../../configs/action_rubric.yaml): bốn hành vi `raise_hand`, `stand_up`, `move_to_board`, `write_board`; posture gồm ngồi, đứng, chuyển tư thế, không xác định.
- Posture loại trừ nhau trong cùng người/thời gian; action có thể chồng nhau. Một người có thể vừa đứng vừa giơ tay.
- Đã tạo pilot Thành: **19 episode nháp, 35 khoảng coverage nháp, 4 người cục bộ, 6 span và 6 box tham chiếu**.
- Nhãn do trợ lý xem RGB lấy mẫu, ghi rõ `origin=assistant`, `review=draft`, có provenance và mốc nativePTS. Chưa phải GT do nhóm xác nhận.
- Nhánh `codex-draft` giữ đề xuất ban đầu; `pilot-review` có bản sao để nhóm sửa/duyệt. Hai nhánh group-a/group-b vẫn trống tại lần kiểm cuối.
- Hai nhánh có nháp vẫn `evaluation_ready=false`, GT export trả 409; chưa biến coverage nháp thành âm đã duyệt.
- Ba episode chưa chắc: hai khoảng nghi viết bảng và một khoảng di chuyển khi người đã ở gần bảng.
- Chưa nhập nhãn hành vi mới cho Khánh/Sơn/Phong/Đô trong các lượt demo vừa rồi.

[Mở pilot Thành để duyệt](http://127.0.0.1:8090/?job=3826fbafed4e4f5da170f546c38dfbac&reviewer=pilot-review). Bằng chứng lưu tại [thanh_draft_verification.json](../../runs/demo/pilot_annotations_20261103/thanh_draft_verification.json) và [live_review_gate_verification.json](../../runs/demo/pilot_annotations_20261103/live_review_gate_verification.json).

## 3. Pose và logic hành vi: đã có đến đâu

**Cập nhật theo quy ước demo người dùng:** có chế độ xuất `board_demo_proximity=true`, khoanh mặt bảng trắng và vùng đứng cạnh bảng riêng. Người đang tiến về vùng bảng được gợi ý đi lên bảng; ở cạnh bảng được đặt nhãn viết bảng theo đúng quy ước vừa yêu cầu; đứng lên giữ logic chuyển ngồi→đứng. Đây là alias phục vụ demo, không xác nhận động tác tạo nét viết và không thay rubric/GT nghiên cứu. [Video Thành 18 giây có vùng bảng và ba nhãn](../../runs/demo/smooth_all_20261103/thanh_board_actions_smooth.mp4).

| Nội dung | Đã làm | Chưa làm xong / chưa chứng minh |
|---|---|---|
| Giơ tay | Có luật cũ dựa mũi/hai tay và cue tùy chọn cổ tay trên vai khi quay lưng; có dwell và ngưỡng vào/ra | Chưa phân biệt tin cậy mọi trường hợp chạm đầu, vươn vai, vẫy tay, tay nghỉ trên bàn |
| Không giơ tay | Chỉ phát quan sát tay hạ khi cả hai tay đủ bằng chứng | Thiếu khớp hoặc `no_suggestion` không được coi là âm |
| Đứng lên | Có proxy chuỗi ngồi → nâng thân → đứng từ lower-body Pose | Chưa đủ GT để chốt ngưỡng, recall hoặc độ chính xác |
| Di chuyển lên bảng | Có proxy quỹ đạo và đến ROI sàn theo thời gian nguồn | Cần đặt vùng đúng từng camera; đi ngang hoặc đứng sẵn gần bảng có thể gây nhầm |
| Giơ tay rồi lên bảng | Quan hệ ngữ cảnh riêng giữa hai bằng chứng cùng ID/cảnh và lịch sử hỗ trợ liên tục | Trên clip Thành hiện không tạo quan hệ vì không đủ bằng chứng liên tục; không tự ép nối |
| Viết bảng | Có định nghĩa nhãn và chỗ gán uncertainty/episode | **Chưa có detector hành vi viết bảng tự động đủ tin cậy**; gần bảng/giơ tay không chứng minh viết |
| Outlier Pose | Có kiểm finite/confidence/raster/hình học; gate chuyển động nhân quả tùy chọn | Ngưỡng chưa được hiệu chỉnh trên dev; không sửa được lỗi model hoặc danh tính bằng gate alone |

Gói `classroom-engagement (1).zip` đã được đọc như tài liệu tham khảo. Dữ liệu Pose đi kèm là tổng hợp, không phải GT của các video thật. Logic ở bảng và combo trong archive không tự trở thành nhãn hành vi đã xác nhận.

Ví dụ lỗi thật đã thấy: RGB pilot Thành có hai lần giơ tay khoảng 6,003–8,003 s và 8,503–19,508 s; cue vai/cổ tay lại gộp thành một đoạn khoảng 6–19 s. Nhãn nháp RGB giữ hai episode; không chỉnh nhãn để khớp kết quả luật.

## 4. Tracking và ổn định: đã sửa gì, còn gì

Đã có:

- ByteTrack, ID chỉ có nghĩa trong cảnh/phiên; không tự nối người qua cảnh.
- Mapping detection index → raw Pose được kiểm bằng fixture đảo thứ tự, giao nhau và mất quan sát.
- Buffer mất quan sát; provenance tái bắt, `gap_seconds`, cờ bất định ngắn trước khi dùng lại hành vi.
- Tách ngưỡng detector, high/low association và tạo ID mới; baseline giữ nguyên, tùy chọn thử được ghi vào cấu hình.
- Bản thử Khánh dùng NMS 0,50 và ngưỡng tạo ID 0,10 để giảm box trùng.
- Sửa đường xuất MP4 vốn vẽ tọa độ thô: thêm **làm mượt box/Pose chỉ cho hiển thị**, theo thời gian nguồn, reset khi mất quan sát/cut/ID bất định/nhảy lớn.
- Bộ lọc hiển thị không bù khớp thiếu, không sửa raw17 hoặc bằng chứng hành vi. Chưa giữ nhãn dương qua khoảng Pose không rõ để che lỗi.

So sánh cùng clip Khánh(1), 321 frame:

| Quan sát model | Bản đầu | Bản lọc |
|---|---:|---:|
| Tổng ID từng confirmed | 21 | 10 |
| Tổng quan sát confirmed | 1.747 | 1.314 |
| Cặp-frame có box IoU > 0,5 | 539 | 0 |

**Bảng này không phải IDF1, HOTA hoặc số ID-switch thật.** Ít box/ID hơn có thể đồng thời bỏ người yếu thật. ID còn nhiều hơn số người nhìn thấy trong clip; chưa chứng minh giữ đúng một ID cho mỗi người suốt video. Làm mượt giảm rung hình, không sửa danh tính khi association sai.

Còn thiếu: rà RGB các đoạn đổi/mất ID, nhãn tracking độc lập, kiểm người bị bỏ sau lọc và đánh giá ghép lại sau che khuất. Chưa chốt cần thêm phương pháp nhận diện lại theo ngoại hình hay đổi tracker; chưa có bằng chứng để tuyên bố một phương án mới tốt hơn.

Bằng chứng: [stability_trial.json](../../runs/demo/extra_demos_20261103/stability_trial.json), [tracking_review_20261103.json](../../runs/demo/tracking_review_20261103.json), [pose_quality_review_20261103.json](../../runs/demo/pose_quality_review_20261103.json).

## 5. Video demo đã lưu — tất cả đã có bản làm mượt

Cập nhật theo yêu cầu **“bật mượt all”**: cả 7 clip đều có phiên bản làm mượt box/Pose. Backend và hàm xuất MP4 mặc định `stabilize_display=true`, gồm xuất thủ công và tự lưu sau Full/Pose. Các bản cũ vẫn giữ để đối chiếu.

| Video mới | Thời lượng | Trạng thái |
|---|---:|---|
| [Khánh góc 2 — đã lọc trùng + làm mượt](../../runs/demo/smooth_all_20261103/khanh_02_filtered_smooth.mp4) | 10.711 s | Đã làm mượt hiển thị |
| [Khánh góc 1](../../runs/demo/smooth_all_20261103/khanh_01_smooth.mp4) | 8.008 s | Đã làm mượt hiển thị |
| [Khánh góc 2 — giữ cấu hình bản đầu](../../runs/demo/smooth_all_20261103/khanh_02_smooth.mp4) | 10.711 s | Đã làm mượt hiển thị |
| [Đô](../../runs/demo/smooth_all_20261103/do_main_smooth.mp4) | 7.975 s | Đã làm mượt hiển thị |
| [Thành 22 giây](../../runs/demo/smooth_all_20261103/thanh_22s_smooth.mp4) | 22.000 s | Đã làm mượt hiển thị |
| [Thành 20 giây](../../runs/demo/smooth_all_20261103/thanh_20s_smooth.mp4) | 20.008 s | Đã làm mượt hiển thị |
| [Thành 10 giây](../../runs/demo/smooth_all_20261103/thanh_10s_smooth.mp4) | 10.000 s | Đã làm mượt hiển thị |

**Làm mượt hiển thị không đồng nghĩa sửa hết ID-switch hoặc nhấp nháy nhãn.** Các bản giữ tracker/cue/ngưỡng của nguồn tương ứng; chỉ bản Khánh đã lọc có thêm NMS0,50/newID0,10. Raw Pose và bằng chứng hành vi không bị thay đổi. Không bù khớp thiếu hoặc kéo khung xương qua người khác.

[Manifest các bản mới](../../runs/demo/smooth_all_20261103/manifest.json) giữ link web, report, hash, cấu hình và xác nhận bản cũ không đổi. Sáu bản xuất mới đều kiểm nativePTS/bộ lọc; bản Khánh đã mượt được giữ nguyên và sao chép đúng hash. Không chạy AI lại trong lượt bật mượt toàn bộ.

## 6. Đối chiếu sáu nội dung nghiên cứu chính thức

Các số thứ tự dưới đây tham chiếu nguồn chính thức, không tạo mục tiêu mới.

| Mục | Trạng thái hiện tại | Điều còn thiếu để khép |
|---|---|---|
| 1 — khảo sát/tiêu chí | Có rubric hành vi bản nháp và nguyên tắc diễn giải | Nhóm/giảng viên thống nhất, ghi rõ tiêu chí và xử lý bất đồng |
| 2 — dữ liệu | Có video, dữ liệu ảnh cũ, công cụ gán và pilot nháp | GT người duyệt; visibility Pose; tổ chức/khóa split theo nguồn/buổi quay |
| 3 — phát hiện/tracking/chuỗi | Có pipeline và ba proxy hành vi, demo thực | Viết bảng; ổn định ID; hiệu chỉnh và so sánh phương pháp trên GT |
| 4 — tổng hợp | Có event/coverage, tần suất/thời lượng/ngữ cảnh và JSON cấp lớp | Đánh giá trên nhãn thật, đối chiếu giảng viên; chưa có điểm tương tác được duyệt |
| 5 — tích hợp/độ trễ | Có web local, API/JSON, replay và một số số đo xử lý | Tích hợp CMC để sau; chưa nghiệm thu độ trễ camera → hành vi dưới 3 giây |
| 6 — lớp thật/chất lượng | Có kiểm phần mềm và thử clip thật | Thử lớp học, test độc lập, chất lượng Detect/Pose/tracking/hành vi, tải dài và điều kiện riêng tư |

**Chưa hoàn tất toàn bộ sáu mục.** Demo hoặc test phần mềm xanh không thay nghiệm thu nghiên cứu.

## 7. Việc có thể tiếp tục và việc cần người trong nhóm

### 7.1. Có thể tiếp tục bằng công cụ/code hiện có

- [x] Bật làm mượt box/Pose mặc định cho mọi lần xuất MP4 và tạo bản làm mượt cho 7/7 clip đã bàn giao. Bộ lọc vẫn có thể tắt tường minh qua API; phần ổn định ID/nhãn còn mở.
- [ ] Rà lỗi ID/Pose theo các cửa sổ cụ thể, xuất cặp video so sánh và giữ provenance cấu hình.
- [ ] Chuẩn bị thêm nhãn nháp RGB cho nguồn khác để giảm công gán, luôn tách nháp khỏi GT.
- [ ] Sau khi GT/rubric được duyệt: chạy baseline/evaluator, hiệu chỉnh trên dev rồi khóa trước test.
- [ ] Bổ sung đánh giá riêng Detect/Pose/tracking; evaluator sự kiện hiện có không tự cung cấp OKS/IDF1.
- [ ] Nghiên cứu baseline viết bảng có đủ RGB/ngữ cảnh mặt bảng sau khi có nhãn mẫu phù hợp.
- [ ] Đo tải dài, cold/warm start, tài nguyên và độ trễ thật theo giao thức được chốt.

Đây là các bước khả thi tiếp theo, không phải cam kết mọi bước tự hoàn thành trong tháng. Thời gian phụ thuộc nhãn thật, chất lượng video và điều kiện lớp/camera.

### 7.2. Cần người trong nhóm/giảng viên hoặc đầu vào bên ngoài

- [ ] Duyệt rubric và pilot Thành; xác nhận người, biên episode, âm khó, uncertainty và coverage đã thực sự xem.
- [ ] Gán độc lập/xử lý bất đồng; không chỉ đổi hàng loạt mọi nhãn nháp thành reviewed.
- [ ] Xác nhận quan hệ giữa video/buổi quay/camera để chia train/dev/test không rò rỉ.
- [ ] Bổ sung hoặc chỉ rõ vị trí các gói nhãn của thành viên khác nếu đang nằm ngoài khu vực đã kiểm.
- [ ] Gán visibility thật nếu muốn train Pose13 từ ảnh cũ.
- [ ] Có mẫu rõ cho đủ bốn hành vi, nhất là viết bảng và các ca dễ nhầm.
- [ ] Tổ chức thử lớp học, đối chiếu với giảng viên và xác nhận quyền sử dụng/lưu/chia sẻ video.
- [ ] Khi tiếp tục tích hợp CMC: có hệ thống đích, đặc tả API/xác thực và quyền trao đổi dữ liệu. Hiện được để sau theo yêu cầu.

### 7.3. Thứ tự ưu tiên đề nghị khi làm tiếp

1. Duyệt pilot và rubric trước; chốt các ca giơ tay/chạm đầu/tay trên bàn, đứng lên và viết bảng chưa rõ.
2. Kiểm tracking ở các đoạn che khuất/giao nhau; đánh giá cả mất người lẫn nhầm người, không chỉ số ID giảm.
3. Mở rộng GT trên các nguồn và khóa source-group/split hợp lệ.
4. Hoàn thiện baseline bốn hành vi và đánh giá định lượng trên test đã khóa.
5. Thử lớp thật, đo độ trễ/tải dài và đối chiếu giảng viên; tích hợp CMC khi nhóm mở lại phạm vi đó.

## 8. Những điều chưa được phép kết luận từ hiện trạng

- Chưa thể nói toàn bộ ảnh/video đã đủ nhãn hành vi hoặc sẵn sàng train model chuỗi.
- Chưa thể nói ID ổn định hoàn toàn, hết nháy, hết nhầm hoặc mỗi người luôn có duy nhất một ID.
- Chưa có detector viết bảng đã kiểm chứng.
- Chưa có kết quả chất lượng độc lập cho đủ bốn hành vi hoặc nghiệm thu độ trễ dưới 3 giây từ camera thật.
- Chưa có thang điểm/phân mức tương tác được giảng viên duyệt; một hành động không chứng minh mức độ năng nổ.
- Chưa tích hợp CMC và chưa nghiệm thu toàn bộ đề tài.

## 9. Kiểm chứng phần mềm đã ghi nhận

- Đợt hồi quy toàn cây trước phần làm mượt hiển thị: **762 Python tests qua**, 1 skip Windows, 1 warning Starlette; **163 frontend tests qua**.
- Sau sửa cuối phần proposals: 3 API regression liên quan raster/legacy qua riêng.
- Sau thêm bộ làm mượt: **41 tests filter/renderer/export qua**; Ruff các file liên quan qua. Chưa ghi nhận chạy lại toàn bộ cây sau thay đổi này.
- Các số test ở đây là kết quả đã chạy trong các lượt triển khai, không phải vừa chạy lại để viết tài liệu và không phải metric chất lượng nhận dạng.

File code chính để người tiếp tục tra cứu: `src/classroom_ai/interfaces/web.py`, `web.html`; `video/actions.py`, `pose_rules.py`, `pose_quality.py`, `tracking.py`, `display_stability.py`, `rendering.py`, `review.py`, `action_evaluation.py`.

Các file source/nhãn/raw và video cũ được giữ. Đã có một số đợt dọn fixture thử nghiệm được ghi trong `runs/demo/cleanup_inventory_20261103.json`; không tuyên bố đã dọn hết file thừa hoặc hoàn tất mọi refactor của repository.

Cập nhật kiểm thử sau khi bật mặc định: 42 tests filter/renderer/export và 9 tests API xuất video qua; chưa chạy lại toàn bộ cây cho thay đổi mặc định này.
