# Kế hoạch v1: dữ liệu lớp học thật, gán nhãn và AI hỗ trợ

Ngày lập/đối chiếu nguồn: **15/10/2026 — Asia/Saigon**. Trạng thái: **đề xuất để nhóm xem xét; chưa triển khai**. Các con số về dữ liệu, công sức và ngưỡng qua vòng là ước tính thiết kế, không phải kết quả thực nghiệm hoặc cam kết chất lượng.

Nguồn mục tiêu duy nhất: [mục chính thức trong plan.md](../plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026). Tài liệu này cụ thể hóa công việc dữ liệu, không sửa tên đề tài hoặc chọn thay nhóm kiến trúc, công cụ, ngân sách. Checklist nằm ngay trong tài liệu này theo phạm vi được giao; không thay `tasks/todo.md` đang có công việc khác.

## 1. Hướng đi nên bắt đầu

**Có thể tự gán nhãn, dùng AI tạo nhãn nháp rồi sửa thành bộ dữ liệu huấn luyện model riêng.** Khó nhất thường là thống nhất nhãn, tìm người bị bỏ sót và kiểm chất lượng, không phải thao tác kéo box. Chỉ dùng output dịch vụ thương mại cho training khi đã xác minh quyền sử dụng cụ thể; việc sở hữu output không tự giải quyết mọi hạn chế hợp đồng.

Luồng mặc định đề xuất cho nhóm 2–3 sinh viên:

1. Xin phép và xác minh điều kiện ghi hình/sử dụng dữ liệu với trường; lưu dữ liệu nội bộ.
2. Chọn 50–100 ảnh đại diện để viết hướng dẫn và gán độc lập bằng tay; tạo bộ đo lỗi ban đầu.
3. Dùng **CVAT Community tự host** để gán `person`, thử detector local tạo box nháp trên dữ liệu train. Người sửa mọi ảnh và rà toàn cảnh.
4. Khi detector đã được đo, gán một tập video ngắn có ID theo chuỗi và các khoảng hành vi. Chỉ bổ sung pose nếu thí nghiệm chứng minh cần.
5. Khóa tập kiểm thử thủ công độc lập; so pretrained n/s và checkpoint hiện có công bằng; chọn cải tiến theo lỗi thực tế.

Chưa cài CVAT, tải model, trích ảnh, gửi video/ảnh thật ra ngoài, trả tiền, huấn luyện, triển khai tracking hay webcam trong nhiệm vụ lập kế hoạch này.

### Bằng chứng nền hiện có

Theo [báo cáo chẩn đoán](../../reports/video/recall_diagnosis_20261015/README.md), model E3 bỏ sót ngay khi gọi Ultralytics trực tiếp trên ba frame. Lượt được kiểm dùng conf thực tế 0,25; mặc định mã video hiện là 0,05. Hạ conf/tăng imgsz/đổi nhánh NMS chưa chứng minh giải quyết vấn đề; số box không phải Recall. Validation lịch sử không đại diện video lớp của nhóm. Dataset Project8 v05 lịch sử chưa hoàn tất semantic review.

[Video CLI/TUI](../../docs/guides/video_processing.md) hiện có full xử lý mọi frame và fast lấy mẫu mặc định 5 FPS; chưa tracking. Box chỉ vẽ trên frame đã xử lý. Kế hoạch dữ liệu này không tự đổi các mặc định đó. Máy hiện có PyTorch CPU; RTX 5060 Laptop mới là thiết bị demo có thể dùng, chưa có benchmark. Dưới 3 giây là hướng độ trễ xử lý, không giới hạn lịch sử quan sát hành vi.

## 2. Bốn tầng nhãn, không thay thế nhau

| Đầu ra cần nghiên cứu | Nhãn tối thiểu | Chưa cần ngay | Định dạng giữ lâu dài |
|---|---|---|---|
| Phát hiện người trong ảnh | Mỗi người đủ điều kiện một bbox + class; ảnh không người đã kiểm | ID, tên, pose, hành vi | CVAT native + YOLO detection |
| Phát hiện vật trong 8 lớp | Box đầy đủ mọi instance của từng lớp đã công bố | Hành vi từ một vật đơn lẻ | Native + YOLO detection 8 lớp |
| Theo một người xuyên đoạn video | Bbox theo frame + ID cục bộ + trạng thái nhìn thấy/che/mất dấu | Danh tính thật, ReID xuyên mọi buổi | Native video; MOT cho bài đo bbox-ID, sidecar thuộc tính |
| Nhận biết giơ tay/đứng lên/đi lên bảng/viết bảng | Video liên tục; khoảng thời gian hoặc event; liên kết track khi đủ bằng chứng; vùng bảng/ngữ cảnh | Pose nếu cách dùng ảnh/track đủ tốt | `events.jsonl` + native video + vùng cảnh |
| Học/đánh giá phương pháp pose | Keypoint theo skeleton cố định, trạng thái visibility và chuỗi | Gán toàn bộ dataset pose trước khi chọn hướng | Native skeleton + COCO Keypoints/YOLO pose phù hợp |

**ID track là mã tạm trong một chuỗi**, ví dụ `S01_C01_clip03_T007`; không phải mã sinh viên hoặc tên. Không cam kết ID giữ nguyên qua che khuất dài, ra vào khung, đổi camera hay đổi buổi. Tracking có thể làm ổn định chuỗi, nhưng không biến người hoàn toàn không có bằng chứng thành detection đúng.

**Đề xuất person-only giai đoạn đầu:** tạo dataset/checkpoint nghiên cứu riêng `classroom_person_v1`, `names: {0: person}` để giảm công gán và đo trực tiếp lỗi bỏ sót người. Đây không phải quyết định bỏ 7 lớp của ứng dụng. Không nạp checkpoint một lớp vào ứng dụng 8 lớp khi chưa có adapter/class contract được kiểm. Nếu tiếp tục training 8 lớp, phải gán đầy đủ cả 8 lớp trong các ảnh dùng train; không để vật không gán bị học thành nền. Không trộn trực tiếp class ID COCO với Project8; COCO dining table cũng không tự đồng nghĩa mọi loại bàn của Project8.

## 3. Thu dữ liệu và pilot vừa sức

### Giả định v1

Nhóm 2–3 người, lớp khoảng 20–40 người, 1 camera cố định, tiếp cận 3–5 buổi và có máy lưu dữ liệu nội bộ. Chưa biết quyền dùng cloud; mặc định lập luồng local. Nếu chỉ có một buổi/clip, vẫn làm được pilot nhưng không được tuyên bố tổng quát sang buổi/người/camera mới.

### Ghi hình có chủ đích

- Thử một góc chéo đủ cao nhìn được đầu/thân/tay và lối đi; kiểm cả hàng xa và vùng bảng. Chụp thử trước, phóng ảnh nguồn để xem người xa còn bao nhiêu pixel. Tăng độ phân giải xử lý không khôi phục được chi tiết camera không ghi được.
- Giữ camera ổn định trong một đoạn; ghi mã camera, góc, độ cao ước tính, độ phân giải, FPS, codec, ánh sáng và thay đổi cấu hình. Với nhiều camera, mỗi nguồn có timestamp/đồng bộ rõ; chưa nối ID liên camera ở v1.
- Bao phủ sáng/tối/ngược sáng, gần/xa, ngồi/đứng, các mức che, biên khung, mờ chuyển động, hàng đông, lối đi và bảng. Thêm đoạn không có hành vi mục tiêu để đo báo nhầm.
- Không chỉ quay bạn đứng diễn rõ ở hàng đầu. Có thể dựng tình huống có đồng thuận để bổ sung hành vi hiếm nhưng gắn `staged=true`; báo riêng đánh giá tự nhiên và dàn dựng. Không tạo hành vi gây gián đoạn/bất lợi cho người học.
- Thu đủ ngữ cảnh trước/sau hành động, ví dụ clip 20–60 giây chứa sự kiện và khoảng nền. Không cắt đúng một ảnh “tay giơ” rồi dùng để huấn luyện thời lượng hoặc diễn biến.
- Không thu âm nếu mục tiêu v1 không cần âm thanh. Kiểm cả bảng tên, màn hình, danh sách lớp và thông tin trên bảng trước khi chia sẻ.

### Quy mô theo vòng, không đặt quota ảnh làm bảo đảm Recall

| Vòng | Khối lượng dự kiến | Mục đích/điều kiện mở rộng |
|---|---|---|
| R0: thống nhất rubric | 50–100 ảnh từ nhiều cảnh + 3–5 clip ngắn | Hai người gán độc lập, giải quyết mọi bất đồng về ontology trước |
| R1: detection pilot | Khoảng 300–500 ảnh khác biệt, từ ít nhất 3 buổi nếu có | Có đủ cảnh khó và bộ đo độc lập; đo thời gian/ảnh thực |
| R2: học theo lỗi | Thêm từng lô 200–300 ảnh; có thể tới 1.000–2.000 ảnh nếu nguồn lực cho phép | Chỉ thêm để lấp nhóm lỗi/độ đa dạng; vẽ learning curve trên cùng val |
| T1: tracking/event pilot | 10–20 clip × 20–60 giây; 2–3 clip hoàn chỉnh để kiểm MOT trước | Bắt đầu với một hành vi dễ thống nhất; mở rộng tới 4 hành vi sau QA |
| P1: pose tùy chọn | Một tập nhỏ các người/frame khó đã chọn từ train/val | Chỉ làm khi phân tích lỗi hoặc ablation có lý do cần pose |

500 ảnh × 25 người đã khoảng 12.500 box; 500 ảnh không tương đương 500 lần kéo chuột. Với hành vi hiếm, ước tính 20–30 sự kiện/lớp trong pilot chỉ giúp thử pipeline; không đủ mặc định để kết luận Recall cao. Ghi cả số sự kiện và số người/buổi độc lập.

### Lấy mẫu không tạo ảnh gần trùng

Detection: bắt đầu khoảng một frame mỗi 2–5 giây ở các đoạn đã chọn; sau đó bỏ ảnh gần trùng và bổ sung frame có đổi tư thế/che/đi vào khung. Đây là khoảng khởi đầu, không ràng buộc cứng. Giữ mẫu ngẫu nhiên theo phân tầng bên cạnh mẫu lỗi để không chỉ đo tập hard mining. Không lấy toàn bộ video 25/30 FPS làm tập ảnh độc lập.

Tracking và event: giữ video liên tục ở FPS nguồn cho ground truth/pilot; có thể đặt keyframe thưa khi gán rồi kiểm các frame nội suy. Ghi mọi ánh xạ frame/timestamp nếu đổi FPS hoặc tạo proxy. Với variable frame rate dùng timestamp trình chiếu (PTS), không suy mọi timestamp từ `frame/FPS`. Giữ source hash và manifest; bản đã vẽ box của model không làm ảnh nguồn để gán nhãn.

## 4. Ontology và hướng dẫn gán nhãn v1

Viết thành `annotation_guide_v1.md` trong đợt triển khai, kèm ít nhất 15–20 ví dụ dễ/khó đã đồng thuận. Quy tắc dưới đây là đề xuất xuất phát để thử R0.

### 4.1 Bbox person

**Đề xuất chọn visible bbox**: hình chữ nhật song song trục bao kín toàn bộ phần cơ thể nhìn thấy của từng người trong ảnh, kể cả các phần rời nhau do che; không suy thêm chân sau bàn hoặc thân ngoài khung. Box vẫn chứa khoảng trống/vật che giữa các phần nhìn thấy, vì bbox không phải mask. Không chọn amodal (ước lượng toàn thân bị che) lẫn với visible trong cùng dataset. Phải kiểm độ tương thích khi ghép dữ liệu lịch sử/pretrained; nếu cần amodal cho một thí nghiệm, dùng version nhãn riêng và báo khác protocol.

| Trường hợp | Quy tắc thao tác |
|---|---|
| Hai người chồng nhau | Vẽ riêng từng người; box giao nhau là hợp lệ. Không gộp hàng ghế thành một box |
| Chỉ lộ đầu/vai | Nếu xác định được một người riêng biệt, vẫn gán bbox phần nhìn thấy, không chỉ gán người toàn thân |
| Che khoảng 50–70% | Gán phần nhìn thấy; thuộc tính `occlusion=heavy` nếu ước lượng che trên nửa. Đây là đánh giá thô, không là phép đo diện tích chính xác |
| Bị cắt biên ảnh | Clip box trong kích thước ảnh, đánh dấu `truncated=true`; không kéo ra ngoài ảnh |
| Tay rời hoặc cụm pixel chưa rõ thuộc ai | Không bịa người riêng; đánh dấu vùng `uncertain`, chuyển reviewer. Dùng video lân cận hỗ trợ nhưng không suy ra phần không nhìn thấy |
| Hoàn toàn bị che/ra ngoài khung | Không tạo bbox detection GT tưởng tượng. Ghi khoảng mất quan sát trong metadata track |
| Quá nhỏ/mờ, không thể xác định là người | Đưa vùng cần phân xử; tiêu chí ignore phải khóa trước khi đo. Không loại chỉ vì model không phát hiện |
| Ảnh thực sự không có người | Chỉ đánh dấu background sau khi rà toàn ảnh; label rỗng đã kiểm khác với chưa gán |

V1 đề xuất gán mọi người có thể phân biệt, không đặt ngưỡng kích thước loại người xa trước khi xem pilot. Nếu reviewer thấy không thể nhất quán, công bố tiêu chí “không thể xác định”, vùng ignore và coverage theo kích thước/cảnh trước vòng benchmark. Không dùng ignore như cách né nhóm lỗi.

Thuộc tính native/sidecar: `occlusion={none,partial,heavy,unknown}`, `truncated`, `uncertain`, `review_status`, `source={manual,ai_corrected}`. Không thêm `confidence` vào YOLO GT detection. Score của AI là score của dự đoán; độ chắc chắn do LLM tự khai không phải xác suất đã hiệu chuẩn.

### 4.2 Track và mất dấu

- Một track/một người trong đoạn liên tục có bằng chứng. Kiểm danh sách ID khi đi cắt nhau, đứng lên, xuống hàng, ra/vào biên.
- `visible`: có bằng chứng đủ để gán box; `occluded_partial`: còn phần người; `lost_full`: không còn thấy; `uncertain_identity`: thấy người nhưng không đủ căn cứ nối với ID cũ.
- Với `lost_full`, tắt box quan sát ở khoảng đó (`outside` trong cách biểu diễn CVAT phù hợp), giữ bảng gap riêng; không để nội suy sinh box GT qua khoảng che hoàn toàn. `occluded` của công cụ không tự đồng nghĩa trạng thái lost/unknown của rubric.
- Nối lại ID khi video liên tục cung cấp bằng chứng đáng tin; nếu không chắc, tạo track mới và ghi quan hệ `possible_same_as` để reviewer quyết định. Không dựa tên/khuôn mặt để ép nối. Giữ mapping giữa job khi cắt clip; không coi cùng số ID ở hai task là cùng người.
- Nếu muốn đánh giá MOT theo một convention full-body/occlusion khác, tạo protocol và dữ liệu đánh giá tương ứng riêng. Không xuất nhãn visible không box ở gap rồi so trực tiếp với benchmark khác convention.

### 4.3 Hành vi là khoảng thời gian và sự kiện

Định nghĩa thời gian theo giây từ đầu clip; interval **[start_s, end_s)**, liên kết `track_uid` hoặc `null` nếu chưa chắc ID. Giữ `boundary_uncertainty_s`, `left_censored/right_censored` khi hành vi đã bắt đầu trước clip hoặc chưa kết thúc. Không ép nhãn âm tính khi bị che; dùng `unobservable/uncertain`.

| Nhãn đề xuất | Bắt đầu/kết thúc quan sát | Dễ nhầm cần ghi trong rubric |
|---|---|---|
| `hand_raise` | Bắt đầu khi tay vào tư thế giơ theo rubric lớp; kết thúc khi hạ khỏi tư thế. Ghi event onset để đếm một lần | Gãi đầu/vươn vai/chạm tóc; không kết luận “muốn phát biểu” nếu không có ngữ cảnh |
| `stand_up` | Chuyển từ ngồi sang đứng: onset khi bắt đầu nâng thân, offset khi đạt đứng ổn định; thêm state `standing` nếu cần đo thời lượng đứng | Người đã đứng ở đầu clip không tạo một onset mới; đứng không tự là tương tác tích cực |
| `move_to_board` | Bắt đầu rời chỗ trên quỹ đạo hướng vùng bảng; kết thúc tới vùng bảng hoặc dừng/hủy. Lưu trạng thái `reached/aborted/uncertain` | Đi ngang bảng/ra cửa; chỉ xác nhận đích khi đủ diễn biến. Online có thể phải phát dự đoán tạm rồi cập nhật |
| `write_on_board` | Có chuyển động tay/dụng cụ và bằng chứng viết lên bảng; kết thúc khi ngừng theo quy tắc tạm dừng đã chốt | Đứng/chỉ tay ở bảng chưa đủ; không nhìn rõ thì unknown |

Trong R0, nhóm/giảng viên chốt ví dụ tư thế “giơ tay”, ngưỡng duy trì ban đầu thử 0,5 giây, quy tắc gộp khoảng ngắt ngắn thử 0,5 giây và dung sai biên thử ±0,5 giây. Đây là tham số rubric đề xuất, phải hiệu chỉnh bằng bất đồng annotator trên train/pilot và khóa trước test. Không loại hành vi ngắn khỏi báo cáo mà không ghi coverage. Hành vi có thể đồng thời (đứng và giơ tay); không ép bốn nhãn thành softmax độc quyền.

Tổng hợp tần suất = số onset hợp lệ; thời lượng = tổng khoảng quan sát xác nhận, không cộng thời gian mất quan sát như hành vi thật. Báo phần thời gian quan sát được và ngữ cảnh; giảng viên diễn giải. Không suy ngủ từ cúi đầu ngắn hay tính năng nổ từ một hành động.

### 4.4 Pose chỉ khi cần

Nếu chọn COCO 17 điểm, khóa thứ tự, trái/phải theo cơ thể người, topology, visibility và ánh xạ export. Điểm thấy được đặt đúng vị trí; điểm bị che dùng convention của format/rubric; điểm không thể suy được đánh dấu không gán, không bịa. Pose AI xuất sẵn chỉ là prediction nếu chưa sửa. Gán toàn bộ skeleton trên mọi frame rất tốn công và không tự sinh nhãn hành vi; thử trước trên tập nhỏ có tay/khớp bị che.

## 5. Chọn phần mềm: tài liệu hiện hành và giới hạn

Các trang web là tài liệu online ngày đối chiếu, không chứng minh tính năng trong bản cài của nhóm. Trước triển khai ghi version/tag/edition thực, thử round-trip 5 ảnh và 1 clip. Không suy mọi chức năng QA hoặc AI đều có trong bản Community.

| Công cụ | Bbox, video/ID, pose/event | Export và nơi chạy | Chi phí/ràng buộc cần biết | Đánh giá cho nhóm |
|---|---|---|---|---|
| **CVAT** | Rectangle ảnh; track video có keyframe/nội suy; skeleton. Event nên giữ sidecar vì không coi box export là interval dataset | Community tự host; Online cloud. Native giữ chi tiết; Ultralytics YOLO cho box/pose, MOT tùy bài | Community miễn phí, tự lo máy/lưu trữ/vận hành. Online Free có quota riêng | Mặc định vì cùng luồng ảnh và video; thao tác sửa keyframe hợp tình huống lớp đông |
| **Label Studio** | Rectangle ảnh, `VideoRectangle` cho tracking; keypoints ảnh; cấu hình cho nhãn thời gian cần thử template/tag | Open Source tự host; dịch vụ thương mại/cloud. Native JSON giữ cấu trúc, YOLO chủ yếu cho tag ảnh được hỗ trợ | OSS không đồng nghĩa mọi review/enterprise workflow miễn phí; tốn công cấu hình/chuyển video | Thay thế khi ưu tiên rubric đa dạng/event; dùng JSON làm gốc |
| **Roboflow** | Công cụ annotation ảnh, dataset versioning; video thường cần trích frame. Không chọn làm nguồn GT ID/event liên tục khi chưa thử tính năng cụ thể | Cloud; YOLO export theo version. Không coi local Inference là tự host toàn bộ Annotate | Free/Public không phù hợp mặc định cho dữ liệu người học; private phải kiểm plan/hợp đồng | Dễ cho dataset ảnh nếu trường cho cloud và có plan phù hợp; không mặc định dùng cho lớp học |

Nguồn: [CVAT track mode](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/), [CVAT native](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/), [CVAT Ultralytics YOLO](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/), [CVAT Community/overview](https://docs.cvat.ai/docs/getting_started/overview/). [Label Studio VideoRectangle](https://labelstud.io/tags/videorectangle), [export](https://labelstud.io/guide/export), [cài OSS](https://labelstud.io/guide/install). [Roboflow giới hạn free/public](https://roboflow.com/common-computer-vision-questions), [pricing](https://roboflow.com/pricing). Các nhận định “mặc định/thay thế/phù hợp” là lựa chọn thiết kế của kế hoạch này.

Tại thời điểm đọc, [CVAT Online pricing](https://www.cvat.ai/pricing/cvat-online) hiển thị Free **1 project, 3 tasks, 1 GB**, export annotations-only; Team Monthly **33 USD/người/tháng**, owner cũng cần tính theo [subscription management](https://docs.cvat.ai/docs/account_management/subscription-management/). Không áp quota Online vào Community tự host; không coi giá này là báo giá mua, chưa gồm thuế và có thể đổi. Với 3 seat giả định, Team Monthly khoảng 99 USD/tháng trước thuế, chỉ để so ngân sách. Luồng đề xuất không cần mua gói này.

**Cảnh báo export:** native là bản gốc; YOLO detection thông thường chỉ giữ 5 cột/class-box, không giữ rubric/event/thuộc tính. CVAT có extension track ID, nhưng không tự thêm cột thứ sáu vào dữ liệu trainer detection. Label Studio tài liệu YOLO nói đến các tag ảnh `RectangleLabels/KeyPointLabels`; không mặc định `VideoRectangle` export thẳng thành YOLO training đúng. Giữ JSON/video + kế hoạch converter có kiểm thử nếu chọn luồng đó. CVAT YOLO không giữ attributes theo tài liệu; phải giữ sidecar/native.

## 6. Cách làm từng bước với CVAT local

Đây là hướng dẫn thao tác dự kiến cho bản được khóa khi triển khai; tên nút có thể đổi. [Hướng dẫn cài Community](https://docs.cvat.ai/docs/administration/community/basics/installation/) là điểm vào cho người phụ trách máy; không cài trong đợt này.

1. **Chuẩn bị máy:** một người phụ trách Docker/WSL2 hoặc máy Linux nội bộ theo tài liệu CVAT. Tạo tài khoản riêng cho annotator/reviewer, backup volume và file. Chỉ mở local/LAN đã kiểm quyền; tắt tích hợp cloud/AI ngoài nếu chưa được duyệt. Nếu máy không chạy nổi CVAT, thử Label Studio OSS cho pilot ảnh trước; không vì khó cài mà upload lớp học lên website công cộng.
2. **Chuẩn bị nguồn:** copy ảnh/video được phép vào kho riêng; tạo `session_id/video_id`, hash, manifest timestamp; giữ nguyên bản gốc. Chọn split theo buổi trước khi AI tạo nhãn. Không đưa raw video hoặc consent vào Git.
3. **Tạo Project:** tên `classroom_person_pilot_v1`; label `person` và attributes của mục 4. Gắn hướng dẫn mẫu. Nếu cần 8 lớp tạo project/ontology riêng với mapping rõ, không sửa lớp ngầm giữa chừng.
4. **Import ảnh:** tạo Task theo buổi/cảnh, upload bản local vào instance nội bộ; chọn thứ tự có manifest. Mở Job, phóng to hàng xa, Rectangle → Shape, mỗi người một box. Save thường xuyên; xác nhận background đã kiểm. Bản đầu 50 ảnh gán tay không hiện box teacher.
5. **Import video:** tạo Task clip nguyên FPS; Rectangle → Track tại frame đầu nhìn thấy, đặt keyframe khi chuyển động/che đổi. CVAT nội suy giữa keyframe; kiểm đoạn giữa bằng xem chậm và sửa, đặc biệt khi người giao nhau. Dùng Outside khi không còn box quan sát; không kéo nội suy qua gap. Track numbering không tự nối giữa task.
6. **AI draft sau pilot:** chạy teacher local ngoài CVAT rồi import annotations đúng class map/format, hoặc tích hợp local đã kiểm bản cài. Không mặc định nút AI trên giao diện luôn xử lý local. Lưu prediction nguyên bản riêng; chỉ đưa train pool vào bước này, reviewer có chế độ xem ảnh sạch.
7. **Review:** annotator xong chuyển reviewer hoặc dùng sổ bàn giao nếu edition không có workflow tương ứng. Reviewer rà lưới toàn ảnh trước, rồi box chi tiết; thêm FN, xóa FP/trùng, sửa class/biên và ID. Ghi lỗi bất đồng, sửa guide và quay lại ảnh cũ khi rule đổi.
8. **Gán event:** xem clip đầy đủ, ghi `events.jsonl` với mốc giây và track UID. Có thể dùng attributes/tag để trợ xem nhưng sidecar là bản interval chuẩn đã thiết kế. Hai người kiểm onset/offset và unknown. Không lấy tên sinh viên làm ID.
9. **Export:** tải native/backup trước; xuất Ultralytics YOLO Detection cho ảnh person đã review; skeleton chỉ khi đã chọn pose; MOT khi cần evaluator tracking tương ứng. Chọn ảnh kèm export nếu edition cho phép, nếu không ghép từ kho nguồn theo manifest. Lưu tool version, schema version, ngày và checksum.
10. **Validate trước training:** reimport native vào task thử; vẽ overlay 20–50 ảnh ngẫu nhiên và tất cả nhóm khó, đối chiếu timestamp/frame một clip. Kiểm file YOLO từng dòng, số ảnh/label, class map, split và leakage. Chỉ đánh dấu dataset `curated_v1` sau khi reviewer ký checklist. Sau này khi có quyền train: chạy loader/smoke 1 epoch rồi xem training batch overlay; chưa chạy trong nhiệm vụ này.

Khởi đầu dễ thực hiện nhất là 20 ảnh, hai người gán riêng và ngồi thống nhất 30–60 phút; sau đó mới xử lý 300–500 ảnh. Đừng bắt đầu bằng gán hàng nghìn ảnh theo một quy tắc chưa kiểm.

## 7. AI thương mại lớn: dùng vào đâu và chưa biết gì

### Vai trò kỹ thuật

| Phương án | Phần hữu ích | Giới hạn/đề xuất thử |
|---|---|---|
| OpenAI API có vision | Soạn rubric không cần ảnh thật; mô tả cảnh, đề xuất trường hợp khó trên ảnh được phép; JSON draft khi endpoint/model hỗ trợ | Tài liệu nêu hạn chế định vị không gian và đếm. Không mặc định tọa độ pixel chính xác hoặc upload MP4 trực tiếp như image input; luồng v1 dùng frame có timestamp nếu được phép |
| Gemini API | Tài liệu có image object detection và video input; có thể gợi ý box hoặc mốc clip | Kiểm đúng model/endpoint và sampling; sự kiện ngắn có thể mất khi lấy mẫu. Dùng làm draft, không làm ID/event GT |
| Detector local / Grounding DINO | Detector cho box; grounding theo prompt `person` giúp tìm ứng viên | So với teacher pretrained trên tập nhỏ, không khẳng định model lớn luôn tốt hơn trên hàng xa/che khuất |
| SAM 2 local | Mask theo prompt và lan truyền video; có thể lấy bbox từ mask | Không tự quyết định mọi người/lớp/hành vi; prompt thiếu người thì vẫn thiếu. Mask lan sai khi che/giao nhau cần sửa |

Nguồn trực tiếp: [OpenAI Images and vision](https://developers.openai.com/api/docs/guides/images-vision), [Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding), [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding), [Grounding DINO chính chủ](https://github.com/IDEA-Research/GroundingDINO), [SAM 2 chính chủ](https://github.com/facebookresearch/sam2). Không cần dùng cả hai API thương mại; local teacher + sửa tay là luồng mặc định để đánh giá trước.

Gemini tài liệu minh họa box `[ymin,xmin,ymax,xmax]` trong miền 0–1000; đây **khác** schema xyxy chuẩn nội bộ dưới đây. Phải có adapter có kiểm thử. Video understanding là khả năng nhận video đầu vào, không bảo đảm đọc từng frame hoặc nối ID như MOT. Tài liệu mô tả static mặc định 1 FPS, có cấu hình sampling; cần thử hành vi nhanh riêng. Các model/API đổi theo thời gian nên kế hoạch chưa khóa model thương mại hay mua subscription.

### Dữ liệu, retention, quyền dùng output

| Nguồn | Điều đã kiểm được | Điều phải xác minh trước dùng thật |
|---|---|---|
| OpenAI API | Dữ liệu API không dùng train theo mặc định trừ opt-in. Abuse logs có thể lưu tới 30 ngày theo mặc định; application state/file có quy tắc riêng. `store=false` không tự đồng nghĩa zero retention | Endpoint, tổ chức/tài khoản, lưu file và xóa, ngoại lệ, vùng xử lý, ZDR eligibility; **quyền dùng output gán nhãn để train YOLO của nhóm: UNKNOWN trong tài liệu đã kiểm**, cần đối chiếu hợp đồng/terms áp dụng |
| Gemini Developer API | Terms phân biệt unpaid/paid; unpaid có thể dùng nội dung cải thiện sản phẩm và người rà; paid không dùng prompt/output để cải thiện sản phẩm nhưng vẫn có logging có giới hạn. Terms có hạn chế phát triển model cạnh tranh | Phạm vi hạn chế áp vào detector của nhóm: **UNKNOWN**, không tự kết luận YOLO được miễn. Kiểm exact plan/region/DPA/retention; không dùng free để gửi PII |
| Local open models | SAM 2 repo công bố code/checkpoint Apache 2.0; Grounding DINO repo có Apache 2.0 | Kiểm license đúng checkpoint tải, dataset nguồn, dependency, và Ultralytics/license cho phát hành; không suy license repo miễn mọi quyền hình ảnh |

Nguồn: [OpenAI Data controls](https://developers.openai.com/api/docs/guides/your-data), [cổng Terms and policies được liên kết từ docs](https://developers.openai.com/api/docs/guides/your-data) và [Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms). Bảng không phải kết luận pháp lý về quyền training. Không đánh đồng OpenAI API với tài khoản chat cá nhân, hoặc Gemini Developer API với mọi sản phẩm Google/Vertex. Với OpenAI, cần đọc hợp đồng actual qua cổng chính thức trước quyết định output-training; nguồn kỹ thuật đã đọc chưa đủ xác nhận điều đó.

Không trích/publish dataset hay model chỉ vì nhãn được người sửa: sự sửa chữa không tự xóa nghĩa vụ đối với input, output nguồn, license và quyền người học. Chỉ đưa output thương mại vào train sau khi có kết quả kiểm quyền được lưu, nếu chưa rõ thì dùng manual/local phù hợp.

## 8. Teacher → nhãn nháp → GT kiểm duyệt → student

1. **Chọn teacher** trên lô train/pilot, ví dụ detector person pretrained lớn hơn hoặc grounding local; giữ E3 làm đối chứng. Ghi weights/hash/version/params. Chưa tải/chọn kiến trúc cụ thể.
2. **Tạo pseudo-label** trên train pool; giữ toàn bộ raw output, conf, tọa độ, prompt/version và lỗi. Không loại ảnh trống dự đoán vì có thể chính là FN.
3. **Người rà từ ảnh sạch:** quét từng hàng ghế/biên trước, đánh dấu mọi người thực; sau đó mới so/sửa box AI. Có thể ẩn overlay luân phiên để giảm bám nhãn. Phải bổ sung người AI chưa từng phát hiện.
4. **Review độc lập:** kiểm gold sample, hard cases và một phần lô thường. Chỉ bản đã sửa/phân xử mới chuyển `reviewed`; nhãn chưa xem vẫn là pseudo-label và bị loại khỏi curated GT ở v1.
5. **Train student** bằng nhãn đã kiểm; ghi provenance AI-assisted. Validation/test do người gán và review độc lập, không lấy prediction teacher làm đáp án.
6. **Lặp theo lỗi:** dùng train/unlabeled pool cho active learning; không đưa ảnh test hay lỗi test vào vòng chọn model. Nếu cần phân tích test sâu để sửa hệ thống, coi nó là development và thu holdout mới.

Đây là **học có giám sát từ nhãn do AI hỗ trợ**, thường gọi pseudo-labeling có kiểm duyệt. Knowledge distillation theo nghĩa tối ưu thêm loss khớp logits/features/quan hệ teacher-student là thiết kế khác; không gọi quá trình kéo sửa box là distillation thực. Quyền sử dụng vẫn phải kiểm dù gọi tên phương pháp nào.

### Prompt thiết kế (chỉ dùng sau khi được phép gửi dữ liệu)

```text
Bạn đề xuất nhãn nháp cho một ảnh lớp học. Không nhận diện tên/danh tính.
Chỉ class person. Theo guide v1: box visible, mỗi người phân biệt được một box,
kể cả chỉ thấy đầu/vai và người ở biên. Không ước lượng phần thân bị che hoàn toàn.
Ảnh gốc W={width}, H={height}; tọa độ pixel xyxy, gốc trái trên, biên phải/dưới exclusive.
Không tạo track ID hoặc hành vi từ ảnh đơn. Không chắc thì ghi uncertain và lý do.
Trả JSON theo schema. Không tuyên bố đã tìm đủ mọi người. Người sẽ rà lại toàn ảnh.
```

Schema chuẩn nội bộ để thiết kế adapter; chưa phải request body SDK của nhà cung cấp:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["image_id", "width", "height", "coord", "objects", "needs_review"],
  "properties": {
    "image_id": {"type": "string"},
    "width": {"type": "integer", "minimum": 1},
    "height": {"type": "integer", "minimum": 1},
    "coord": {"const": "xyxy_pixels_half_open"},
    "needs_review": {"const": true},
    "objects": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["class", "bbox", "uncertain", "reason"],
        "properties": {
          "class": {"const": "person"},
          "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
          "uncertain": {"type": "boolean"},
          "reason": {"type": "string"}
        }
      }
    }
  }
}
```

Schema chỉ ràng buộc cấu trúc. Validator tiếp tục kiểm đúng image ID/W/H, số hữu hạn, `0≤x1<x2≤W`, `0≤y1<y2≤H`; không đảo x/y. Nếu ảnh resize/crop/letterbox, đảo transform về ảnh nguồn trước. Box hơi vượt biên có thể clip vào [0,W]×[0,H] nhưng phải giữ raw + cờ sửa; tọa độ sai đơn vị, đảo thứ tự hoặc box rỗng phải reject/review, không âm thầm “sửa” thành GT. Không xóa box overlap chỉ vì IoU cao: có thể hai người khác nhau. Dùng overlay để phát hiện cả duplicate lẫn người bị mất.

Ví dụ ảnh 1000×500, xyxy `[100,50,300,250]` → YOLO `0 0.2 0.3 0.2 0.4`; công thức `cx=(x1+x2)/(2W)`, `cy=(y1+y2)/(2H)`, `w=(x2-x1)/W`, `h=(y2-y1)/H`. Nguồn định dạng [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect/). Không dùng ví dụ này làm nhãn thật.

### Thử AI có đáng công không

Trên 30–50 ảnh train/pilot gồm dễ/khó, so hai nhóm cân bằng độ khó: manual từ ảnh sạch và AI+review. Nếu cùng ảnh, đổi thứ tự/người để giảm hiệu ứng nhớ. Gold được phân xử độc lập. Đo tổng phút/ảnh (kể cả import, FN và review), số FN/FP/box phải sửa, API usage/retries. Chỉ mở rộng AI nếu giảm khoảng **20% tổng thời gian** mà chất lượng sau review không kém ngưỡng QA; ngưỡng 20% là đề xuất kinh tế, không bắt buộc. Nếu teacher bỏ gần hết người, sửa nhãn có thể chậm hơn vẽ tay.

## 9. Quyền riêng tư và điều kiện trước ghi hình/cloud

### Căn cứ đã tra và phần còn cần trường xác minh

[Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=) có hiệu lực từ **01/01/2026**. [Nghị định 356/2025/NĐ-CP — bản chính thức](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/01/356-nd.signed.pdf) là văn bản hướng dẫn cần đối chiếu trong năm 2026; không dùng duy nhất Nghị định 13/2023 làm kết luận hiện hành. [Bộ luật Dân sự 91/2015/QH13, Điều 32](https://vbpl.moj.gov.vn/toaannhandantoicao/Pages/vbpq-toanvan.aspx?ItemID=95942&Keyword=91%2F2015%2FQH13) quy định quyền hình ảnh và nguyên tắc đồng ý, có ngoại lệ theo luật. Không tự coi lớp học là hoạt động công cộng đủ điều kiện ngoại lệ.

Phần trên xác định văn bản/nguyên tắc cần rà; kế hoạch không kết luận nhóm đã đủ điều kiện pháp lý ghi hình, upload, chuyển dữ liệu ra nước ngoài hoặc công bố. Nội dung thủ tục áp dụng, vai trò trường/nhóm/nhà cung cấp, hồ sơ đánh giá tác động và chuyển dữ liệu xuyên biên giới phải nhờ bộ phận phụ trách của trường xác minh theo thực tế. Bản PDF luật/nghị định là bản scan; chưa trích/diễn giải từng điều về trẻ em hay ngoại lệ để tránh đưa kết luận thiếu căn cứ.

### Biện pháp vận hành đề xuất, phải được trường duyệt theo bối cảnh

- Có văn bản của đơn vị/giảng viên cho phép tổ chức ghi hình và thông tin cho người tham gia về mục đích, dữ liệu thu, người xem, thời hạn, quyền yêu cầu/rút lại theo cơ chế áp dụng. Không coi đồng ý ghi hình đồng nghĩa đồng ý upload hoặc công bố dataset.
- Xác minh tuổi; nếu có người chưa thành niên/trẻ em, trường xác định yêu cầu đồng ý của chính người học/người đại diện và bảo vệ bổ sung theo pháp luật/quy định trường. Không tự dùng một mốc tuổi duy nhất cho mọi thủ tục. Có phương án tránh ghi hình/loại vùng cho người không tham gia, không gắn việc tham gia nghiên cứu với điểm học tập.
- Không thu tên/mã sinh viên, face embedding hoặc bảng nối tên-ID nếu v1 chỉ cần track. Dùng ID giả danh; video còn khuôn mặt vẫn có thể là dữ liệu cá nhân, chưa phải ẩn danh chỉ nhờ bỏ tên.
- Kho raw riêng có mã hóa thiết bị, tài khoản riêng, quyền tối thiểu; bảng consent lưu nơi khác; log truy cập và người chịu trách nhiệm. Không đưa raw, crops, backup, nhãn có đường dẫn nhận diện, khóa API vào repo/public drive.
- Đề xuất thời hạn xem xét: raw đến khi nghiệm thu + 90 ngày; bản gán nhãn tối đa 12 tháng rồi rà nhu cầu; backup có lịch xóa đồng bộ. Đây là thời hạn quản trị để trường/nhóm chọn, **không phải thời hạn bắt buộc theo luật**. Phải có owner/ngày xóa, quy trình yêu cầu và xử lý việc dữ liệu đã đi vào model trước khi bắt đầu train.
- Trước cloud: phê duyệt rõ nhà cung cấp, endpoint, region/subprocessor, quyền output-training, training opt-in/opt-out, log/file/cache retention, xóa/backup, DPA và chi phí. Nếu chưa rõ, tiếp tục local. Che tên/mặt có thể giảm rủi ro nhưng phải kiểm mức tái nhận diện và ảnh hưởng nhãn; không coi blur là giấy phép upload.
- Bản báo cáo/demo chỉ dùng dữ liệu được phép; ưu tiên hình đã giảm định danh/số liệu tổng hợp. Không tự công bố ảnh mặt/nguyên video của lớp hoặc chấm điểm tự động từng sinh viên.

## 10. QA, split và bộ đo độc lập

### Phân vai và kiểm nhãn

Một người gán; người khác review; người thứ ba hoặc giảng viên phân xử rubric khi cần. R0 gán độc lập toàn bộ 50 ảnh trước khi nhìn nhãn nhau. R1 double-label ngẫu nhiên phân tầng khoảng 10–20% và review tất cả trường hợp khó; mọi ảnh đều có ít nhất một lượt rà toàn cảnh. Test được hai người rà toàn bộ và phân xử, không dùng AI prediction làm đáp án ban đầu.

Checklist một job:

- [ ] Rà toàn cảnh/hàng xa/biên: FN, đầu/vai, người bị che; không chỉ kiểm box có sẵn.
- [ ] Box đúng người, đủ phần nhìn thấy, không có FP trên ghế/ảnh poster/bóng; không trùng instance.
- [ ] Class/visibility/truncation nhất quán; background thực đã kiểm; unknown/ignore có lý do.
- [ ] Video không nhảy ID/đổi người ở chỗ giao nhau, không nội suy qua gap không quan sát.
- [ ] Event đủ diễn biến, onset/offset/unknown/censored đúng; quan sát tách suy luận.
- [ ] Native/export overlay khớp ảnh, frame, timestamp, class map và version.

Đề xuất gate QA trên phần double-label sau matching một-một person tại IoU 0,5: agreement P/R trước phân xử đều ≥0,95; median IoU của cặp hợp lệ ≥0,85; event agreement F1 ở protocol đã khóa ≥0,85. Đây là **độ đồng thuận nhãn**, không phải chất lượng model. Với mẫu nhỏ, luôn báo số lỗi/n và phân tầng, không chỉ tỷ lệ. Nếu không đạt, sửa guide, huấn luyện annotator và rà lại lô liên quan; không hạ ngưỡng âm thầm. Các ví dụ có bất đồng đều phải phân xử trước đóng version.

### Chia dữ liệu chống leakage

- Đơn vị split là buổi/video/cảnh gốc, không random ảnh. Hai camera cùng thời điểm/buổi và các crop/augment/clip chồng nhau của cùng nguồn đi cùng split.
- V1 gợi ý train/val/test khoảng 60/20/20 theo nhóm nguồn nếu đủ **ít nhất 5 buổi**; cân đối độ khó/hành vi bằng chọn nhóm. Chỉ có 3 buổi có thể 1/1/1 hoặc cross-validation theo buổi + giữ holdout mới, nhưng train/test nhỏ nên kết luận hạn chế. Không ép tỷ lệ ảnh bằng cách xé một buổi sang ba tập.
- Nếu mục tiêu tổng quát sang người mới, cần cohort/người độc lập, có thể cần nhóm lớp khác. Cùng lớp ở mọi split chỉ kiểm sang buổi/góc khác, không chứng minh sang người mới. Không thu PII chỉ để split; dùng nhóm tham gia giả danh khi thực sự cần và được phép.
- Gold val/test gán thủ công từ nguồn độc lập với teacher; khóa manifest/hash và quyền xem test. QA gold dùng luyện annotator không đồng thời là test giữ kín. Pilot đã dùng so teacher/tune rule là development, không đổi tên thành test độc lập.
- Không dùng pseudo-label làm test GT. Không chọn conf/imgsz/checkpoint dựa test. Không augment/cân bằng test. Có mẫu representative và challenge subset báo riêng; không trộn mẫu lỗi có chủ đích rồi gọi Recall toàn lớp.

### Ignore và coverage

YOLO detection TXT thường không biểu diễn ignore region. Vì vậy giữ vùng ignore ở native/sidecar, không đổi vùng chưa biết thành ảnh nền sạch. Ảnh train có người không xác định phải được phân xử trước; nếu chưa giải được, đưa cả ảnh vào quarantine và báo số lượng/lý do, hoặc xây pipeline loss có ignore riêng được kiểm sau này. Không chỉ bỏ một box rồi train phần ảnh đó như negative.

Evaluator cần quy tắc loại GT/prediction tương ứng với ignore được khóa trước và kiểm bằng vài ví dụ; báo metric thô cùng metric có ignore nếu có. Báo coverage: tổng người/frame/giây ứng viên, số đủ điều kiện, số unknown/ignore/quarantine theo cảnh/kích thước. Không bỏ outlier sau khi thấy model sai để nâng metric.

## 11. Lộ trình training/evaluation sau khi dữ liệu qua gate

### Detection trước

1. Đo không fine-tune E3, R04 nếu còn, pretrained **n** và pretrained **s** của cùng family tương thích phiên bản đã khóa. Ghi class mapping, weights/hash, input, conf, max_det và NMS/end-to-end setting; không đổi tên checkpoint n thành s.
2. Với fine-tune, n/s đều bắt đầu pretrained tương ứng, cùng split, recipe/chính sách chọn trên val, seeds và ngân sách được ghi. E3 đã fine-tune là nhánh lịch sử riêng, không quy chênh lệch với s mới hoàn toàn cho kích thước kiến trúc. Nếu chỉ đủ một seed, báo bất định; khi chọn ứng viên cuối thử thêm seed nếu nguồn lực cho phép.
3. Chọn threshold trên val bằng đường P–R và ràng buộc báo nhầm/độ trễ. Đo person P/R/F1 với matching một-một **IoU ≥0,5**, AP50 và AP50:95; ghi conf và cách xử lý ignore. Report FN theo gần/xa, box nhỏ, che, blur, biên. Phân biệt micro theo instance với trung bình theo buổi; bootstrap theo buổi/clip để ước lượng CI, không coi frame lân cận độc lập.
4. Mốc **Recall 90–95%** chỉ là mục tiêu thử đề xuất trên tập đủ điều kiện, đi kèm Precision, ví dụ P≥90%, latency và CI. Chưa biết khả thi; không hứa Recall 100%. Nếu không đạt, quyết định thêm dữ liệu/đổi góc/cải nhãn/model dựa lỗi; không chỉ tăng imgsz.

### Tracking và hành vi

- Sau detector baseline, thử ByteTrack/BoT-SORT hoặc phương án khác như thí nghiệm riêng, chưa lựa chọn mặc định. Ghi ngưỡng track, buffer/FPS, ReID có/không và license. Đo IDF1, HOTA, ID switches, fragmentation/mất dấu theo clip bằng GT có ID và protocol rõ. Để tách nguyên nhân, có thể so tracker trên detector cố định và trên GT box.
- Với hành vi, có baseline đơn giản theo chuỗi ảnh/track/rule và phương án pose-temporal khi đủ nhãn. YOLO Pose, RTMPose, ST-GCN, PoseC3D chỉ là ứng viên; chưa triển khai tất cả.
- Đề xuất event matching một-một cùng class/người khi có ID đáng tin, temporal IoU≥0,5; thêm onset tolerance ±0,5 giây khi đo onset. Báo P/R/F1 từng hành vi, macro-F1, duration error và false events/giờ; trường hợp không có event thật thì Recall là N/A, vẫn báo FP. Không tính hai dự đoán trùng thành hai TP.
- Bài end-to-end đòi đúng track để gán cho người; bài event với oracle track đo riêng để biết lỗi do ID hay do hành vi. Nếu chưa chắc ID, báo event cấp cảnh riêng và không quy cho sinh viên.
- Chốt cửa sổ quan sát theo hành vi. Đo độ trễ từ thời điểm bằng chứng đủ tới kết quả và end-to-end capture→output p50/p95; báo riêng chờ ngữ cảnh, look-ahead và cold start. Không dùng model xem tương lai rồi tuyên bố realtime dưới 3 giây.

Nguồn khả năng tracker/ReID: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/). Chỉ dùng chỉ số với bộ GT/protocol phù hợp; hiện chưa có bằng chứng các mốc trên đã đạt.

### Tối ưu triển khai sau baseline

Chỉ thử FP16/INT8/ONNX/TensorRT sau khi baseline chất lượng đủ dùng; ghi artifact/runtime/device có hỗ trợ thực. So lại model FP32 với FP16/INT8 trên cùng holdout sau khóa lựa chọn; calibration INT8 lấy train/calibration riêng, không lấy test. Ghi P/R/F1, lỗi từng nhóm và latency/RAM/VRAM trước/sau. Không cam kết tốc độ RTX 5060 Laptop hoặc INT8 giữ nguyên Recall. Pipeline hiện CPU; phải xác minh CUDA/runtime và đo thật ở bước triển khai sau.

## 12. Cấu trúc bàn giao và định dạng

Layout **đề xuất**, chưa tạo dataset trong đợt lập kế hoạch. Kho dữ liệu nằm ngoài Git; repo chỉ chứa tài liệu/schema/script không có dữ liệu định danh sau khi triển khai được cho phép.

```text
<private_data_root>/classroom_v1/
  governance/                 # quyền, phạm vi sử dụng, lịch xóa; quyền riêng
  raw/S01/C01/                # video gốc bất biến, không tên người
  manifests/sources.jsonl     # hash, session, camera, PTS/timebase, metadata cảnh
  manifests/splits_v1.json    # nhóm nguồn và lý do chia
  frames/S01_C01_V01/         # ảnh nguồn đã chọn + mapping PTS
  annotations/guide_v1.md
  annotations/native/        # CVAT backup/XML hoặc JSON nguồn
  annotations/pseudo/        # teacher raw, chưa là GT
  annotations/reviewed/       # nhãn đã phân xử + provenance
  annotations/tracks/        # MOT + gap/uncertainty sidecar
  annotations/events.jsonl
  annotations/scene_regions.json
  quarantine/                # chưa giải quyết, lý do; không đưa vào train ngầm
  exports/person_v1/          # data.yaml, images/{train,val,test}, labels/...
  qa/                        # double-label, lỗi, adjudication, coverage
  experiments/               # config, metrics, hashes; không token API
```

Ví dụ event thiết kế:

```json
{"schema_version":"1.0","video_id":"S01_C01_V01","clip_id":"clip03","track_uid":"S01_C01_clip03_T007","event":"hand_raise","start_s":4.2,"end_s":7.8,"boundary_uncertainty_s":0.2,"left_censored":false,"right_censored":false,"status":"reviewed","annotator_id":"A02","reviewer_id":"A01","guide_version":"1.0"}
```

Ví dụ trên không phải quan sát thật. Manifest frame cần `source_sha256`, frame index/PTS, clip offset, dimensions, sampling/resize transform; annotation provenance có teacher/model/prompt hash khi dùng AI, người sửa, reviewer, guide/tool version và lịch sử sửa. Không lưu tên người học. Schema event phải cho phép track null/unknown và observation gaps trong bản triển khai.

Dataset card mỗi version: nguồn/giấy phép sử dụng, phạm vi lớp/camera/buổi, ontology, convention box/pose/event, split, coverage, unknown/ignore, phân bố, phương thức AI/human, QA, hạn chế và thời hạn. Giữ native để không mất ID/attributes khi xuất YOLO.

## 13. Checklist công việc, phụ thuộc và gate

Mỗi dòng là một đầu việc có sản phẩm và cách kiểm. Nhân lực tính **giờ công**, không phải thời gian lịch; gate là đề xuất để chốt sau pilot.

| Thứ tự | Việc và phụ thuộc | Deliverable | Điều kiện qua gate / cách kiểm | Ước tính |
|---|---|---|---|---|
| G0 | Xác minh quyền/phạm vi; chưa phụ thuộc dữ liệu mới | `data_governance_v1.md`, owner, danh mục quyền | Trường xác nhận phạm vi ghi hình/local; cloud có phê duyệt riêng nếu muốn dùng | 2–4 giờ chuẩn bị; thời gian trường phản hồi chưa biết |
| G1 | Khảo sát camera + rubric, sau G0 | Guide, manifest mẫu, 50–100 ảnh/3–5 clip đã chọn | Hai người gán độc lập và phân xử; đủ ví dụ che/xa/biên/hành vi | 6–12 giờ |
| G2 | Công cụ/export round-trip, sau G1 | Version công cụ, backup, 5 ảnh + 1 clip thử | Overlay khớp nguồn, ID/timestamp không lệch, phục hồi được backup | 4–8 giờ, máy khó cài có thể lâu hơn |
| G3 | Gold development + split, sau G1/G2 | Gold val/test theo nguồn, split hash, QA report | Không leakage; test độc lập; QA/coverage đạt mục 10 hoặc ghi rõ chưa đạt | 8–16 giờ cho tập nhỏ; tăng theo mật độ |
| G4 | So AI/manual lô nhỏ, sau G3 | Time study, error/cost table, kiểm quyền teacher | Nhãn sau review đạt QA; chỉ dùng AI nếu có lợi; cloud chưa rõ quyền thì giữ local | 3–6 giờ |
| G5 | Curate 300–500 ảnh, sau G4 | `person_v1`, dataset card, native + YOLO | Rà toàn ảnh, double-label 10–20%, không nhãn chưa xem vào GT; validate loader sau này | 20–40 giờ gán+review, có thể vượt nếu đông/che |
| G6 | Baseline detection, sau G5 | Bảng E3/pretrained n/s, config, lỗi/CI | Có P/R/F1 và coverage/latency; gate mở rộng theo lỗi, không dựa số box | 4–8 giờ người; compute đo sau smoke |
| G7 | Tracking/event pilot, sau G1–G3 và baseline | 10–20 clip, tracks/gaps/events, QA | Hai người thống nhất event, đo ID/event riêng; nếu không đủ sample ghi chưa kết luận | 16–40 giờ hoặc hơn tùy mật độ |
| G8 | Mở rộng theo learning curve, sau G6/G7 | Version mới và báo cáo so sánh | Gain trên val theo nhóm lỗi, test chưa tune; dừng/đổi hướng nếu không tăng | Mỗi lô lập lại dự toán theo số đo |
| G9 | Tối ưu/demo thiết bị, sau baseline chọn | FP32/export parity, latency report | Đo chất lượng và độ trễ thật, calibration không lấy test | Lập dự toán sau khi biết phần cứng |

Theo dõi tiến độ riêng kế hoạch này:

- [ ] G0 quyền dữ liệu/phạm vi.
- [ ] G1 rubric và pilot thủ công.
- [ ] G2 công cụ và export round-trip.
- [ ] G3 split/gold/QA độc lập.
- [ ] G4 time study AI so manual và kiểm quyền dùng output.
- [ ] G5 curated detection v1.
- [ ] G6 baseline có kiểm soát.
- [ ] G7 tracks/events v1.
- [ ] G8 mở rộng theo lỗi/learning curve.
- [ ] G9 tối ưu và demo có benchmark.

Không đánh dấu các việc trên đã hoàn tất vì tài liệu đã được viết. Luồng phụ thuộc chính: **quyền → rubric → gán thử/export → gold/split → curate → baseline → mở rộng**; track/event có rubric và nhãn riêng, pose là nhánh tùy kết quả.

### Công sức và tiền cần dự trù

Giả định gán+review trung bình 2–4 phút/ảnh lớp đông: 400 ảnh khoảng 13–27 giờ thao tác; thêm 30–50% cho QA/sửa guide/import/backup thành khoảng 18–40 giờ. 20 clip × 30 giây = 10 phút video; kiểm/gán tracking-event ở hệ số 50–150 lần thời lượng đã là 8–25 giờ, chưa tính double-label/cài máy. Phải thay bằng time study R0/G4; có thể cao hơn nhiều ở che khuất/mất ID.

Hai người cùng làm tổng 12 giờ công/tuần thì riêng 60 giờ công là khoảng 5 tuần, chưa tính lịch quay/xin phép. Không cộng máy chạy training thành giờ công nếu người không cần trực.

| Khoản | Dự toán v1 có giả định |
|---|---|
| CVAT Community/Label Studio OSS tự host | 0 phí thuê bao phần mềm cho luồng OSS; dùng máy/điện/ổ lưu sẵn có, chưa định giá |
| Raw video | 1 giờ ở 8–16 Mbit/s ≈3,6–7,2 GB, chưa cộng proxy/backup; 5 giờ khoảng18–36 GB, lưu hai bản cần gấp đôi |
| Cloud annotation | Chưa mua; ví dụ CVAT Team 3 seat khoảng99 USD/tháng trước thuế theo giá trang đã đọc; quyền dữ liệu vẫn phải kiểm |
| API AI | Chưa chọn model nên **chưa có báo giá/1.000 ảnh**. Đề xuất trần thử 5–10 USD nếu được duyệt; đây là giới hạn chi, không ước lượng phí thực |
| GPU train | Dùng máy sẵn có trước; giờ GPU/điện hoặc cloud chưa biết, phải đo smoke rồi dự toán, không tự đặt job tính tiền |

Công thức API: `cost = (input_tokens × input_rate + output_tokens × output_rate)/1e6 + phí công cụ/lưu trữ nếu có`; image/video tokens, thinking/retry và region/tier phải tính đúng API. Dùng usage thực lô nhỏ rồi dự phóng × số lô với buffer 20–30%. Tra [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) hoặc [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) lúc chọn model. Không lấy giá subscription chat làm giá API, không giả định miễn phí vì chưa hết quota. Tổng chi phí phải cộng thời gian người sửa; tiết kiệm token mà tăng FN có thể đắt hơn.

## 14. Rủi ro và cách quyết định vòng tiếp

| Rủi ro | Dấu hiệu cần kiểm | Xử lý trong kế hoạch |
|---|---|---|
| Camera không thấy tay/thân hàng xa | Hai annotator không thống nhất ngay ảnh nguồn | Sửa góc/độ phân giải/quy trình quan sát trước tăng model |
| AI tạo cảm giác đã đủ nhãn | Nhiều FN được phát hiện khi ẩn overlay | Rà ảnh sạch bắt buộc; giữ manual gold |
| Dữ liệu lặp cùng một buổi | Val đẹp nhưng buổi mới giảm mạnh | Group split/holdout buổi/camera; mở rộng theo domain |
| Guide mâu thuẫn với dataset cũ | Visible/amodal/class semantics khác | Version riêng; audit trước trộn, không sửa dữ liệu lịch sử ngầm |
| ID sai làm thống kê sai người | Người đổi chỗ/giao nhau, gap dài | Báo uncertain/gap, đánh giá ID riêng, hạn chế quy kết cá nhân |
| Hành vi ít hoặc khó quan sát | Không đủ event/annotator disagreement | Bổ sung thu có đồng thuận, báo sample nhỏ, chỉnh rubric trước test |
| Cloud/điều khoản chưa rõ | Chưa xác định plan/retention/quyền training | Dùng manual/local; đánh dấu unknown, chưa upload |

## 15. Những thông tin tối thiểu cần chốt để làm v2

Kế hoạch v1 vẫn dùng các giả định ở mục 3, không chờ câu trả lời mới có thể bắt đầu đọc/rà rubric. Trước triển khai, nhóm cần trả lời:

1. Lớp khoảng bao nhiêu người, có người dưới 18 tuổi không; trường cho phép ghi hình và dùng dữ liệu nghiên cứu ở phạm vi nào?
2. Có bao nhiêu camera, độ phân giải/góc quay; dự kiến có bao nhiêu buổi khác nhau và clip gốc dài bao lâu?
3. Nhóm có mấy người gán/review và bao nhiêu giờ mỗi tuần?
4. Có bắt buộc giữ đủ 8 lớp ngay không, hay chấp nhận dataset person-only thử riêng trước? Cần ID trong clip hay xuyên cả buổi/camera?
5. Ngân sách và phần cứng train/demo thực; **có quyền upload nhà cung cấp nào** hay chỉ được local?

**Bước đầu thực tế:** sau khi xác minh quyền, lấy 20 ảnh đại diện và một clip có hành vi, gán tay độc lập trong CVAT local, chốt cách xử lý đầu/vai/che/biên. Sau đó tăng tới 50–100 ảnh để đo lỗi model và thời gian gán. AI làm tốt vai trò soạn rubric, đề xuất box/mask/mốc thời gian và chọn mẫu cần xem; con người vẫn chịu trách nhiệm tính đầy đủ, ngữ nghĩa, ID, biên sự kiện và phê duyệt GT. Chưa có bước nào trong kế hoạch bảo đảm Recall 100%, ID vĩnh viễn hoặc kết luận mức tương tác của sinh viên từ một hành vi đơn lẻ.
