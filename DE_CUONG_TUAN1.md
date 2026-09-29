# Đề cương một trang — đề tài 8

**Tên đề xuất:** Hệ thống phát hiện và theo dõi người cùng đồ vật trong phòng học qua camera.

**Trạng thái phê duyệt:** Chưa có phản hồi giảng viên được cung cấp. **Sĩ số, tên thành viên, ngày nộp:** chờ nhóm điền và xác nhận.

## Vấn đề và mục tiêu

Nhóm xây dựng nguyên mẫu nhận hình từ một camera cố định trong phòng học/phòng thực hành. Hệ thống nhận biết bốn lớp `person`, `bottle`, `cell phone`, `laptop`, vẽ hộp/tên/độ tin cậy, theo dõi người qua các frame, đếm lượt qua một vạch và cảnh báo khi một người ở trong vùng đã chỉ định đủ thời gian. Kết quả hỗ trợ quan sát; không nhận diện danh tính hay kết luận hành vi của cá nhân.

## Phương pháp

YOLO26n pretrained là baseline và bản demo CPU; nhóm sẽ fine-tune trên ảnh có nhãn trong bối cảnh demo. ByteTrack gán ID tạm thời cho người; luật vạch và vùng được viết, kiểm và đánh giá riêng. YOLO26s là đối chứng kích thước nếu có tài nguyên Colab/GPU, không chặn bản chính. Ứng dụng local dùng Python, OpenCV; giao diện Streamlit nằm ở giai đoạn sau.

## Dữ liệu và đánh giá

Nhóm dự kiến thu 1.000–1.500 ảnh, mục tiêu khoảng 1.200, chia theo **phiên quay** thành train/validation/test; chuẩn nhãn và nguồn/quyền sử dụng được ghi trong data card/manifest. Tuần 1 dùng COCO128 chỉ để kiểm pipeline và tập quy tắc nhãn, không dùng điểm trên COCO128 làm kết quả cuối kỳ. Video phát triển và video test độc lập, có sự kiện đúng để chấm đếm/cảnh báo. So sánh pretrained với fine-tuned trên cùng tập test bốn lớp; báo mAP, lỗi theo lớp, đếm/cảnh báo, FPS và độ trễ trên máy demo thực.

## Đầu ra và kế hoạch

Đầu ra: mã nguồn, cấu hình/weights, dataset có nguồn, báo cáo PDF, 10–12 slide, video demo dự phòng và hướng dẫn tái lập. Tuần 1 chạy baseline camera/video và thử nhãn; tuần 2 hoàn thiện dữ liệu/split; tuần 3 fine-tune và evaluator; tuần 4 tracking/đếm/UI; tuần 5 test/đo/phân tích lỗi; tuần 6 báo cáo/bảo vệ. Các mục tiêu kỹ thuật trong kế hoạch là mục tiêu cần đo, chưa phải kết quả.

**Cần giảng viên xác nhận:** (1) sĩ số áp dụng; (2) phạm vi bốn lớp và tracking/đếm có phù hợp đề 8; (3) dùng pretrained rồi fine-tune có đáp ứng yêu cầu; (4) hạn nộp chính thức.
