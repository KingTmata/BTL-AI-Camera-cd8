> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../../../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

# Tuần 3 — readiness và dữ liệu đã khóa

Cập nhật 04/10/2026. Người dùng chọn **CPU local**, yêu cầu **review/khóa trước mọi lần train**, và đã xác nhận review ảnh/hộp nhãn toàn bộ 5.002 ảnh, đủ tám lớp, cùng nguồn/quyền và gần trùng/nhóm phiên.

## Đã hoàn thành trong bước chuẩn bị

- Ghi annotation_status=approved, review_status=approved, reviewer=human_user và near_duplicate_status=approved trong bốn bảng metadata hiện có. Giữ nguyên annotator nguồn, ảnh/TXT, số box và split.
- Khóa snapshot ngay tại project8_v0.3; không sao chép, khôi phục hoặc thêm ảnh. RELEASE.json và checksums nhận diện release.
- Validator release đạt đủ **5.002 ảnh**; checksum cuối xác minh toàn bộ media không đổi.
- **29/29 kiểm thử đạt**, không skip; compile và git diff --check đạt. Cache YOLO chỉ được bỏ qua đúng ba file dẫn xuất, còn thay ảnh/TXT/metadata hoặc thêm file khác vẫn bị phát hiện.
- Cấu hình dataset, mapping tám lớp, weights có sẵn và môi trường CPU được kiểm; chưa inference baseline hoặc training.

| Tập | Ảnh/TXT | Box | Review người được ghi nhận |
|---|---:|---:|---:|
| train | 3,883 | 19,718 | 3,883 |
| val | 798 | 5,382 | 798 |
| test | 321 | 1,466 | 321 |

Tổng 5.002 ảnh/TXT, 26.566 box; development 4.681 ảnh. Train/val có nhãn cả tám lớp. Test thiếu nhãn backpack/cup, chưa thể công bố đánh giá đủ tám lớp trên test.

4.566 cờ needs_review từ nguồn được giữ trong metadata, nhưng tất cả hiện có review người approved. Không còn cờ chưa có review. Mã phiên unknown và khai báo giấy phép gốc được giữ trung thực; nguồn/quyền và độc lập nhóm được người dùng xác nhận, không phải kết luận pháp lý hoặc phép đo của AI.

## Bước tiếp theo trên CPU

1. Chấm YOLO26n pretrained trên val đã khóa bằng evaluator chung.
2. Smoke 1–3 epoch, lưu cấu hình/log/checkpoint và đo thời gian thực.
3. Fine-tune YOLO26n bằng toàn bộ train, checkpoint chọn bằng val.
4. Chấm best.pt bằng cùng evaluator/protocol và lập bảng đối chứng. Test giữ ngoài tối ưu; YOLO26s chỉ khi đủ tài nguyên.

Chưa chạy các bước này trong lượt chuẩn bị. Mọi mAP/loss/thời gian training hiện là **chưa đo**.

## Hồ sơ

- [readiness.json](../../../../reports/results/week3_readiness_20261004/readiness.json): môi trường, config, hashes, mapping/protocol và trạng thái mới.
- [human_review_confirmation.json](../../../../reports/results/week3_readiness_20261004/human_review_confirmation.json): xác nhận trực tiếp của người dùng và phạm vi review.
- [release_validation.json](../../../../reports/results/week3_readiness_20261004/release_validation.json): validator release trên 5.002 ảnh.
- [review_queue.csv](../../../../reports/results/week3_readiness_20261004/review_queue.csv): danh sách tham chiếu hiện đã duyệt, không chứa bản sao ảnh.
- [RELEASE.json](../../../../data/dataset/project8_v0.3/RELEASE.json): snapshot khóa tại chỗ.
- [hướng dẫn CPU tuần 3](../../../../README/docs/WEEK3_START.md): lệnh dự kiến, cách lưu kết quả và G3.
