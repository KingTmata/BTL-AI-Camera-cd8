# Hồ sơ dataset chính — project8_v0.3

Cập nhật 04/10/2026: **5.002 ảnh / 5.002 TXT / 26.566 bounding box**, train/val/test **3.883 / 798 / 321**. Không còn pending.

Nguồn: 1.800 ảnh bộ chính cũ, 566 COCO bổ sung, 600 sách/điện thoại và 2.036 classroom sau lọc. Lớp bàn: **1.017 ảnh / 2.479 box**; `with-student` đã map về `table` theo quyết định người dùng.

Người dùng đã xác nhận review toàn bộ 5.002 ảnh kèm hộp nhãn, đủ các vật thuộc tám lớp; đã kiểm và chấp nhận nguồn/quyền cùng gần trùng/nhóm phiên. Metadata ghi human_user là người duyệt, giữ nguyên media/nhãn/split. Snapshot đã khóa tại chỗ ngày 04/10/2026; chưa chạy baseline validation hoặc fine-tune. Test vẫn thiếu nhãn backpack/cup. Hai ZIP hành vi nguồn giữ riêng cho tuần 4–6.

Xem [hồ sơ hiện hành](../docs/PROJECT8_V0_3_DATA_PROFILE.md), `dataset/project8_v0.3/DATA_CARD.md` và [bảng báo cáo mới](../../README_DA_HOAN_THANH/reports/results/project_progress_20261004/REPORT.md). Audit xóa tại `reports/results/primary_cleanup_20261004/`; báo cáo trước khi dọn là snapshot lịch sử.
