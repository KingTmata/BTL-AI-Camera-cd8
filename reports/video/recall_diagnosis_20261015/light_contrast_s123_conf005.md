# Đối chứng LIGHT_CONTRAST seed123/conf0,05 — 15/10/2026

Người dùng đính chính: riêng person Recall, LIGHT_CONTRAST seed123 đạt **78,61%**, cao nhất trong nhóm E3 gốc và sáu bản train thêm hai epoch. Con số và phạm vi xếp hạng này do người dùng cung cấp; chưa tìm thấy bảng per-class local để xác minh độc lập. Không đồng nhất với Recall tổng trainer hoặc Recall tại conf0,05 trên video hiện tại.

## Trọng số đã xác nhận

- ZIP nguồn: `C:/Users/aawm8/Downloads/E3_AdamW_lr1e4_LIGHT_CONTRAST_S123_1280_E2_B32.zip`.
- Đọc riêng entry `weights/best.pt`, không giải nén đường dẫn tùy ý hoặc ghi đè file khác.
- Đích: `weights/seed_repeats_20261014/E3_LIGHT_CONTRAST_seed123_E2_best.pt`.
- SHA256: `5487978ad53e7c5fc9d5cbe0c043edab8497f4e83dc248a7c0f785775654e816`.
- Tên run trong checkpoint: `E3_AdamW_lr1e4_LIGHT_CONTRAST_S123_1280_E2_B32`.
- Class mapping project8 đúng, person=0. Recall tổng trong checkpoint: 0,70885; đây không phải person Recall 0,7861.

Lần kiểm trước chỉ xét các file trong weights, bỏ sót ZIP ở Downloads; do đó chưa đủ để xếp hạng person Recall trong cả nhóm người dùng nêu. Đã thêm đính chính vào báo cáo E2.

## Cấu hình lượt thử

Full, cùng video lớp học 898×506/25FPS/541frame, CPU, imgsz1280, confidence0,05, một lượt, không NMS bổ sung. Không sửa mặc định E3 gốc hoặc Streamlit/Training; checkpoint chọn tường minh cho lượt đối chứng này.

Thư mục kết quả: `runs/video_light123_conf005/20261015_221242_1e371cdaa7`; console `.temp/video_light123_conf005.log`. File MP4 cuối chỉ có sau kiểm tra xuất thành công; backend giữ cửa sổ xem/tua đến khi người dùng đóng, report cuối có sau đó.

Mở TUI với trọng số này cho các lượt tiếp:

```powershell
.\run_video.cmd --tui --weights "weights/seed_repeats_20261014/E3_LIGHT_CONTRAST_seed123_E2_best.pt" --confidence 0.05 --imgsz 1280
```

Không kết luận Recall trên video lớp học đạt 78,61% khi chưa có ground truth và protocol tương ứng. Không suy ra chênh lệch chất lượng chỉ từ số box nhiều hơn.

## Kết quả thực thi

Đã suy luận đủ **541/541 frame**, MP4 cuối được xuất/kiểm tra thành công và tự phát bằng OpenCV (bắt đầu ở khoảng 64,3 giây từ khởi chạy). Không chạy lại inference khi tua. Tại frame0/250/500, số box person là **2/3/1**, so E3 gốc cùng imgsz1280/conf0,05 trước đó **1/2/0**, E2 **0/0/0**. Đây chỉ là số dự đoán trên ba mẫu, chưa xác minh TP/FP/FN hoặc Recall video.
