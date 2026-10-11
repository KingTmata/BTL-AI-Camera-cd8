# R04 cộng: trọng số R04 + TTA ba lượt + NMS

## So sánh với R04 chạy mặc định

Ở đây, mặc định nghĩa là suy luận một lượt từ checkpoint R04, không bật explicit TTA và không thêm NMS gộp views. Không phải checkpoint COCO chưa fine-tune.

| Nội dung | R04 một lượt | R04 cộng |
|---|---|---|
| Trọng số học được | Checkpoint R04 | Cùng checkpoint R04 |
| Train thêm | Không | Không |
| Lượt suy luận mỗi batch | 1 | 3 |
| Views | Ảnh đầu vào | (1.0, không lật), (0.83, lật ngang), (0.67, không lật) |
| Gộp kết quả | Native end-to-end head | Đưa box về cùng tọa độ, class-aware NMS IoU 0.5 |
| Tính toán | Một lượt model | Ba lượt ở các kích thước khác nhau + NMS; không suy ra tốc độ giảm đúng ba lần |
| Điểm đã đo | GPU một lượt: AP50 76.893780%, AP50–95 54.623630% | CPU FP32: AP50 80.505213%, AP50–95 56.569637% |

Hai hàng điểm khác thiết bị và quy trình; không dùng chúng để quy toàn bộ chênh lệch cho riêng TTA. Chưa đo FPS camera. Điểm R04 cộng đo tại 1280 trên 2500 validation/8359 GT, chưa chấm 500 test.

## Dùng R04 cộng

Checkpoint đang giữ: `weights/campaign_20261010/BEST_Project8_v05.pt`. Đây vẫn là raw R04, chưa chứa wrapper TTA/NMS.

Implementation đã có: `kaggle_tools.inference_variants.EndToEndViews`. Nạp một detection model R04 và tạo wrapper với `views=((1.0, False), (0.83, True), (0.67, False))`, `iou=0.5`, `conf=0.001`, `max_det=300`. Wrapper nhận tensor batch đã được tiền xử lý đúng như Ultralytics; không đưa trực tiếp ảnh uint8 vào forward.

Để tái hiện validation, dùng `ExplicitViewsValidator` với đúng views và `merge_iou=0.5`; giữ protocol CPU FP32, imgsz1280, batch8, rectTrue, conf0.001, max_det300, augmentFalse, nmsFalse. NMS bổ sung đã nằm trong wrapper; không thay bằng augment=True hoặc bật thêm NMS một lần khác.

## Có thể dùng như một file trọng số không?

Có thể lưu wrapper cùng model vào một PT và nạp bằng YOLO. Việc này đã được kiểm chứng bằng full2500 validation: AP50/AP50–95/precision/recall và tám lớp khớp trong 1e-6. Source kiểm chứng: `notebooks/kaggle/project8_v05_local_best_pipeline_reload_20261010.py`.

Đây là file đóng gói model cùng logic suy luận, không phải bộ tham số mới được train bằng TTA. Cần module `kaggle_tools.inference_variants` và môi trường tương thích (đã kiểm với Ultralytics8.4.165) để nạp lớp Python trong PT. Chưa kiểm ONNX/TensorRT hoặc ứng dụng camera. Không mặc định cho rằng file dùng được ở mọi nơi như checkpoint YOLO chuẩn.

File wrapper thử đã được xoá sau kiểm chứng; checkpoint công bố chưa bị thay. Tài liệu này không tạo PT thứ hai và không khởi chạy train. Nếu quyết định phát hành R04 cộng, tạo file tạm, nạp lại/kiểm chứng rồi mới thay file công bố, giữ một artifact cuối theo yêu cầu người dùng.

## Giới hạn

AP50 không phải phần trăm mọi dự đoán đúng. Ghế AP50 70.269%, điện thoại 74.586% còn yếu. Có nhóm cảnh/video chung giữa các split, nên chưa chứng minh độ chính xác trên cảnh hoàn toàn mới. Kết quả 1280 không tự chứng minh chất lượng khi app chạy640. Xem `reports/PROJECT8_V05_LOG.md` để biết provenance và các phép đo đầy đủ.
