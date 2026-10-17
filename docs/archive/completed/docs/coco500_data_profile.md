> **Hồ sơ lưu trữ:** đợt công việc ghi trong file đã kết thúc. Xem [phạm vi hoàn thành](../README.md); số liệu và hạn chế bên dưới thuộc thời điểm lập hồ sơ.

# COCO500 — Hồ sơ đặc tính dữ liệu

Phiên bản: `coco500_v0.1`; cập nhật 02/10/2026. **500 ảnh thật, trạng thái draft chờ review người thứ hai.**

## Nguồn, loại dữ liệu và cách chọn

Nguồn COCO2017, 400 ảnh train2017 và 100 ảnh val2017, seed 42. Chọn ngẫu nhiên không cân bằng từ ảnh có ít nhất một trong tám lớp, loại crowd mục tiêu/annotation lỗi và ID COCO128. Giữ split gốc; nguồn không cung cấp session_id nên không khẳng định độc lập theo phiên/cảnh.

Ảnh RGB/JPEG theo nguồn, nhãn gốc COCO JSON có category_id, bbox xywh pixel, area, iscrowd và segmentation. Dataset nhóm xuất YOLO TXT: class_id cx cy w h chuẩn hóa; sử dụng detection bbox, không dùng segmentation. Metadata nguồn gồm image ID, kích thước, URL ảnh/Flickr, license và date_captured khi có. Giữ annotation gốc và danh sách lựa chọn trong data/raw/coco500_v0.1/.

Tám lớp: person, table, chair, laptop, cell phone, backpack, book, cup. COCO dining table → table là đối chứng gần đúng; book theo phạm vi sách/vở nhóm nhưng nhãn COCO không tách riêng vở. Ảnh vẫn có thể chứa đối tượng ngoài tám lớp; các đối tượng đó không tham gia nhãn YOLO. Bộ chọn chỉ lấy ảnh dương tính nên không có tập ảnh âm tính chủ động.

## Phân bố lớp thực tế

Số ảnh các lớp có thể cộng vượt 500 vì một ảnh chứa nhiều lớp.

| Lớp | Train: ảnh / box | Validation: ảnh / box | Tổng ảnh / box |
|---|---:|---:|---:|
| person | 340 / 1069 | 85 / 297 | 425 / 1366 |
| table | 75 / 86 | 11 / 12 | 86 / 98 |
| chair | 63 / 162 | 12 / 38 | 75 / 200 |
| laptop | 12 / 15 | 6 / 6 | 18 / 21 |
| cell phone | 29 / 35 | 10 / 11 | 39 / 46 |
| backpack | 24 / 36 | 4 / 6 | 28 / 42 |
| book | 26 / 93 | 4 / 26 | 30 / 119 |
| cup | 48 / 106 | 10 / 29 | 58 / 135 |

## Thuộc tính ảnh

| Thuộc tính | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|
| width | 281.00 | 640.00 | 640.00 | 640.00 |
| height | 169.00 | 480.00 | 640.00 | 640.00 |
| file_bytes | 41892.00 | 149773.00 | 265030.25 | 403707.00 |
| brightness | 34.47 | 112.93 | 166.11 | 219.86 |
| contrast | 5.44 | 59.93 | 80.68 | 103.86 |
| laplacian_variance | 23.99 | 1640.92 | 7235.92 | 18593.48 |
| box_count | 1.00 | 2.00 | 12.00 | 25.00 |

Brightness/contrast tính từ grayscale 0–255; laplacian_variance là chỉ báo độ chi tiết phụ thuộc độ phân giải/nội dung, không phải kết luận ảnh mờ. CSV image_properties.csv lưu từng ảnh.

Định dạng thực: {"JPEG": 500}; chế độ màu: {"RGB": 498, "L": 2}; tỷ lệ width/height: {"min": 0.5609375, "median": 1.3333333333333333, "p95": 1.6161616161616161, "max": 3.78698224852071}; date_captured nguồn: {"available": 500, "min": "2013-11-14 12:30:11", "max": "2013-11-25 21:33:27", "note": "COCO metadata date_captured; not independently verified shooting sessions"}

## Kích thước vật thể

| Lớp | Box area/ảnh: median | Cạnh ngắn khi resize 640: median px | BBox small / medium / large |
|---|---:|---:|---:|
| person | 0.02765 (train) | 62.48 (train) | 320 / 431 / 615 |
| table | 0.17888 (train) | 155.25 (train) | 4 / 14 / 80 |
| chair | 0.02663 (train) | 70.66 (train) | 24 / 101 / 75 |
| laptop | 0.06255 (train) | 125.89 (train) | 0 / 7 / 14 |
| cell phone | 0.00895 (train) | 37.85 (train) | 16 / 23 / 7 |
| backpack | 0.01034 (train) | 48.35 (train) | 13 / 19 / 10 |
| book | 0.00310 (train) | 19.38 (train) | 59 / 47 / 13 |
| cup | 0.00490 (train) | 31.53 (train) | 64 / 51 / 20 |

Phân nhóm dùng diện tích **bounding box** <32², 32²–96², ≥96² pixel; không đồng nhất với phân nhóm COCO theo diện tích annotation/mask. Chi tiết min/median/p95/max từng lớp/tập và bảng đồng xuất hiện lớp ở statistics.json.

## Outlier và lỗi

Các cờ sau chỉ tạo ứng viên để người xem, không tự loại ảnh:

| Cờ | Số ảnh |
|---|---:|
| very_small_object_at_640 | 46 |
| low_laplacian_variance | 5 |
| bright | 2 |
| low_resolution | 10 |
| dark | 4 |
| extreme_aspect_ratio | 1 |
| low_contrast | 1 |

Ngưỡng: dark mean<40; bright mean>215; low contrast std<20; Laplacian variance<50; tỷ lệ dài/ngắn>3; cạnh ảnh<320; >30 box mục tiêu; cạnh ngắn vật sau resize640<8px. Một ảnh có thể có nhiều cờ. Đây là heuristic để review; giữ nguyên mẫu ngẫu nhiên.

Có 64 ảnh có ít nhất một cờ. Review gốc 100 ảnh (80 train/20 val, đủ đại diện tám lớp), thêm 48 ca có cờ chưa nằm trong danh sách gốc; tổng 148 ca pending. Không ghi reviewer giả.

Kiểm tự động: 500 ảnh giải mã được, đúng kích thước nguồn, 500 TXT hợp lệ, mọi box mục tiêu khớp annotation gốc, không trùng bytes/ID xuyên split, không trùng COCO128, đủ tám lớp trong cả hai split. Máy không xác nhận COCO không bỏ sót vật hoặc mọi nhãn đúng về ngữ nghĩa.

### Loại trước chọn và thay thế sau tải

- train: {"target_positive_images": 77518, "excluded_bad_images": 6161, "excluded_reasons": {"target crowd annotation": 6157, "duplicate target box": 3, "invalid geometry": 1}, "excluded_reference_images": 74, "eligible_images": 71290}
- Ảnh thay thế sau tải train: 0; chi tiết trong download_train.json.

- val: {"target_positive_images": 3272, "excluded_bad_images": 275, "excluded_reasons": {"target crowd annotation": 275}, "excluded_reference_images": 0, "eligible_images": 2997}
- Ảnh thay thế sau tải val: 0; chi tiết trong download_val.json.

## Giấy phép và ghi công

Giấy phép ảnh lấy từ metadata từng ảnh, không coi giấy phép annotation/code/model là giấy phép của mọi ảnh. Manifest lưu URL nguồn và license. Giữ nghĩa vụ attribution/điều khoản của từng nguồn khi chia sẻ.

| URL giấy phép ảnh | Số ảnh |
|---|---:|
| http://creativecommons.org/licenses/by-nc-sa/2.0/ | 155 |
| http://creativecommons.org/licenses/by-sa/2.0/ | 41 |
| http://creativecommons.org/licenses/by-nc-nd/2.0/ | 127 |
| http://creativecommons.org/licenses/by-nc/2.0/ | 64 |
| http://creativecommons.org/licenses/by/2.0/ | 83 |
| http://creativecommons.org/licenses/by-nd/2.0/ | 28 |
| http://flickr.com/commons/usage/ | 2 |

## Giới hạn và sử dụng

Bộ này là dataset đầu tiên để kiểm công cụ và phát triển, có nhãn nguồn sẵn; chưa chứng minh chất lượng camera phòng học. Pretrained YOLO26 dùng COCO; train2017 có overlap nguồn pretraining, val2017 là benchmark công khai, không phải test phòng học độc lập. Không suy ra fine-tune tốt hơn trên miền mới từ điểm COCO.

Phân bố lớp không cân bằng theo thiết kế; ảnh âm tính chưa được lấy chủ động, crowd mục tiêu bị loại, bối cảnh COCO chưa được phân loại thủ công thành ánh sáng/địa điểm/phiên. Mean độ sáng không xác định ngày/đêm. Chưa có review người thứ hai hoặc video thật. Không gọi bản draft là release đã khóa.

Khi nhận thêm dữ liệu nhóm, giữ bản này để truy thí nghiệm, kiểm nguồn/nhãn/trùng trước gộp và tạo phiên bản mới. Ghi số ảnh COCO, số ảnh nhóm và tổng ảnh duy nhất riêng.

## Hồ sơ bàn giao

Trong data/dataset/coco500_v0.1/: images, labels, manifest.csv, data.yaml, splits, checksums.json, DATA_CARD.md, statistics.json, validation.json, image_properties.csv, review.csv. Checksums là snapshot draft; các lần review/sửa cần phiên bản mới trước khi khóa và đo baseline.

Nguồn tham khảo: [COCO](https://cocodataset.org/#download), [FiftyOne COCO2017](https://docs.voxel51.com/dataset_zoo/datasets/coco_2017.html).
