# Thành phần bên thứ ba

| Thành phần | Vai trò | Nguồn | Ghi chú |
|---|---|---|---|
| Ultralytics | YOLO26, ByteTrack tích hợp | https://github.com/ultralytics/ultralytics | Xem license và phiên bản thực trong `requirements-demo-cpu.lock.txt` sau khi cài. |
| PyTorch | Suy luận và huấn luyện | https://pytorch.org/ | Bản CPU cho máy demo; bản GPU cần cấu hình riêng. |
| COCO128 | Ảnh và nhãn tham khảo để thử tuần 1 | https://docs.ultralytics.com/datasets/detect/coco128/ | Tải từ URL trong tài liệu Ultralytics; chỉ dùng kỹ thuật, không phải test độc lập. Xem `data/reference/coco128/LICENSE` trong bản tải. |

Chưa sao chép mã từ các repo tham khảo trong kế hoạch. Khi sử dụng mã bên thứ ba, ghi URL, commit, file, license và thay đổi tại đây.
