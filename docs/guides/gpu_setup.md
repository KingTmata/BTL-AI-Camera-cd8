# Thiết bị xử lý CPU / GPU

Trong sidebar **Nguồn & mô hình**, ngay dưới **Checkpoint**, chọn **Thiết bị xử lý**.
CPU là mặc định. GPU NVIDIA xuất hiện theo tên card và chỉ số CUDA khi PyTorch
chạy xong phép tính thử trên card đó. **Kiểm tra lại GPU** xóa cache kiểm tra;
sau khi cài PyTorch/driver cần khởi động lại backend Streamlit.

Lựa chọn áp dụng cho ảnh, frame video và webcam, cả một/ba lượt suy luận.
Cache model tách theo checkpoint, thời gian sửa và thiết bị. Đổi thiết bị làm
kết quả ảnh cũ hết hiệu lực; bấm **Phân tích ảnh** lại. Webcam dừng phiên cũ,
rồi bấm **Bật webcam** để dùng thiết bị mới. JSON ghi thiết bị đã chọn.
Nếu GPU đã chọn biến mất khi kiểm lại, UI chuyển CPU, dừng webcam và bỏ kết quả cũ.
Nếu CUDA lỗi trong suy luận, app báo lỗi; không âm thầm đổi thiết bị.

Chỉ số `cuda:0` là GPU CUDA đầu tiên PyTorch nhìn thấy, không phải số GPU trong
Task Manager. iGPU Intel/AMD không xuất hiện trong danh sách CUDA. Bản này chưa
tích hợp XPU, OpenVINO hoặc DirectML; không gộp RTX và iGPU để chạy một model.

Máy có RTX 5060 cần driver NVIDIA và bản PyTorch CUDA hỗ trợ Blackwell, không
phải bản `+cpu`. Dùng bộ torch/torchvision tương thích theo
[hướng dẫn cài PyTorch](https://pytorch.org/get-started/locally/).
[PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) bắt đầu hỗ trợ Blackwell với
CUDA 12.8; các bản mới có ma trận CUDA/driver khác, kiểm hướng dẫn hiện hành.
Không sao chép `.venv` từ máy khác. Trong venv của máy chạy backend, kiểm:

```powershell
& .venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Thiết lập R04, confidence 0,25, imgsz 640, một lượt và NMS tắt vẫn giữ mặc định.
Chỉ thêm chọn CPU/CUDA; không đưa Precision/TensorRT hay các chế độ khác của
worktree vào thay đổi này. Training vẫn dùng cấu hình riêng, không chịu tác động
từ lựa chọn thiết bị của Recognition.

Đã kiểm tra thực trên máy AMD Radeon 780M với PyTorch CPU: chỉ hiện CPU và lý do
CUDA chưa có. Nhánh CUDA được kiểm thử bằng mô phỏng; cần kiểm trên máy RTX 5060
để xác nhận tốc độ và khả năng tương thích model thực.
