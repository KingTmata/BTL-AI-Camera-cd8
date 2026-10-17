# Nhịp preview webcam Streamlit

## Nguyên nhân và cách sửa (12/10/2026)

`src/classroom_ai/interfaces/web/app.py` dùng `@st.fragment(run_every=0.3)`: timer yêu cầu tối đa
`1 / 0.3 = 3,33` lần/giây, hoàn toàn không đọc `runtime.preview_fps`.
Mỗi tick còn gọi `manager.active()`, lấy OS guard chung với `inference_slot()`.
Worker giữ guard suốt lượt suy luận nên UI có thể chờ/retry (tối đa 5 giây),
tranh khóa với camera. Chỉ sửa timer chưa đủ: mẫu thử webcam sau sửa timer
nhưng trước bỏ guard ở UI chỉ thấy khoảng 1,21 ảnh mới/giây trong 15 giây.
Đây là số đo của một lần chạy, không phải benchmark chung cho mọi máy.

Fragment ảnh nay được đăng ký trên mỗi full app run với `run_every=1/preview_fps`;
controls, metrics và báo cáo dùng fragment riêng ở 1 Hz để giảm tải UI.
Đọc riêng ba khóa runtime trong `configs/app.yaml`; giá trị thiếu dùng mặc định
10 FPS, tối đa 640px, JPEG quality 75. Giá trị sai báo lỗi trước khi đăng ký timer.
Sau khi sửa YAML, reload trang để đăng ký lại fragment. Không thay `model`,
checkpoint R04, CPU, confidence 0,25, imgsz 640, một lượt suy luận hay NMS tắt.

Mỗi fragment lấy một snapshot mỗi tick, không lấy training guard. Trang chính vẫn kiểm
training, chuyển sang Training vẫn dừng camera; worker vẫn kiểm `_active()`
trong `inference_slot()` trước mỗi dự đoán. Nếu training bắt đầu từ phiên khác,
worker từ chối suy luận, chuyển lỗi và giải phóng camera.

Camera publish frame/kết quả mới nhất, không xếp hàng theo nhịp UI. Snapshot
copy tối đa 300 latency dưới lock, rồi tính percentile ngoài lock. Resize, scale
tọa độ box, vẽ và encode JPEG đều ở UI, ngoài camera lock; không sửa frame/box
worker đang chia sẻ. Ảnh nhỏ không bị phóng lớn; `width="stretch"` chỉ đổi bố cục.
Metadata phiên bản package được cache, tránh đọc distribution mỗi tick. UI giữ
một JPEG mới nhất; chỉ encode lại khi frame, lần mở camera, bộ lọc lớp hoặc
thiết lập preview đổi. JPEG truyền bằng data URI qua Streamlit, tránh request
`/media` riêng và ảnh cũ bị thu hồi trước khi browser tải xong. Base64 tăng
dung lượng khoảng 33% so với JPEG gốc; ảnh vẫn giới hạn max_width/quality.
Không lưu frame vào hàng đợi.

## Đo FPS hiển thị

Phân biệt ba số:

- **FPS mục tiêu**: `preview_fps`, là yêu cầu timer, chưa phải số đo.
- **FPS xử lý** trên UI/JSON: số frame xử lý / thời gian phiên, gồm mở camera
  và warmup. p95 chỉ đo từ nhận frame đến kết quả, chưa gồm browser.
- **FPS ảnh mới ở browser**: đếm URL ảnh preview mới đã tải thành công;
  không đếm lại cùng URL và không dùng số lần fragment chạy thay thế.

Giữ tab ở foreground, chọn Webcam trực tiếp, Bật webcam, đợi warmup 10–15 giây.
Mở DevTools Console của chính tab ứng dụng rồi chạy đoạn dưới, đợi 30 giây.
Đo cùng model/camera, cùng lớp hiển thị, kích thước cửa sổ và tải CPU. Có thể
so 5/10/20 FPS bằng sửa riêng `preview_fps` và reload; không đổi model để so tốc độ.

```javascript
(() => {
  const start = performance.now();
  let previous = document.querySelector('[data-testid="stImage"] img')?.currentSrc;
  let count = 0;
  const timer = setInterval(() => {
    const img = document.querySelector('[data-testid="stImage"] img');
    if (img?.complete && img.naturalWidth > 0 && img.currentSrc && img.currentSrc !== previous) {
      previous = img.currentSrc;
      count++;
    }
    const seconds = (performance.now() - start) / 1000;
    if (seconds >= 30) {
      clearInterval(timer);
      console.table({seconds, newLoadedImages: count, displayedImageFps: count / seconds,
                     tabVisibility: document.visibilityState});
    }
  }, 10);
})();
```

Selector chỉ dành cho trang webcam có một ảnh `stImage`. Polling 10ms có thể bỏ
lỡ ảnh hoặc bị throttle khi tab nền; kết quả là ước lượng ảnh mới đã decode/load,
không bảo đảm mỗi ảnh được paint ra màn hình. Để xác nhận paint/drop frame,
ghi DevTools Performance với Screenshots trong 30 giây, xem filmstrip và main
thread; ghi lại trạng thái tab, FPS xử lý và p95. JSON tải xuống ghi target,
interval, max_width và quality, không gán FPS xử lý thành FPS hiển thị.

## Giới hạn còn lại

FPS ảnh mới không vượt tốc độ có kết quả mới từ camera/model; timer có thể gửi
lại cùng ảnh nếu worker chậm. Streamlit vẫn rerender ảnh mỗi tick và controls,
metrics, nút tải báo cáo mỗi giây. JPEG/resize, base64, WebSocket, React, tải CPU,
số phiên và timer của browser đều có thể làm FPS thấp hơn mục tiêu; không có
bảo đảm realtime hoặc end-to-end latency. Worker hiện đọc camera rồi suy luận
nối tiếp, chưa có luồng capture riêng, nên backend webcam vẫn có thể buffer.
JPEG giảm dung lượng nhưng mất chất lượng; resize preview không giảm imgsz
suy luận. Không thêm sleep, queue vô hạn hay đổi thiết lập model để đạt FPS.

Tham khảo API chính thức:
[st.fragment](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment),
[st.image](https://docs.streamlit.io/develop/api-reference/media/st.image).

## Kết quả kiểm tra và giới hạn của phép đo

- Bộ test hiện có: **56 test pass** bằng
  `.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests`.
  Sandbox Windows chặn ghi thư mục tạm ở lần đầu; chạy ngoài sandbox đã qua.
- Test preview kiểm tra đăng ký timer 5/10/20 FPS và controls 1 Hz; YAML sai;
  resize/box không mutate dữ liệu worker; JPEG được dùng lại khi frame chưa đổi;
  ảnh truyền bằng data URI; UI không lấy training guard; percentile ngoài lock.
- Webcam thật, R04/CPU/640/confidence 0,25/1 lượt/NMS tắt, JPEG 640px/75,
  preview mục tiêu 10 FPS, browser tích hợp ở `localhost:8517`: mẫu cuối dài
  **29,974 giây**, **657 lần đọc DOM**, **8 ảnh mới đã tải**, tức **0,267 FPS**
  theo công cụ lấy mẫu. UI cuối cửa sổ báo FPS xử lý **0,9**, p95 **2386 ms**.
  Đã xác nhận `currentSrc` là data URI, không có lỗi tải ảnh mới trong cửa sổ đó.
  Công cụ đọc DOM có chi phí/độ trễ riêng; mẫu này không chứng minh nhịp paint
  hay nhịp fragment thực tế và chưa xác nhận được 10 FPS hiển thị trên máy này.
  Cần dùng Console/Performance trực tiếp như trên để đo lại trong điều kiện
  tải máy ổn định; không so các lần mở lại camera như benchmark kiểm soát.
- Console vẫn ghi `Download Button source error - 404` cho báo cáo cập nhật;
  data URI chỉ xử lý đường ảnh preview. Đường tải báo cáo Streamlit còn dùng
  media storage và cần kiểm tra riêng nếu lỗi này tái hiện khi người dùng tải.
