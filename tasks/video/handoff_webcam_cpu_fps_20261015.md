# Bàn giao: điều tra và tối ưu webcam CPU / Streamlit

## Yêu cầu của người dùng
Trong cùng app, ảnh tĩnh được báo khoảng200ms lần đầu và dưới100ms những lần sau, nhưng webcam dưới1FPS và p95 khoảng3000ms. Người dùng muốn tận dụng CPU tốt hơn để xử lý webcam, xác định nghẽn ở inference, khóa/thread, camera hay Streamlit và sửa bằng số đo thực tế. Không dùng CPU100% làm tiêu chí thành công: ưu tiên FPS xử lý thực, độ trễ thấp, frame mới và UI vẫn điều khiển được.

Repo: C:/Users/aawm8/Downloads/BTL-AI-Camera-cd8
Python: .venv/Scripts/python.exe
Entrypoint: python -m streamlit run src/classroom_ai/interfaces/web/app.py
URL phiên vừa bật: http://127.0.0.1:8501 (cần kiểm tra còn sống, không launch trùng).
App lần kiểm tra gần nhất chỉ thấy CPU, chưa có CUDA khả dụng; chưa thu được CPU model/core count do CIM bị chặn. Không mặc định GPU/hardware cụ thể.

Checkpoint mới đã thêm/chọn: weights/seed_repeats_20261014/E3_INVERT_seed2026_E2_best.pt
Nguồn: E3 gốc + INVERT, seed2026, train thêm2epoch ở1280; đã xác minh tải được và mapping8lớp đúng. SHA256 e68d0f4c04248c946ea58e02b11ca75c5da83a508243f85ccbde727bcb9e5d1b.
Model mặc định trong code vẫn R04; E2/E3 gốc còn trong menu. Khi so hiệu năng phải ghi/check đúng checkpoint thực được chọn, không dùng mặc định để suy ra.

## Sự thật đã đọc từ mã, chưa phải nguyên nhân đã xác nhận
- src/classroom_ai/inference/detector.py: ImageInspector.predict(), dòng112–142: khóa self.lock trước, rồi mới bắt đầu timer processing_ms; gồm YOLO predict, extraNMS nếu có, chuyển kết quả thành Pythonlist. Không tính thời gian chờ vào self.lock, cũng không tính manager.inference_slot ở ngoài. Chế độ3lượt dùng wrapper/model khác; không so với1lượt.
- src/classroom_ai/interfaces/web/app.py: dòng374–399: ảnh chạy trên threadscript Streamlit, qua manager.inference_slot rồi engine.predict; metric Xử lý ảnh chỉ hiển thị output.processing_ms cho một lần, KHÔNG phải p95.
- src/classroom_ai/inference/camera.py: CameraSession._run(), dòng96–127: thread daemon webcam đọc capture.read(); timestamp received được lấy SAU khi read trả frame; gọi self.predict(frame,confidence), rồi tính latency. Vì callback có manager.inference_slot và predict có self.lock, p95 webcam bao gồm thời gian chờ các khóa này.
- Webcam p95 là percentile95 của tối đa300mẫu từ lúc bắt đầu phiên, không warmup-excluded hiện tại; KHÔNG trực tiếp tính capture.read, JPEG hay browser rendering. CPU/UI contention vẫn có thể tác động gián tiếp.
- processing_fps ở camera.py dòng89 = tổng frames_processed / elapsed kể từ start, gồm startup và đọc camera; không phải FPS tức thời/slidingwindow. Ít frame có thể làm thống kê bị startup chi phối.
- src/classroom_ai/training/manager.py dòng194: inference_slot lấy OSfilelock guard rồi kiểm _active và giữ khóa suốt inference; kiểm chi phí lockwait/_active, số process và session tranh chấp. Không bỏ bảo vệ training/inference tùy tiện.
- src/classroom_ai/inference/camera.py: camera Windows thử DSHOW, MSMF, ANY; request640x480 và BUFFERSIZE1. Driver có thể không chấp nhận: cần đọc giá trị thực. Capture và inference hiện cùng1worker tuần tự; chỉ output được giữ mới nhất, chưa có capturethread độc lập bảo đảm không đọng buffer driver.
- src/classroom_ai/interfaces/web/app.py dòng161–187: webcam_panel fragment1giây/lần; webcam_image fragment theo preview.fps (mặc định10). JPEG được cache theo frame/token và vẫn gửi mỗi tick bằng dataURIbase64 qua st.image. Preview10FPS không có nghĩa inference10FPS.
- src/classroom_ai/interfaces/web/preview.py: render box, resizepreview, JPEG. configs/app.yaml runtime: preview_fps10, preview_max_width640, jpeg_quality75. pending_frames1 có trong YAML nhưng phải trace xem có thật sự được dùng trước khi dựa vào nó.
- Chỉ tìm thấy torch.set_num_threads(min(8,physicalcores)) và interop2 trong src/classroom_ai/training/worker.py cho TRAINING. Đừng suy ra inference đã được tune; đo threadsettings thực tế trong đúng thread/process.

## Điều tra bắt buộc trước khi chỉnh
1. Chụp baseline: checkpoint/hash, imgsz, passes, NMS, confidence, actualcameraresolution/backend/FPS, previewsettings; CPUmodel/physical+logicalcores, torch/ultralytics/OpenCV versions, torchintra/inter-op threads, OpenCVthreads, OMP/MKL/OPENBLAS settings, process/threadcounts, CPUpercore/process, RAM và dấu hiệu thermal/power throttling. Không đọc/in log secrets.
2. So cùng frame/cùng model/cùng settings: ảnh mainthread, frame replay ở backgroundthread KHÔNGStreamlit, camera khôngpreview, và camera cópreview. Warmup riêng, đo sustained30–60s mỗi cấu hình thay vì một ảnh. Bắt đầu onepass640, khôngextraNMS; ghi riêng cấu hình người dùng thực tế để không che nguyên nhân bằng giảm chất lượng.
3. Instrument riêng: capture.read time, managerlockwait + activecheck, inspectorlockwait, preprocess, forward, postprocess/resultconversion, publishlock, JPEG/base64, UIrender/browser FPS, tuổi frame khi hiển thị. Ghi timestamps/frameID và counters, tránh log mỗi frame gây nghẽn.
4. Đặt các metric cạnh nhau: output.processing_ms versus webcamlatency cùng frame. Chênh lớn => đo lock/callback overhead; cảhai chậm => profile inference/thread/shape. Camera/UI chậm phải có số đo hỗ trợ. Đo p50,p95,frames/sec slidingwindow, startup/warmup và samplecount riêng. Cố định p95definition trước/sau.
5. Kiểm nhiều Streamlit/Pythonprocess, tab/session sử dụng chung camera/model, lockcontention, firstmodelinit/warmup trong mainthread nhưng predict ở threadkhác, Torch/OpenMPpool oversubscription, kích thước tensor thực và imagecontent ảnh hưởng postprocess.

## Tối ưu theo bằng chứng
- Thử threadbudget1/2/4/... đến physicalcorecount/logicalcorecount thích hợp, tránh hardcode dùng toàn logicalcores. Setinterop trước khi parallelwork bắt đầu; đổi thiết lập cần processfresh khi runtime yêu cầu. Đo chọn cấu hình tốt nhất, chừa UI/capture xử lý; CPU% cao hơn nhưng FPSgiảm thì loại.
- Thử giới hạn OpenCV/BLASpool để tránh oversubscription, modelreused + inference_mode/warmup thích hợp, đọc kết quả tensor theo lô nếu profile cho thấy Pythonconversion tốn.
- Nếu inference worker thread là nghẽn đã đo: cân nhắc inferenceprocess riêng, modelowned bởi worker, mailboxbounded latestframe; không tạo một YOLO/process mỗi frame hay hàng đợi vô hạn. Bảo toàn đóngcam/stop/restart/heartbeat và trainingexclusiveguard.
- Nếu camera buffered: tách capture latestframe và inference, capacity1, bỏframe cũ có counters; đo tuổi frame. Không cho phép camera chiếm RAM hoặc tích hàng đợi.
- Nếu preview/Streamlit gây contention: giảm việc encode/gửi frame trùng, hạn chế fullrerun, cache publish phù hợp; thử tắtpreview để chứng minh. Chỉ đề xuất đổi transport/UI khi có số đo. Không rewrite toàn app vì phỏng đoán.
- Có thể thử CPUruntime ONNXRuntime/OpenVINO nếu stack hỗ trợ, nhưng kiểm output/classmapping/boxes và tốc độ so cùng model/input trước khi tích hợp. Không tự train, đổi checkpoint/mô hình lớn hơn hay giảm imgsize/passes để gọi là tối ưu cùng chất lượng. Các tradeoff chất lượng đưa thành option rõ ràng.

## Kết quả bàn giao cần có
Sửa thực tế sau profiling; bảng trước/sau cùng điều kiện gồm FPS xử lý, FPSpreview, p50/p95 inference, lockwait, app latency và frameage nếu đo được, CPUpercore/process, RAM, threadcount. Nêu bottleneck đã xác nhận, khác biệt metricảnh/webcam, tối ưu giữ/lùi và lý do. Chỉ giữ thay đổi có cải thiện hơn nhiễu đo. Kiểm stop/restart, đổi checkpoint/settings, traininglock, một cameraowner và chức năng ảnh tĩnh. Không hứa FPS cụ thể trước khi đo. Giữ quyền riêng tư: không ghi/phát tán ảnh webcam mặc định; cần replay thì ưu tiên ảnh có sẵn, không tự lưu ảnh camera cá nhân.

Người dùng yêu cầu công việc này được xử lý ở chatkhác. Chat hiện tại mới chuẩn bị bàn giao và đọc mã, chưa benchmark webcam hay xác nhận nguyên nhân; chưa có tên chatđích để gửi bằng tool.
