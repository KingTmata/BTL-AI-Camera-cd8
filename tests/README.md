# Kiểm thử

Chạy từ gốc repo: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

`test_inspection.py` kiểm mapping, nhãn, pixel và crop. `test_app.py` chạy giao diện Streamlit bằng AppTest và suy luận thật; tự skip nếu chưa tải COCO128/weights. Hướng dẫn đầy đủ ở [docs/KIEM_THU.md](../docs/KIEM_THU.md). Kiểm webcam, tracking và chất lượng test set chưa nằm trong các bài này.
