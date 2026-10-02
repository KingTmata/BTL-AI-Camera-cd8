"""Mở webcam trên máy chạy Python, thử backend và xác nhận đọc được hình."""

import sys

import cv2


def open_camera(index: int):
    if index < 0:
        raise ValueError("Chỉ số webcam phải từ 0 trở lên.")
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Một số driver báo mở thành công nhưng không trả frame.
                for _ in range(3):
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size:
                        return capture
        except cv2.error:
            pass
        capture.release()
    raise RuntimeError(
        f"Không đọc được webcam {index}. Đóng Camera/Teams/Zoom, kiểm quyền camera cho ứng dụng desktop, "
        "kiểm nắp che/cáp USB hoặc thử chỉ số camera khác (0, 1, 2)."
    )
