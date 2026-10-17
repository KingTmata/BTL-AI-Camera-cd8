"""Main-thread HighGUI display. No model work or web transport here."""

from __future__ import annotations

import cv2
import numpy as np


class NativeDisplay:
    def __init__(self):
        self.title = "AI Camera | Video"
        self.opened = False
        self.seek_enabled = False
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 1100, 740)
        self.status("Preparing video...")
        self.opened = True

    def status(self, text):
        image = np.zeros((160, 1000, 3), dtype=np.uint8)
        cv2.putText(
            image,
            text[:105],
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Space: pause/resume | Q/Esc: stop | Preview has no audio",
            (15, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.title, image)

    def show(self, packet, result, latest, text):
        # Overlay only the exact frame, never the previous frame's boxes.
        rows = (
            result["detections"] if result and result["frame_index"] == packet.frame_index else []
        )
        height, width = packet.image.shape[:2]
        scale = min(1.0, 1280 / width, 800 / height)
        image = packet.image
        if scale < 1:
            image = cv2.resize(
                image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
            )
            rows = [{**r, "bbox_xyxy": [v * scale for v in r["bbox_xyxy"]]} for r in rows]
        if rows:
            from classroom_ai.inference.detector import draw_boxes

            image = draw_boxes(image, rows)
        else:
            image = image.copy()
        footer = np.zeros((110, image.shape[1], 3), dtype=np.uint8)
        lines = [
            f"Frame {packet.frame_index} | t={packet.timestamp_seconds:.3f}s | {text}",
            (
                "Full result rendered in video"
                if text.startswith("full")
                else "Not inferred on this frame"
            )
            if result is None
            else f"YOLO: {len(rows)} detections on this frame",
            "Space: pause | Q/Esc: stop | silent preview",
        ]
        if self.seek_enabled:
            lines[2] = "Slider: seek | A/D: -/+5s | Space: pause/replay | Q/Esc: stop"
        if latest is not None:
            age = max(0.0, packet.timestamp_seconds - latest["timestamp_seconds"])
            lines[1] += (
                f" | Latest: t={latest['timestamp_seconds']:.3f}s, age={age:.3f}s, n={len(latest['detections'])}"
            )
        for index, line in enumerate(lines):
            cv2.putText(
                footer,
                line,
                (8, 25 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(self.title, np.vstack((image, footer)))

    def enable_seek(self, frame_count, fps):
        self.seek_enabled = True
        self.frame_count, self.fps = frame_count, float(fps)
        self.current_frame = self.slider_position = 0
        self.seek_request = None
        self.at_end = False
        if frame_count > 1:
            cv2.createTrackbar("Frame", self.title, 0, frame_count - 1, self._slider_changed)

    def _slider_changed(self, position):
        if position != self.slider_position:
            self.slider_position = position
            self.seek_request = position

    def update_position(self, frame_index):
        self.current_frame = frame_index
        if self.frame_count > 1 and self.seek_request is None:
            self.slider_position = frame_index
            cv2.setTrackbarPos("Frame", self.title, frame_index)

    def take_seek(self):
        target, self.seek_request = self.seek_request, None
        if target is not None:
            self.at_end = False
        return target

    def mark_end(self):
        self.at_end = True

    def poll(self):
        # On Windows waitKey(1) can sleep for a full OS timer quantum on every
        # poll, including between frames. pollKey pumps events without that wait.
        key = cv2.pollKey()
        if key in (27, ord("q"), ord("Q")):
            return "quit"
        if self.opened:
            try:
                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    return "quit"
            except cv2.error:
                return "quit"
        if getattr(self, "seek_enabled", False):
            if key in (ord("a"), ord("A"), 2424832, 65361):
                self.seek_request = max(0, self.current_frame - round(5 * self.fps))
            elif key in (ord("d"), ord("D"), 2555904, 65363):
                self.seek_request = min(
                    self.frame_count - 1, self.current_frame + round(5 * self.fps)
                )
            elif key == ord(" ") and self.at_end:
                self.seek_request = 0
                return "replay"
        return "pause" if key == ord(" ") else None

    def close(self):
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            # The window manager may already have destroyed this window.
            pass
        self.opened = False
