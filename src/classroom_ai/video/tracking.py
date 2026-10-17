"""Session-local person association and display geometry; no model or GUI ownership."""

import math
from itertools import count
from types import SimpleNamespace

import cv2
import numpy as np


def clip_box(box, width, height):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    return box.tolist() if box[2] > box[0] and box[3] > box[1] else None


def center_size(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def corners(box):
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def box_iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1e-12, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class PersonTracker:
    def __init__(self, fps, confidence, max_lost_seconds, metrics):
        # Check explicitly before Ultralytics import: never let its AutoUpdate install at runtime.
        try:
            import lap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Tracking cần lap: pip install -r requirements/video_tracking.txt"
            ) from exc
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker, STrack

        allocator = count(1)

        class SessionTrack(STrack):
            @staticmethod
            def next_id():
                return next(allocator)

        class SessionByteTracker(BYTETracker):
            track_class = SessionTrack

            @staticmethod
            def reset_id():
                # SessionTrack owns its allocator; scene resets must not recycle IDs.
                pass

        self.settings = dict(
            track_high_thresh=confidence,
            track_low_thresh=0.8 * confidence,
            new_track_thresh=confidence,
            match_thresh=0.8,
            fuse_score=False,
            track_buffer=math.ceil(float(fps) * max_lost_seconds),
        )
        self.tracker = SessionByteTracker(SimpleNamespace(**self.settings))
        self.boxes_class = Boxes
        self.metrics, self.max_lost = metrics, max_lost_seconds
        self.states = {}
        self.segment = 0

    def finish(self):
        self.metrics.count("tracks_ended", len(self.states))
        self.states.clear()

    def reset(self):
        self.finish()
        self.tracker.reset()
        self.segment += 1
        self.metrics.count("track_scene_resets")

    def update(self, frame_index, timestamp, detections):
        # Expire before association: the native tracker otherwise can rebind a stale
        # lost track before its end-of-update expiry check.
        expired = {
            key
            for key, state in self.states.items()
            if state["frame"] < frame_index - 1 and timestamp - state["time"] > self.max_lost + 1e-9
        }
        for attr in ("tracked_stracks", "lost_stracks"):
            setattr(
                self.tracker,
                attr,
                [t for t in getattr(self.tracker, attr) if t.track_id not in expired],
            )
        for key in expired:
            del self.states[key]
        self.metrics.count("tracks_ended", len(expired))
        selected = [(i, r) for i, r in enumerate(detections) if r["class_id"] == 0]
        data = np.asarray(
            [[*r["bbox_xyxy"], r["confidence"], 0] for _, r in selected], dtype=np.float32
        ).reshape(-1, 6)
        self.tracker.update(self.boxes_class(data, orig_shape=(1, 1)))
        rows = []
        for track in self.tracker.tracked_stracks:
            if track.frame_id != self.tracker.frame_id:
                continue
            key = int(track.track_id)
            detection_index, detection = selected[int(track.idx)]
            previous = self.states.get(key)
            if previous is None:
                previous = dict(frame=-2, streak=0, confirmed=False, time=timestamp)
                self.metrics.count("tracks_created")
            consecutive = previous["frame"] == frame_index - 1
            streak = previous["streak"] + 1 if consecutive else 1
            confirmed = previous["confirmed"] or streak >= 2
            if confirmed and not previous["confirmed"]:
                self.metrics.count("tracks_confirmed")
            if previous["confirmed"] and not consecutive:
                self.metrics.count("tracks_refound")
            self.states[key] = dict(
                frame=frame_index, streak=streak, confirmed=confirmed, time=timestamp
            )
            rows.append(
                {
                    **detection,
                    "track_id": key,
                    "confirmed": confirmed,
                    "detection_index": detection_index,
                }
            )
        live = {t.track_id for t in self.tracker.tracked_stracks + self.tracker.lost_stracks}
        removed = set(self.states) - live
        for key in removed:
            del self.states[key]
        self.metrics.count("tracks_ended", len(removed))
        self.metrics.values["track_state_peak"] = max(
            self.metrics.values.get("track_state_peak", 0), len(self.states)
        )
        return dict(
            frame_index=frame_index,
            timestamp_seconds=timestamp,
            segment_id=self.segment,
            tracks=rows,
        )


class BoxSmoother:
    def __init__(self, alpha, width, height):
        self.alpha, self.width, self.height = alpha, width, height
        self.previous = {}

    def update(self, key, frame_index, raw_box):
        raw = clip_box(raw_box, self.width, self.height)
        if raw is None:
            return None
        prev = self.previous.get(key)
        result = raw
        if prev is not None and prev[0] == frame_index - 1:
            result = clip_box(
                corners(self.alpha * center_size(raw) + (1 - self.alpha) * center_size(prev[1])),
                self.width,
                self.height,
            )
            if box_iou(raw, result) < 0.5:
                result = raw
        self.previous[key] = (frame_index, result)
        return result

    def prune(self, frame_index):
        self.previous = {
            key: value for key, value in self.previous.items() if value[0] >= frame_index - 1
        }


class SceneCuts:
    def __init__(self):
        self.previous = None

    def update(self, image):
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cut = False
        if self.previous is not None:
            old_hist, old_gray = self.previous
            cut = (
                cv2.compareHist(old_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.6
                and float(np.abs(gray.astype(float) - old_gray).mean()) / 255 > 0.25
            )
        self.previous = hist, gray
        return cut
