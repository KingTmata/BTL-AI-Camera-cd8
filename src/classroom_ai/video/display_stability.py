"""Causal display-only smoothing. Never used as behavior evidence or identity repair."""

import copy
import math


class PoseDisplayStabilizer:
    config = dict(
        version="causal_display_ema_v1",
        time_constant_s=0.08,
        max_gap_s=0.2,
        jump_bbox_diagonals=0.75,
        imputation=False,
        scope="Current observed same-ID display only; raw behavior inputs unchanged",
    )

    def __init__(self):
        self.previous = {}
        self.segment = None
        self.time = None

    def update(self, tracks, *, segment, t):
        if self.segment != segment or self.time is not None and not self.time < t:
            self.previous.clear()
        self.segment, self.time = segment, t
        current, result = {}, []
        counts = {}
        for row in tracks:
            key = row.get("track_id")
            counts[key] = counts.get(key, 0) + 1
        for original in tracks:
            row = copy.deepcopy(original)
            key = row.get("track_id")
            box = row.get("bbox_xyxy")
            supported = (
                row.get("state") == "observed"
                and row.get("confirmed") is True
                and not row.get("identity_uncertain")
                and not row.get("uncertain")
                and counts[key] == 1
                and isinstance(box, list)
                and len(box) == 4
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                and box[2] > box[0]
                and box[3] > box[1]
            )
            previous = self.previous.get(key) if supported else None
            points = copy.deepcopy(row.get("keypoints13"))
            if previous:
                dt, old = t - previous[0], previous[1]
                a = old["bbox_xyxy"]
                scale = math.hypot(a[2] - a[0], a[3] - a[1])
                jump = math.hypot(
                    (box[0] + box[2] - a[0] - a[2]) / 2, (box[1] + box[3] - a[1] - a[3]) / 2
                )
                if 0 < dt <= self.config["max_gap_s"] and jump <= 0.75 * scale:
                    alpha = -math.expm1(-dt / self.config["time_constant_s"])
                    row["bbox_xyxy"] = [u + alpha * (v - u) for u, v in zip(a, box)]
                    old_points = old.get("display_keypoints13")
                    if (
                        isinstance(points, list)
                        and isinstance(old_points, list)
                        and len(points) == len(old_points) == 13
                    ):
                        for i, (p, q) in enumerate(zip(points, old_points)):
                            if (
                                isinstance(p, list)
                                and isinstance(q, list)
                                and len(p) == len(q) == 3
                                and all(
                                    isinstance(v, (int, float)) and math.isfinite(v) for v in p + q
                                )
                                and p[2] >= 0.05
                                and q[2] >= 0.05
                            ):
                                points[i] = [
                                    q[0] + alpha * (p[0] - q[0]),
                                    q[1] + alpha * (p[1] - q[1]),
                                    p[2],
                                ]
            row["display_keypoints13"] = points
            if supported:
                current[key] = (t, row)
            result.append(row)
        self.previous = current
        return result
