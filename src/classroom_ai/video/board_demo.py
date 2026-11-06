"""User-defined demo aliases: near board=writing, approaching board=going up.

These cues do not assert visual writing or GT. Uses source time, never face ID.
"""

import math


class BoardDemoState:
    def __init__(self, floor_roi, *, offscreen=False):
        self.roi = floor_roi
        self.offscreen = offscreen
        self.previous = {}
        self.segment = None

    def update(self, tracks, *, segment, t, width, height):
        if segment != self.segment:
            self.previous.clear()
        self.segment = segment
        current, result = {}, {}
        counts = {}
        for row in tracks:
            key = row.get("track_id")
            counts[key] = counts.get(key, 0) + 1
        a, b, c, d = self.roi
        for row in tracks:
            key = row.get("track_id")
            if (
                row.get("state") != "observed"
                or row.get("confirmed") is not True
                or row.get("identity_uncertain")
                or row.get("uncertain")
                or counts[key] != 1
            ):
                continue
            box = row.get("bbox_xyxy", [])
            if len(box) != 4 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in box
            ):
                continue
            x, y = (box[0] + box[2]) / 2 / width, box[3] / height
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                continue
            distance = math.hypot(max(a - x, 0, x - c), max(b - y, 0, y - d))
            if self.offscreen:
                distance = max(0, d - y)
            near = a <= x <= c and b <= y <= d
            old = self.previous.get(key)
            moving = bool(old and 0 < t - old[0] <= 0.2 and old[1] - distance > 0.01)
            if old and 0 < t - old[0] <= 0.2:
                baseline = old[2] if t - old[2][0] <= 0.7 else (old[0], old[1])
                moving = moving or (0.2 <= t - baseline[0] <= 0.7 and baseline[1] - distance > 0.01)
            else:
                baseline = (t, distance)
            common = dict(
                origin="rule",
                review="draft",
                calibrated=False,
                observed_through_s=t,
                rule="user_board_proximity_demo_v1",
                evidence_scope="user_demo_proximity_not_visual_writing",
            )
            result[key] = [
                dict(
                    common,
                    label="move_to_board",
                    status="candidate"
                    if moving and (not near or self.offscreen)
                    else "no_suggestion",
                ),
                dict(
                    common,
                    label="write_board",
                    status="active" if near and not self.offscreen else "no_suggestion",
                ),
            ]
            current[key] = (t, distance, baseline)
        self.previous = current
        return result
