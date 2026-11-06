"""Causal input-quality checks for raw Pose rules, not correction or GT.

Temporal thresholds are opt-in, uncalibrated safeguards. A rejection is unknown,
never a negative action observation. Callers own scene/gap/identity resets.
"""

import math


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite number")
    return float(value)


def frame_bounds(frame):
    if "width" not in frame and "height" not in frame:
        return None
    width, height = (_finite(frame.get(key)) for key in ("width", "height"))
    if min(width, height) <= 0:
        raise ValueError("frame width/height must be positive")
    return width, height


class PoseQualityGate:
    def __init__(self, config=None):
        defaults = dict(temporal_guard=False, max_joint_speed=6.0, max_scale_ratio=3.0)
        if config is not None and (not isinstance(config, dict) or set(config) - set(defaults)):
            raise ValueError("unsupported pose_quality config")
        self._config = {**defaults, **(config or {})}
        if not isinstance(self._config["temporal_guard"], bool):
            raise ValueError("temporal_guard must be boolean")
        if (
            _finite(self._config["max_joint_speed"]) <= 0
            or _finite(self._config["max_scale_ratio"]) <= 1
        ):
            raise ValueError("require max_joint_speed>0 and max_scale_ratio>1")
        self._states = {}

    @property
    def config(self):
        return dict(
            gate_version="causal_pose_quality_v1",
            calibrated=False,
            scale="bbox_diagonal_or_twice_torso",
            speed_units="scale_units_per_source_second_after_common_translation",
            **self._config,
        )

    @property
    def state_count(self):
        return len(self._states)

    def clear(self):
        self._states.clear()

    def retain(self, actors):
        self._states = {actor: state for actor, state in self._states.items() if actor in actors}

    def _result(self, reason, supported=False):
        return dict(
            status="supported" if supported else "unsupported",
            reason=reason,
            gate_version="causal_pose_quality_v1",
            temporal_guard=self._config["temporal_guard"],
        )

    def check(self, actor, track, indices, *, t, confidence, bounds=None):
        def reject(reason):
            self._states.pop(actor, None)
            return self._result(reason)

        if (
            track.get("state") != "observed"
            or track.get("confirmed") is not True
            or track.get("identity_uncertain", False)
            or track.get("uncertain", False)
        ):
            return reject("unknown_pose_or_identity")
        raw = track.get("keypoints17")
        if not isinstance(raw, list) or len(raw) != 17:
            return reject("missing_keypoints")
        try:
            points = {}
            for i in indices:
                if len(raw[i]) != 3:
                    return reject("invalid_keypoint")
                x, y, score = map(_finite, raw[i])
                if not confidence <= score <= 1:
                    return reject("low_confidence")
                if x < 0 or y < 0 or (bounds is not None and (x > bounds[0] or y > bounds[1])):
                    return reject("outside_raster")
                points[i] = (x, y)
            shoulder_width = math.dist(points[5], points[6])
            if shoulder_width <= 1e-9:
                return reject("degenerate_shoulders")
            for a, b in (
                (5, 7),
                (7, 9),
                (6, 8),
                (8, 10),
                (5, 11),
                (6, 12),
                (11, 13),
                (13, 15),
                (12, 14),
                (14, 16),
            ):
                if a in points and b in points and math.dist(points[a], points[b]) <= 1e-9:
                    return reject("degenerate_limb")
            if not self._config["temporal_guard"]:
                return self._result("valid_geometry", True)
            center = tuple((points[5][j] + points[6][j]) / 2 for j in (0, 1))
            box = track.get("bbox_xyxy")
            if isinstance(box, list) and len(box) == 4:
                x1, y1, x2, y2 = map(_finite, box)
                if not (x1 < x2 and y1 < y2):
                    return reject("invalid_scale")
                scale = math.hypot(x2 - x1, y2 - y1)
            elif all(len(raw[i]) == 3 and confidence <= _finite(raw[i][2]) <= 1 for i in (11, 12)):
                hip = tuple((_finite(raw[11][j]) + _finite(raw[12][j])) / 2 for j in (0, 1))
                scale = 2 * math.dist(center, hip)
            else:
                return reject("missing_robust_scale")
            if not math.isfinite(scale) or scale <= 1e-9:
                return reject("invalid_scale")
            relative = tuple((points[i][0] - center[0], points[i][1] - center[1]) for i in indices)
            previous = self._states.get(actor)
            # Store the current geometry even on an outlier, so a stable new
            # baseline can recover; no replacement coordinates are returned.
            self._states[actor] = (t, relative, scale)
            if previous is not None:
                dt = t - previous[0]
                if dt <= 0:
                    return reject("nonpositive_time")
                ratio = max(scale / previous[2], previous[2] / scale)
                speed = (
                    max(math.dist(a, b) for a, b in zip(relative, previous[1], strict=True))
                    / previous[2]
                    / dt
                )
                if (
                    ratio > self._config["max_scale_ratio"]
                    or speed > self._config["max_joint_speed"]
                ):
                    return self._result("temporal_outlier")
            return self._result("valid_geometry", True)
        except (TypeError, ValueError, IndexError, KeyError):
            return reject("invalid_keypoint")
