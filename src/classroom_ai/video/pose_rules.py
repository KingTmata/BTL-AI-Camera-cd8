"""Uncalibrated, causal raw-Pose proxies; never empirical GT or student scores.

The floor ROI is a human-selected encoded-raster region, not a calibrated world
plane. Perspective, camera motion and occlusion remain limitations. No writing,
intent, participation, generic standing or sitting classification is emitted.
"""

from __future__ import annotations

import math

from classroom_ai.video.actions import RaiseHandState, _actor, _json_copy, _number
from classroom_ai.video.pose_quality import PoseQualityGate, frame_bounds

RULES = {"stand_up": "coco17_sit_to_stand_v1", "move_to_board": "coco17_floor_roi_arrival_v1"}
DEFAULTS = dict(
    confidence=0.5,
    seated_knee_deg=125.0,
    seated_hip_deg=135.0,
    standing_knee_deg=155.0,
    standing_hip_deg=150.0,
    baseline_dwell_s=0.4,
    standing_dwell_s=0.4,
    hip_rise_leg_fraction=0.15,
    max_gap_s=0.25,
    outside_dwell_s=0.4,
    progress_diagonal=0.03,
    arrival_dwell_s=1.0,
    roi_enter_margin=0.01,
    roi_leave_margin=0.02,
    max_sequence_s=15.0,
    recent_completed_s=1.0,
)


def validate_board_roi(value, *, enter_margin=0.01):
    """Return detached normalized [left, top, right, bottom], or None."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("board_roi requires four normalized coordinates")
    roi = [_number(v, "board_roi") for v in value]
    x1, y1, x2, y2 = roi
    if not (
        0 <= x1 < x2 <= 1
        and 0 <= y1 < y2 <= 1
        and x2 - x1 > 2 * enter_margin
        and y2 - y1 > 2 * enter_margin
    ):
        raise ValueError("board_roi must be ordered within [0,1], with a nonempty inner region")
    return roi


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _angle(a, b, c):
    ab, cb = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    left, right = math.hypot(*ab), math.hypot(*cb)
    if min(left, right) <= 1e-9 or not all(math.isfinite(v) for v in (left, right)):
        return None
    dot = sum((ab[i] / left) * (cb[i] / right) for i in (0, 1))
    return math.degrees(math.acos(max(-1, min(1, dot))))


def _body(track, cfg, bounds=None):
    if (
        track.get("state") != "observed"
        or track.get("confirmed") is not True
        or track.get("identity_uncertain", False)
        or track.get("uncertain", False)
    ):
        return None
    raw = track.get("keypoints17")
    if not isinstance(raw, list) or len(raw) != 17:
        return None
    try:
        points = {}
        for index in (5, 6, 11, 12, 13, 14, 15, 16):
            if len(raw[index]) != 3:
                return None
            x, y, score = [_number(v, "keypoint") for v in raw[index]]
            if not (cfg["confidence"] <= score <= 1 and x >= 0 and y >= 0):
                return None
            if bounds is not None and (x > bounds[0] or y > bounds[1]):
                return None
            points[index] = (x, y)
        knees = [
            _angle(points[h], points[k], points[a]) for h, k, a in ((11, 13, 15), (12, 14, 16))
        ]
        hips = [_angle(points[s], points[h], points[k]) for s, h, k in ((5, 11, 13), (6, 12, 14))]
        if any(v is None for v in knees + hips):
            return None
        hip, shoulder = _mid(points[11], points[12]), _mid(points[5], points[6])
        torso = math.dist(hip, shoulder)
        leg = (
            sum(
                math.dist(points[h], points[k]) + math.dist(points[k], points[a])
                for h, k, a in ((11, 13, 15), (12, 14, 16))
            )
            / 2
        )
        if torso <= 1e-9 or leg <= 1e-9 or not all(math.isfinite(v) for v in (torso, leg)):
            return None
        return dict(
            hip=hip,
            axis=((shoulder[0] - hip[0]) / torso, (shoulder[1] - hip[1]) / torso),
            leg=leg,
            foot=_mid(points[15], points[16]),
            seated=all(v < cfg["seated_knee_deg"] for v in knees)
            and all(v < cfg["seated_hip_deg"] for v in hips),
            standing=all(v > cfg["standing_knee_deg"] for v in knees)
            and all(v > cfg["standing_hip_deg"] for v in hips),
        )
    except (TypeError, ValueError, IndexError):
        return None


class BasicPoseActionState:
    """Three independent causal rule/draft hints, bounded to current actors.

    config overrides declared DEFAULTS only. Board input frames require positive
    width/height of the unchanged encoded raster. Completed transition/arrival
    hints expire in source time and cannot be recovered from an old event log.
    """

    def __init__(
        self,
        board_roi=None,
        *,
        config=None,
        pose_quality=None,
        relation_window_s=60,
        hand_cue="nose_both_arms_v1",
    ):
        if config is not None and (not isinstance(config, dict) or set(config) - set(DEFAULTS)):
            raise ValueError("unsupported basic_pose config")
        self._cfg = {**DEFAULTS, **(config or {})}
        for key, value in self._cfg.items():
            if _number(value, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if (
            self._cfg["confidence"] > 1
            or not self._cfg["seated_knee_deg"] < self._cfg["standing_knee_deg"] < 180
            or not self._cfg["seated_hip_deg"] < self._cfg["standing_hip_deg"] < 180
            or not self._cfg["roi_enter_margin"] < self._cfg["roi_leave_margin"] < 0.5
            or self._cfg["progress_diagonal"] >= 1
        ):
            raise ValueError("inconsistent basic_pose config thresholds")
        self._roi = validate_board_roi(board_roi, enter_margin=self._cfg["roi_enter_margin"])
        if not 0 < _number(relation_window_s, "relation_window_s") <= 60:
            raise ValueError("relation_window_s must be in (0,60]")
        self._relation_window = relation_window_s
        self._body_quality = PoseQualityGate(pose_quality)
        self._raise = RaiseHandState(pose_quality=pose_quality, hand_cue=hand_cue)
        self._actors, self._people = {}, []
        self._t, self._segment = None, None
        self._finished = False
        self._stats = dict(
            stand_supported_observations=0,
            board_supported_observations=0,
            incomplete_stand_candidates=0,
            incomplete_board_candidates=0,
            invalidated_stand_baselines=0,
            stand_events=0,
            board_events=0,
        )

    @property
    def config(self):
        return _json_copy(
            dict(
                engine="basic_pose_v1",
                calibrated=False,
                board_roi=self._roi,
                thresholds=self._cfg,
                rules=RULES,
                raise_hand=self._raise.config,
                hand_cue=self._raise.config["hand_cue"],
                pose_quality=self._body_quality.config,
                context_relation=dict(
                    rule="raise_hand_then_move_to_board_v1",
                    max_separation_s=self._relation_window,
                    calibrated=False,
                ),
            )
        )

    @property
    def people(self):
        return _json_copy(self._people)

    @property
    def counts(self):
        statuses = [set(a["status"] for a in p["actions"]) for p in self._people]
        return dict(
            active_actor_count=sum(bool(s & {"active", "release_pending"}) for s in statuses),
            pending_actor_count=sum("candidate" in s for s in statuses),
            recent_completed_actor_count=sum("completed" in s for s in statuses),
        )

    @property
    def observation_counts(self):
        return {**self._raise.observation_counts, **self._stats}

    def _drop(self, actor):
        state = self._actors.pop(actor, {})
        for phase, count in (
            ("stand", "incomplete_stand_candidates"),
            ("board", "incomplete_board_candidates"),
        ):
            if state.get(phase, {}).get("start") is not None:
                self._stats[count] += 1

    def _hint(self, label, t, status="no_suggestion", **extra):
        return dict(
            label=label,
            status=status,
            origin="rule",
            review="draft",
            calibrated=False,
            rule=RULES[label],
            observed_through_s=t,
            **extra,
        )

    def _event(self, actor, label, start, t):
        return dict(
            segment_id=actor[0],
            track_id=actor[1],
            label=label,
            start_s=start,
            end_s=t,
            origin="rule",
            review="draft",
            uncertain=False,
            evidence=dict(
                rule=RULES[label],
                calibrated=False,
                thresholds=dict(self._cfg),
                pose_quality=self._body_quality.config,
                board_roi=list(self._roi) if label == "move_to_board" else None,
                boundary_truncated=dict(start=False, end=False),
                boundary_reason=dict(
                    start="observed_transition",
                    end="confirmed_standing"
                    if label == "stand_up"
                    else "confirmed_floor_roi_arrival",
                ),
            ),
        )

    def _stand(self, state, body, actor, t, events):
        cfg, phase = self._cfg, state.setdefault("stand", {})
        recent = state.get("stand_recent")
        if recent and (
            not body["standing"] or t - recent["confirmed_at_s"] >= cfg["recent_completed_s"] - 1e-9
        ):
            state.pop("stand_recent")
            recent = None
        if recent:
            return self._hint("stand_up", t, "completed", **recent)
        if body["seated"]:
            if phase.get("transition_since") is not None:
                self._stats["incomplete_stand_candidates"] += int(phase.get("start") is not None)
                phase.clear()
            phase.setdefault("seated_since", t)
            if t - phase["seated_since"] + 1e-9 >= cfg["baseline_dwell_s"]:
                phase["baseline"] = dict(hip=body["hip"], axis=body["axis"], leg=body["leg"])
            return self._hint("stand_up", t, reason="observing_seated_baseline")
        phase.pop("seated_since", None)
        baseline = phase.get("baseline")
        if baseline is None:
            return self._hint("stand_up", t, reason="no_observed_sitting_baseline")
        phase.setdefault("transition_since", t)
        if t - phase["transition_since"] > cfg["max_sequence_s"]:
            self._stats["incomplete_stand_candidates"] += int(phase.get("start") is not None)
            self._stats["invalidated_stand_baselines"] += 1
            phase.clear()
            return self._hint("stand_up", t, reason="transition_timeout")
        rise = sum((body["hip"][i] - baseline["hip"][i]) * baseline["axis"][i] for i in (0, 1))
        if rise <= cfg["hip_rise_leg_fraction"] * baseline["leg"]:
            phase.pop("standing_since", None)
            if body["standing"]:
                phase.setdefault("unraised_standing_since", t)
                if t - phase["unraised_standing_since"] + 1e-9 >= cfg["standing_dwell_s"]:
                    self._stats["incomplete_stand_candidates"] += int(
                        phase.get("start") is not None
                    )
                    self._stats["invalidated_stand_baselines"] += 1
                    phase.clear()
                    return self._hint(
                        "stand_up", t, reason="already_standing_without_observed_rise"
                    )
            else:
                phase.pop("unraised_standing_since", None)
            return (
                self._hint("stand_up", t, "candidate", start_s=phase["start"])
                if phase.get("start") is not None
                else self._hint("stand_up", t)
            )
        phase.pop("unraised_standing_since", None)
        phase.setdefault("start", t)
        if body["standing"]:
            phase.setdefault("standing_since", t)
            if t - phase["standing_since"] + 1e-9 >= cfg["standing_dwell_s"]:
                events.append(self._event(actor, "stand_up", phase["start"], t))
                self._stats["stand_events"] += 1
                recent = dict(start_s=phase["start"], confirmed_at_s=t)
                state["stand_recent"] = recent
                phase.clear()
                return self._hint("stand_up", t, "completed", **recent)
        else:
            phase.pop("standing_since", None)
        return self._hint("stand_up", t, "candidate", start_s=phase["start"])

    def _inside(self, point, margin):
        x1, y1, x2, y2 = self._roi
        return x1 + margin <= point[0] <= x2 - margin and y1 + margin <= point[1] <= y2 - margin

    def _distance(self, point):
        x1, y1, x2, y2 = self._roi
        return math.hypot(
            max(x1 - point[0], 0, point[0] - x2), max(y1 - point[1], 0, point[1] - y2)
        ) / math.sqrt(2)

    def _board(self, state, point, upright, actor, t, events):
        cfg, phase = self._cfg, state.setdefault("board", {})
        outer, inner = (
            self._inside(point, -cfg["roi_leave_margin"]),
            self._inside(point, cfg["roi_enter_margin"]),
        )
        recent = state.get("board_recent")
        if recent and (
            not outer or t - recent["confirmed_at_s"] >= cfg["recent_completed_s"] - 1e-9
        ):
            state.pop("board_recent")
            recent = None
        if recent:
            return self._hint("move_to_board", t, "completed", **recent)
        if phase.get("anchor") is None:
            if not outer:
                phase.setdefault("outside_since", t)
                if t - phase["outside_since"] + 1e-9 >= cfg["outside_dwell_s"]:
                    phase["anchor"] = point
                    phase["anchor_distance"] = self._distance(point)
            else:
                phase.clear()
            return self._hint("move_to_board", t, reason="observing_outside_baseline")
        progress = phase["anchor_distance"] - self._distance(point)
        net = math.dist(point, phase["anchor"]) / math.sqrt(2)
        if phase.get("start") is None:
            if progress < cfg["progress_diagonal"] or net < cfg["progress_diagonal"]:
                return self._hint("move_to_board", t, reason="insufficient_net_progress")
            phase["start"] = t
        if t - phase["start"] > cfg["max_sequence_s"]:
            self._stats["incomplete_board_candidates"] += 1
            phase.clear()
            return self._hint("move_to_board", t, reason="approach_timeout")
        if not upright:
            phase.pop("arrival_since", None)
        elif inner:
            phase.setdefault("arrival_since", t)
        elif not outer:
            phase.pop("arrival_since", None)
        if (
            phase.get("arrival_since") is not None
            and t - phase["arrival_since"] + 1e-9 >= cfg["arrival_dwell_s"]
            and min(progress, net) >= cfg["progress_diagonal"]
        ):
            events.append(self._event(actor, "move_to_board", phase["start"], t))
            self._stats["board_events"] += 1
            recent = dict(start_s=phase["start"], confirmed_at_s=t)
            state["board_recent"] = recent
            phase.clear()
            return self._hint("move_to_board", t, "completed", **recent)
        return self._hint("move_to_board", t, "candidate", start_s=phase["start"])

    def update(self, frame):
        if self._finished:
            raise ValueError("BasicPoseActionState is finished")
        # The shared hand state validates strict increasing source indices/PTS,
        # actors and duplicate semantics. It never reads future frames.
        events = self._raise.update(frame)
        t, segment = frame["timestamp_seconds"], frame["segment_id"]
        bounds = frame_bounds(frame)
        if self._roi is not None:
            if bounds is None:
                raise ValueError("board_roi requires positive frame width/height")
            width, height = bounds
        if self._t is not None and (
            segment != self._segment or t - self._t > self._cfg["max_gap_s"]
        ):
            for actor in list(self._actors):
                self._drop(actor)
            self._body_quality.clear()
        seen, duplicates = {}, set()
        for track in frame["tracks"]:
            actor = _actor(dict(segment_id=segment, track_id=track["track_id"]))
            if actor in seen:
                duplicates.add(actor)
            seen[actor] = track
        for actor in list(self._actors):
            if actor not in seen or actor in duplicates:
                self._drop(actor)
        self._body_quality.retain(set(seen) - duplicates)
        hands = {_actor(p): p for p in self._raise.people}
        self._people = []
        for actor, track in seen.items():
            quality = (
                dict(
                    status="unsupported",
                    reason="duplicate_track",
                    gate_version="causal_pose_quality_v1",
                    temporal_guard=self._body_quality.config["temporal_guard"],
                )
                if actor in duplicates
                else self._body_quality.check(
                    actor,
                    track,
                    (5, 6, 11, 12, 13, 14, 15, 16),
                    t=t,
                    confidence=self._cfg["confidence"],
                    bounds=bounds,
                )
            )
            body = _body(track, self._cfg, bounds) if quality["status"] == "supported" else None
            if body is None:
                if quality["status"] == "supported":
                    quality.update(status="unsupported", reason="invalid_body_geometry")
                self._drop(actor)
                stand = self._hint("stand_up", t, "unsupported", reason="unknown_pose_or_identity")
                board = self._hint(
                    "move_to_board", t, "unsupported", reason="unknown_pose_or_identity"
                )
            else:
                self._stats["stand_supported_observations"] += 1
                state = self._actors.setdefault(actor, {})
                state.setdefault("context_supported_since", t)
                stand = self._stand(state, body, actor, t, events)
                if self._roi is None:
                    board = None
                else:
                    point = (body["foot"][0] / width, body["foot"][1] / height)
                    if not all(0 <= v <= 1 for v in point):
                        state.pop("board", None)
                        state.pop("board_recent", None)
                        board = self._hint(
                            "move_to_board", t, "unsupported", reason="feet_outside_raster"
                        )
                    else:
                        self._stats["board_supported_observations"] += 1
                        board = self._board(state, point, body["standing"], actor, t, events)
            if self._roi is None:
                board = self._hint("move_to_board", t, "not_configured", reason="missing_floor_roi")
            stand["pose_quality"] = dict(quality)
            board["pose_quality"] = dict(quality)
            hand = dict(hands[actor], label="raise_hand")
            relations = self._context_relations(actor, hand, board, t, body is not None)
            self._people.append(
                dict(
                    segment_id=actor[0],
                    track_id=actor[1],
                    observed_through_s=t,
                    actions=[hand, stand, board],
                    context_relations=relations,
                )
            )
        self._t, self._segment = t, segment
        return events

    def _context_relations(self, actor, hand, board, t, body_supported):
        state = self._actors.get(actor)
        if state is None:
            return []
        if not body_supported or hand["pose_quality"]["status"] != "supported":
            state.pop("raise_context", None)
            state["context_supported_since"] = t
            return []
        prior = state.get("raise_context")
        if prior is not None and t - prior["raise_observed_through_s"] > self._relation_window:
            state.pop("raise_context")
            prior = None
        if hand["status"] == "active":
            if hand["start_s"] < state["context_supported_since"]:
                return []
            if prior is None or prior["raise_start_s"] != hand["start_s"]:
                prior = dict(raise_start_s=hand["start_s"], raise_confirmed_at_s=t)
                state["raise_context"] = prior
            prior["raise_observed_through_s"] = t
        if (
            prior is None
            or board["status"] != "completed"
            or prior["raise_confirmed_at_s"] > board["start_s"]
        ):
            return []
        return [
            dict(
                relation="raise_hand_then_move_to_board",
                status="observed_sequence",
                rule="raise_hand_then_move_to_board_v1",
                origin="rule",
                review="draft",
                calibrated=False,
                **prior,
                board_start_s=board["start_s"],
                board_confirmed_at_s=board["confirmed_at_s"],
                max_separation_s=self._relation_window,
                continuity="same_scene_track_continuous_supported_pose",
            )
        ]

    def finish(self, duration_s=None, *, reason="source_end"):
        events = self._raise.finish(duration_s, reason=reason)
        for actor in list(self._actors):
            self._drop(actor)
        self._people.clear()
        self._body_quality.clear()
        self._finished = True
        return events
