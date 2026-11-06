"""Local, observable interval annotations; no student scores or identity profiles.

JSON contract (all intervals are half-open source seconds, ``[start_s, end_s)``):

* Event: ``segment_id`` and ``track_id`` (nonnegative int or nonempty string),
  ``label``, finite ``start_s < end_s``, ``origin``
  (human|rule|assistant), ``review`` (draft|reviewed), ``uncertain`` (bool), ``evidence``
  (JSON object). These nine keys are required; put supplementary data in evidence.
  Optional ``evidence.boundary_truncated={"start":bool,"end":bool}`` records
  censored temporal endpoints. True means the episode may extend beyond the
  observed interval; it does not make the observed action content uncertain.
* track_segments: list of ``{segment_id, track_id, start_s, end_s}`` extents.
  Each actor is the pair (segment_id, track_id), never a cross-cut identity.
* coverage: list of ``{segment_id, track_id, label, start_s, end_s, review}``.
  This is an explicit assertion of annotation coverage for ONE label. ``reviewed``
  means a human checked both presence and absence of that label in that extent;
  ``draft`` means only an observation/annotation working area. Adjacent intervals
  may form continuous coverage; gaps may not. An empty event list alone NEVER
  asserts absence. Empty coverage means unreviewed, not a negative example.

Posture labels are mutually exclusive per actor, including unknown. Independent
action labels may overlap posture and one another. No posture is inferred here.
All thresholds in the optional pose rule are uncalibrated demonstration settings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

POSTURE_LABELS = ("sitting", "standing", "transition", "unknown")
ACTION_LABELS = ("raise_hand", "stand_up", "move_to_board", "write_board")
EVENT_LABELS = ACTION_LABELS
LABELS = POSTURE_LABELS + ACTION_LABELS
_EVENT_KEYS = {
    "segment_id",
    "track_id",
    "label",
    "start_s",
    "end_s",
    "origin",
    "review",
    "uncertain",
    "evidence",
}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _actor(row):
    segment, track = row.get("segment_id"), row.get("track_id")
    for value, name in ((segment, "segment_id"), (track, "track_id")):
        if isinstance(value, bool) or not (
            isinstance(value, int) and value >= 0 or isinstance(value, str) and value.strip()
        ):
            raise ValueError(f"{name} must be a nonnegative int or nonempty string")
    return segment, track


def _interval(row, duration):
    start = _number(row.get("start_s"), "start_s")
    end = _number(row.get("end_s"), "end_s")
    if not 0 <= start < end <= duration:
        raise ValueError("interval must satisfy 0 <= start_s < end_s <= duration_s")
    return start, end


def _duration(value):
    value = _number(value, "duration_s")
    if value <= 0:
        raise ValueError("duration_s must be positive")
    return value


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("annotation must contain finite JSON values") from exc


def _events(events, duration):
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    result = []
    for row in events:
        if not isinstance(row, dict) or set(row) != _EVENT_KEYS:
            raise ValueError(f"event requires exactly these keys: {sorted(_EVENT_KEYS)}")
        _actor(row)
        _interval(row, duration)
        if row["label"] not in LABELS:
            raise ValueError("unsupported observable label")
        if row["origin"] not in ("human", "rule", "assistant"):
            raise ValueError("origin must be human, rule or assistant")
        if row["review"] not in ("draft", "reviewed"):
            raise ValueError("review must be draft or reviewed")
        if not isinstance(row["uncertain"], bool) or not isinstance(row["evidence"], dict):
            raise ValueError("uncertain must be bool and evidence must be an object")
        if "boundary_truncated" in row["evidence"]:
            boundary = row["evidence"]["boundary_truncated"]
            if (
                not isinstance(boundary, dict)
                or set(boundary) != {"start", "end"}
                or any(not isinstance(value, bool) for value in boundary.values())
            ):
                raise ValueError("evidence.boundary_truncated requires start and end booleans")
        result.append(_json_copy(row))
    for i, row in enumerate(result):
        for other in result[:i]:
            if _actor(row) != _actor(other) or not _overlap(row, other):
                continue
            if row["label"] in POSTURE_LABELS and other["label"] in POSTURE_LABELS:
                raise ValueError("overlapping posture intervals for the same local actor")
            if row["label"] == other["label"]:
                raise ValueError("duplicate/overlapping intervals for the same actor and label")
    return result


def _extents(rows, duration, *, coverage=False):
    if not isinstance(rows, list):
        raise ValueError("coverage/track_segments must be lists")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("coverage/track segment must be an object")
        if coverage and set(row) != {
            "segment_id",
            "track_id",
            "label",
            "start_s",
            "end_s",
            "review",
        }:
            raise ValueError(
                "coverage requires exactly segment_id, track_id, label, start_s, end_s, review"
            )
        _actor(row)
        _interval(row, duration)
        if coverage and (
            row.get("label") not in LABELS or row.get("review") not in ("draft", "reviewed")
        ):
            raise ValueError("coverage needs a supported label and draft/reviewed state")
    return _json_copy(rows)


def _overlap(a, b):
    return max(a["start_s"], b["start_s"]) < min(a["end_s"], b["end_s"])


def _covered(row, extents, *, label=False, reviewed=False):
    intervals = sorted(
        (e["start_s"], e["end_s"])
        for e in extents
        if _actor(e) == _actor(row)
        and (not label or e["label"] == row["label"])
        and (not reviewed or e["review"] == "reviewed")
    )
    cursor = row["start_s"]
    for start, end in intervals:
        if start > cursor:
            break
        cursor = max(cursor, end)
        if cursor >= row["end_s"]:
            return True
    return False


def validate_events(events, *, duration_s, track_segments, coverage) -> list[dict]:
    """Return a detached canonical JSON copy, or raise ValueError.

    Every event must fit actor extents and same-label coverage. Reviewed events
    require reviewed coverage. Coverage itself must fit actor extents. No event
    or actor is stitched across different segment_id values.
    """
    duration = _duration(duration_s)
    segments = _extents(track_segments, duration)
    areas = _extents(coverage, duration, coverage=True)
    for area in areas:
        if not _covered(area, segments):
            raise ValueError("coverage exceeds its local actor segment")
    result = _events(events, duration)
    for row in result:
        if not _covered(row, segments):
            raise ValueError("event exceeds its local actor segment")
        if not _covered(row, areas, label=True, reviewed=row["review"] == "reviewed"):
            raise ValueError("event exceeds its label/review coverage")
    return result


def save_reviewed_events(events, destination) -> Path:
    """Create a new UTF-8 JSON snapshot exclusively; never overwrite any file.

    Caller must first validate against source context with validate_events. This
    helper checks schema/review status but cannot certify source boundaries or
    coverage without that context. Origin stays rule when a human reviews a rule
    proposal. ``uncertain=True`` is allowed and remains excluded from metrics.
    """
    rows = _events(events, float("inf"))
    if any(row["review"] != "reviewed" for row in rows):
        raise ValueError("only reviewed events can be saved in a reviewed snapshot")
    payload = {
        "schema_version": 1,
        "events": rows,
        "coverage_note": "Events alone do not assert reviewed negative coverage.",
    }
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path = Path(destination)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return path


def _pose_flags(track, confidence, enter_margin, exit_margin):
    if (
        track.get("state") != "observed"
        or track.get("confirmed") is not True
        or track.get("identity_uncertain", False)
        or track.get("uncertain", False)
    ):
        return None
    points = track.get("keypoints17")
    if not isinstance(points, list) or len(points) != 17:
        return None
    try:
        visible = {}
        for index in (0, 5, 6, 7, 8, 9, 10):
            point = points[index]
            if len(point) != 3:
                return None
            x, y, score = (_number(v, "keypoint") for v in point)
            if not confidence <= score <= 1:
                return None
            visible[index] = (x, y)
        width = math.dist(visible[5], visible[6])
        if width <= 0:
            return None
        head_y = visible[0][1]
        flags = []
        for margin in (enter_margin, exit_margin):
            flags.append(
                any(
                    visible[wrist][1] < head_y - margin * width
                    and visible[wrist][1] < visible[elbow][1]
                    and visible[elbow][1] < visible[shoulder][1]
                    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10))
                )
            )
        return flags
    except (TypeError, ValueError, IndexError):
        return None


class RaiseHandState:
    """Past/present COCO17 rule with bounded actor state and no event history.

    update returns newly closed rule/draft intervals. finish censors at the last
    actual observation, never extending to duration_s. Native source indices may
    skip; source timestamps and indices must increase. Raw keypoints17 only.
    """

    def __init__(
        self,
        *,
        min_dwell_s=None,
        release_dwell_s=0.3,
        max_gap_s=0.25,
        confidence=0.5,
        enter_margin=None,
        exit_margin=None,
        pose_quality=None,
        hand_cue="nose_both_arms_v1",
    ):
        from classroom_ai.video.pose_quality import PoseQualityGate

        if hand_cue not in ("nose_both_arms_v1", "shoulder_any_arm_v1"):
            raise ValueError("unsupported hand_cue")
        alternate = hand_cue == "shoulder_any_arm_v1"
        min_dwell_s = (0.5 if alternate else 0.4) if min_dwell_s is None else min_dwell_s
        enter_margin = (0.45 if alternate else 0.1) if enter_margin is None else enter_margin
        exit_margin = (0.2 if alternate else 0.05) if exit_margin is None else exit_margin
        for value, name in (
            (min_dwell_s, "min_dwell_s"),
            (release_dwell_s, "release_dwell_s"),
            (max_gap_s, "max_gap_s"),
        ):
            if _number(value, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < _number(confidence, "confidence") <= 1:
            raise ValueError("confidence must be in (0,1]")
        if not 0 <= _number(exit_margin, "exit_margin") < _number(enter_margin, "enter_margin"):
            raise ValueError("require 0 <= exit_margin < enter_margin")
        self._quality = PoseQualityGate(pose_quality)
        self._arm_quality = {side: PoseQualityGate(pose_quality) for side in ("left", "right")}
        self._config = dict(
            rule="coco17_shoulder_any_arm_v1" if alternate else "coco17_wrist_above_nose_v1",
            hand_cue=hand_cue,
            calibrated=False,
            min_dwell_s=min_dwell_s,
            release_dwell_s=release_dwell_s,
            max_gap_s=max_gap_s,
            confidence=confidence,
            enter_margin=enter_margin,
            exit_margin=exit_margin,
            pose_quality=self._quality.config,
            limitations=[
                "Uncalibrated Pose cue; cannot disambiguate head touching, stretching or intent without RGB review."
            ],
        )
        self._states, self._known_absent = {}, set()
        self._previous_t, self._previous_index, self._previous_segment = -1, -1, None
        self._finished = False
        self._people = []
        self._observations = dict(
            frames_seen=0,
            track_rows_seen=0,
            supported_actor_observations=0,
            unsupported_actor_observations=0,
            duplicate_track_rows=0,
            observation_gap_resets=0,
            scene_cut_resets=0,
        )

    @property
    def config(self):
        return _json_copy(self._config)

    @property
    def counts(self):
        active = sum(state["active"] for state in self._states.values())
        return dict(active_actor_count=active, pending_actor_count=len(self._states) - active)

    @property
    def observation_counts(self):
        """Lifetime integer counters of rule input support, never reviewed coverage."""
        return dict(self._observations)

    @property
    def people(self):
        """Detached current-observation rule hints, never closed-event history.

        Candidate/release_pending are provisional. Unsupported/no_suggestion
        never assert reviewed action absence. Only current actors are retained.
        """
        return _json_copy(self._people)

    def _shoulder_flags(self, actor, track, t, confidence, bounds, enter_margin, exit_margin):
        arms = {"left": (5, 7, 9), "right": (6, 8, 10)}
        qualities = {
            side: self._arm_quality[side].check(
                actor, track, (5, 6, elbow, wrist), t=t, confidence=confidence, bounds=bounds
            )
            for side, (_, elbow, wrist) in arms.items()
        }
        valid = [side for side, quality in qualities.items() if quality["status"] == "supported"]
        quality = dict(qualities[valid[0] if valid else "left"], supported_arms=valid)
        if not valid:
            return quality, None, False
        points = track["keypoints17"]
        width = math.dist(points[5][:2], points[6][:2])
        flags = [
            any(
                points[arms[side][2]][1] < points[arms[side][0]][1] - margin * width
                for side in valid
            )
            for margin in (enter_margin, exit_margin)
        ]
        return quality, flags, len(valid) == 2

    def _close(self, actor, *, right_truncated=True, reason="unknown_observation"):
        state = self._states.pop(actor, None)
        if state and state["active"]:
            end = state["release"] if state["release"] is not None else state["last"]
            if end > state["start"]:
                return dict(
                    segment_id=actor[0],
                    track_id=actor[1],
                    label="raise_hand",
                    start_s=state["start"],
                    end_s=end,
                    origin="rule",
                    review="draft",
                    uncertain=False,
                    evidence=dict(
                        rule=self._config["rule"],
                        hand_cue=self._config["hand_cue"],
                        calibrated=False,
                        min_dwell_s=self.config["min_dwell_s"],
                        release_dwell_s=self.config["release_dwell_s"],
                        max_gap_s=self.config["max_gap_s"],
                        confidence=self.config["confidence"],
                        enter_margin=self.config["enter_margin"],
                        exit_margin=self.config["exit_margin"],
                        pose_quality=self.config["pose_quality"],
                        boundary_truncated={
                            "start": state["left_truncated"],
                            "end": right_truncated,
                        },
                        boundary_reason={
                            "start": "first_actor_observation"
                            if state["left_truncated"]
                            else "observed_absence_before_onset",
                            "end": reason,
                        },
                    ),
                )
        return None

    def update(self, frame) -> list[dict]:
        from classroom_ai.video.pose_quality import frame_bounds

        if self._finished:
            raise ValueError("RaiseHandState is finished")
        states, known_absent = self._states, self._known_absent
        previous_t, previous_index, previous_segment = (
            self._previous_t,
            self._previous_index,
            self._previous_segment,
        )
        max_gap_s, min_dwell_s, release_dwell_s = (
            self._config[k] for k in ("max_gap_s", "min_dwell_s", "release_dwell_s")
        )
        confidence, enter_margin, exit_margin = (
            self._config[k] for k in ("confidence", "enter_margin", "exit_margin")
        )
        emitted = []

        def finish(actor, **kwargs):
            event = self._close(actor, **kwargs)
            if event is not None:
                emitted.append(event)

        if not isinstance(frame, dict):
            raise ValueError("pose frame must be an object")
        t = _number(frame.get("timestamp_seconds"), "timestamp_seconds")
        index, segment = frame.get("frame_index"), frame.get("segment_id")
        if (
            t < 0
            or t <= previous_t
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index <= previous_index
        ):
            raise ValueError(
                "pose frames require increasing source timestamps/indices and segment_id"
            )
        _actor(dict(segment_id=segment, track_id=0))
        tracks = frame.get("tracks")
        if not isinstance(tracks, list):
            raise ValueError("frame tracks must be a list")
        bounds = frame_bounds(frame)
        if segment != previous_segment or t - previous_t > max_gap_s:
            if previous_t >= 0:
                self._observations[
                    "scene_cut_resets" if segment != previous_segment else "observation_gap_resets"
                ] += 1
            for actor in list(states):
                finish(
                    actor, reason="scene_cut" if segment != previous_segment else "observation_gap"
                )
            known_absent.clear()
            self._quality.clear()
            for gate in self._arm_quality.values():
                gate.clear()
        seen, duplicates = {}, set()
        for track in tracks:
            if not isinstance(track, dict):
                raise ValueError("track must be an object")
            actor = _actor(dict(segment_id=segment, track_id=track.get("track_id")))
            if actor in seen:
                duplicates.add(actor)
            seen[actor] = track
        self._observations["frames_seen"] += 1
        self._observations["track_rows_seen"] += len(tracks)
        self._observations["duplicate_track_rows"] += len(tracks) - len(seen)
        for actor in list(states):
            if actor not in seen or actor in duplicates:
                finish(actor, reason="missing_or_duplicate_track")
        known_absent.intersection_update(set(seen) - duplicates)
        self._quality.retain(set(seen) - duplicates)
        for gate in self._arm_quality.values():
            gate.retain(set(seen) - duplicates)
        supported = set()
        qualities, observations = {}, {}
        for actor, track in seen.items():
            quality = (
                dict(
                    status="unsupported",
                    reason="duplicate_track",
                    gate_version="causal_pose_quality_v1",
                    temporal_guard=self._quality.config["temporal_guard"],
                )
                if actor in duplicates
                else self._quality.check(
                    actor,
                    track,
                    (0, 5, 6, 7, 8, 9, 10),
                    t=t,
                    confidence=confidence,
                    bounds=bounds,
                )
            )
            qualities[actor] = quality
            flags = (
                None
                if quality["status"] != "supported"
                else _pose_flags(
                    track,
                    confidence,
                    enter_margin,
                    exit_margin,
                )
            )
            negative_supported = True
            if self._config["hand_cue"] == "shoulder_any_arm_v1" and actor not in duplicates:
                quality, flags, negative_supported = self._shoulder_flags(
                    actor, track, t, confidence, bounds, enter_margin, exit_margin
                )
                qualities[actor] = quality
            observations[actor] = ("unknown", quality["reason"])
            if flags is None:
                self._observations["unsupported_actor_observations"] += 1
                finish(actor, reason="unknown_pose_or_identity")
                known_absent.discard(actor)
                continue
            self._observations["supported_actor_observations"] += 1
            supported.add(actor)
            enter, keep = flags
            if enter:
                observations[actor] = ("raised", "raised_pose")
            else:
                points = track["keypoints17"]
                shoulder_width = math.dist(points[5][:2], points[6][:2])
                lowered = negative_supported and all(
                    points[wrist][1] > points[shoulder][1] + exit_margin * shoulder_width
                    for shoulder, wrist in ((5, 9), (6, 10))
                )
                observations[actor] = (
                    ("not_raised", "both_wrists_lowered")
                    if lowered
                    else ("unknown", "ambiguous_geometry")
                )
            if observations[actor][0] == "unknown":
                known_absent.discard(actor)
                if not keep:
                    # A failed positive cue is not an observed negative or a
                    # completed release. Do not extend to this unknown endpoint.
                    finish(actor, reason="ambiguous_pose_geometry")
                    continue
            state = states.get(actor)
            if state is None:
                if enter:
                    states[actor] = dict(
                        start=t,
                        last=t,
                        active=False,
                        release=None,
                        left_truncated=actor not in known_absent,
                    )
                    known_absent.discard(actor)
                elif observations[actor][0] == "not_raised":
                    known_absent.add(actor)
                continue
            state["last"] = t
            if not state["active"]:
                if not enter:
                    finish(actor)
                    if observations[actor][0] == "not_raised":
                        known_absent.add(actor)
                elif t - state["start"] + 1e-9 >= min_dwell_s:
                    state["active"] = True
            elif keep:
                state["release"] = None
            else:
                if state["release"] is None:
                    state["release"] = t
                if t - state["release"] + 1e-9 >= release_dwell_s:
                    finish(actor, right_truncated=False, reason="observed_release")
                    known_absent.add(actor)
        self._people = []
        for actor in seen:
            state = states.get(actor)
            if actor not in supported:
                status = "unsupported"
            elif state is None:
                status = "no_suggestion"
            elif not state["active"]:
                status = "candidate"
            else:
                status = "release_pending" if state["release"] is not None else "active"
            person = dict(
                segment_id=actor[0],
                track_id=actor[1],
                status=status,
                label="raise_hand" if state is not None else None,
                origin="rule",
                review="draft",
                calibrated=False,
                rule=self._config["rule"],
                observed_through_s=t,
                pose_observation=observations[actor][0],
                pose_observation_reason=observations[actor][1],
                pose_observation_scope="uncalibrated_raw_pose_geometry",
                pose_quality=qualities[actor],
            )
            if state is not None:
                person["start_s"] = state["start"]
            if status == "unsupported":
                person["reason"] = (
                    "duplicate_track" if actor in duplicates else "unknown_pose_or_identity"
                )
            self._people.append(person)
        self._previous_t, self._previous_index, self._previous_segment = t, index, segment
        return emitted

    def finish(self, duration_s=None, *, reason="source_end") -> list[dict]:
        if reason not in ("source_end", "cancelled", "failed"):
            raise ValueError("invalid finish reason")
        if duration_s is not None:
            duration = _number(duration_s, "duration_s")
            if duration < max(0, self._previous_t):
                raise ValueError("duration_s precedes last observation")
        if self._finished:
            return []
        emitted = []
        for actor in list(self._states):
            event = self._close(actor, reason=reason)
            if event is not None:
                emitted.append(event)
        self._known_absent.clear()
        self._quality.clear()
        for gate in self._arm_quality.values():
            gate.clear()
        self._people.clear()
        self._finished = True
        return emitted


def propose_raise_hand_events(
    pose_frames,
    *,
    min_dwell_s=None,
    release_dwell_s=0.3,
    max_gap_s=0.25,
    confidence=0.5,
    enter_margin=None,
    exit_margin=None,
    pose_quality=None,
    hand_cue="nose_both_arms_v1",
) -> list[dict]:
    """Opt-in, uncalibrated COCO17 rule; returns rule/draft raise_hand only.

    Input is a list of frames with frame_index, timestamp_seconds, segment_id,
    tracks=[{track_id,keypoints17:[[x,y,confidence],...],state:'observed',
    confirmed:true}]. Coordinates must share the same image coordinate system
    with y increasing downward. Both arms, shoulders and nose must be confident;
    a hidden arm, missing frame/track, unconfirmed/uncertain ID, cut, or large
    timestamp gap resets state. A wrist above the nose and elbow above shoulder
    is only a pose proxy for observable hand raising, not intent/participation.

    Enter/exit margins use shoulder distance. Release dwell bridges only short
    *observed* lowering, never unknown gaps. Ends are last observed timestamps;
    no duration is extrapolated beyond the final sample. Boundary evidence marks
    a first already-raised actor observation as left-truncated, and end/cut/gap/
    unknown termination as right-truncated. Only observed prior absence and
    confirmed release establish complete boundaries. Endpoint uncertainty stays
    in evidence, separately from certainty about the observed action content.
    No write_board,
    move_to_board, stand_up or posture claim is generated from this pose rule.
    """
    state = RaiseHandState(
        min_dwell_s=min_dwell_s,
        release_dwell_s=release_dwell_s,
        max_gap_s=max_gap_s,
        confidence=confidence,
        enter_margin=enter_margin,
        exit_margin=exit_margin,
        pose_quality=pose_quality,
        hand_cue=hand_cue,
    )
    result = []
    for frame in pose_frames:
        result.extend(state.update(frame))
    result.extend(state.finish())
    return sorted(result, key=lambda row: row["start_s"])


def _union_duration(intervals, start, end):
    total, cursor = 0.0, start
    for left, right in sorted(intervals):
        left, right = max(left, start, cursor), min(right, end)
        if right > left:
            total += right - left
            cursor = right
    return total


def _masks(row, mask):
    if _actor(mask) != _actor(row) or not _overlap(mask, row):
        return False
    if mask["label"] in POSTURE_LABELS:
        if mask["label"] == "unknown" and mask["review"] == "reviewed" and not mask["uncertain"]:
            return row["label"] in POSTURE_LABELS and row["label"] != "unknown"
        return row["label"] in POSTURE_LABELS
    return mask["label"] == row["label"]


def _reviewed_areas(areas, masks):
    """Subtract uncertain reference spans, preserving independent action labels."""
    result = []
    for area in areas:
        if area["review"] != "reviewed":
            continue
        spans = [(area["start_s"], area["end_s"])]
        for mask in masks:
            if not _masks(area, mask):
                continue
            remaining = []
            for start, end in spans:
                if mask["end_s"] <= start or mask["start_s"] >= end:
                    remaining.append((start, end))
                else:
                    if mask["start_s"] > start:
                        remaining.append((start, mask["start_s"]))
                    if mask["end_s"] < end:
                        remaining.append((mask["end_s"], end))
            spans = remaining
        result.extend(dict(area, start_s=start, end_s=end) for start, end in spans)
    return result


def _coverage_seconds(areas, label, start, end):
    by_actor = {}
    for area in areas:
        if area["label"] == label and area["review"] == "reviewed":
            by_actor.setdefault(_actor(area), []).append((area["start_s"], area["end_s"]))
    return sum(_union_duration(spans, start, end) for spans in by_actor.values())


def _boundary_flags(row, areas):
    explicit = row["evidence"].get("boundary_truncated")
    if explicit is not None:
        return explicit["start"], explicit["end"]
    # Without endpoint evidence, reviewed presence/absence on both sides is
    # needed to call the whole episode observed. Source and coverage edges have
    # no such evidence. Explicit human metadata can attest a boundary onset.
    same = [
        area
        for area in areas
        if _actor(area) == _actor(row)
        and area["label"] == row["label"]
        and area["review"] == "reviewed"
    ]
    return (
        not any(area["start_s"] < row["start_s"] <= area["end_s"] for area in same),
        not any(area["start_s"] <= row["end_s"] < area["end_s"] for area in same),
    )


def aggregate_events(events, *, duration_s, coverage, window_s=60) -> dict:
    """Class totals, actor-seconds and onset counts; no individual output or scores.

    Only certain reviewed events wholly inside reviewed label coverage contribute.
    A window with no reviewed label coverage has count/duration None, not zero.
    Partial coverage is reported in actor-seconds; zero within that coverage is
    not a claim of no activity throughout the class/session. Duration sums across
    actors, so it can exceed wall-clock duration. Windows count event ONSETS;
    duration is clipped to each window (ongoing events do not add new onsets).
    Left-truncated episodes never count as observed onsets. Without explicit
    boundary evidence, source/reviewed-coverage edges are conservatively censored.
    Explicit start:false asserts a reviewed onset even at a boundary.
    truncated_episode_count counts censored episodes intersecting each window
    (it can repeat across windows); observed_duration_s/duration_s report only
    the observed clipped portion, never an extrapolated complete-episode length.
    """
    duration = _duration(duration_s)
    window = _number(window_s, "window_s")
    if window <= 0:
        raise ValueError("window_s must be positive")
    rows = _events(events, duration)
    masks = [
        row
        for row in rows
        if row["review"] != "reviewed" or row["uncertain"] or row["label"] == "unknown"
    ]
    # Unknown is a posture label; it must not erase a separately observed hand
    # action. Unknown itself remains visible as an uncertainty duration below.
    areas = _reviewed_areas(_extents(coverage, duration, coverage=True), masks)
    usable = [
        row
        for row in rows
        if row["review"] == "reviewed"
        and not row["uncertain"]
        and _covered(row, areas, label=True, reviewed=True)
    ]

    def labels(start, end):
        totals = {}
        for label in LABELS:
            covered = _coverage_seconds(areas, label, start, end)
            selected = [row for row in usable if row["label"] == label]
            active = [row for row in selected if row["start_s"] < end and row["end_s"] > start]
            observed_duration = (
                sum(max(0, min(end, row["end_s"]) - max(start, row["start_s"])) for row in selected)
                if covered
                else None
            )
            totals[label] = dict(
                count=sum(
                    start <= row["start_s"] < end and not _boundary_flags(row, areas)[0]
                    for row in selected
                )
                if covered
                else None,
                truncated_episode_count=sum(any(_boundary_flags(row, areas)) for row in active)
                if covered
                else None,
                duration_s=observed_duration,
                observed_duration_s=observed_duration,
                reviewed_coverage_actor_s=covered,
            )
        return totals

    windows = [
        dict(
            start_s=start,
            end_s=min(start + window, duration),
            labels=labels(start, min(start + window, duration)),
        )
        for start in (i * window for i in range(math.ceil(duration / window)))
    ]
    return dict(
        session=labels(0, duration),
        windows=windows,
        excluded_events=len(rows) - len(usable),
        caveat="Reviewed observable events only; partial coverage is not whole-class absence. "
        "Unknown posture is uncertainty, not an action or a negative example. "
        "Counts are observed onsets only; censored episodes are separate. "
        "Durations cover observed intervals, not complete truncated episodes. "
        "Durations are actor-seconds; no interaction scores or per-student profiles.",
    )


def evaluate_event_f1(
    predicted,
    reference,
    *,
    duration_s,
    track_segments,
    coverage,
    iou_threshold=0.5,
    boundary_tolerance_s=None,
    include_matches=False,
) -> dict:
    """Maximum one-to-one event matching by local actor+label and temporal IoU.

    Coverage MUST explicitly mark reviewed positive AND negative regions per
    actor/label. Predictions outside it are ignored, not false positives. Draft,
    uncertain and unknown reference intervals mask overlapping predictions for
    that actor/label (unknown/uncertain posture masks posture only). Draft rule predictions may be tested;
    only certain reviewed reference events are GT. No reference supplied by this
    library is empirical GT. F1 is None when no eligible event exists, including
    all-negative evaluated regions; TP/FP/FN and coverage remain visible.
    IoU compares observed source intervals, including censored ones; it does not
    evaluate unobserved complete-episode boundaries or extrapolate their lengths.
    """
    duration = _duration(duration_s)
    threshold = _number(iou_threshold, "iou_threshold")
    if not 0 < threshold <= 1:
        raise ValueError("iou_threshold must be in (0,1]")
    tolerance = None
    if boundary_tolerance_s is not None:
        tolerance = _number(boundary_tolerance_s, "boundary_tolerance_s")
        if tolerance < 0:
            raise ValueError("boundary_tolerance_s must be nonnegative")
    segments = _extents(track_segments, duration)
    areas = _extents(coverage, duration, coverage=True)
    for area in areas:
        if not _covered(area, segments):
            raise ValueError("coverage exceeds its local actor segment")
    # Predictions may duplicate one another: duplicates must become FP rather
    # than be rejected as invalid GT annotation intervals.
    preds = [_events([row], duration)[0] for row in predicted]
    refs = _events(reference, duration)
    for row in preds + refs:
        if not _covered(row, segments):
            raise ValueError("event exceeds its local actor segment")
    masks = [
        row
        for row in refs
        if row["review"] != "reviewed" or row["uncertain"] or row["label"] == "unknown"
    ]
    reviewed_areas = _reviewed_areas(areas, masks)

    def eligible(row):
        return (
            not row["uncertain"]
            and row["label"] != "unknown"
            and _covered(row, reviewed_areas, label=True, reviewed=True)
            and not any(_masks(row, mask) for mask in masks)
        )

    pred_indices = [i for i, row in enumerate(preds) if eligible(row)]
    ref_indices = [i for i, row in enumerate(refs) if row["review"] == "reviewed" and eligible(row)]
    good_preds = [preds[i] for i in pred_indices]
    good_refs = [refs[i] for i in ref_indices]
    edges = []
    for pred in good_preds:
        matches = []
        for i, ref in enumerate(good_refs):
            if _actor(pred) != _actor(ref) or pred["label"] != ref["label"]:
                continue
            if tolerance is not None:
                censored = _boundary_flags(ref, reviewed_areas)
                if any(
                    not truncated and abs(pred[side] - ref[side]) > tolerance + 1e-9
                    for side, truncated in zip(("start_s", "end_s"), censored)
                ):
                    continue
            overlap = max(
                0, min(pred["end_s"], ref["end_s"]) - max(pred["start_s"], ref["start_s"])
            )
            union = max(pred["end_s"], ref["end_s"]) - min(pred["start_s"], ref["start_s"])
            if overlap / union >= threshold:
                matches.append(i)
        edges.append(matches)
    assigned = {}

    def assign(index):
        # Iterative alternating paths preserve maximum-cardinality matching
        # without depending on Python's recursion limit on long sessions.
        queue, parents, visited = [index], {index: None}, set()
        for prediction_index in queue:
            for ref_index in edges[prediction_index]:
                if ref_index in visited:
                    continue
                visited.add(ref_index)
                if ref_index not in assigned:
                    current_pred, current_ref = prediction_index, ref_index
                    while True:
                        assigned[current_ref] = current_pred
                        previous = parents[current_pred]
                        if previous is None:
                            return True
                        current_pred, current_ref = previous
                child = assigned[ref_index]
                if child not in parents:
                    parents[child] = prediction_index, ref_index
                    queue.append(child)
        return False

    tp = sum(assign(i) for i in range(len(good_preds)))
    fp, fn = len(good_preds) - tp, len(good_refs) - tp
    result = dict(
        tp=tp,
        fp=fp,
        fn=fn,
        f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        ignored_predictions=len(preds) - len(good_preds),
        ignored_references=len(refs) - len(good_refs),
        reviewed_coverage_actor_s={
            label: _coverage_seconds(reviewed_areas, label, 0, duration) for label in LABELS
        },
        iou_threshold=threshold,
        caveat="Requires explicit reviewed label coverage and independent human GT; "
        "unreviewed/uncertain intervals are excluded, not negative examples. "
        "Temporal IoU uses observed intervals, not unobserved complete-episode boundaries.",
    )
    if boundary_tolerance_s is not None:
        result["boundary_tolerance_s"] = tolerance
    if include_matches:
        matches = []
        for ref_index, pred_index in sorted(assigned.items()):
            ref, pred = good_refs[ref_index], good_preds[pred_index]
            censored = _boundary_flags(ref, reviewed_areas)
            intersection = max(
                0, min(ref["end_s"], pred["end_s"]) - max(ref["start_s"], pred["start_s"])
            )
            union = max(ref["end_s"], pred["end_s"]) - min(ref["start_s"], pred["start_s"])
            matches.append(
                dict(
                    prediction_index=pred_indices[pred_index],
                    reference_index=ref_indices[ref_index],
                    temporal_iou=intersection / union,
                    start_error_s=None if censored[0] else pred["start_s"] - ref["start_s"],
                    end_error_s=None if censored[1] else pred["end_s"] - ref["end_s"],
                )
            )
        result["matches"] = matches
        result["eligible_prediction_indices"] = pred_indices
        result["eligible_reference_indices"] = ref_indices
        result["matching_policy"] = "maximum_cardinality_deterministic_input_order"
    return result
