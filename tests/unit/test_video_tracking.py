"""Real ByteTrack association and bounded offline rendering contracts."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from classroom_ai.video.metrics import Metrics
from classroom_ai.video.rendering import render_tracks, track_color
from classroom_ai.video.tracking import BoxSmoother, PersonTracker, SceneCuts


def person(x=10, score=0.05):
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": score,
        "bbox_xyxy": [x, 10, x + 40, 90],
    }


def anchor(i, x=10, segment=0, track=1, confirmed=True):
    return {
        "frame_index": i,
        "timestamp_seconds": i / 30,
        "segment_id": segment,
        "tracks": [{"track_id": track, "confirmed": confirmed, "detection_index": 0, **person(x)}],
    }


class TrackingTests(unittest.TestCase):
    def test_sampled_observations_confirm_without_renaming_source_indices(self):
        tracker = PersonTracker(5, 0.05, 0.1, Metrics(), max_observation_gap_s=0.3)
        a = tracker.update(0, 0, [person()], observation_index=0)
        b = tracker.update(6, 0.2, [person()], observation_index=1)
        self.assertEqual(b["frame_index"], 6)
        self.assertEqual(b["observation_index"], 1)
        self.assertEqual(a["tracks"][0]["track_id"], b["tracks"][0]["track_id"])
        self.assertTrue(b["tracks"][0]["confirmed"])
        c = tracker.update(30, 1, [person()], observation_index=2)
        self.assertNotEqual(c["tracks"][0]["track_id"], b["tracks"][0]["track_id"])

    def tracker(self, fps=30, lost=3):
        return PersonTracker(fps, 0.05, lost, Metrics())

    def test_two_consecutive_hits_including_first_frame_and_person_only(self):
        tracker = self.tracker()
        obj = {**person(), "class_id": 1, "class_name": "table"}
        a = tracker.update(0, 0, [obj, person()])
        self.assertEqual(len(a["tracks"]), 1)
        self.assertFalse(a["tracks"][0]["confirmed"])
        b = tracker.update(1, 1 / 30, [obj, person(11)])
        self.assertTrue(b["tracks"][0]["confirmed"])
        self.assertEqual(a["tracks"][0]["track_id"], b["tracks"][0]["track_id"])
        self.assertEqual(b["tracks"][0]["detection_index"], 1)

    def test_missing_detection_hidden_recovery_keeps_id_and_expiry_creates_new(self):
        tracker = self.tracker(lost=0.1)
        tracker.update(0, 0, [person()])
        first = tracker.update(1, 1 / 30, [person()])["tracks"][0]["track_id"]
        self.assertEqual(tracker.update(2, 2 / 30, [])["tracks"], [])
        found = tracker.update(3, 3 / 30, [person()])["tracks"][0]
        self.assertEqual(found["track_id"], first)
        self.assertTrue(found["confirmed"])
        for i in range(4, 10):
            tracker.update(i, i / 30, [])
        after = tracker.update(10, 10 / 30, [person()])["tracks"][0]
        self.assertNotEqual(first, after["track_id"])
        self.assertFalse(after["confirmed"])

    def test_zero_lost_budget_keeps_continuous_track_but_not_a_gap(self):
        tracker = self.tracker(lost=0)
        first = tracker.update(0, 0, [person()])["tracks"][0]["track_id"]
        second = tracker.update(1, 1 / 30, [person()])["tracks"][0]
        self.assertEqual(second["track_id"], first)
        self.assertTrue(second["confirmed"])
        tracker.update(2, 2 / 30, [])
        self.assertNotEqual(tracker.update(3, 3 / 30, [person()])["tracks"][0]["track_id"], first)

    def test_reset_and_two_independent_sessions_do_not_reuse_or_interfere_ids(self):
        a, b = self.tracker(), self.tracker()
        first = a.update(0, 0, [person()])["tracks"][0]["track_id"]
        b.update(0, 0, [person()])
        self.assertEqual(a.update(1, 1 / 30, [person()])["tracks"][0]["track_id"], first)
        a.reset()
        after = a.update(2, 2 / 30, [person()])
        self.assertNotEqual(after["tracks"][0]["track_id"], first)
        self.assertEqual(after["segment_id"], 1)

    def test_smoother_reduces_stationary_jitter_and_resets_after_gap(self):
        smoother = BoxSmoother(0.65, 200, 100)
        rng = np.random.default_rng(123)
        raw, smooth = [], []
        for i, noise in enumerate(rng.normal(0, 2, 300)):
            box = person(50 + noise)["bbox_xyxy"]
            raw.append(box[0])
            smooth.append(smoother.update(1, i, box)[0])
        self.assertLess(np.std(smooth[20:]), 0.8 * np.std(raw[20:]))
        self.assertEqual(smoother.update(1, 302, person(80)["bbox_xyxy"]), person(80)["bbox_xyxy"])
        self.assertEqual(
            smoother.update(1, 303, person(140)["bbox_xyxy"]), person(140)["bbox_xyxy"]
        )

    def test_scene_cut_requires_both_signals(self):
        cuts = SceneCuts()
        black = np.zeros((90, 160, 3), dtype=np.uint8)
        red = np.full_like(black, (0, 0, 255))
        self.assertFalse(cuts.update(black))
        self.assertFalse(cuts.update(black))
        self.assertTrue(cuts.update(red))


class TrackingRecoveryTests(unittest.TestCase):
    def test_resolve_settings_is_pure_and_validates_without_native_imports(self):
        from unittest.mock import patch

        from classroom_ai.video.tracking import resolve_tracking_settings

        with patch.dict("sys.modules", {"lap": None, "ultralytics": None}):
            settings = resolve_tracking_settings(0.05, new_track_thresh=0.1)
            self.assertEqual(
                settings,
                dict(
                    detector_confidence=0.05,
                    track_high_thresh=0.05,
                    track_low_thresh=0.8 * 0.05,
                    new_track_thresh=0.1,
                ),
            )
            with self.assertRaises(ValueError):
                resolve_tracking_settings(0.05, recovery_confirm_observations=False)

    def test_optional_thresholds_keep_legacy_and_publish_effective_settings(self):
        metrics = Metrics()
        tracker = PersonTracker(30, 0.05, 3, metrics)
        self.assertEqual(tracker.settings["track_high_thresh"], 0.05)
        self.assertAlmostEqual(tracker.settings["track_low_thresh"], 0.04)
        self.assertEqual(tracker.settings["new_track_thresh"], 0.05)
        self.assertFalse(tracker.settings["low_stage_reachable"])
        self.assertEqual(tracker.settings["recovery_confirm_observations"], 2)
        self.assertEqual(metrics.values["tracking_settings"], tracker.settings)

    def test_low_score_can_maintain_observed_track_but_cannot_seed_one(self):
        tracker = PersonTracker(
            30,
            0.05,
            3,
            Metrics(),
            track_high_thresh=0.1,
            track_low_thresh=0.05,
            new_track_thresh=0.2,
        )
        self.assertTrue(tracker.settings["low_stage_reachable"])
        first = tracker.update(0, 0, [person(score=0.9)])["tracks"][0]
        tracker.update(1, 1 / 30, [person(score=0.9)])
        weak = {**person(11, score=0.07), "keypoints17": [[17, 21, 0.7]] * 17}
        found = tracker.update(2, 2 / 30, [weak, person(200, score=0.07)])["tracks"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["track_id"], first["track_id"])
        self.assertEqual(found[0]["detection_index"], 0)
        self.assertEqual(found[0]["keypoints17"], weak["keypoints17"])
        self.assertFalse(found[0]["identity_uncertain"])

    def test_bad_thresholds_fail_before_native_tracker_import(self):
        from unittest.mock import patch

        invalid = [
            {"track_high_thresh": True},
            {"track_low_thresh": float("nan")},
            {"new_track_thresh": float("inf")},
            {"track_low_thresh": -0.01},
            {"track_high_thresh": 0.04},
            {"track_high_thresh": 0.2},
            {"track_low_thresh": 0.06},
            {"new_track_thresh": 1.01},
            {"recovery_confirm_observations": True},
            {"recovery_confirm_observations": 0},
            {"recovery_confirm_observations": 1.5},
        ]
        for options in invalid:
            with self.subTest(options=options), patch.dict("sys.modules", {"lap": None}):
                with self.assertRaises(ValueError):
                    PersonTracker(30, 0.05, 3, Metrics(), **options)

    def test_gap_recovery_preserves_id_but_gates_behavior_until_reconfirmed(self):
        tracker = PersonTracker(30, 0.05, 3, Metrics(), recovery_confirm_observations=3)
        first = tracker.update(0, 0, [person()])["tracks"][0]
        self.assertFalse(first["confirmed"])
        self.assertFalse(first["identity_uncertain"])
        self.assertFalse(first["reacquired_after_gap"])
        self.assertEqual(first["gap_seconds"], 0)
        second = tracker.update(1, 1 / 30, [person()])["tracks"][0]
        self.assertTrue(second["confirmed"])
        tracker.update(2, 2 / 30, [])
        recovered = tracker.update(3, 3 / 30, [person()])["tracks"][0]
        self.assertEqual(recovered["track_id"], first["track_id"])
        self.assertTrue(recovered["confirmed"])
        self.assertTrue(recovered["reacquired_after_gap"])
        self.assertAlmostEqual(recovered["gap_seconds"], 2 / 30)
        self.assertTrue(recovered["identity_uncertain"])
        next_row = tracker.update(4, 4 / 30, [person()])["tracks"][0]
        self.assertFalse(next_row["reacquired_after_gap"])
        self.assertEqual(next_row["gap_seconds"], 0)
        self.assertTrue(next_row["identity_uncertain"])
        stable = tracker.update(5, 5 / 30, [person()])["tracks"][0]
        self.assertFalse(stable["identity_uncertain"])
        tracker.update(6, 6 / 30, [])
        self.assertTrue(tracker.update(7, 7 / 30, [person()])["tracks"][0]["identity_uncertain"])

    def test_vfr_interval_and_regular_sampled_source_skips_are_not_missing_rows(self):
        tracker = PersonTracker(5, 0.05, 3, Metrics())
        tracker.update(0, 0, [person()], observation_index=0)
        row = tracker.update(6, 0.4, [person()], observation_index=1)["tracks"][0]
        self.assertTrue(row["confirmed"])
        self.assertFalse(row["reacquired_after_gap"])
        self.assertFalse(row["identity_uncertain"])
        self.assertEqual(row["gap_seconds"], 0)

    def test_elapsed_cadence_gap_is_not_a_missing_tracker_observation(self):
        tracker = PersonTracker(5, 0.05, 3, Metrics(), max_observation_gap_s=0.3)
        tracker.update(0, 0, [person()], observation_index=0)
        tracker.update(6, 0.2, [person()], observation_index=1)
        row = tracker.update(7, 0.8, [person()], observation_index=2)["tracks"][0]
        self.assertTrue(row["confirmed"])
        self.assertFalse(row["reacquired_after_gap"])
        self.assertFalse(row["identity_uncertain"])
        self.assertEqual(row["gap_seconds"], 0)

    def test_expiry_and_scene_reset_cannot_recover_or_reuse_previous_id(self):
        tracker = PersonTracker(30, 0.05, 0.1, Metrics())
        first = tracker.update(0, 0, [person()])["tracks"][0]
        tracker.update(1, 1 / 30, [person()])
        tracker.update(2, 2 / 30, [])
        expired = tracker.update(9, 0.3, [person()])["tracks"][0]
        self.assertNotEqual(expired["track_id"], first["track_id"])
        self.assertFalse(expired["reacquired_after_gap"])
        self.assertFalse(expired["identity_uncertain"])
        tracker.reset()
        reset = tracker.update(10, 1 / 3, [person()])["tracks"][0]
        self.assertNotEqual(reset["track_id"], expired["track_id"])
        self.assertFalse(reset["reacquired_after_gap"])

    def test_reacquired_uncertainty_reaches_existing_behavior_guard(self):
        from classroom_ai.video.actions import RaiseHandState

        tracker = PersonTracker(20, 0.05, 3, Metrics())
        state = RaiseHandState(min_dwell_s=0.1)
        points = [[20, 20, 0.9] for _ in range(17)]
        for i, point in {
            5: [10, 40, 0.9],
            6: [30, 40, 0.9],
            7: [10, 25, 0.9],
            8: [30, 25, 0.9],
            9: [10, 10, 0.9],
            10: [30, 10, 0.9],
        }.items():
            points[i] = point
        detection = {**person(score=0.9), "keypoints17": points}
        for i in range(6):
            row = tracker.update(i, i / 20, [] if i == 4 else [detection])
            for track in row["tracks"]:
                track["state"] = "observed"
            state.update(row)
            if i == 3:
                self.assertEqual(state.people[0]["status"], "active")
            if i == 5:
                self.assertTrue(row["tracks"][0]["identity_uncertain"])
                self.assertEqual(state.people[0]["status"], "unsupported")

    def test_overlapping_observed_people_are_not_blanket_identity_uncertain(self):
        tracker = PersonTracker(30, 0.05, 3, Metrics())
        detections = [person(10, score=0.9), person(20, score=0.8)]
        tracker.update(0, 0, detections)
        rows = tracker.update(1, 1 / 30, detections)["tracks"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(t["confirmed"] and not t["identity_uncertain"] for t in rows))

    def test_full_tracking_forwards_optional_config_and_records_effective_settings(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from classroom_ai.video.tracked_processing import process_tracked

        config = SimpleNamespace(
            confidence=0.05,
            track_max_lost_seconds=3,
            track_gap_max_ms=300,
            track_smooth_alpha=0.65,
            track_high_thresh=0.1,
            track_low_thresh=0.05,
            new_track_thresh=0.15,
            recovery_confirm_observations=3,
        )
        metrics = Metrics()
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(
                config=config,
                metrics=metrics,
                directory=Path(directory),
                progress=SimpleNamespace(set=lambda text: None),
                stop=SimpleNamespace(is_set=lambda: True),
            )
            with (
                patch("classroom_ai.video.tracked_processing.frames", return_value=iter(())),
                patch(
                    "classroom_ai.video.tracked_processing.PersonTracker", wraps=PersonTracker
                ) as factory,
            ):
                process_tracked(session, SimpleNamespace(fps=30))
        self.assertEqual(
            factory.call_args.kwargs,
            dict(
                track_high_thresh=0.1,
                track_low_thresh=0.05,
                new_track_thresh=0.15,
                recovery_confirm_observations=3,
            ),
        )
        self.assertTrue(metrics.values["tracking_settings"]["low_stage_reachable"])
        self.assertEqual(metrics.values["tracking_settings"]["new_track_thresh"], 0.15)


class PoseAssociationContractTests(unittest.TestCase):
    """Scripted ByteTrack output tests adapter wiring, not identity accuracy."""

    def fixture(self):
        def pose(actor, x, frame):
            score = 0.71 if actor == "a" else 0.89
            points = [[x + j / 100, frame + j / 100, score] for j in range(17)]
            reduced = copy.deepcopy(points[:13])
            reduced[3 if actor == "a" else 8] = None
            return {
                **person(x),
                "keypoints17": points,
                "keypoints13": reduced,
                "keypoints13_missing": [p is None for p in reduced],
            }

        table = {**person(45), "class_id": 1, "class_name": "table"}
        # A moves left->right; B moves right->left. Detector order changes.
        detections = [
            [table, pose("a", 5, 0), pose("b", 85, 0)],
            [pose("b", 55, 1), table, pose("a", 35, 1)],
            [table, pose("b", 25, 2)],  # A is not observed during the crossing.
            [pose("b", 5, 3), table, pose("a", 85, 3)],
            [],
            [pose("a", 95, 5), pose("b", 0, 5), table],
        ]
        # Explicit oracle is original detection index, not output-list position.
        expected = [{22: 2, 11: 1}, {11: 2, 22: 0}, {22: 1}, {11: 2, 22: 0}, {}, {22: 1, 11: 0}]
        scripts = []
        for frame, (rows, mapping) in enumerate(zip(detections, expected, strict=True)):
            selected = [i for i, row in enumerate(rows) if row["class_id"] == 0]
            output = []
            for track_id, detection_index in mapping.items():
                # Predicted geometry deliberately points at the OTHER actor. A
                # nearest-box rematch must fail even though returned idx is valid.
                other = next((i for i in selected if i != detection_index), detection_index)
                output.append(
                    SimpleNamespace(
                        track_id=track_id,
                        idx=selected.index(detection_index),
                        frame_id=frame + 1,
                        xyxy=rows[other]["bbox_xyxy"],
                    )
                )
            if frame in (2, 4):
                # A stale Kalman row is not a current observation or Pose source.
                output.append(SimpleNamespace(track_id=11, idx=99, frame_id=frame))
            scripts.append(output)

        class ScriptedByteTrack:
            frame_id = 0
            tracked_stracks = []
            lost_stracks = []

            def update(self, boxes):
                self.tracked_stracks = scripts[self.frame_id]
                self.frame_id += 1

        # Isolate the production adapter from NN/native association. Existing
        # TrackingTests above separately exercise the real ByteTrack runtime.
        adapter = PersonTracker.__new__(PersonTracker)
        adapter.tracker = ScriptedByteTrack()
        adapter.boxes_class = lambda data, orig_shape: data
        adapter.metrics = Metrics()
        adapter.max_lost = 3
        adapter.states = {}
        adapter.segment = 0
        adapter.max_observation_gap_s = None
        adapter.recovery_confirm_observations = 2
        return adapter, detections, expected

    def assert_association(self, result, detections, expected):
        self.assertEqual({track["track_id"] for track in result["tracks"]}, set(expected))
        for track in result["tracks"]:
            index = expected[track["track_id"]]
            self.assertEqual(track["detection_index"], index)
            for field in ("bbox_xyxy", "keypoints17", "keypoints13", "keypoints13_missing"):
                self.assertEqual(track[field], detections[index][field], field)

    def test_crossing_reordered_and_missing_observations_keep_matched_raw_pose(self):
        adapter, frames, expected = self.fixture()
        original = copy.deepcopy(frames)
        for frame, (detections, mapping) in enumerate(zip(frames, expected, strict=True)):
            with self.subTest(frame=frame):
                result = adapter.update(frame, frame / 30, detections)
                self.assert_association(result, detections, mapping)
                if frame == 2:
                    self.assertNotIn(11, {track["track_id"] for track in result["tracks"]})
                if frame == 1:
                    self.assertTrue(all(track["confirmed"] for track in result["tracks"]))
        self.assertEqual(frames, original, "Adapter must not rewrite raw input Pose")

    def test_crossing_oracle_rejects_list_order_and_nearest_box_pose_rematches(self):
        adapter, frames, expected = self.fixture()
        correct = adapter.update(0, 0, frames[0])
        self.assert_association(correct, frames[0], expected[0])
        selected = [row for row in frames[0] if row["class_id"] == 0]
        native = {track.track_id: track for track in adapter.tracker.tracked_stracks}
        for matcher in ("list_order", "nearest_box"):
            bad = copy.deepcopy(correct)
            for position, track in enumerate(bad["tracks"]):
                wrong = (
                    selected[position]
                    if matcher == "list_order"
                    else min(
                        selected,
                        key=lambda row: np.linalg.norm(
                            np.asarray(row["bbox_xyxy"]) - native[track["track_id"]].xyxy
                        ),
                    )
                )
                for field in ("keypoints17", "keypoints13", "keypoints13_missing"):
                    track[field] = copy.deepcopy(wrong[field])
            with self.subTest(matcher=matcher), self.assertRaises(AssertionError):
                self.assert_association(bad, frames[0], expected[0])


class RenderTests(unittest.TestCase):
    def test_track_colors_are_distinct_reproducible_and_not_frame_dependent(self):
        self.assertEqual(track_color(7), track_color(7))
        self.assertEqual(len({track_color(i) for i in range(1, 101)}), 100)
        for color in (track_color(i) for i in range(1, 10)):
            self.assertGreater(max(color), 150)

    def render(self, rows, gap=300, alpha=1):
        metrics = Metrics()
        result = list(render_tracks(iter(rows), 200, 100, alpha, gap, metrics))
        return result, metrics

    def test_linear_gap_exact_anchors_and_no_extrapolation(self):
        rows = [
            anchor(0, 10),
            anchor(1, 20),
            {**anchor(2), "tracks": []},
            anchor(3, 40),
            {**anchor(4), "tracks": []},
        ]
        out, metrics = self.render(rows)
        row = out[2]["tracks"][0]
        self.assertEqual(row["bbox_xyxy"], [30, 10, 70, 90])
        self.assertEqual(row["state"], "interpolated")
        self.assertIsNone(row["confidence"])
        self.assertEqual(row["anchor_frames"], [1, 3])
        self.assertEqual(out[4]["tracks"], [])
        self.assertEqual(metrics.counts["track_boxes_interpolated"], 1)

    def test_no_bridge_before_confirmation_across_cut_or_different_id(self):
        for final in (anchor(2, segment=1), anchor(2, track=2)):
            out, _ = self.render([anchor(0), {**anchor(1), "tracks": []}, final])
            self.assertEqual(out[1]["tracks"], [])
        out, _ = self.render([anchor(0, confirmed=False), {**anchor(1), "tracks": []}, anchor(2)])
        self.assertEqual(out[0]["tracks"], [])
        self.assertEqual(out[1]["tracks"], [])

    def test_timestamp_boundary_fractional_fps_and_disabled_gap(self):
        for fps in (25, 30000 / 1001, 30):
            for delta, expected in ((0.299999, True), (0.3, True), (0.300001, False)):
                a, mid, b = anchor(0), anchor(1), anchor(2)
                mid.update(timestamp_seconds=1 / fps, tracks=[])
                b["timestamp_seconds"] = delta
                out, _ = self.render([a, mid, b])
                self.assertEqual(bool(out[1]["tracks"]), expected)
        out, _ = self.render([anchor(0), {**anchor(1), "tracks": []}, anchor(2)], gap=0)
        self.assertEqual(out[1]["tracks"], [])

    def test_large_jump_rejected_and_lookahead_history_bounded(self):
        out, metrics = self.render([anchor(0), {**anchor(1), "tracks": []}, anchor(2, 150)])
        self.assertEqual(out[1]["tracks"], [])
        self.assertEqual(metrics.counts["track_gaps_rejected"], 1)
        rows = ({**anchor(i, track=i), "timestamp_seconds": i / 30} for i in range(2000))
        for _ in render_tracks(rows, 200, 100, 0.65, 300, metrics):
            pass
        self.assertLessEqual(metrics.values["track_metadata_peak_frames"], 12)
        self.assertLessEqual(metrics.values["track_render_peak_history"], 12)


def pose_anchor(index, x=20, timestamp=None, **kwargs):
    row = anchor(index, **kwargs)
    if timestamp is not None:
        row["timestamp_seconds"] = timestamp
    row["tracks"][0].update(
        keypoints13=[[x, 30, 0.8] for _ in range(13)],
        keypoints17=[[x, 30, 0.8] for _ in range(17)],
    )
    return row


class PoseDisplayTests(unittest.TestCase):
    def render(self, rows, **kwargs):
        return list(render_tracks(rows, 200, 100, 0.65, 300, Metrics(), **kwargs))

    def test_display_pose_filters_head_jump_without_mutating_raw_points_or_scores(self):
        rows = [pose_anchor(0, 20), pose_anchor(1, 29)]
        rows[1]["tracks"][0]["keypoints13"][0][2] = 0.31
        original = copy.deepcopy(rows)
        shown = self.render(rows)[1]["tracks"][0]
        self.assertGreater(shown["display_keypoints13"][0][0], 20)
        self.assertLess(shown["display_keypoints13"][0][0], 29)
        self.assertEqual(shown["display_keypoints13"][0][2], 0.31)
        self.assertEqual(shown["keypoints13"], original[1]["tracks"][0]["keypoints13"])
        self.assertEqual(shown["keypoints17"], original[1]["tracks"][0]["keypoints17"])
        self.assertEqual(rows, original)
        self.assertEqual(
            shown["display_pose"],
            {"algorithm": "timestamp_ema", "tau_ms": 80, "smoothed": True},
        )

    def test_pose_response_uses_elapsed_source_time_for_variable_frame_intervals(self):
        split = self.render(
            [pose_anchor(0, 20, 0), pose_anchor(1, 40, 0.04), pose_anchor(2, 40, 0.08)]
        )
        single = self.render([pose_anchor(0, 20, 0), pose_anchor(1, 40, 0.08)])
        self.assertAlmostEqual(
            split[-1]["tracks"][0]["display_keypoints13"][0][0],
            single[-1]["tracks"][0]["display_keypoints13"][0][0],
        )

    def test_missing_joint_stays_missing_and_only_that_joint_restarts(self):
        rows = [pose_anchor(0, 20), pose_anchor(1, 30), pose_anchor(2, 40)]
        rows[1]["tracks"][0]["keypoints13"][0] = None
        shown = self.render(rows)
        self.assertIsNone(shown[1]["tracks"][0]["display_keypoints13"][0])
        points = shown[2]["tracks"][0]["display_keypoints13"]
        self.assertEqual(points[0], [40, 30, 0.8])
        self.assertLess(points[1][0], 40)

    def test_scene_time_gap_and_nonpositive_time_restart_pose_state(self):
        for final in (
            pose_anchor(1, 40, segment=1),
            pose_anchor(1, 40, timestamp=0.5),
            pose_anchor(1, 40, timestamp=0),
        ):
            with self.subTest(final=final):
                shown = self.render([pose_anchor(0, 20), final])[-1]["tracks"][0]
                self.assertEqual(shown["display_keypoints13"], shown["keypoints13"])
                self.assertFalse(shown["display_pose"]["smoothed"])

    def test_track_gap_resets_pose_and_interpolation_does_not_invent_joints(self):
        rows = [pose_anchor(0, 20), {**anchor(1), "tracks": []}, pose_anchor(2, 40)]
        shown = self.render(rows)
        self.assertEqual(shown[1]["tracks"][0]["state"], "interpolated")
        self.assertNotIn("display_keypoints13", shown[1]["tracks"][0])
        self.assertNotIn("keypoints13", shown[1]["tracks"][0])
        last = shown[2]["tracks"][0]
        self.assertEqual(last["display_keypoints13"], last["keypoints13"])

    def test_ids_have_independent_pose_histories(self):
        first, second = pose_anchor(0, 20), pose_anchor(1, 40)
        first["tracks"].extend(pose_anchor(0, 80, track=2)["tracks"])
        second["tracks"].extend(pose_anchor(1, 60, track=2)["tracks"])
        tracks = self.render([first, second])[1]["tracks"]
        self.assertLess(tracks[0]["display_keypoints13"][0][0], 40)
        self.assertGreater(tracks[1]["display_keypoints13"][0][0], 60)

    def test_zero_tau_disables_filter_and_invalid_tau_is_rejected(self):
        shown = self.render([pose_anchor(0), pose_anchor(1, 40)], pose_tau_ms=0)[1]["tracks"][0]
        self.assertEqual(shown["display_keypoints13"], shown["keypoints13"])
        self.assertFalse(shown["display_pose"]["smoothed"])
        for tau in (-1, float("nan"), float("inf")):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                self.render([pose_anchor(0)], pose_tau_ms=tau)


if __name__ == "__main__":
    unittest.main()
