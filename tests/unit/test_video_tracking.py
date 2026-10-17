"""Real ByteTrack association and bounded offline rendering contracts."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
