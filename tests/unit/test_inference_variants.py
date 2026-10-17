import unittest

import torch
from ultralytics.nn.autobackend import AutoBackend

from classroom_ai.inference.variants import EndToEndViews, merge_detections, restore_boxes


class BrightRectangleDetector(torch.nn.Module):
    """A real tensor detector with analytically known geometry, no mock calls."""

    def __init__(self):
        super().__init__()
        self.register_buffer("stride", torch.tensor([32.0]))
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.names = {0: "object"}
        self.yaml = {"channels": 3}
        self.end2end = True

    def forward(self, images):
        predictions = []
        for image in images:
            ys, xs = (image[0] > 0.9).nonzero(as_tuple=True)
            predictions.append(
                torch.stack(
                    [
                        xs.min(),
                        ys.min(),
                        xs.max() + 1,
                        ys.max() + 1,
                        self.weight * 0.9,
                        self.weight * 0,
                    ]
                ).reshape(1, 6)
            )
        return torch.stack(predictions), None


class InferenceGeometryTests(unittest.TestCase):
    def test_restores_asymmetric_flipped_box_after_integer_resize(self):
        # Original box [20, 10, 60, 30] in H=80, W=200.
        # Resize to H=39, W=99, then flip; padded canvas width is 128.
        prediction = torch.tensor([[69.3, 4.875, 89.1, 14.625, 0.9, 7]])
        original = prediction.clone()
        restored = restore_boxes(prediction, (80, 200), (39, 99), flipped=True)
        torch.testing.assert_close(restored, torch.tensor([[20.0, 10.0, 60.0, 30.0, 0.9, 7]]))
        torch.testing.assert_close(prediction, original)

    def test_class_aware_merge_suppresses_duplicates_before_global_limit(self):
        views = [
            torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0, 0.9, 0],
                    [0.0, 0.0, 10.0, 10.0, 0.8, 0],
                    [0.0, 0.0, 10.0, 10.0, 0.7, 1],
                    [20.0, 20.0, 30.0, 30.0, 0.6, 0],
                ]
            )
        ]
        merged = merge_detections(views, iou=0.6, max_det=3)
        torch.testing.assert_close(merged[:, 4], torch.tensor([0.9, 0.7, 0.6]))
        self.assertEqual(merged[:, 5].tolist(), [0, 1, 0])

    def test_empty_views_and_rejects_invalid_shapes_or_nonfinite_values(self):
        empty = torch.empty((0, 6))
        self.assertEqual(tuple(merge_detections([empty, empty]).shape), (0, 6))
        for invalid in [torch.zeros((3, 5)), torch.tensor([[0.0, 0.0, 1.0, 1.0, float("nan"), 0]])]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    merge_detections([invalid])
        with self.assertRaises(ValueError):
            restore_boxes(empty, (80, 200), (0, 99))

    def test_half_precision_large_boxes_do_not_overflow_iou_areas(self):
        view = torch.tensor(
            [
                [0.0, 0.0, 1000.0, 1000.0, 0.9, 0],
                [0.0, 0.0, 1000.0, 1000.0, 0.8, 0],
                [500.0, 500.0, 1200.0, 1200.0, 0.7, 0],
            ],
            dtype=torch.float16,
        )
        merged = merge_detections([view], iou=0.6)
        self.assertEqual(len(merged), 2)
        torch.testing.assert_close(merged[:, :4], view[[0, 2], :4].float())


class InferenceBackendTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros((2, 3, 128, 192))
        self.image[:, :, 16:48, 24:72] = 1

    def test_actual_backend_original_flip_and_scale_recover_one_box_per_image(self):
        wrapper = EndToEndViews(
            [BrightRectangleDetector()], views=((1.0, False), (1.0, True), (0.5, True)), max_det=3
        )
        backend = AutoBackend(model=wrapper, device=torch.device("cpu"), fuse=False, verbose=False)
        with torch.inference_mode():
            result = backend(self.image)
        self.assertTrue(backend.end2end)
        self.assertEqual(tuple(result.shape), (2, 3, 6))
        expected = torch.tensor([24.0, 16.0, 72.0, 48.0, 0.9, 0.0])
        for item in result:
            torch.testing.assert_close(item[0], expected)
            self.assertEqual(item[:, 4].gt(0.001).sum().item(), 1)

    def test_single_view_has_native_prediction_parity_and_ensemble_retains_classes(self):
        detector = BrightRectangleDetector()
        native = detector(self.image)[0]
        wrapper = EndToEndViews([detector], max_det=1)
        torch.testing.assert_close(wrapper(self.image), native)
        second = BrightRectangleDetector()
        ensemble = EndToEndViews([detector, second], max_det=2)
        merged = ensemble(self.image)
        torch.testing.assert_close(merged[:, :1], native)
        self.assertEqual(merged[:, 1, 4].tolist(), [0.0, 0.0])
        second.names = {0: "different"}
        with self.assertRaises(ValueError):
            EndToEndViews([detector, second])
        detector.end2end = False
        with self.assertRaises(ValueError):
            EndToEndViews([detector])


if __name__ == "__main__":
    unittest.main()
