import unittest

import torch

from kaggle_tools.validation_replay import replay_metrics


class ValidationReplayTests(unittest.TestCase):
    def sample(self):
        # Two real overlapping objects, plus a duplicate of the first object.
        return dict(im_name='two_objects.jpg',
                    target_boxes=torch.tensor([[0., 0., 10., 10.], [2., 0., 12., 10.]]),
                    target_cls=torch.tensor([0., 0.]),
                    predictions=torch.tensor([[0., 0., 10., 10., .9, 0.],
                                              [0., 0., 10., 10., .8, 0.],
                                              [2., 0., 12., 10., .7, 0.]]))

    def test_looser_nms_retains_two_true_overlapping_objects(self):
        strict = replay_metrics([self.sample()], {0: 'chair'}, iou=.6)
        loose = replay_metrics([self.sample()], {0: 'chair'}, iou=.8)
        self.assertEqual(strict.nt_per_class.tolist(), [2])
        self.assertGreater(loose.box.map50, strict.box.map50)
        self.assertAlmostEqual(loose.box.map50, .995, places=6)
        self.assertGreater(loose.box.mr, strict.box.mr)

    def test_missing_predictions_still_count_ground_truth_and_empty_images(self):
        missing = self.sample()
        missing['predictions'] = torch.empty((0, 6))
        empty = dict(im_name='empty.jpg', target_boxes=torch.empty((0, 4)),
                     target_cls=torch.empty(0), predictions=torch.empty((0, 6)))
        metrics = replay_metrics([missing, empty], {0: 'chair'}, iou=.6)
        self.assertEqual(metrics.nt_per_class.tolist(), [2])
        self.assertEqual(metrics.box.map50, 0.)

    def test_rejects_duplicate_images_or_mismatched_ground_truth_shapes(self):
        for samples in ([self.sample(), self.sample()],
                        [dict(self.sample(), target_cls=torch.tensor([0.]))]):
            with self.assertRaises(ValueError):
                replay_metrics(samples, {0: 'chair'})


if __name__ == '__main__':
    unittest.main()
