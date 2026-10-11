import copy
import unittest
from types import SimpleNamespace

import torch
from ultralytics.nn.autobackend import AutoBackend

from kaggle_tools.inference_variants import EndToEndViews
from kaggle_tools.validation_replay import replay_metrics
from kaggle_tools.view_replay import RawViewCollector, CacheViewsValidator


class RectangleDetector(torch.nn.Module):
    def __init__(self, fixed=False):
        super().__init__()
        self.register_buffer('stride', torch.tensor([32.]))
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.names, self.yaml, self.end2end = {0: 'object'}, {'channels': 3}, True
        self.fixed = fixed

    def forward(self, images):
        if self.fixed:
            raw = images.new_tensor([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0],
                                     [2, 0, 12, 10, .7, 0]])
            return raw.repeat(len(images), 1, 1), None
        rows = []
        for image in images:
            ys, xs = (image[0] > .9).nonzero(as_tuple=True)
            rows.append(torch.stack([xs.min(), ys.min(), xs.max()+1, ys.max()+1,
                                     self.weight*.9, self.weight*0]).reshape(1, 6))
        return torch.stack(rows), None


class ViewReplayTests(unittest.TestCase):
    def ensemble_models(self):
        models = [RectangleDetector(), RectangleDetector()]
        with torch.no_grad():
            models[1].weight.fill_(.5)
        return models

    def test_multimodel_rows_preserve_view_model_order_geometry_and_pre_cap_predictions(self):
        models = self.ensemble_models()
        views = ((1., False), (1., True), (.5, True))
        ordinary = EndToEndViews(copy.deepcopy(models), views=views, max_det=1)
        collector = RawViewCollector(models, views=views, max_det=1)
        with torch.inference_mode():
            torch.testing.assert_close(collector(self.images()), ordinary(self.images()))
        for raw, box in zip(collector.last_raw_predictions, [[24.,16.,72.,48.], [96.,64.,144.,96.]]):
            self.assertEqual(tuple(raw.shape), (6, 6))
            torch.testing.assert_close(raw[:, :4], torch.tensor([box]).repeat(6, 1))
            torch.testing.assert_close(raw[:, 4], torch.tensor([.9, .45]*3))
        old = [raw.clone() for raw in collector.last_raw_predictions]
        with torch.inference_mode():
            collector(self.images()[1:])
        self.assertEqual(len(collector.last_raw_predictions), 1)
        torch.testing.assert_close(collector.last_raw_predictions[0], old[1])

    def test_multimodel_backend_copy_caches_all_rows_with_unchanged_gt(self):
        with torch.inference_mode():
            original = RawViewCollector(self.ensemble_models(),
                views=((1., False), (1., True), (.5, True)), max_det=1)
            backend = AutoBackend(model=original, device=torch.device('cpu'), fuse=False, verbose=False)
            result = backend(self.images())
        self.assertIsNot(backend.backend.model, original)
        validator = CacheViewsValidator(args=dict(task='detect', plots=False, conf=.001, nms=False, max_det=1))
        validator.training, validator.device = False, torch.device('cpu')
        validator.data = {'val':'val'}
        validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(labels=[]))
        validator.init_metrics(backend)
        batch = dict(img=self.images(), batch_idx=torch.tensor([0,1]), cls=torch.zeros(2,1),
            bboxes=torch.tensor([[.25,.25,.25,.25],[.625,.625,.25,.25]]),
            ori_shape=[(128,192)]*2, ratio_pad=[((1.,1.),(0.,0.))]*2,
            im_file=['/val/first.jpg','/val/second.jpg'])
        validator.update_metrics(validator.postprocess(result), batch)
        self.assertEqual(len(validator.cached_samples), 2)
        for sample, box in zip(validator.cached_samples, [[24.,16.,72.,48.],[96.,64.,144.,96.]]):
            self.assertEqual(len(sample['predictions']), 6)
            torch.testing.assert_close(sample['target_boxes'], torch.tensor([box]))
        replay = replay_metrics(validator.cached_samples, {0:'object'}, iou=.6, max_det=1)
        self.assertAlmostEqual(replay.box.map50, .995, places=6)

    def test_multimodel_rejects_reusing_one_instance_with_duplicate_hooks(self):
        model = RectangleDetector()
        with self.assertRaisesRegex(ValueError, 'distinct model instances'):
            RawViewCollector([model, model])

    def test_validator_uses_the_actual_inference_backend_copy_and_preserves_targets(self):
        detector = RectangleDetector()
        with torch.inference_mode():
            original = RawViewCollector([detector], views=((1., False), (1., True), (.5, True)), max_det=1)
            backend = AutoBackend(model=original, device=torch.device('cpu'), fuse=False, verbose=False)
            result = backend(self.images())
        self.assertIsNot(backend.backend.model, original)
        validator = CacheViewsValidator(args=dict(task='detect', plots=False, conf=.001, nms=False, max_det=1))
        validator.collector = original
        validator.training, validator.device = False, torch.device('cpu')
        validator.data = {'val': 'val'}
        validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(labels=[]))
        validator.init_metrics(backend)
        self.assertIs(validator.collector, backend.backend.model)
        batch = dict(img=self.images(), batch_idx=torch.tensor([0, 1]), cls=torch.zeros(2, 1),
                     bboxes=torch.tensor([[.25, .25, .25, .25], [.625, .625, .25, .25]]),
                     ori_shape=[(128, 192)]*2, ratio_pad=[((1., 1.), (0., 0.))]*2,
                     im_file=['/val/first.jpg', '/val/second.jpg'])
        validator.update_metrics(validator.postprocess(result), batch)
        self.assertEqual(len(validator.cached_samples), 2)
        for sample, expected in zip(validator.cached_samples, [[24.,16.,72.,48.], [96.,64.,144.,96.]]):
            self.assertEqual(len(sample['predictions']), 3)
            torch.testing.assert_close(sample['target_boxes'], torch.tensor([expected]))
        replay = replay_metrics(validator.cached_samples, {0:'object'}, iou=.6)
        self.assertAlmostEqual(replay.box.map50, .995, places=6)

    def images(self):
        images = torch.zeros((2, 3, 128, 192))
        images[0, :, 16:48, 24:72] = 1
        images[1, :, 64:96, 96:144] = 1
        return images

    def test_capture_is_before_merge_limit_and_restores_each_image_canvas(self):
        wrapper = RawViewCollector([RectangleDetector()],
            views=((1., False), (1., True), (.5, True)), max_det=1)
        backend = AutoBackend(model=wrapper, device=torch.device('cpu'), fuse=False, verbose=False)
        with torch.inference_mode():
            result = backend(self.images())
        self.assertEqual(tuple(result.shape), (2, 1, 6))
        expected = [[24, 16, 72, 48, .9, 0], [96, 64, 144, 96, .9, 0]]
        self.assertEqual(len(wrapper.last_raw_predictions), 2)
        for index, raw in enumerate(wrapper.last_raw_predictions):
            self.assertEqual(raw.device.type, 'cpu')
            self.assertEqual(tuple(raw.shape), (3, 6))
            torch.testing.assert_close(raw, torch.tensor(expected[index]).repeat(3, 1).float())
            torch.testing.assert_close(result[index], raw[:1])

    def test_capture_does_not_change_inference_or_retain_previous_batch(self):
        ordinary = EndToEndViews([RectangleDetector()], views=((1., False), (1., True)))
        collector = RawViewCollector([RectangleDetector()], views=((1., False), (1., True)))
        with torch.inference_mode():
            torch.testing.assert_close(ordinary(self.images()), collector(self.images()))
            previous = collector.last_raw_predictions
            previous_values = [value.clone() for value in previous]
            collector(self.images()[1:])
        self.assertEqual(len(collector.last_raw_predictions), 1)
        for old, expected in zip(previous, previous_values):
            torch.testing.assert_close(old, expected)

    def test_raw_capture_can_recover_objects_removed_by_stricter_merge_and_cap(self):
        collector = RawViewCollector([RectangleDetector(fixed=True)], max_det=1)
        with torch.inference_mode():
            merged = collector(torch.zeros(1, 3, 32, 32))
        sample = dict(im_name='two_objects.jpg', target_boxes=torch.tensor([[0., 0., 10., 10.], [2., 0., 12., 10.]]),
                      target_cls=torch.zeros(2), predictions=collector.last_raw_predictions[0])
        strict = replay_metrics([sample], {0: 'object'}, iou=.6)
        loose = replay_metrics([sample], {0: 'object'}, iou=.8)
        self.assertEqual(int(merged[0, :, 4].gt(.001).sum()), 1)
        self.assertGreater(loose.box.map50, strict.box.map50)
        self.assertAlmostEqual(loose.box.map50, .995, places=6)


if __name__ == '__main__':
    unittest.main()
