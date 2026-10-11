"""Capture restored TTA rows before NMS so thresholds can be replayed fairly."""
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from kaggle_tools.inference_variants import EndToEndViews, restore_boxes
from kaggle_tools.validation_replay import CacheDetectionValidator


class RawViewCollector(EndToEndViews):
    """Preserve ordinary merged output; retain only the last batch's raw CPU rows."""
    def __init__(self, models, *args, **kwargs):
        if not models or len({id(model) for model in models}) != len(models):
            raise ValueError('Raw collector requires distinct model instances')
        super().__init__(models, *args, **kwargs)
        self.last_raw_predictions = ()
        self._raw_groups = []
        self._call_index = 0
        for model in self.models:
            model.register_forward_hook(self._capture)

    def _capture(self, model, inputs, output):
        predictions = output[0] if isinstance(output, tuple) else output
        model_index = self._call_index % len(self.models)
        view_index = self._call_index // len(self.models)
        if view_index >= len(self.views) or model is not self.models[model_index]:
            raise ValueError('Native model/view capture order differs')
        scale, flipped = self.views[view_index]
        self._call_index += 1
        resized = tuple(max(1, int(size * scale)) for size in self._original_hw)
        if predictions.ndim != 3 or predictions.shape[0] != len(self._raw_groups) or predictions.shape[2] != 6:
            raise ValueError('Unexpected native rows in raw capture')
        for index, rows in enumerate(predictions):
            restored = restore_boxes(rows.float(), self._original_hw, resized, flipped)
            self._raw_groups[index].append(restored.detach().cpu().clone())

    def forward(self, images):
        self.last_raw_predictions = ()
        self._original_hw = tuple(images.shape[-2:])
        self._raw_groups = [[] for _ in images]
        self._call_index = 0
        output = super().forward(images)
        if self._call_index != len(self.views) * len(self.models):
            raise ValueError('Incomplete explicit view capture')
        self.last_raw_predictions = tuple(torch.cat(rows) for rows in self._raw_groups)
        self._raw_groups = []
        return output


class CacheViewsValidator(CacheDetectionValidator):
    """Score the .6 anchor normally while caching pre-merge rows with official GT."""
    def __init__(self, *args, views=((1., False), (.83, True), (.67, False)), companions=(), merge_iou=.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.views, self.companions, self.merge_iou = views, tuple(companions), merge_iou

    def get_model(self, model, trainer=None):
        if trainer is not None or self.args.nms or self.args.augment or self.args.agnostic_nms:
            raise ValueError('Raw view cache requires standalone native-head validation')
        models = [DetectionValidator.get_model(self, model, trainer)]
        models.extend(YOLO(str(path)).model for path in self.companions)
        self.collector = RawViewCollector(models, views=self.views, iou=self.merge_iou,
                                          max_det=self.args.max_det, conf=self.args.conf)
        return self.collector

    def update_metrics(self, preds, batch):
        raw = self.collector.last_raw_predictions
        if len(raw) != len(preds):
            raise ValueError('Raw capture batch order/count differs')
        start = len(self.cached_samples)
        super().update_metrics(preds, batch)
        for sample, detections in zip(self.cached_samples[start:], raw):
            sample['predictions'] = detections.clone()

    def init_metrics(self, model):
        # AutoBackend may copy a wrapper created inside inference_mode. Its
        # retained native model is the one whose hooks actually run.
        active = model.model if getattr(model, 'format', None) == 'pt' else model
        if not isinstance(active, RawViewCollector):
            raise ValueError('Expected active raw-view PyTorch backend')
        self.collector = active
        super().init_metrics(model)
