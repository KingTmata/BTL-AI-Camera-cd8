"""Cache native validation predictions and replay actual Ultralytics metrics."""
from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator

from kaggle_tools.inference_variants import merge_detections


class CacheDetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_samples = []

    def update_metrics(self, preds, batch):
        if self.args.single_cls or self.args.nms or self.args.augment:
            raise ValueError('Cache requires native, unchanged class-aware predictions')
        for index, prediction in enumerate(preds):
            targets = self._prepare_batch(index, batch)
            detections = torch.cat((prediction['bboxes'], prediction['conf'][:, None], prediction['cls'][:, None]), dim=1)
            self.cached_samples.append(dict(im_name=Path(targets['im_file']).name,
                                            target_boxes=targets['bboxes'].detach().cpu().clone(),
                                            target_cls=targets['cls'].detach().cpu().clone(),
                                            predictions=detections.detach().cpu().clone()))
        super().update_metrics(preds, batch)


def replay_metrics(samples, names, iou=None, device='cpu', conf=.001, max_det=300):
    """Reuse the exact detection matcher/AP implementation; no model rerun.

    Targets and predictions both use the preprocessed batch canvas coordinates.
    A None IoU preserves native row order; other values perform class-aware NMS.
    """
    if not samples or set(names) != set(range(len(names))):
        raise ValueError('Missing samples or noncontiguous class mapping')
    validator = DetectionValidator(args=dict(plots=False, conf=conf, max_det=max_det))
    validator.metrics.names = dict(names)
    seen = set()
    for sample in samples:
        if sample['im_name'] in seen:
            raise ValueError('Duplicate cached image')
        seen.add(sample['im_name'])
        boxes, classes = sample['target_boxes'], sample['target_cls']
        predictions = sample['predictions']
        if boxes.ndim != 2 or boxes.shape[1] != 4 or classes.ndim != 1 or len(boxes) != len(classes):
            raise ValueError('Target shapes differ')
        if predictions.ndim != 2 or predictions.shape[1] != 6:
            raise ValueError('Expected N x 6 cached predictions')
        if not all(torch.isfinite(value).all() for value in (boxes, classes, predictions)):
            raise ValueError('Nonfinite cache values')
        for values in (classes, predictions[:, 5]):
            if ((values < 0) | (values >= len(names)) | (values != values.round())).any():
                raise ValueError('Cache class ID differs from model mapping')
        detections = predictions.to(device).float()
        if iou is None:
            detections = detections[detections[:, 4] > conf][:max_det]
        else:
            detections = merge_detections([detections], iou=iou, max_det=max_det, conf=conf)
        prediction = dict(bboxes=detections[:, :4], conf=detections[:, 4], cls=detections[:, 5])
        targets = dict(bboxes=boxes.to(device), cls=classes.to(device))
        target_cls = classes.cpu().numpy()
        validator.metrics.update_stats(dict(**validator._process_batch(prediction, targets),
                                            target_cls=target_cls, target_img=np.unique(target_cls),
                                            conf=detections[:, 4].cpu().numpy(),
                                            pred_cls=detections[:, 5].cpu().numpy(), im_name=sample['im_name']))
    validator.metrics.process(plot=False)
    return validator.metrics
