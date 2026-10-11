"""Explicit end-to-end YOLO26 TTA; keep the native head and class mapping."""
import math

import torch
import torch.nn.functional as F
from ultralytics.utils.nms import TorchNMS
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator


def _check_detections(detections):
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError('Expected N x 6 end-to-end detections')
    if not torch.isfinite(detections).all():
        raise ValueError('Nonfinite detections')


def restore_boxes(detections, original_hw, resized_hw, flipped=False):
    """Invert resize/flip before right/bottom padding using actual integer sizes."""
    _check_detections(detections)
    if any(not math.isfinite(v) or v <= 0 for v in (*original_hw, *resized_hw)):
        raise ValueError('Image dimensions must be positive and finite')
    restored = detections.clone()
    original_h, original_w = original_hw
    resized_h, resized_w = resized_hw
    if flipped:
        restored[:, [0, 2]] = resized_w - restored[:, [2, 0]]
    restored[:, [0, 2]] *= original_w / resized_w
    restored[:, [1, 3]] *= original_h / resized_h
    return restored


def merge_detections(views, iou=.6, max_det=300, conf=.001):
    """Class-aware NMS across all views, then apply the global detection limit."""
    if not views or not 0 <= iou <= 1 or not 0 <= conf <= 1 or max_det < 1:
        raise ValueError('Invalid merge configuration')
    for view in views:
        _check_detections(view)
    detections = torch.cat(views).float()
    detections = detections[detections[:, 4] > conf]
    selected = []
    for cls in detections[:, 5].unique():
        ids = (detections[:, 5] == cls).nonzero(as_tuple=False).flatten()
        keep = TorchNMS.nms(detections[ids, :4], detections[ids, 4], iou)
        selected.append(ids[keep])
    if not selected:
        return detections
    keep = torch.cat(selected)
    keep = keep[detections[keep, 4].argsort(descending=True)][:max_det]
    return detections[keep]


class EndToEndViews(torch.nn.Module):
    """Sequential native-head inference across explicit views and checkpoints.

    Accepts already loaded detection models. Output remains B x max_det x 6;
    zero-confidence padding is discarded by Ultralytics' native postprocessor.
    NMS runs in FP32 to avoid half-precision area overflow at large image sizes.
    """
    def __init__(self, models, views=((1., False),), iou=.6, max_det=300, conf=.001):
        super().__init__()
        if not models or any(not getattr(model, 'end2end', False) for model in models):
            raise ValueError('Expected native end-to-end detection models')
        self.names = dict(models[0].names)
        if any(dict(model.names) != self.names for model in models):
            raise ValueError('Ensemble class mappings differ')
        if not views or len(set(views)) != len(views):
            raise ValueError('Expected distinct explicit inference views')
        if any(not math.isfinite(scale) or scale <= 0 or not isinstance(flip, bool) for scale, flip in views):
            raise ValueError('Invalid inference view')
        if not 0 <= iou <= 1 or not 0 <= conf <= 1 or max_det < 1:
            raise ValueError('Invalid merge configuration')
        self.models = torch.nn.ModuleList(models)
        self.register_buffer('stride', torch.tensor([max(int(model.stride.max()) for model in models)]))
        self.yaml = dict(models[0].yaml)
        self.end2end = True
        self.views, self.iou, self.max_det, self.conf = tuple(views), iou, max_det, conf

    def fuse(self):
        for model in self.models:
            if hasattr(model, 'fuse'):
                model.fuse(verbose=False)
        return self

    def set_head_attr(self, **attributes):
        for model in self.models:
            if hasattr(model, 'set_head_attr'):
                model.set_head_attr(**attributes)

    def forward(self, images):
        original_hw = tuple(images.shape[-2:])
        collected = [[] for _ in images]
        stride = int(self.stride.max())
        for scale, flipped in self.views:
            resized_hw = tuple(max(1, int(size * scale)) for size in original_hw)
            view = images if resized_hw == original_hw else F.interpolate(
                images, size=resized_hw, mode='bilinear', align_corners=False)
            if flipped:
                view = view.flip(-1)
            pad_h, pad_w = (-resized_hw[0]) % stride, (-resized_hw[1]) % stride
            if pad_h or pad_w:
                view = F.pad(view, (0, pad_w, 0, pad_h), value=114 / 255)
            for model in self.models:
                predictions = model(view)
                predictions = predictions[0] if isinstance(predictions, tuple) else predictions
                if predictions.ndim != 3 or predictions.shape[0] != len(images) or predictions.shape[2] != 6:
                    raise ValueError('Unexpected native detection output')
                for index, detection in enumerate(predictions):
                    collected[index].append(restore_boxes(detection.float(), original_hw, resized_hw, flipped))
        output = images.new_zeros((len(images), self.max_det, 6), dtype=torch.float32)
        for index, views in enumerate(collected):
            merged = merge_detections(views, self.iou, self.max_det, self.conf)
            output[index, :len(merged)] = merged
        return output


class ExplicitViewsValidator(DetectionValidator):
    """Standalone validation using the ordinary dataloader and metric matcher."""
    def __init__(self, *args, views=((1., False),), companions=(), merge_iou=.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.views, self.companions, self.merge_iou = views, companions, merge_iou

    def get_model(self, model, trainer=None):
        if trainer is not None or self.args.nms or self.args.augment or self.args.agnostic_nms:
            raise ValueError('Explicit views require standalone native-head, class-aware validation')
        models = [super().get_model(model)]
        models.extend(YOLO(str(path)).model for path in self.companions)
        return EndToEndViews(models, self.views, self.merge_iou, self.args.max_det, self.args.conf)
