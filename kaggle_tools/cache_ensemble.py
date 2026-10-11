"""Merge native prediction caches only when their validation identity matches."""
import torch

from kaggle_tools.inference_variants import merge_detections


def ensemble_cached_predictions(caches, iou=.6, conf=.001, max_det=300):
    """Keep exact batch-canvas targets; reject different image order or geometry."""
    if len(caches) < 2:
        raise ValueError('An ensemble requires at least two caches')
    first = caches[0]
    if not first.get('manifest_sha256') or not isinstance(first.get('protocol'), dict):
        raise ValueError('Missing cache validation identity')
    names = first['names']
    if not names or set(names) != set(range(len(names))):
        raise ValueError('Invalid class mapping')
    image_ids = [sample['im_name'] for sample in first['samples']]
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError('Missing or duplicate cached images')
    for cache in caches[1:]:
        for key in ('manifest_sha256', 'protocol', 'names'):
            if cache.get(key) != first[key]:
                raise ValueError('Different cache ' + key)
        if [sample['im_name'] for sample in cache['samples']] != image_ids:
            raise ValueError('Different cached image order or membership')
    output = []
    for index, sample in enumerate(first['samples']):
        predictions = []
        for cache in caches:
            other = cache['samples'][index]
            for key in ('target_boxes', 'target_cls'):
                if not torch.equal(sample[key], other[key]):
                    raise ValueError('Different ground truth or batch canvas: ' + sample['im_name'])
            detections = other['predictions']
            if detections.ndim != 2 or detections.shape[1] != 6:
                raise ValueError('Expected N x 6 cached predictions')
            classes = detections[:, 5]
            if ((classes < 0) | (classes >= len(names)) | (classes != classes.round())).any():
                raise ValueError('Prediction class differs from mapping')
            predictions.append(detections)
        output.append(dict(im_name=sample['im_name'], target_boxes=sample['target_boxes'].clone(),
                           target_cls=sample['target_cls'].clone(),
                           predictions=merge_detections(predictions, iou=iou, conf=conf, max_det=max_det)))
    return output
