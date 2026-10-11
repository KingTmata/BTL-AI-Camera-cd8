"""Measure identity-view clipping against native predictions on the full val split."""
import csv
import datetime
import json
import traceback
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from kaggle_tools.experiment_campaign import file_sha
from kaggle_tools.validation_replay import CacheDetectionValidator, replay_metrics


class CanvasCache(CacheDetectionValidator):
    def update_metrics(self, preds, batch):
        start = len(self.cached_samples)
        super().update_metrics(preds, batch)
        for sample in self.cached_samples[start:]:
            sample['canvas_hw'] = tuple(batch['img'].shape[-2:])


def scores(metrics):
    return dict(mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                per_class={metrics.names[int(cls)]: dict(AP50=float(metrics.box.ap50[j]),
                    AP50_95=float(metrics.box.ap[j]))
                    for j, cls in enumerate(metrics.box.ap_class_index)})


def differences(left, right):
    result = {key: right[key] - left[key]
              for key in ('mAP50', 'mAP50_95', 'precision', 'recall')}
    result['per_class'] = {name: {key: right['per_class'][name][key] - value
        for key, value in values.items()} for name, values in left['per_class'].items()}
    return result


if __name__ == '__main__':
    root = Path.cwd().resolve()
    selection_path = root / '.temp/single_best_selection_20261010.json'
    selection = json.loads(selection_path.read_text())
    checkpoint = Path(selection['checkpoint'])
    protocol = dict(selection['protocol'])
    assert ultralytics.__version__ == '8.4.165'
    assert protocol['device'] == 'cpu' and protocol['quantize'] is None
    assert not protocol['nms'] and not protocol['augment']
    assert file_sha(checkpoint) == selection['checkpoint_sha256']
    data = yaml.safe_load(Path(protocol['data']).read_text())
    dataset = Path(data['path'])
    assert 'test' not in data
    assert file_sha(dataset / 'manifest.csv') == selection['manifest_sha256']
    with (dataset / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['split'] == 'val']
    assert len(rows) == 2500 and sum(int(row['boxes']) for row in rows) == 8359
    for row in rows:
        for path_key, hash_key in [('image_path', 'image_sha256'), ('label_path', 'label_sha256')]:
            assert file_sha(dataset / row[path_key]) == row[hash_key]
    torch.set_num_threads(4)
    folder = root / '.temp' / ('identity_clip_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(exist_ok=False)
    record = dict(status='running', scope='Full val identity clipping audit; CPU FP32, no test',
                  source_sha256=file_sha(__file__), checkpoint_sha256=file_sha(checkpoint),
                  manifest_sha256=selection['manifest_sha256'], protocol=protocol,
                  validation_images=2500, validation_boxes=8359, published_model_replaced=False)

    def log(title):
        with (root / 'reports/PROJECT8_V05_LOG.md').open('a', encoding='utf-8') as stream:
            stream.write('\n\n### ' + title + '\n\n```json\n' + json.dumps(record, indent=2) + '\n```\n')
        print(title, json.dumps(record), flush=True)

    log('Identity clipping — full validation bắt đầu')
    try:
        model = YOLO(str(checkpoint))
        native = model.val(validator=CanvasCache, project=str(folder), name='native', **protocol)
        samples = model.validator.cached_samples
        assert len(samples) == len({s['im_name'] for s in samples}) == 2500
        assert sum(len(s['target_cls']) for s in samples) == 8359
        native_scores = scores(native)
        replay_native = scores(replay_metrics(samples, native.names))
        parity = differences(native_scores, replay_native)
        assert all(abs(parity[key]) <= 1e-6 for key in ('mAP50', 'mAP50_95', 'precision', 'recall'))
        assert all(abs(value) <= 1e-6 for values in parity['per_class'].values() for value in values.values())
        original = scores(replay_metrics(samples, native.names, iou=.6))
        changed_images = changed_rows = 0
        for sample in samples:
            height, width = sample['canvas_hw']
            before = sample['predictions'][:, :4].clone()
            sample['predictions'][:, [0, 2]] = before[:, [0, 2]].clamp(0, width)
            sample['predictions'][:, [1, 3]] = before[:, [1, 3]].clamp(0, height)
            count = int((before != sample['predictions'][:, :4]).any(dim=1).sum())
            changed_images += count > 0
            changed_rows += count
        clipped = scores(replay_metrics(samples, native.names, iou=.6))
        delta = differences(original, clipped)
        all_deltas = [delta[k] for k in ('mAP50', 'mAP50_95', 'precision', 'recall')]
        all_deltas += [value for values in delta['per_class'].values() for value in values.values()]
        record.update(status='PASS', native=native_scores, native_replay_parity=parity,
                      extra_nms_06_unclipped=original, extra_nms_06_identity_clipped=clipped,
                      clipped_minus_unclipped=delta, changed_images=changed_images,
                      changed_prediction_rows=changed_rows, equivalent_at_1e6=all(abs(v) <= 1e-6 for v in all_deltas),
                      raw_cache_saved=False)
    except Exception:
        record.update(status='ERROR', error=traceback.format_exc())
    finally:
        assert file_sha(checkpoint) == selection['checkpoint_sha256']
        assert json.loads(selection_path.read_text()) == selection
        assert len(list(checkpoint.parent.glob('*.pt'))) == 1
        log('Identity clipping — full validation kết thúc')
    if record['status'] != 'PASS':
        raise RuntimeError(record.get('error', 'Probe failed'))
